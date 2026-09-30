"""DYN job granularity, ordered selection, and process cleanup."""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import Manager, active_children, get_context
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration._proposals import ProposalSequence
from epydemix.calibration._scheduler import DynamicScheduler


class Arguments:
    """An unlimited candidate sequence with an optional proposal failure."""

    def __init__(self, fail=False):
        self.fail = fail

    def __getitem__(self, index):
        if self.fail:
            raise IndexError("proposal failure")
        return (index,)


def every_third(index):
    """Accept every third proposal, making retries observable."""
    return {"accepted": index % 3 == 2, "params": [index]}


def fail_model(index):
    """Fail inside a worker, before completion accounting."""
    raise RuntimeError("model failure")


def synchronized_accept(index, barrier, later_finished):
    """Gate candidate 0 on candidate 1 reaching the end of its model evaluation."""
    barrier.wait(timeout=10)
    if index == 0:
        assert later_finished.wait(timeout=10)
    else:
        later_finished.set()
    return {"accepted": True, "params": [index]}


def test_dynamic_retries_inside_one_worker_job():
    """One worker job returns three accepted particles after nine evaluations."""
    with ProcessPoolExecutor(max_workers=1) as pool:
        with patch.object(pool, "submit", wraps=pool.submit) as submit:
            result = DynamicScheduler().run_until_n_accepted(
                pool, every_third, Arguments(), 3
            )
        assert submit.call_count == 1
    assert result["n_simulations"] == 9
    assert [r["params"][0] for r in result["accepted_results"]] == [2, 5, 8]


def test_dynamic_drains_surplus_and_keeps_earliest_candidate():
    """W > N must use both workers and retain ID 0 after draining both results.

    A barrier forces two physical evaluations for one retained particle. This
    fails if DYN caps in-flight work by remaining acceptances. Gating the model
    also delays ID 0, but the assertion does not depend on OS completion order;
    the retained ID must always be 0.
    """
    with Manager() as manager:
        barrier, finished = manager.Barrier(2), manager.Event()
        args = [(i, barrier, finished) for i in range(2)]
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as pool:
            result = DynamicScheduler().run_until_n_accepted(
                pool, synchronized_accept, args, 1
            )
    assert result["n_simulations"] == 2
    assert result["accepted_results"] == [{"accepted": True, "params": [0]}]


@pytest.mark.parametrize("failure", ["model", "proposal", "progress"])
def test_failure_drains_jobs_and_preserves_external_pool(failure):
    """Propagate callback errors, reap the manager, and leave the caller pool usable.

    In particular, IndexError from a proposal must not be mistaken for sequence
    exhaustion. No hard worker termination is required for these finite jobs.
    """

    def progress(*args):
        raise RuntimeError("progress failure")

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        list(pool.map(int, ["1", "2"]))
        children = {child.pid for child in active_children()}
        with pytest.raises((RuntimeError, IndexError), match=failure + " failure"):
            DynamicScheduler().run_until_n_accepted(
                pool,
                fail_model if failure == "model" else every_third,
                Arguments(fail=failure == "proposal"),
                3,
                progress=progress if failure == "progress" else None,
            )
        assert pool.submit(int, "7").result() == 7
        assert {child.pid for child in active_children()} == children


def test_explicit_physical_budget_and_expired_deadline():
    """Keep an explicitly requested physical cap even when N cannot be reached."""
    scheduler = DynamicScheduler()
    with ProcessPoolExecutor(max_workers=2) as pool:
        result = scheduler.run_until_n_accepted(
            pool, every_third, Arguments(), 10, max_simulations=5
        )
        assert result["n_simulations"] == 5
        assert result["accepted_results"] == [{"accepted": True, "params": [2]}]
        result = scheduler.run_until_n_accepted(
            pool,
            every_third,
            Arguments(),
            10,
            deadline=datetime.now() - timedelta(seconds=1),
        )
        assert result == {"accepted_results": [], "n_simulations": 0}


def test_indexed_candidates_preserve_previous_spawn_streams():
    """Random access must preserve the old sequential SeedSequence.spawn mapping."""
    entropy = [12, 34, 56, 78]
    candidates = ProposalSequence({"x": stats.norm()}, ["x"], entropy, True)
    children = np.random.SeedSequence(entropy).spawn(21)
    for index in (20, 0, 3):
        reference = np.random.default_rng(children[index])
        params, _, rng = candidates[index]
        assert params == [stats.norm().rvs(random_state=reference)]
        np.testing.assert_array_equal(rng.random(10), reference.random(10))


@pytest.mark.parametrize("cutoff", ["target", "budget", "deadline", "abort"])
def test_reporting_after_cutoff_counts_drained_evaluations(cutoff):
    """A combined report/claim must count in-flight work after stopping new work.

    Two workers have already claimed candidates when the cutoff is reached.
    Both completions must be counted, even though neither may receive another ID.
    This preserves physical evaluation accounting for DYN's surplus and failures.
    """
    from epydemix.calibration._scheduler import _SchedulerState

    state = _SchedulerState(
        target=1 if cutoff == "target" else 3,
        budget=2 if cutoff == "budget" else None,
        deadline=None,
    )
    assert state.report_and_claim() == 0
    assert state.report_and_claim() == 1
    if cutoff == "abort":
        state.abort()
    elif cutoff == "deadline":
        state.deadline = datetime.now() - timedelta(seconds=1)
    assert state.report_and_claim(accepted=True) is None
    assert state.report_and_claim(accepted=False) is None
    assert state.snapshot() == (2, 2, 1, cutoff == "abort")
