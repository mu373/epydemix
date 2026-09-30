"""Scheduler queue bounds, ordering, partial results, and failure cleanup."""

from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from epydemix._execution import map_tasks
from epydemix.calibration._scheduler import (
    DynamicScheduler,
    SequentialScheduler,
    create_particle_scheduler,
    validate_parallel_strategy,
)


def test_batch_bounds_lazy_input_and_preserves_order():
    """Check lazy inputs stay within the queue bound and results keep input order."""
    consumed = 0
    retrieved = 0

    def arguments():
        nonlocal consumed
        for index in range(25):
            consumed += 1
            assert consumed - retrieved <= 4
            yield (index,)

    def submit(fn, index):
        future = Future()
        future.set_result(fn(index))
        original_result = future.result

        def result():
            nonlocal retrieved
            retrieved += 1
            return original_result()

        future.result = result
        return future

    # Completed futures make the queue bound deterministic, with no timing sleeps.
    with ProcessPoolExecutor(max_workers=2) as executor:
        with patch.object(executor, "submit", side_effect=submit):
            assert map_tasks(executor, str, arguments()) == [str(i) for i in range(25)]
    assert consumed == retrieved == 25


def test_fixed_progress_reports_completion_without_waiting_for_input_order():
    """A later candidate finishes first; progress still advances with aligned data.

    Controlled Futures model two in-flight candidates without timing sleeps. Returned
    parameter/distance/trajectory tuples stay in candidate order even though progress
    sees the second candidate first. This catches submission-count progress and ordered
    iteration hiding already-completed work behind a slow leading candidate.
    """
    submitted, updates = [], []
    values = [(10, 0.1, "trajectory-10"), (20, 0.2, "trajectory-20")]

    def submit(fn, value):
        future = Future()
        if submitted:
            future.set_result(value)
        submitted.append(future)
        return future

    def observe(completed, result):
        updates.append((completed, result))
        if completed == 1:
            submitted[0].set_result(values[0])

    with ProcessPoolExecutor(max_workers=1) as executor:
        with patch.object(executor, "submit", side_effect=submit):
            results = map_tasks(
                executor, tuple, ((value,) for value in values), on_completed=observe
            )
    assert results == values
    assert updates == [(1, values[1]), (2, values[0])]


@pytest.mark.parametrize("failure", ["worker", "input"])
def test_batch_cancels_pending_work_on_failure(failure):
    """Check pending tasks are cancelled after worker or input-iterator failure."""
    submitted = []

    def arguments():
        for index in range(20):
            if failure == "input" and index == 2:
                raise RuntimeError("bad input")
            yield (index,)

    def submit(*args):
        future = Future()
        if failure == "worker" and not submitted:
            future.set_exception(RuntimeError("bad worker"))
        submitted.append(future)
        return future

    with ProcessPoolExecutor(max_workers=2) as executor:
        with patch.object(executor, "submit", side_effect=submit):
            with pytest.raises(RuntimeError, match="bad " + failure):
                map_tasks(executor, str, arguments())
    assert len(submitted) == (4 if failure == "worker" else 2)
    assert all(
        future.cancelled() for future in submitted[1 if failure == "worker" else 0 :]
    )


def test_sequential_cutoffs_do_not_consume_extra_candidates():
    """The serial loop stays lazy, with no work after a cutoff.

    Local callbacks need no pickling. Progress is emitted after each evaluation;
    exhausted inputs and physical budgets return the accepted prefix unchanged.
    """
    consumed, progress = [], []

    def arguments():
        for index in range(4):
            consumed.append(index)
            yield (index,)

    def evaluate(index):
        return {"accepted": index % 2 == 0, "params": [index]}

    for options in (
        {"max_simulations": 0},
        {"deadline": datetime.now() - timedelta(seconds=1)},
    ):
        result = SequentialScheduler().run_until_n_accepted(
            evaluate, arguments(), 3, **options
        )
        assert result == {"accepted_results": [], "n_simulations": 0}
        assert consumed == []
    result = SequentialScheduler().run_until_n_accepted(
        evaluate,
        arguments(),
        3,
        max_simulations=3,
        progress=lambda completed, accepted: progress.append((completed, accepted)),
    )
    assert consumed == [0, 1, 2]
    assert progress == [(1, 1), (2, 1), (3, 2)]
    assert result == {
        "accepted_results": [evaluate(0), evaluate(2)],
        "n_simulations": 3,
    }
    exhausted = SequentialScheduler().run_until_n_accepted(evaluate, arguments(), 3)
    assert exhausted == {**result, "n_simulations": 4}


def test_strategy_name_selects_scheduler():
    """Select by executor availability while keeping name validation separate."""
    with patch("epydemix.calibration._scheduler.DynamicScheduler") as constructor:
        validate_parallel_strategy("dynamic")
        constructor.assert_not_called()
        assert isinstance(create_particle_scheduler(None), SequentialScheduler)
        constructor.assert_not_called()
    with ProcessPoolExecutor(max_workers=1) as pool:
        scheduler = create_particle_scheduler(pool, "dynamic")
        assert isinstance(scheduler, DynamicScheduler)
        assert scheduler.executor is pool
        for strategy in ("static", "non_speculative", "dyn", "unknown"):
            with pytest.raises(ValueError, match="Unknown parallel strategy"):
                create_particle_scheduler(pool, strategy)
    for strategy in ("static", "non_speculative", "dyn", "unknown"):
        with pytest.raises(ValueError, match="Unknown parallel strategy"):
            create_particle_scheduler(None, strategy)


def test_dynamic_scheduler_requires_executor():
    """DYN must never silently fall back to sequential execution."""
    with pytest.raises(ValueError, match="requires an executor"):
        DynamicScheduler(None)
