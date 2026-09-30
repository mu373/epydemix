"""Ensemble execution preserves trial streams, isolation, and pool ownership."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from epydemix import _execution
from epydemix.model.predefined_models import create_sir
from epydemix.utils.random_utils import rng_for_index


class _TrialRate:
    """Detect leaked callback state and nested native threads inside a simulation."""

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def __call__(self, params, data):
        assert self.calls == data["t"]
        self.calls += 1
        assert {pool["num_threads"] for pool in threadpool_info()} == {1}
        if self.fail:
            raise ValueError("expected transition failure")
        return data["parameters"][params][data["t"]]


def _model(fail=False):
    model = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    model.register_transition_kind("spontaneous", _TrialRate(fail))
    return model


def _run(model, **options):
    return model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict={
            "Susceptible": np.array([99000]),
            "Infected": np.array([1000]),
            "Recovered": np.array([0]),
        },
        **options,
    )


def _assert_equal(left, right):
    assert left.Nsim == right.Nsim
    assert left.parameters == right.parameters
    for a, b in zip(left.trajectories, right.trajectories):
        np.testing.assert_array_equal(a.dates, b.dates)
        assert a.compartment_idx == b.compartment_idx
        assert a.transitions_idx == b.transitions_idx
        for field in ("compartments", "transitions", "parameters"):
            aa, bb = getattr(a, field), getattr(b, field)
            assert aa.keys() == bb.keys()
            for key in aa:
                np.testing.assert_array_equal(aa[key], bb[key])


@pytest.mark.parametrize("workers", [1, 2, -1, -3, -6])
def test_trial_results_and_parent_rng_match_sequential(workers, monkeypatch):
    capacity = min(2, _execution._get_available_cpu_count())
    monkeypatch.setattr(_execution, "_get_available_cpu_count", lambda: capacity)
    if workers > capacity:
        pytest.skip("CPU capacity too small")
    left_rng, right_rng = (np.random.default_rng(43) for _ in range(2))
    serial, parallel = _model(), _model()
    # Repeated calls must advance the supplied generators in the same way.
    for _ in range(2):
        previous_state = left_rng.bit_generator.state
        expected = _run(serial, Nsim=5, rng=left_rng)
        actual = _run(parallel, Nsim=5, rng=right_rng, n_workers=workers)
        _assert_equal(expected, actual)
        assert left_rng.bit_generator.state == right_rng.bit_generator.state
        assert left_rng.bit_generator.state != previous_state
    assert serial.transition_functions["spontaneous"].calls == 0
    assert parallel.transition_functions["spontaneous"].calls == 0


def _thread_counts():
    return [pool["num_threads"] for pool in threadpool_info()]


def test_spawn_pool_survives_success_and_failure_and_restores_threads():
    with threadpool_limits(limits=2):
        before = threadpool_info()
        expected = _run(_model(), Nsim=4, rng=43)
        with ProcessPoolExecutor(
            max_workers=1, mp_context=get_context("spawn")
        ) as pool:
            worker_before = pool.submit(_thread_counts).result()
            actual = _run(_model(), Nsim=4, rng=43, executor=pool, n_workers=999)
            _assert_equal(expected, actual)
            with pytest.raises(RuntimeError, match="expected transition failure"):
                _run(_model(fail=True), Nsim=4, rng=43, executor=pool)
            assert pool.submit(int, "7").result() == 7
            assert pool.submit(_thread_counts).result() == worker_before
        with pytest.raises(RuntimeError, match="expected transition failure"):
            _run(_model(fail=True), Nsim=2, rng=43)
        assert threadpool_info() == before


def test_invalid_execution_and_empty_ensemble_do_not_advance_rng(monkeypatch):
    monkeypatch.setattr(_execution, "_get_available_cpu_count", lambda: 1)
    model, rng = _model(), np.random.default_rng(43)
    state = rng.bit_generator.state
    contacts = model.Cs.copy()
    with ThreadPoolExecutor(1) as threads, ProcessPoolExecutor(2) as oversized:
        for options in (
            {"Nsim": -1},
            {"Nsim": True},
            {"Nsim": 1.5},
            {"n_workers": 0},
            {"n_workers": 2},
            {"executor": threads},
            {"executor": oversized},
        ):
            with pytest.raises((TypeError, ValueError)):
                _run(model, rng=rng, **options)
    assert _run(model, Nsim=0, rng=rng).Nsim == 0
    assert rng.bit_generator.state == state
    assert model.Cs == contacts


def test_indexed_rng_matches_spawn_without_mutating_parent():
    parent = np.random.SeedSequence(43, spawn_key=(7, 3), pool_size=8)
    children = parent.spawn(5)
    state = parent.state.copy()
    # Random access must not depend on which indices were already requested.
    for index in (4, 0, 2, 4):
        np.testing.assert_array_equal(
            rng_for_index(parent, index).integers(2**32, size=20),
            np.random.default_rng(children[index]).integers(2**32, size=20),
        )
    assert parent.state == state
