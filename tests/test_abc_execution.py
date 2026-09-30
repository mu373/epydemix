"""Public calibration execution: strategies, executors, budgets, and callbacks."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from epydemix import _execution
from epydemix._execution import (
    _get_available_cpu_count,
)
from epydemix.calibration.abc import ABCSampler
from tests.fixtures.calibration import (
    make_seeded_sampler,
)

_CPU_CAPACITY = _get_available_cpu_count()


def _mock_simulate(params):
    """Simple deterministic mock simulation."""
    beta = params.get("beta", 0.3)
    gamma = params.get("gamma", 0.1)
    return {"data": np.array([100 * np.exp(-beta * gamma * t) for t in range(10)])}


@pytest.fixture
def basic_abc_sampler():
    """Build a deterministic two-parameter sampler for execution tests."""
    priors = {
        "beta": stats.uniform(0.1, 0.5),
        "gamma": stats.uniform(0.05, 0.2),
    }
    return ABCSampler(
        simulation_function=_mock_simulate,
        priors=priors,
        parameters={"dt": 0.1},
        observed_data=np.array([90, 82, 75, 68, 62, 57, 52, 48, 44, 40]),
    )


class TestSMCParallel:
    """Smoke-test SMC output; exact parity is checked by TestParallelEquivalence."""

    @pytest.mark.parametrize("workers", [min(2, _CPU_CAPACITY), -1, -3, -6])
    def test_smc_parallel_basic(self, basic_abc_sampler, workers, monkeypatch):
        """Produce complete SMC generations using the available process capacity."""
        monkeypatch.setattr(
            _execution, "_get_available_cpu_count", lambda: min(2, _CPU_CAPACITY)
        )
        results = basic_abc_sampler.calibrate(
            strategy="smc",
            num_particles=10,
            num_generations=2,
            verbose=False,
            n_workers=workers,
            parallel_strategy="dynamic",
        )
        assert len(results.posterior_distributions) == 2
        posterior = results.get_posterior_distribution()
        assert "beta" in posterior.columns
        assert "gamma" in posterior.columns
        assert len(posterior) == 10


class TestRejectionParallel:
    """Smoke-test rejection output; exact parity is checked by TestParallelEquivalence."""

    @pytest.mark.parametrize("workers", [min(2, _CPU_CAPACITY), -1, -3, -6])
    def test_rejection_parallel_basic(self, basic_abc_sampler, workers, monkeypatch):
        """Produce the requested rejection sample using the available process capacity."""
        monkeypatch.setattr(
            _execution, "_get_available_cpu_count", lambda: min(2, _CPU_CAPACITY)
        )
        results = basic_abc_sampler.calibrate(
            strategy="rejection",
            epsilon=100.0,
            num_particles=10,
            verbose=False,
            n_workers=workers,
            parallel_strategy="dynamic",
        )
        assert len(results.posterior_distributions) == 1
        posterior = results.posterior_distributions[0]
        assert "beta" in posterior.columns
        assert "gamma" in posterior.columns
        assert len(posterior) == 10


class TestTopFractionParallel:
    """Smoke-test top-fraction output; exact parity is checked by TestParallelEquivalence."""

    @pytest.mark.parametrize("workers", [min(2, _CPU_CAPACITY), -1, -3, -6])
    def test_top_fraction_parallel_basic(self, basic_abc_sampler, workers, monkeypatch):
        """Select posterior samples from a fixed batch evaluated in a process pool."""
        monkeypatch.setattr(
            _execution, "_get_available_cpu_count", lambda: min(2, _CPU_CAPACITY)
        )
        results = basic_abc_sampler.calibrate(
            strategy="top_fraction",
            top_fraction=0.5,
            Nsim=20,
            verbose=False,
            n_workers=workers,
        )
        assert len(results.posterior_distributions) == 1
        posterior = results.posterior_distributions[0]
        assert "beta" in posterior.columns
        assert len(posterior) >= 1


class TestProjectionsParallel:
    """Smoke-test projections; exact parity is checked by TestParallelEquivalence."""

    @pytest.mark.parametrize("workers", [min(2, _CPU_CAPACITY), -1, -3, -6])
    def test_projections_parallel_basic(self, basic_abc_sampler, workers, monkeypatch):
        """Produce the requested projection trajectories using a process pool."""
        monkeypatch.setattr(
            _execution, "_get_available_cpu_count", lambda: min(2, _CPU_CAPACITY)
        )
        # First calibrate
        basic_abc_sampler.calibrate(
            strategy="rejection",
            epsilon=100.0,
            num_particles=10,
            verbose=False,
        )
        # Then project in parallel
        results = basic_abc_sampler.run_projections(
            parameters={"dt": 0.1},
            iterations=10,
            n_workers=workers,
        )
        assert "baseline" in results.projections
        assert len(results.projections["baseline"]) == 10


class TestSequentialExecution:
    """Check calibration with no worker pool requested."""

    @pytest.mark.parametrize(
        "strategy, options",
        [("smc", {"num_generations": 2}), ("rejection", {"epsilon": 100.0})],
    )
    def test_no_workers_runs_sequentially(self, basic_abc_sampler, strategy, options):
        """Sequential calibration must not construct or enter a DYN scheduler."""
        with patch(
            "epydemix.calibration._scheduler.DynamicScheduler",
            side_effect=AssertionError("Sequential calibration must not use DYN"),
        ):
            results = basic_abc_sampler.calibrate(
                strategy=strategy,
                num_particles=10,
                verbose=False,
                **options,
            )
        posterior = results.get_posterior_distribution()
        assert len(posterior) == 10


class TestInvalidStrategy:
    """Check validation of the scheduler strategy name."""

    @pytest.mark.parametrize(
        "parallel_strategy", ["dyn", "nonexistent", "static", "non_speculative"]
    )
    def test_invalid_parallel_strategy_raises(
        self, basic_abc_sampler, parallel_strategy
    ):
        """Reject unknown strategies, including the former abbreviated name."""
        with pytest.raises(ValueError, match="Unknown parallel strategy"):
            basic_abc_sampler.calibrate(
                strategy="rejection",
                epsilon=100.0,
                num_particles=10,
                verbose=False,
                n_workers=min(2, _CPU_CAPACITY),
                parallel_strategy=parallel_strategy,
            )


class TestCustomExecutor:
    """Check calibration through a caller-owned process pool."""

    def test_user_provided_executor(self, basic_abc_sampler):
        """User-provided executor should be used and NOT shut down."""
        with ProcessPoolExecutor(
            max_workers=min(2, _CPU_CAPACITY), mp_context=get_context("spawn")
        ) as executor:
            results = basic_abc_sampler.calibrate(
                strategy="rejection",
                epsilon=100.0,
                num_particles=10,
                verbose=False,
                executor=executor,
            )
            # Executor should still be usable after calibrate returns
            future = executor.submit(int, "42")
            assert future.result() == 42

        posterior = results.posterior_distributions[0]
        assert len(posterior) == 10


class TestStoppingConditions:
    """Check time and simulation-budget cutoffs during calibration."""

    def test_rejection_parallel_respects_budget(self, basic_abc_sampler):
        """Parallel rejection should stop when total_simulations_budget is hit."""
        results = basic_abc_sampler.calibrate(
            strategy="rejection",
            epsilon=0.001,  # Very strict; almost nothing accepted
            num_particles=10000,
            total_simulations_budget=20,
            verbose=False,
            n_workers=min(2, _CPU_CAPACITY),
        )
        # Should have stopped early, not run 10000 particles
        posterior = results.posterior_distributions[0]
        assert len(posterior) < 10000

    def test_rejection_parallel_respects_max_time(self, basic_abc_sampler):
        """Parallel rejection should stop when max_time is hit."""
        results = basic_abc_sampler.calibrate(
            strategy="rejection",
            epsilon=0.001,  # Very strict
            num_particles=10000,
            max_time=timedelta(milliseconds=100),
            verbose=False,
            n_workers=min(2, _CPU_CAPACITY),
        )
        posterior = results.posterior_distributions[0]
        assert len(posterior) < 10000

    def test_smc_parallel_respects_budget(self, basic_abc_sampler):
        """Parallel SMC should stop when total_simulations_budget is hit."""
        results = basic_abc_sampler.calibrate(
            strategy="smc",
            num_particles=10000,
            num_generations=5,
            total_simulations_budget=30,
            verbose=False,
            n_workers=min(2, _CPU_CAPACITY),
        )
        # Should have stopped early; either no generations completed or very few
        total_particles = sum(
            len(df) for df in results.posterior_distributions.values()
        )
        assert total_particles < 10000


class TestCumulativeSimCount:
    """n_simulations should accumulate across SMC generations."""

    @pytest.mark.parametrize("n_workers", [None, 1, 2, 4])
    def test_smc_cumulative_sim_count(self, n_workers, tmp_path):
        """Verify n_simulations accumulates rather than resetting each generation."""
        if n_workers is not None and n_workers > _CPU_CAPACITY:
            pytest.skip("Insufficient CPU capacity for this worker count")
        sampler = make_seeded_sampler(43, "argument-int")
        sampler.simulation_function = _recorded_simulate
        sampler.parameters["record_dir"] = str(tmp_path)
        # DYN surplus may leave too little budget to complete generation 1.
        results = sampler.calibrate(
            strategy="smc",
            num_particles=5,
            num_generations=4,
            epsilon_schedule=[float("inf")] * 4,
            total_simulations_budget=10,
            verbose=False,
            n_workers=n_workers,
        )
        assert len(list(tmp_path.iterdir())) == 10
        assert len(results.posterior_distributions) in (1, 2)
        if n_workers in (None, 1):
            assert len(results.posterior_distributions) == 2


def _recorded_simulate(params):
    """Record each evaluation in a unique file to expose extra calls or reused RNGs."""
    # Exclusive creation detects reused streams as well as counting actual calls.
    draw = params["rng"].random()
    with (Path(params["record_dir"]) / str(draw)).open("x"):
        pass
    return {"data": np.ones(8)}


@pytest.mark.parametrize("budget", [0, 13])
def test_budget_bounds_actual_simulations(budget, tmp_path):
    """Count simulation calls to check the budget, including a zero budget."""
    sampler = make_seeded_sampler(0, "argument-int")
    sampler.simulation_function = _recorded_simulate
    sampler.parameters["record_dir"] = str(tmp_path)
    with ProcessPoolExecutor(max_workers=min(4, _CPU_CAPACITY)) as executor:
        result = sampler.calibrate(
            strategy="rejection",
            num_particles=10,
            epsilon=-1,
            total_simulations_budget=budget,
            executor=executor,
            verbose=False,
        )
    assert len(list(tmp_path.iterdir())) == budget
    assert len(result.get_posterior_distribution()) == 0


def _unseeded_simulate(params):
    """Reject unexpected RNG injection and return a constant trajectory."""
    assert "rng" not in params
    return {"data": np.zeros(8)}


def test_unseeded_simulation_receives_no_rng():
    """Check that unseeded calibration does not inject an RNG into the simulation."""
    with ProcessPoolExecutor(max_workers=min(2, _CPU_CAPACITY)) as executor:
        for pool in (None, executor):
            ABCSampler(
                _unseeded_simulate, {"beta": stats.uniform()}, {}, np.zeros(8)
            ).calibrate(
                strategy="top_fraction",
                Nsim=4,
                executor=pool,
                verbose=False,
            )


@pytest.mark.parametrize(
    "strategy", ["smc", "rejection", "top_fraction", "projections"]
)
def test_public_api_rejects_thread_executor_before_work(strategy):
    """Reject thread pools before submission or RNG advancement, leaving them open."""
    sampler = make_seeded_sampler(43, "argument-int")
    state = sampler.rng.bit_generator.state
    with ThreadPoolExecutor(max_workers=2) as executor:
        with patch.object(executor, "submit") as submit:
            with pytest.raises(TypeError, match="ProcessPoolExecutor"):
                if strategy == "projections":
                    sampler.run_projections({}, executor=executor)
                else:
                    sampler.calibrate(
                        strategy=strategy, executor=executor, verbose=False
                    )
            submit.assert_not_called()
        assert executor.submit(int, "42").result() == 42
    assert sampler.rng.bit_generator.state == state


def _distance_with_positional_names(observed, predicted, /):
    """Return the threshold exactly; positional-only args reject keyword calls."""
    return 1.0


@pytest.mark.parametrize("mode", ["sequential", "owned", "spawn"])
@pytest.mark.parametrize("strategy", ["rejection", "smc", "top_fraction"])
def test_distance_callback_and_upstream_boundary_rules(mode, strategy):
    """Check positional callback invocation and equality at acceptance thresholds.

    Every candidate has distance 1.0. Rejection and later SMC generations
    exclude equality; the initial SMC generation includes it. Top-fraction
    selection keeps all candidates tied at its quantile threshold.
    """
    sampler = ABCSampler(
        _mock_simulate,
        {"beta": stats.uniform(0.1, 0.5)},
        {},
        np.zeros(10),
        distance_function=_distance_with_positional_names,
        rng=43,
    )
    # The budget exceeds 3 so SMC can attempt generation 1, and bounds runs
    # that cannot accept any further candidates.
    options = {
        "rejection": dict(num_particles=3, epsilon=1.0, total_simulations_budget=7),
        "smc": dict(
            num_particles=3,
            num_generations=2,
            epsilon_schedule=[1.0, 1.0],
            total_simulations_budget=7,
        ),
        "top_fraction": dict(Nsim=3, top_fraction=0.5),
    }[strategy]
    context = (
        ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
        if mode == "spawn"
        else nullcontext()
    )
    with context as executor:
        result = sampler.calibrate(
            strategy=strategy,
            verbose=False,
            executor=executor,
            n_workers=1 if mode == "owned" else None,
            **options,
        )
    assert list(result.posterior_distributions) == [0]
    assert len(result.get_posterior_distribution()) == (
        0 if strategy == "rejection" else 3
    )


@pytest.mark.parametrize("method", ["top_fraction", "projections"])
def test_fixed_batches_reject_acceptance_strategy(basic_abc_sampler, method):
    """Fixed simulation counts do not accept an ABC acceptance-scheduler option."""
    with pytest.raises(TypeError, match="parallel_strategy"):
        if method == "projections":
            basic_abc_sampler.run_projections({}, parallel_strategy="dynamic")
        else:
            basic_abc_sampler.calibrate(
                strategy="top_fraction", parallel_strategy="dynamic"
            )


@pytest.mark.parametrize(
    "counts, completed, accepted",
    [({"n_accepted": 2}, 6, 2), ({"n_evaluations": 5}, 5, 1)],
)
def test_shared_evaluation_stops_on_acceptances_or_evaluations(
    counts, completed, accepted
):
    """Fixed counts must include rejected evaluations; acceptance targets retry."""
    from epydemix.calibration._evaluate import run_particle_evaluations

    calls, progress = [], []

    def simulate(params):
        calls.append(params["theta"])
        return {"data": len(calls) % 3}

    def distance(data, simulation):
        return simulation["data"]

    result = run_particle_evaluations(
        (simulate, {}, ["theta"], None, distance),
        {"theta": stats.uniform()},
        np.random.default_rng(43),
        True,
        epsilon=0.5,
        progress=lambda done, kept: progress.append((done, kept)),
        **counts,
    )
    assert len(calls) == result["n_simulations"] == completed
    assert len(result["accepted_results"]) == accepted
    assert all(item["accepted"] for item in result["accepted_results"])
    assert progress[-1] == (completed, accepted)


@pytest.mark.parametrize(
    "counts, message",
    [
        ({}, "exactly one"),
        ({"n_accepted": 1, "n_evaluations": 1}, "exactly one"),
        ({"n_accepted": 0}, "positive"),
        ({"n_evaluations": 0}, "positive"),
    ],
)
def test_shared_evaluation_rejects_invalid_counts_before_rng_advances(counts, message):
    from epydemix.calibration._evaluate import run_particle_evaluations

    rng = np.random.default_rng(43)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match=message):
        run_particle_evaluations(None, None, rng, True, **counts)
    assert rng.bit_generator.state == state
