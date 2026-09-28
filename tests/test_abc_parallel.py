"""Tests for parallel ABC calibration (DYN scheduling)."""

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from itertools import count
from multiprocessing import get_context
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from scipy import stats

from epydemix.calibration.abc import ABCSampler
from epydemix.calibration.parallel import DynamicParticleScheduler, _available_cpu_count
from tests.fixtures.calibration import assert_exact_calibration, make_seeded_sampler

_CPU_CAPACITY = _available_cpu_count()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_simulate(params):
    """Simple deterministic mock simulation."""
    beta = params.get("beta", 0.3)
    gamma = params.get("gamma", 0.1)
    return {"data": np.array([100 * np.exp(-beta * gamma * t) for t in range(10)])}


@pytest.fixture
def basic_abc_sampler():
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


# ---------------------------------------------------------------------------
# DynamicParticleScheduler unit tests
# ---------------------------------------------------------------------------


class TestDynOrdering:
    """Verify DYN ordering correctness: results counted from contiguous prefix only."""

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dyn_ordering_respects_submission_order(self, parallel):
        """Fast-completing tasks should not get preferential acceptance
        over earlier-submitted tasks that haven't completed yet."""
        sampler = DynamicParticleScheduler(queue_factor=2)

        def evaluate(idx):
            # All tasks accepted
            return {
                "params": [idx],
                "distance": 0.5,
                "simulation": {"data": np.array([idx])},
                "accepted": True,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.collect_particles(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=5,
            )

        accepted = result["accepted_results"]
        assert len(accepted) == 5
        # Verify results are in submission order
        indices = [r["params"][0] for r in accepted]
        assert indices == sorted(indices)

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dyn_stops_at_n_target(self, parallel):
        """Should stop once n_target accepted particles in contiguous prefix."""
        sampler = DynamicParticleScheduler(queue_factor=1)

        def evaluate(idx):
            # Alternate accepted/rejected
            accepted = idx % 2 == 0
            return {
                "params": [idx],
                "distance": 0.5 if accepted else 999.0,
                "simulation": {"data": np.array([idx])} if accepted else None,
                "accepted": accepted,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.collect_particles(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=3,
            )

        assert len(result["accepted_results"]) == 3


# ---------------------------------------------------------------------------
# Integration tests: parallel execution with n_workers
# ---------------------------------------------------------------------------


class TestSMCParallel:
    def test_smc_parallel_basic(self, basic_abc_sampler):
        """SMC with n_workers=2 should produce valid results."""
        results = basic_abc_sampler.calibrate(
            strategy="smc",
            num_particles=10,
            num_generations=2,
            verbose=False,
            n_workers=min(2, _CPU_CAPACITY),
        )
        assert len(results.posterior_distributions) == 2
        posterior = results.get_posterior_distribution()
        assert "beta" in posterior.columns
        assert "gamma" in posterior.columns
        assert len(posterior) == 10


class TestRejectionParallel:
    def test_rejection_parallel_basic(self, basic_abc_sampler):
        """Rejection with n_workers=2 should produce valid results."""
        results = basic_abc_sampler.calibrate(
            strategy="rejection",
            epsilon=100.0,
            num_particles=10,
            verbose=False,
            n_workers=min(2, _CPU_CAPACITY),
        )
        assert len(results.posterior_distributions) == 1
        posterior = results.posterior_distributions[0]
        assert "beta" in posterior.columns
        assert "gamma" in posterior.columns
        assert len(posterior) == 10


class TestTopFractionParallel:
    def test_top_fraction_parallel_basic(self, basic_abc_sampler):
        """Top fraction with n_workers=2 should produce valid results."""
        results = basic_abc_sampler.calibrate(
            strategy="top_fraction",
            top_fraction=0.5,
            Nsim=20,
            verbose=False,
            n_workers=min(2, _CPU_CAPACITY),
        )
        assert len(results.posterior_distributions) == 1
        posterior = results.posterior_distributions[0]
        assert "beta" in posterior.columns
        assert len(posterior) >= 1


class TestProjectionsParallel:
    def test_projections_parallel_basic(self, basic_abc_sampler):
        """Projections with n_workers=2 should produce valid results."""
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
            n_workers=min(2, _CPU_CAPACITY),
        )
        assert "baseline" in results.projections
        assert len(results.projections["baseline"]) == 10


# ---------------------------------------------------------------------------
# Sequential fallback + error handling
# ---------------------------------------------------------------------------


class TestSequentialFallback:
    def test_no_workers_runs_sequentially(self, basic_abc_sampler):
        """n_workers=None should give identical behavior to current code."""
        results = basic_abc_sampler.calibrate(
            strategy="rejection",
            epsilon=100.0,
            num_particles=10,
            verbose=False,
        )
        posterior = results.posterior_distributions[0]
        assert len(posterior) == 10


class TestInvalidStrategy:
    def test_invalid_parallel_strategy_raises(self, basic_abc_sampler):
        """Unknown parallel_strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown parallel strategy"):
            basic_abc_sampler.calibrate(
                strategy="rejection",
                epsilon=100.0,
                num_particles=10,
                verbose=False,
                n_workers=min(2, _CPU_CAPACITY),
                parallel_strategy="nonexistent",
            )


class TestCustomExecutor:
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
    def test_rejection_parallel_respects_budget(self, basic_abc_sampler):
        """Parallel rejection should stop when total_simulations_budget is hit."""
        results = basic_abc_sampler.calibrate(
            strategy="rejection",
            epsilon=0.001,  # Very strict — almost nothing accepted
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
        # Should have stopped early — either no generations completed or very few
        total_particles = sum(
            len(df) for df in results.posterior_distributions.values()
        )
        assert total_particles < 10000


class TestDynPartialResults:
    """DynamicParticleScheduler should return partial results on budget/deadline cutoff."""

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dyn_returns_partial_on_budget(self, parallel):
        """When max_simulations < n_target, accepted particles should not be lost."""
        sampler = DynamicParticleScheduler(queue_factor=1)

        def evaluate(idx):
            return {
                "params": [idx],
                "distance": 0.5,
                "simulation": {"data": np.array([idx])},
                "accepted": True,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.collect_particles(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=100,
                max_simulations=5,
            )

        accepted = result["accepted_results"]
        # Budget only allows 5 sims, but all are accepted — should get them back
        assert len(accepted) > 0
        assert len(accepted) <= 5

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dyn_returns_partial_on_mixed_results(self, parallel):
        """Budget cutoff with some rejected particles still returns accepted ones."""
        sampler = DynamicParticleScheduler(queue_factor=1)

        def evaluate(idx):
            accepted = idx % 2 == 0  # Every other particle rejected
            return {
                "params": [idx],
                "distance": 0.5 if accepted else 999.0,
                "simulation": {"data": np.array([idx])} if accepted else None,
                "accepted": accepted,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.collect_particles(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=100,
                max_simulations=6,
            )

        accepted = result["accepted_results"]
        # 6 sims, indices 0-5, accepted are 0,2,4 = 3 particles
        assert len(accepted) > 0
        assert all(r["accepted"] for r in accepted)


class TestCumulativeSimCount:
    """n_simulations should accumulate across SMC generations."""

    @pytest.mark.parametrize("n_workers", [None, 1, 2, 4])
    def test_smc_cumulative_sim_count(self, n_workers):
        """Verify n_simulations accumulates rather than resetting each generation."""
        if n_workers is not None and n_workers > _CPU_CAPACITY:
            pytest.skip("Insufficient CPU capacity for this worker count")
        priors = {
            "beta": stats.uniform(0.1, 0.5),
            "gamma": stats.uniform(0.05, 0.2),
        }
        sampler = ABCSampler(
            simulation_function=_mock_simulate,
            priors=priors,
            parameters={"dt": 0.1},
            observed_data=np.array([90, 82, 75, 68, 62, 57, 52, 48, 44, 40]),
        )
        # Exactly two generations fit, irrespective of queue size or completion order.
        results = sampler.calibrate(
            strategy="smc",
            num_particles=5,
            num_generations=4,
            epsilon_schedule=[float("inf")] * 4,
            total_simulations_budget=10,
            verbose=False,
            n_workers=n_workers,
        )
        # Should have completed 2 generations
        assert len(results.posterior_distributions) == 2


@pytest.mark.parametrize(
    "source",
    [
        "argument-int",
        "argument-generator",
        "parameters-int",
        "parameters-generator",
    ],
)
@pytest.mark.parametrize(
    "strategy, options",
    [
        ("smc", dict(num_particles=9, num_generations=3)),
        ("rejection", dict(num_particles=9, epsilon=1.0)),
        ("top_fraction", dict(Nsim=25, top_fraction=0.4)),
    ],
)
def test_seed_matches_every_worker_count(source, strategy, options):
    def calibrate(seed, workers):
        return make_seeded_sampler(seed, source).calibrate(
            strategy=strategy,
            verbose=False,
            n_workers=workers,
            **options,
        )

    reference = calibrate(43, None)
    assert not reference.get_posterior_distribution().equals(
        calibrate(44, None).get_posterior_distribution()
    )
    for workers in (None, 1, 2, 4):
        if workers is not None and workers > _CPU_CAPACITY:
            continue
        for _ in range(2):
            assert_exact_calibration(reference, calibrate(43, workers))


def test_rng_state_and_projections_match_after_repeated_calibration():
    serial = make_seeded_sampler(0, "argument-generator")
    parallel = make_seeded_sampler(0, "argument-generator")
    with ProcessPoolExecutor(
        max_workers=min(3, _CPU_CAPACITY), mp_context=get_context("spawn")
    ) as executor:
        for _ in range(2):
            left = serial.calibrate(
                strategy="smc", num_particles=7, num_generations=2, verbose=False
            )
            right = parallel.calibrate(
                strategy="smc",
                num_particles=7,
                num_generations=2,
                verbose=False,
                executor=executor,
            )
            assert_exact_calibration(left, right)
            assert serial.rng.bit_generator.state == parallel.rng.bit_generator.state
        # Inherited seed, parameter seed, and explicit seed precedence.
        for params, rng in (({}, None), ({"rng": 7}, None), ({"rng": 99}, 7)):
            left = serial.run_projections(params, iterations=10, rng=rng)
            right = parallel.run_projections(
                params, iterations=10, rng=rng, executor=executor
            )
            assert_frame_equal(
                left.projection_parameters["baseline"],
                right.projection_parameters["baseline"],
                check_exact=True,
            )
            np.testing.assert_array_equal(
                left.get_projection_trajectories()["data"],
                right.get_projection_trajectories()["data"],
            )
        assert executor.submit(int, "7").result() == 7


def test_dyn_out_of_order_work_has_no_speculative_simulations():
    later_started = Event()
    submitted = []

    def evaluate(index):
        if index == 0:
            assert later_started.wait(timeout=5)
        else:
            later_started.set()
        return {"params": [index], "accepted": index % 2 == 0}

    with ThreadPoolExecutor(max_workers=2) as executor:

        def arguments():
            for index in count():
                submitted.append(index)
                yield (index,)

        result = DynamicParticleScheduler().collect_particles(
            executor,
            evaluate,
            arguments(),
            n_target=3,
        )
    assert [r["params"][0] for r in result["accepted_results"]] == [0, 2, 4]
    assert result["n_simulations"] == len(submitted) == 5


def _recorded_simulate(params):
    # Exclusive creation detects reused streams as well as counting actual calls.
    draw = params["rng"].random()
    with (Path(params["record_dir"]) / str(draw)).open("x"):
        pass
    return {"data": np.ones(8)}


@pytest.mark.parametrize("budget", [0, 13])
def test_budget_bounds_actual_simulations(budget, tmp_path):
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


@pytest.mark.parametrize("strategy", ["smc", "rejection"])
@pytest.mark.parametrize("budget", [0, 4, 19])
def test_partial_budget_results_match(strategy, budget):
    options = dict(num_generations=3) if strategy == "smc" else dict(epsilon=1.0)

    def run(workers):
        return make_seeded_sampler(43, "parameters-int").calibrate(
            strategy=strategy,
            num_particles=7,
            verbose=False,
            total_simulations_budget=budget,
            n_workers=workers,
            **options,
        )

    reference = run(None)
    for workers in sorted({1, min(4, _CPU_CAPACITY)}):
        assert_exact_calibration(reference, run(workers))


def _unseeded_simulate(params):
    assert "rng" not in params
    return {"data": np.zeros(8)}


def test_unseeded_simulation_receives_no_rng():
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


def test_batch_bounds_lazy_input_and_preserves_order():
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
            assert DynamicParticleScheduler().run_batch(executor, str, arguments()) == [
                str(i) for i in range(25)
            ]
    assert consumed == retrieved == 25


@pytest.mark.parametrize("failure", ["worker", "input"])
def test_batch_cancels_pending_work_on_failure(failure):
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
                DynamicParticleScheduler().run_batch(executor, str, arguments())
    assert len(submitted) == (4 if failure == "worker" else 2)
    assert all(
        future.cancelled() for future in submitted[1 if failure == "worker" else 0 :]
    )


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
