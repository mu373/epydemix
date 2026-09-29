"""Shared exact-equivalence contract for sequential and parallel calibration."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from itertools import count
from multiprocessing import get_context
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from epydemix.calibration.parallel import (
    _get_available_cpu_count,
    create_particle_scheduler,
)
from tests.fixtures.calibration import (
    assert_exact_calibration,
    make_runtime_skewed_sampler,
    make_seeded_sampler,
)

_CPU_CAPACITY = _get_available_cpu_count()


@pytest.fixture(params=["dynamic"])
def scheduler_strategy(request):
    """Add each registered scheduler name here to run the same equivalence contract."""
    return request.param


class TestParallelEquivalence:
    """Require exact sequential results from every registered parallel scheduler.

    Covers all ABC strategies, seed sources, repeated calls, projections, budget
    cutoffs, candidate evaluation counts, and parameter-dependent runtimes.
    Real-time cutoffs are excluded because they can change the evaluated prefix.
    Add scheduler names to the scheduler_strategy fixture, not copied test classes.
    """

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
    def test_seed_matches_every_worker_count(
        self, scheduler_strategy, source, strategy, options
    ):
        """Check exact repeatability across seed sources and available worker counts."""

        def calibrate(seed, workers):
            return make_seeded_sampler(seed, source).calibrate(
                strategy=strategy,
                verbose=False,
                n_workers=workers,
                parallel_strategy=scheduler_strategy
                if workers is not None
                else "dynamic",
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

    def test_rng_state_and_projections_match_after_repeated_calibration(
        self, scheduler_strategy
    ):
        """Check repeated calibration, RNG state, and projection seeding against spawn."""
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
                    parallel_strategy=scheduler_strategy,
                )
                assert_exact_calibration(left, right)
                assert (
                    serial.rng.bit_generator.state == parallel.rng.bit_generator.state
                )
            # Inherited seed, parameter seed, and explicit seed precedence.
            for params, rng in (({}, None), ({"rng": 7}, None), ({"rng": 99}, 7)):
                left = serial.run_projections(params, iterations=10, rng=rng)
                right = parallel.run_projections(
                    params,
                    iterations=10,
                    rng=rng,
                    executor=executor,
                    parallel_strategy=scheduler_strategy,
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

    @pytest.mark.parametrize("strategy", ["smc", "rejection"])
    @pytest.mark.parametrize("budget", [0, 4, 19])
    def test_partial_budget_results_match(self, scheduler_strategy, strategy, budget):
        """Check sequential and process results match when a budget interrupts sampling."""
        options = dict(num_generations=3) if strategy == "smc" else dict(epsilon=1.0)

        def run(workers):
            return make_seeded_sampler(43, "parameters-int").calibrate(
                strategy=strategy,
                num_particles=7,
                verbose=False,
                total_simulations_budget=budget,
                n_workers=workers,
                parallel_strategy=scheduler_strategy
                if workers is not None
                else "dynamic",
                **options,
            )

        reference = run(None)
        for workers in sorted({1, min(4, _CPU_CAPACITY)}):
            assert_exact_calibration(reference, run(workers))

    @pytest.mark.parametrize("strategy", ["rejection", "smc"])
    @pytest.mark.parametrize("slow_sign", [-1, 1])
    @pytest.mark.parametrize("mode", ["owned", "spawn"])
    def test_runtime_skew_preserves_sequential_results(
        self, scheduler_strategy, strategy, slow_sign, mode
    ):
        """Preserve every generation and RNG state when either parameter region is slow.

        The synthetic simulator returns theta**2 with observation 1 and a uniform
        prior on [-2, 2], giving equally valid modes near -1 and +1. Sleeping on
        one sign makes runtime depend on the parameter without changing its output
        or acceptance probability. Keeping the first accepted results to finish
        can favor the faster mode; in SMC, that changes the next generation's
        proposal population. Reversing slow_sign exercises both directions.

        For the same seed, require identical particles, weights, distances and
        trajectories in every generation, plus the final root RNG state. Cover
        sampler-owned pools and caller-owned spawn pools. The sequential reference
        skips sleeps because they affect only runtime; neither run has a time limit.

        This small regression checks exact equivalence, not statistical agreement
        with a 50:50 target. It makes no assertion about timing or completion order;
        the event-controlled scheduler test below forces an ordering reversal.
        """
        if _CPU_CAPACITY < 2:
            pytest.skip(
                "Runtime-skew comparison requires two available process workers"
            )
        options = (
            dict(num_generations=2, epsilon_schedule=[0.8, 0.3])
            if strategy == "smc"
            else dict(epsilon=0.3)
        )
        serial = make_runtime_skewed_sampler(43, slow_sign, fast_delay=0, slow_delay=0)
        parallel = make_runtime_skewed_sampler(43, slow_sign)
        reference = serial.calibrate(
            strategy=strategy, num_particles=12, verbose=False, **options
        )
        context = (
            ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn"))
            if mode == "spawn"
            else nullcontext()
        )
        with context as executor:
            result = parallel.calibrate(
                strategy=strategy,
                num_particles=12,
                verbose=False,
                parallel_strategy=scheduler_strategy,
                n_workers=2,
                executor=executor,
                **options,
            )
        assert_exact_calibration(reference, result)
        assert serial.rng.bit_generator.state == parallel.rng.bit_generator.state

    def test_completion_order_preserves_candidates_and_evaluation_count(
        self, scheduler_strategy
    ):
        """Force a later candidate to finish first and retain the sequential sample.

        The done callback releases candidate 0 only after candidate 2's future is
        complete. Both are accepted, so their completion order is always reversed.
        Even candidate IDs are accepted: the sequential target of three particles
        therefore requires IDs [0, 2, 4] and exactly five evaluations. Matching both
        catches completion-order selection and surplus evaluations. Events provide
        a guaranteed reversal to complement the runtime-skew model's sleep delays.
        """

        def candidate_result(index):
            return {"params": [index], "accepted": index % 2 == 0}

        reference = create_particle_scheduler("dynamic").run_until_n_accepted(
            None,
            candidate_result,
            ((index,) for index in count()),
            n_target=3,
        )
        later_completed = Event()
        submitted = []

        def evaluate(index):
            if index == 0:
                assert later_completed.wait(timeout=5)
            return candidate_result(index)

        def arguments():
            for index in count():
                submitted.append(index)
                yield (index,)

        with ThreadPoolExecutor(max_workers=2) as executor:
            submit = executor.submit

            def submit_with_completion_signal(fn, index):
                future = submit(fn, index)
                if index == 2:
                    future.add_done_callback(lambda _: later_completed.set())
                return future

            with patch.object(
                executor, "submit", side_effect=submit_with_completion_signal
            ):
                result = create_particle_scheduler(
                    scheduler_strategy
                ).run_until_n_accepted(
                    executor,
                    evaluate,
                    arguments(),
                    n_target=3,
                )
        assert result == reference
        assert [item["params"][0] for item in result["accepted_results"]] == [0, 2, 4]
        assert result["n_simulations"] == len(submitted) == 5
