"""Shared exact-equivalence contract for sequential and parallel calibration."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from epydemix._execution import _get_available_cpu_count
from tests.fixtures.calibration import (
    assert_exact_calibration,
    make_runtime_skewed_sampler,
    make_sampler,
    make_seeded_sampler,
)

_CPU_CAPACITY = _get_available_cpu_count()


@pytest.fixture(params=["dynamic"])
def scheduler_strategy(request):
    """Add each registered scheduler name here to run the same equivalence contract."""
    return request.param


class TestParallelEquivalence:
    """Require exact sequential results from every registered parallel scheduler.

    Covers all ABC strategies, seed sources, repeated calls, projections, and
    parameter-dependent runtimes. DYN may perform surplus evaluations; binding
    time or physical-budget cutoffs are excluded from the cross-strategy contract.
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

        scheduling = (
            {}
            if strategy == "top_fraction"
            else {"parallel_strategy": scheduler_strategy}
        )

        def calibrate(seed, workers):
            return make_seeded_sampler(seed, source).calibrate(
                strategy=strategy,
                verbose=False,
                n_workers=workers,
                **scheduling,
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

    def test_sir_checkpoint_resume_across_schedulers(
        self, scheduler_strategy, tmp_path
    ):
        """A real SIR model and resumed SMC retain every generation across strategies.

        Resume a one-generation parallel checkpoint using sequential execution.
        Surplus evaluation counts may differ, but RNG state and retained data
        must match an uninterrupted sequential run with in-memory history.
        """
        reference_sampler = make_sampler("sir")
        options = dict(num_particles=6, verbose=False)
        reference = reference_sampler.calibrate(num_generations=3, **options)
        checkpoint = tmp_path / "run.checkpoint"
        parallel = make_sampler("sir")
        parallel.calibrate(
            num_generations=1,
            n_workers=min(2, _CPU_CAPACITY),
            parallel_strategy=scheduler_strategy,
            checkpoint_path=checkpoint,
            **options,
        )
        result = parallel.calibrate(
            num_generations=3,
            checkpoint_path=checkpoint,
            resume=True,
            **options,
        )
        assert_exact_calibration(reference, result)
        assert (
            reference_sampler.rng.bit_generator.state
            == parallel.rng.bit_generator.state
        )

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
        test_particle_jobs checks ordered selection with synchronized workers.
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
