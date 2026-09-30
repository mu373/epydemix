"""Independent statistical regression for ABC rejection and SMC.

Models and exact finite-epsilon targets are documented in statistical_models.
Use 512 particles, three fixed generations and three preselected independent seeds
(11,29,47); no time cutoff or binding simulation budget. Per-run errors are squared
before aggregation, preventing opposite biases from cancelling. A separate maximum
error guard catches a single broken run. These are accuracy bounds, not a KS test
or a claim that p>.05 establishes equivalence. SMC ancestry means particles are not
IID; thresholds do not treat ESS as an exact independent sample count.

RMS bounds are .10 for mean, .16 for variance, .08 for CDF and .06 for discrete
probabilities, with per-run maxima twice these bounds. They are deliberately wider
than an IID standard error, allowing SMC ancestry and multiple generation/metric
checks. Independent pilot seeds and representative negative controls are recorded
in tests/data/STATISTICAL_VALIDATION.md. No retry or seed selection is permitted.
"""

import numpy as np
import pytest

from epydemix._execution import _get_available_cpu_count
from tests.fixtures.statistical_models import (
    DISCRETE_TARGET,
    NORMAL_SCHEDULE,
    PARTICLES,
    SEEDS,
    make_statistical_sampler,
    normal_reference,
    posterior_summary,
)


def assert_statistical_accuracy(estimates, targets, bounds):
    """Limit RMS across independent runs and every individual run's error."""
    errors = np.asarray(estimates) - np.asarray(targets)
    rms = np.sqrt(np.mean(errors**2, axis=0))
    bounds = np.broadcast_to(bounds, rms.shape)
    np.testing.assert_array_less(rms, bounds)
    np.testing.assert_array_less(np.max(abs(errors), axis=0), 2 * bounds)


@pytest.mark.slow
@pytest.mark.parametrize("workers", [None, 2])
@pytest.mark.parametrize("model", ["normal", "discrete"])
@pytest.mark.parametrize("strategy", ["rejection", "smc"])
def test_calibration_targets_independent_abc_distribution(model, strategy, workers):
    """Detect biased calibration using an analytic likelihood, not another sampler.

    Normal moments/CDF cover continuous proposal and importance weighting; three-state
    probabilities cover pmf and discrete kernels. Check every SMC generation, including
    the prior-proposal initial population. Uniform prior output and ignoring importance
    weights must not pass simply because serial and parallel share the same bug.
    Two workers are the representative parallel configuration; exact fast tests
    cover other worker counts/seed sources. Physical surplus is not a target metric.
    """
    if workers is not None and workers > _get_available_cpu_count():
        pytest.skip("Statistical parallel check requires two available CPUs")
    schedule = NORMAL_SCHEDULE if model == "normal" else (0.5,) * 3
    if strategy == "rejection":
        schedule = schedule[-1:]
    targets = np.array(
        [
            normal_reference(epsilon) if model == "normal" else DISCRETE_TARGET
            for epsilon in schedule
        ]
    )
    bounds = (
        np.array([0.10, 0.16, 0.08, 0.08, 0.08])
        if model == "normal"
        else np.full(3, 0.06)
    )
    estimates = []
    for seed in SEEDS:
        sampler = make_statistical_sampler(model, seed)
        options = (
            {"epsilon": schedule[0]}
            if strategy == "rejection"
            else {"num_generations": len(schedule), "epsilon_schedule": schedule}
        )
        result = sampler.calibrate(
            strategy=strategy,
            num_particles=PARTICLES,
            verbose=False,
            n_workers=workers,
            **options,
        )
        assert len(result.posterior_distributions) == len(schedule)
        summaries = []
        for generation, epsilon in enumerate(schedule):
            weights = np.asarray(result.get_weights(generation))
            assert np.isfinite(weights).all() and (weights >= 0).all()
            assert weights.sum() == pytest.approx(1.0)
            assert len(weights) == PARTICLES
            distances = np.asarray(result.get_distances(generation))
            assert (distances <= epsilon).all()
            summaries.append(posterior_summary(result, model, generation))
        estimates.append(summaries)
    assert_statistical_accuracy(estimates, targets, bounds)


def test_models_use_the_injected_rng():
    """A frozen/reseeded simulation would pass superficial seed-repeatability checks.

    Same seed produces the same sequence, while consecutive calls advance that same
    Generator. Check 32 calls deterministically so Bernoulli coincidences do not make
    an assertion depend on a single pair of random outputs.
    """
    from tests.fixtures.statistical_models import simulate_discrete, simulate_normal

    for simulate, latent in (
        (simulate_normal, {"mu": 0.0}),
        (simulate_discrete, {"k": 1}),
    ):

        def draws():
            params = dict(latent, rng=np.random.default_rng(43))
            return [simulate(params)["data"][0] for _ in range(32)]

        first = draws()
        assert first == draws()
        assert len(set(first)) > 1
