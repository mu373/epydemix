"""Small public-API ABC contracts that predate the parallelization refactor.

These tests preserve threshold boundaries, ties and the importance-weight formula.
Models are deliberately tiny and deterministic where randomness is irrelevant.
They do not bake in known budget/callback bugs; those get tests in their fix commits.
"""

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration.abc import ABCSampler
from tests.fixtures.statistical_models import absolute_distance, simulate_normal


@pytest.mark.parametrize("fraction, expected", [(0.5, [0, 1, 1]), (1.0, [0, 1, 1, 3])])
def test_top_fraction_keeps_quantile_ties(fraction, expected):
    """The median of [0,1,1,3] is 1: retaining exactly two rows loses a valid tie."""
    distances = iter([0, 1, 1, 3])
    sampler = ABCSampler(
        lambda parameters: {"data": np.array([next(distances)])},
        {"mu": stats.norm()},
        {},
        np.array([0]),
        absolute_distance,
        rng=43,
    )
    result = sampler.calibrate(
        strategy="top_fraction", Nsim=4, top_fraction=fraction, verbose=False
    )
    np.testing.assert_array_equal(result.get_distances(), expected)
    np.testing.assert_array_equal(
        result.get_calibration_trajectories()["data"][:, 0], expected
    )


def test_acceptance_equality_depends_on_strategy_and_generation():
    """Initial SMC accepts equality; rejection and subsequent SMC reject it.

    Script distances remove stochastic threshold ambiguity. [1,1] must fill the first
    generation at epsilon=1; [1,0,1,0] must yield two zero-distance later particles.
    """

    def sampler_for(values):
        values = iter(values)
        return ABCSampler(
            lambda parameters: {"data": np.array([next(values)])},
            {"mu": stats.norm()},
            {},
            np.array([0]),
            absolute_distance,
            rng=43,
        )

    rejection = sampler_for([1, 0]).calibrate(
        strategy="rejection", epsilon=1, num_particles=1, verbose=False
    )
    np.testing.assert_array_equal(rejection.get_distances(), [0])
    smc = sampler_for([1, 1, 1, 0, 1, 0]).calibrate(
        num_particles=2,
        num_generations=2,
        epsilon_schedule=[1, 1],
        verbose=False,
    )
    np.testing.assert_array_equal(smc.get_distances(0), [1, 1])
    np.testing.assert_array_equal(smc.get_distances(1), [0, 0])


def test_minimum_epsilon_equality_does_not_stop():
    """epsilon<minimum stops after completing a generation; equality continues.

    The simulator returns zero with a nondegenerate random prior population, so every
    positive threshold accepts and the result length isolates the stopping boundary.
    """
    sampler = ABCSampler(
        lambda parameters: {"data": np.array([0])},
        {"mu": stats.norm()},
        {},
        np.array([0]),
        absolute_distance,
        rng=43,
    )
    result = sampler.calibrate(
        num_particles=5,
        num_generations=4,
        epsilon_schedule=[1, 1, 0.5, 0.25],
        minimum_epsilon=1,
        verbose=False,
    )
    assert len(result.posterior_distributions) == 3


def test_public_smc_weights_match_independent_mixture_formula():
    """Expose importance-weight regressions before extracting a weight helper.

    Model: beta~Uniform(0,2), k~Bernoulli(.25), deterministic data=0, epsilon=1.
    For each returned second-generation row, independently expand the two component
    kernel densities over eight prior particles. Normal density is evaluated from its
    closed formula; the discrete transition is .7 to stay and .3 to move. Neither
    sampler helpers nor kernel.pdf are used to compute the expectation.
    """
    sampler = ABCSampler(
        lambda parameters: {"data": np.array([0])},
        {"beta": stats.uniform(0, 2), "k": stats.bernoulli(0.25)},
        {},
        np.array([0]),
        absolute_distance,
        rng=43,
    )
    result = sampler.calibrate(
        num_particles=8, num_generations=2, epsilon_schedule=[1, 1], verbose=False
    )
    previous = result.get_posterior_distribution(0).to_numpy()
    current = result.get_posterior_distribution(1).to_numpy()
    previous_weights = result.get_weights(0)
    np.testing.assert_array_equal(previous_weights, np.full(8, 1 / 8))
    sigma = np.std(previous[:, 0]) * np.sqrt(2)
    raw = []
    for beta, k in current:
        numerator = 0.5 * (0.25 if k == 1 else 0.75)
        continuous = np.exp(-0.5 * ((beta - previous[:, 0]) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
        )
        discrete = np.where(k == previous[:, 1], 0.7, 0.3)
        raw.append(numerator / np.sum(previous_weights * continuous * discrete))
    expected = np.array(raw) / np.sum(raw)
    np.testing.assert_allclose(result.get_weights(1), expected, rtol=1e-14)


def test_explicit_schedule_overrides_adaptation():
    """A fixed schedule must work even if the unused adaptive quantile is invalid.

    This catches accidentally computing adaptive epsilon before consulting the explicit
    schedule; a stochastic normal model supplies nonzero, unequal observed distances.
    """
    sampler = ABCSampler(
        simulate_normal,
        {"mu": stats.norm()},
        {},
        np.array([1.0]),
        absolute_distance,
        rng=43,
    )
    result = sampler.calibrate(
        num_particles=8,
        num_generations=3,
        epsilon_schedule=[1, 0.5, 0.25],
        epsilon_quantile_level=2.0,
        verbose=False,
    )
    assert len(result.posterior_distributions) == 3
    for generation, epsilon in enumerate([1, 0.5, 0.25]):
        assert np.all(result.get_distances(generation) <= epsilon)
