"""Unit tests for the RNG-using ABC-SMC building blocks in
``epydemix.utils.abc_smc_utils``: perturbation kernels and prior sampling.
"""

import numpy as np
import pytest
from scipy import stats

from epydemix.utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
    compute_particle_weights,
    fast_normal_pdf,
    sample_prior,
)

# --- DefaultPerturbationContinuous ------------------------------------------


def test_continuous_propose_is_seed_reproducible():
    """Same seed gives the same proposal; a different seed gives a different one."""
    kernel = DefaultPerturbationContinuous("beta")
    a = kernel.propose(0.5, rng=np.random.default_rng(0))
    b = kernel.propose(0.5, rng=np.random.default_rng(0))
    assert a == b

    c = kernel.propose(0.5, rng=np.random.default_rng(1))
    assert a != c


def test_continuous_propose_centred_on_current_value():
    """Proposals are normally distributed around ``x`` with the kernel's std."""
    kernel = DefaultPerturbationContinuous("beta")
    assert kernel.std == 0.1  # documented default
    rng = np.random.default_rng(0)
    samples = np.array([kernel.propose(0.5, rng=rng) for _ in range(5000)])
    assert samples.mean() == pytest.approx(0.5, abs=0.01)
    assert samples.std() == pytest.approx(kernel.std, abs=0.01)


def test_continuous_pdf_matches_normal():
    """``pdf`` is the normal density centred on ``center`` with the kernel's std."""
    kernel = DefaultPerturbationContinuous("beta")
    assert kernel.pdf(0.5, 0.5) == fast_normal_pdf(0.5, 0.5, kernel.std)
    # Symmetric about the center and peaked there.
    assert kernel.pdf(0.4, 0.5) == pytest.approx(kernel.pdf(0.6, 0.5))
    assert kernel.pdf(0.5, 0.5) > kernel.pdf(0.4, 0.5)


def test_continuous_update_sets_std_to_sqrt2_times_spread():
    """``update`` sets std to ``sqrt(2) * std(particle values)`` (Beaumont 2009)."""
    kernel = DefaultPerturbationContinuous("beta")
    # Two params; "beta" is column 1. Its values are [1.0, 3.0] -> std 1.0.
    particles = np.array([[10.0, 1.0], [20.0, 3.0]])
    weights = np.array([0.5, 0.5])
    kernel.update(particles, weights, param_names=["gamma", "beta"])
    assert kernel.std == pytest.approx(1.0 * np.sqrt(2))


# --- DefaultPerturbationDiscrete --------------------------------------------


def _discrete_kernel(jump_probability=0.3):
    # randint(0, 5) has support {0, 1, 2, 3, 4}.
    prior = stats.randint(0, 5)
    return DefaultPerturbationDiscrete("k", prior, jump_probability=jump_probability)


def test_discrete_propose_is_seed_reproducible():
    """Same seed gives the same proposal for the discrete kernel."""
    kernel = _discrete_kernel()
    a = kernel.propose(2, rng=np.random.default_rng(0))
    b = kernel.propose(2, rng=np.random.default_rng(0))
    assert a == b


def test_discrete_propose_stays_within_support():
    """Every proposal is a value in the kernel's support."""
    kernel = _discrete_kernel()
    rng = np.random.default_rng(0)
    for _ in range(500):
        proposed = kernel.propose(2, rng=rng)
        assert proposed in kernel.support


def test_discrete_propose_always_jumps_to_a_different_value():
    """With jump_probability=1 the reject loop must always land on a new value.

    Exercises the ``while proposed == x`` resample: the proposal can never equal the
    current value when a jump is forced.
    """
    kernel = _discrete_kernel(jump_probability=1.0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        proposed = kernel.propose(2, rng=rng)
        assert proposed != 2
        assert proposed in kernel.support


def test_discrete_propose_never_jumps_when_probability_zero():
    """With jump_probability=0 the value is always kept."""
    kernel = _discrete_kernel(jump_probability=0.0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert kernel.propose(2, rng=rng) == 2


def test_discrete_kernel_rejects_degenerate_support():
    """A single-value support cannot form a jump distribution.

    ``rest_prob = jump_probability / (len(support) - 1)`` is undefined when the support
    has one element, so constructing the kernel over such a prior must fail loudly
    rather than silently produce a broken kernel.
    """
    single_value_prior = stats.randint(3, 4)  # support is just {3}
    with pytest.raises((ZeroDivisionError, ValueError)):
        DefaultPerturbationDiscrete("k", single_value_prior)


def test_discrete_pdf_values():
    """pmf: stay -> 1 - jump; move within support -> rest_prob; outside -> 0."""
    kernel = _discrete_kernel(jump_probability=0.3)
    # rest_prob = jump_probability / (len(support) - 1) = 0.3 / 4
    assert kernel.pdf(2, 2) == pytest.approx(1 - 0.3)
    assert kernel.pdf(3, 2) == pytest.approx(0.3 / 4)
    # A value outside the support has zero transition probability.
    assert kernel.pdf(99, 2) == 0


# --- sample_prior -----------------------------------------------------------


def test_sample_prior_is_seed_reproducible():
    """Same seed gives the same ordered draws; a different seed differs."""
    priors = {
        "beta": stats.uniform(0.1, 0.5),
        "gamma": stats.uniform(0.05, 0.2),
    }
    param_names = ["beta", "gamma"]

    a = sample_prior(priors, param_names, rng=np.random.default_rng(0))
    b = sample_prior(priors, param_names, rng=np.random.default_rng(0))
    assert a == b

    c = sample_prior(priors, param_names, rng=np.random.default_rng(1))
    assert a != c


def test_sample_prior_missing_param_raises():
    """Requesting a param name absent from ``priors`` fails with a clear KeyError."""
    priors = {"beta": stats.uniform(0.1, 0.5)}
    with pytest.raises(KeyError):
        sample_prior(priors, ["beta", "gamma"], rng=np.random.default_rng(0))


def test_sample_prior_respects_param_name_order():
    """The returned list follows ``param_names``, not the prior dict's order."""
    priors = {
        "beta": stats.uniform(0.1, 0.5),
        "gamma": stats.uniform(0.05, 0.2),
    }
    # Same seed, reversed name order: values must swap position accordingly.
    forward = sample_prior(priors, ["beta", "gamma"], rng=np.random.default_rng(0))
    reverse = sample_prior(priors, ["gamma", "beta"], rng=np.random.default_rng(0))

    # First draw off the generator is consumed by whichever name comes first.
    assert forward[0] != reverse[0]
    # Each value lands in its prior's range regardless of order.
    assert 0.1 <= forward[0] <= 0.6 and 0.05 <= forward[1] <= 0.25
    assert 0.05 <= reverse[0] <= 0.25 and 0.1 <= reverse[1] <= 0.6


def test_particle_weights_are_uniform_for_initial_generation():
    particles = np.array([[0.5], [1.0], [1.5]])
    weights = compute_particle_weights(
        particles, None, None, {"beta": stats.uniform(0, 2)}, ["beta"], None
    )
    np.testing.assert_array_equal(weights, np.full(3, 1 / 3))


@pytest.mark.parametrize("param_names", [["beta", "k"], ["k", "beta"]])
def test_particle_weights_match_mixed_prior_kernel_mixture(param_names):
    priors = {"beta": stats.uniform(0, 2), "k": stats.bernoulli(0.25)}
    kernels = {
        "beta": DefaultPerturbationContinuous("beta"),
        "k": DefaultPerturbationDiscrete("k", priors["k"], jump_probability=0.2),
    }
    kernels["beta"].std = 1.0
    previous = np.array([[0.5, 0], [1.5, 1]])
    particles = np.array([[0.5, 1], [1.5, 0]])
    if param_names[0] == "k":
        previous = previous[:, ::-1].copy()
        particles = particles[:, ::-1].copy()
    previous_weights = np.array([0.25, 0.75])
    snapshots = [value.copy() for value in (particles, previous, previous_weights)]
    actual = compute_particle_weights(
        particles, previous, previous_weights, priors, param_names, kernels
    )
    # Two explicit proposal mixtures; the new particles have prior masses .125 and .375.
    same, other = stats.norm.pdf(0), stats.norm.pdf(1)
    expected = np.array(
        [
            0.125 / (0.25 * same * 0.2 + 0.75 * other * 0.8),
            0.375 / (0.25 * other * 0.8 + 0.75 * same * 0.2),
        ]
    )
    np.testing.assert_allclose(actual, expected / expected.sum(), rtol=1e-14)
    for value, snapshot in zip((particles, previous, previous_weights), snapshots):
        np.testing.assert_array_equal(value, snapshot)


@pytest.mark.parametrize("dimensions", [1, 3, 8, 17])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_particle_weights_preserve_scalar_product_rounding(dimensions, dtype):
    """Match the original np.prod formula exactly, including custom scalar dtypes.

    Faster product dispatch must retain the reduction order and precision: an
    altered weight can change parent selection and every later seeded generation.
    """
    rng = np.random.default_rng(43)
    names = [f"p{i}" for i in range(dimensions)]
    priors = {name: stats.norm() for name in names}

    class Kernel:
        def pdf(self, x, center):
            return dtype(np.exp(-0.5 * (x - center) ** 2))

    kernels = {name: Kernel() for name in names}
    particles = rng.normal(size=(7, dimensions))
    previous = rng.normal(size=(11, dimensions))
    previous_weights = rng.random(11)
    previous_weights /= previous_weights.sum()
    expected = np.ones(len(particles))
    for i, params in enumerate(particles):
        numerator = np.prod([priors[p].pdf(params[k]) for k, p in enumerate(names)])
        denominator = np.sum(
            [
                previous_weights[j]
                * np.prod(
                    [
                        kernels[p].pdf(params[k], previous[j, k])
                        for k, p in enumerate(names)
                    ]
                )
                for j in range(len(previous))
            ]
        )
        expected[i] = numerator / denominator
    expected /= expected.sum()
    actual = compute_particle_weights(
        particles, previous, previous_weights, priors, names, kernels
    )
    np.testing.assert_array_equal(actual, expected)
