"""Diagnose undefined ABC arithmetic instead of returning silent NaN populations.

Tiny deterministic models distinguish invalid arithmetic from a useful +infinity
rejection sentinel. Budgets bound the pre-fix reproductions; no statistical oracle,
sleep, or lucky seed is needed. Valid scalar reductions retain their exact order.
"""

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration.abc import ABCSampler
from epydemix.utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    compute_particle_weights,
)


def _constant_observation(parameters):
    return {"data": np.array([parameters["value"]])}


def _observation_distance(data, simulation):
    return float(simulation["data"][0])


def _sampler(value):
    return ABCSampler(
        _constant_observation,
        {"mu": stats.uniform()},
        {"value": value},
        np.zeros(1),
        distance_function=_observation_distance,
        rng=43,
    )


@pytest.mark.parametrize("workers", [None, 1])
def test_nan_distance_fails_instead_of_exhausting_budget(workers):
    """A NaN distance is undefined, not a legitimate rejection or acceptance."""
    with pytest.raises(ValueError, match="distance.*NaN"):
        _sampler(float("nan")).run_rejection(
            num_particles=2,
            total_simulations_budget=1,
            verbose=False,
            n_workers=workers,
        )


def test_nan_threshold_fails_before_running_model():
    """A NaN epsilon would reject forever; fail before consuming simulation RNG."""
    sampler = _sampler(0.0)
    state = sampler.rng.bit_generator.state
    with pytest.raises(ValueError, match="epsilon.*NaN"):
        sampler.run_rejection(
            epsilon=float("nan"), total_simulations_budget=1, verbose=False
        )
    assert sampler.rng.bit_generator.state == state


@pytest.mark.parametrize("value", [1.0, float("inf")])
def test_continuous_kernel_requires_positive_finite_spread(value):
    """Two identical particles have no Gaussian density; do not invent a floor."""
    kernel = DefaultPerturbationContinuous("mu")
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="mu.*variance"):
        kernel.update(np.array([[value], [value]]), np.array([0.5, 0.5]), ["mu"])


class _ConstantKernel:
    def __init__(self, density):
        self.density = density

    def pdf(self, value, center):
        return self.density


@pytest.mark.parametrize("density", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_kernel_mixture_fails_before_weight_normalization(density):
    """One parent isolates zero, negative, and nonfinite importance denominators."""
    with pytest.raises(ValueError, match="kernel mixture"):
        compute_particle_weights(
            np.array([[0.5]]),
            np.array([[0.5]]),
            np.array([1.0]),
            {"mu": stats.uniform()},
            ["mu"],
            {"mu": _ConstantKernel(density)},
        )


def test_all_zero_importance_weights_have_diagnostic():
    """Outside-prior particles have zero mass; total zero cannot be normalized."""
    with pytest.raises(ValueError, match="weights.*positive finite sum"):
        compute_particle_weights(
            np.array([[2.0], [3.0]]),
            np.array([[0.5]]),
            np.array([1.0]),
            {"mu": stats.uniform()},
            ["mu"],
            {"mu": _ConstantKernel(1.0)},
        )


def test_infinite_distance_can_still_represent_rejection():
    """A finite epsilon rejects +infinity without banning the simulator sentinel."""
    result = _sampler(float("inf")).run_rejection(
        num_particles=2, total_simulations_budget=2, verbose=False
    )
    assert result.get_posterior_distribution().empty


@pytest.mark.parametrize("strategy", ["top_fraction", "smc"])
def test_undefined_distance_quantile_fails_explicitly(strategy):
    """Interpolating infinite distances can yield NaN; diagnose before next proposals."""
    options = (
        {"Nsim": 2}
        if strategy == "top_fraction"
        else {"num_particles": 2, "num_generations": 2, "total_simulations_budget": 4}
    )
    with np.errstate(invalid="ignore"), pytest.raises(
        ValueError, match="distance quantile"
    ):
        _sampler(float("inf")).calibrate(strategy=strategy, verbose=False, **options)
