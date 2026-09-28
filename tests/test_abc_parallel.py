"""Candidate evaluation and upstream acceptance boundaries."""

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration.abc import ABCSampler


def _mock_simulate(params):
    """Simple deterministic mock simulation."""
    beta = params.get("beta", 0.3)
    gamma = params.get("gamma", 0.1)
    return {"data": np.array([100 * np.exp(-beta * gamma * t) for t in range(10)])}


def _distance_with_positional_names(observed, predicted, /):
    return 1.0


@pytest.mark.parametrize("strategy", ["rejection", "smc", "top_fraction"])
def test_distance_callback_and_upstream_boundary_rules(strategy):
    sampler = ABCSampler(
        _mock_simulate,
        {"beta": stats.uniform(0.1, 0.5)},
        {},
        np.zeros(10),
        distance_function=_distance_with_positional_names,
        rng=43,
    )
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
    result = sampler.calibrate(strategy=strategy, verbose=False, **options)
    assert list(result.posterior_distributions) == [0]
    assert len(result.get_posterior_distribution()) == (
        0 if strategy == "rejection" else 3
    )
