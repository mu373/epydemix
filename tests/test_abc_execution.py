"""Sequential candidate evaluation and calibration boundary regressions."""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration.abc import ABCSampler
from tests.fixtures.calibration import make_seeded_sampler


def _mock_simulate(params):
    """Simple deterministic mock simulation."""
    beta = params.get("beta", 0.3)
    gamma = params.get("gamma", 0.1)
    return {"data": np.array([100 * np.exp(-beta * gamma * t) for t in range(10)])}


def _distance_with_positional_names(observed, predicted, /):
    """Return the threshold exactly; positional-only args reject keyword calls."""
    return 1.0


@pytest.mark.parametrize("strategy", ["rejection", "smc", "top_fraction"])
def test_distance_callback_and_upstream_boundary_rules(strategy):
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
    result = sampler.calibrate(strategy=strategy, verbose=False, **options)
    assert list(result.posterior_distributions) == [0]
    assert len(result.get_posterior_distribution()) == (
        0 if strategy == "rejection" else 3
    )


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
    result = sampler.calibrate(
        strategy="rejection",
        num_particles=10,
        epsilon=-1,
        total_simulations_budget=budget,
        verbose=False,
    )
    assert len(list(tmp_path.iterdir())) == budget
    assert len(result.get_posterior_distribution()) == 0


def test_smc_cumulative_sim_count():
    """Verify n_simulations accumulates rather than resetting each generation."""
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
    # Exactly two generations fit, with a budget equal to their combined particle count.
    results = sampler.calibrate(
        strategy="smc",
        num_particles=5,
        num_generations=4,
        epsilon_schedule=[float("inf")] * 4,
        total_simulations_budget=10,
        verbose=False,
    )
    # Should have completed 2 generations
    assert len(results.posterior_distributions) == 2
