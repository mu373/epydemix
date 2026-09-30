"""Importable calibration models and assertions shared by tests and spawned workers."""

import numpy as np
from pandas.testing import assert_frame_equal
from scipy import stats

from epydemix.calibration.abc import ABCSampler


def _noisy_simulate(params):
    """Generate a noisy trajectory after a candidate-dependent number of RNG draws."""
    rng = params["rng"]
    # Variable draw counts must not change any other candidate's random stream.
    rng.normal(size=int(params["count"]))
    return {"data": params["beta"] * np.arange(8) + rng.normal(size=8)}


def make_seeded_sampler(seed, source):
    """Build a noisy sampler with an integer or Generator in rng= or parameters["rng"]."""
    value = np.random.default_rng(seed) if "generator" in source else seed
    return ABCSampler(
        simulation_function=_noisy_simulate,
        priors={"beta": stats.uniform(0.1, 0.5), "count": stats.randint(1, 6)},
        parameters={"rng": value} if source.startswith("parameters") else {},
        observed_data=np.arange(8) / 3,
        rng=value if source.startswith("argument") else None,
    )


def assert_exact_calibration(left, right):
    """Assert exact equality of every generation, weight, distance, and trajectory."""
    assert left.posterior_distributions.keys() == right.posterior_distributions.keys()
    for generation in left.posterior_distributions:
        assert_frame_equal(
            left.posterior_distributions[generation],
            right.posterior_distributions[generation],
            check_exact=True,
        )
        np.testing.assert_array_equal(
            left.weights[generation], right.weights[generation]
        )
        np.testing.assert_array_equal(
            left.distances[generation], right.distances[generation]
        )
        left_trajectories = left.get_calibration_trajectories(generation)
        right_trajectories = right.get_calibration_trajectories(generation)
        assert left_trajectories.keys() == right_trajectories.keys()
        for key in left_trajectories:
            np.testing.assert_array_equal(
                left_trajectories[key], right_trajectories[key]
            )
