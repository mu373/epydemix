"""Importable calibration models and assertions shared by tests and spawned workers."""

from time import sleep

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


def runtime_skewed_simulate(parameters):
    """Return theta squared, with a sign-dependent delay and no epidemic simulation.

    Opposite signs of the same magnitude give identical outputs but different
    runtimes. Sleeping does not consume RNG draws or change the model output.
    """
    theta = parameters["theta"]
    delay = (
        parameters["slow_delay"]
        if theta * parameters["slow_sign"] > 0
        else parameters["fast_delay"]
    )
    if delay:
        sleep(delay)
    return {"data": np.array([theta**2])}


def make_runtime_skewed_sampler(seed, slow_sign=1, fast_delay=0.0001, slow_delay=0.002):
    """Build a symmetric two-mode ABC target with parameter-dependent runtimes.

    With theta uniform on [-2, 2] and observation 1, the squared-parameter model
    gives equal target mass near -1 and +1 for the test thresholds. slow_sign=1
    delays the positive side; -1 delays the negative side. Zero delays provide
    the same model for a sequential reference without artificial waiting.
    """
    return ABCSampler(
        runtime_skewed_simulate,
        {"theta": stats.uniform(-2, 4)},
        {"slow_sign": slow_sign, "fast_delay": fast_delay, "slow_delay": slow_delay},
        np.array([1.0]),
        rng=seed,
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
