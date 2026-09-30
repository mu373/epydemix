"""Importable simulation model and sampler helpers for calibration tests."""

import numpy as np
from scipy import stats

from epydemix.calibration.abc import ABCSampler


def _noisy_simulate(params):
    """Generate a noisy trajectory after a candidate-dependent number of RNG draws."""
    rng = params["rng"]
    # Vary draw counts to exercise sequential RNG consumption.
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
