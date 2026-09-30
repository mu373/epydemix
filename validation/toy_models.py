"""Importable toy comparisons, using only the Generator supplied by the caller.

These preserve the continuous and mixed-prior pyabc notebook examples. They are
exploratory models; independent posterior oracles live in tests/fixtures instead.
"""

import numpy as np


def normal_mean(parameters):
    """Uniform-prior mean example: one Normal(mu, standard deviation 0.5) draw."""
    return {"data": np.array([parameters["rng"].normal(parameters["mu"], 0.5)])}


def mixed_location_scale(parameters):
    """Eight observations sharing a discrete offset, with independent Normal noise."""
    rng = parameters["rng"]
    return {
        "data": parameters["p_discrete"]
        + rng.choice([-2, 0, 2], p=[0.2, 0.5, 0.3])
        + parameters["p_continuous"] * rng.normal(size=8)
    }


def absolute_distance(data, simulation):
    return float(np.abs(data["data"] - simulation["data"]).sum())


def squared_distance(data, simulation):
    return float(np.square(data["data"] - simulation["data"]).sum())
