"""Small, stochastic compatibility workload and explicit golden-data recipe.

The prior is beta ~ Uniform(.1, .6), count ~ DiscreteUniform(1, ..., 5).
The simulator consumes count random draws before producing eight noisy observations
of beta * arange(8). Observation arange(8)/3 and RMSE exercise acceptance, mixed
priors, importance weights and parameter-dependent random consumption. Two calls
on each sampler and a projection after each call expose root-RNG state regressions.

This module contains no sampler implementation. A checked-in JSON snapshot is the
reference; pytest never generates it. Regenerate only in the documented source
checkout/environment, then review the changed fields and the reason for changing
them. JSON stores array shapes/dtypes and full float precision without pickle.
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from epydemix.calibration.abc import ABCSampler


def simulate_reference(parameters):
    """Consume a parameter-dependent stream before returning a small trajectory."""
    rng = parameters["rng"]
    rng.normal(size=int(parameters["count"]))
    return {"data": parameters["beta"] * np.arange(8) + rng.normal(size=8)}


def as_json(value):
    """Preserve numerical values, shapes, dtypes and dataframe row alignment."""
    if isinstance(value, pd.DataFrame):
        return {
            "columns": list(value.columns),
            "index": as_json(value.index.to_numpy()),
            "values": {name: as_json(value[name].to_numpy()) for name in value},
        }
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    return value


def environment():
    """Versions governing exact SciPy sampling and NumPy reduction results."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def reference_snapshot():
    """Capture six calibrations, all generations, projections and parent RNG state."""
    output = {}
    cases = [
        ("smc", dict(num_particles=6, num_generations=2)),
        ("rejection", dict(num_particles=6, epsilon=1.0)),
        ("top_fraction", dict(Nsim=12, top_fraction=0.5)),
    ]
    for strategy, options in cases:
        sampler = ABCSampler(
            simulate_reference,
            {"beta": stats.uniform(0.1, 0.5), "count": stats.randint(1, 6)},
            {},
            np.arange(8) / 3,
            rng=np.random.default_rng(43),
        )
        for repeat in range(2):
            result = sampler.calibrate(strategy=strategy, verbose=False, **options)
            value = {
                "particles": result.posterior_distributions,
                "weights": result.weights,
                "distances": result.distances,
                "trajectories": {
                    gen: result.get_calibration_trajectories(gen)
                    for gen in result.posterior_distributions
                },
                "rng": sampler.rng.bit_generator.state,
            }
            projection = sampler.run_projections({}, iterations=3)
            value.update(
                projection=projection.get_projection_trajectories(),
                projected_parameters=projection.projection_parameters,
                rng_after_projection=sampler.rng.bit_generator.state,
            )
            output[f"{strategy}/{repeat}"] = as_json(value)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    expected = subprocess.check_output(
        ["git", "rev-parse", args.source_commit], text=True
    ).strip()
    if actual != expected:
        parser.error(f"Run in source checkout {expected}; current HEAD is {actual}")
    snapshot = {
        "source_commit": actual,
        "environment": environment(),
        "results": reference_snapshot(),
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
