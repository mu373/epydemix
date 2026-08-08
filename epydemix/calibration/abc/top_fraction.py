import numpy as np
import pandas as pd

from . import common
from .common import SamplerContext


def run(
    ctx: SamplerContext,
    top_fraction: float = 0.05,
    Nsim: int = 100,
    verbose: bool = True,
):
    """Run serial ABC top-fraction selection."""
    simulations, distances = [], []
    sampled_params = {p: [] for p in ctx.param_names}

    if verbose:
        print(
            f"Starting ABC top fraction selection with {Nsim} simulations and top {top_fraction * 100:.1f}% selected"
        )

    for n in range(Nsim):
        params = common.sample_params(ctx)
        simulation = common.run_simulation(ctx, params)
        distance = ctx.distance_function(ctx.observed_data, simulation)

        simulations.append(simulation)
        distances.append(distance)
        for i, param_name in enumerate(ctx.param_names):
            sampled_params[param_name].append(params[i])

        if verbose and (n + 1) % max(1, Nsim // 10) == 0:
            print(
                f"\tProgress: {n + 1}/{Nsim} simulations completed ({(n + 1) / Nsim * 100:.1f}%)"
            )

    threshold = np.quantile(distances, top_fraction)
    mask = np.array(distances) <= threshold
    n_selected = int(np.sum(mask))

    if verbose:
        print(
            f"\tSelected {n_selected} particles (top {top_fraction * 100:.1f}%) "
            f"with distance threshold {threshold:.6f}"
        )

    return common.create_results(
        ctx,
        "top_fraction",
        pd.DataFrame(sampled_params)[mask],
        np.ones(n_selected) / n_selected,
        np.array(distances)[mask],
        np.array(simulations)[mask],
    )
