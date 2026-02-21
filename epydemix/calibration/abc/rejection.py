from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from . import common
from .common import SamplerContext


def run(
    ctx: SamplerContext,
    epsilon: float = 0.1,
    num_particles: int = 1000,
    max_time: Optional[timedelta] = None,
    total_simulations_budget: Optional[int] = None,
    verbose: bool = True,
    progress_update_interval: int = 1000,
):
    """Run serial ABC rejection sampling."""
    simulations, distances = [], []
    sampled_params = {p: [] for p in ctx.param_names}

    start_time = datetime.now()
    n_simulations = 0
    last_print = 0

    if verbose:
        print(
            f"Starting ABC rejection sampling with {num_particles} particles and epsilon threshold {epsilon}"
        )

    while len(distances) < num_particles:
        if common.check_stopping_conditions(
            None,
            None,
            start_time,
            max_time,
            n_simulations,
            total_simulations_budget,
        ):
            break

        params = common.sample_params(ctx)
        simulation = common.run_simulation(ctx, params)
        distance = ctx.distance_function(ctx.observed_data, simulation)
        n_simulations += 1

        # Match intended acceptance threshold semantics.
        if distance <= epsilon:
            simulations.append(simulation)
            distances.append(distance)
            for i, param_name in enumerate(ctx.param_names):
                sampled_params[param_name].append(params[i])

        if (
            verbose
            and n_simulations % progress_update_interval == 0
            and n_simulations != last_print
        ):
            last_print = n_simulations
            acceptance_rate = len(distances) / n_simulations * 100
            print(
                f"\tSimulations: {n_simulations}, Accepted: {len(distances)}, "
                f"Acceptance rate: {acceptance_rate:.2f}%"
            )

    if verbose:
        if n_simulations == 0:
            print("\tFinal: 0 particles accepted from 0 simulations (acceptance rate: 0.00%)")
        else:
            print(
                f"\tFinal: {len(distances)} particles accepted from {n_simulations} simulations "
                f"({len(distances) / n_simulations * 100:.2f}% acceptance rate)"
            )

    if distances:
        weights = np.ones(len(distances)) / len(distances)
    else:
        weights = np.array([])

    return common.create_results(
        ctx,
        "rejection",
        pd.DataFrame(sampled_params),
        weights,
        np.array(distances),
        simulations,
    )
