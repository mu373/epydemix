from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.abc_smc_utils import sample_prior
from ..calibration_results import CalibrationResults


@dataclass(frozen=True)
class SamplerContext:
    simulation_function: Callable
    priors: Dict[str, Any]
    parameters: Dict[str, Any]
    observed_data: Dict[str, Any]
    distance_function: Callable
    param_names: List[str]
    continuous_params: List[str]
    discrete_params: List[str]


def sample_params(ctx: SamplerContext) -> List[float]:
    """Sample parameters from priors."""
    return sample_prior(ctx.priors, ctx.param_names)


def run_simulation(ctx: SamplerContext, params: List[float]) -> Dict[str, Any]:
    """Run a single simulation with the provided sampled parameters."""
    full_params = {**ctx.parameters, **dict(zip(ctx.param_names, params))}
    simulation = ctx.simulation_function(full_params)
    if not isinstance(simulation, dict):
        raise ValueError(
            f"Simulation must return dictionary, got {type(simulation)}"
        )
    return simulation


def check_stopping_conditions(
    epsilon: Optional[float],
    minimum_epsilon: Optional[float],
    start_time: datetime,
    max_time: Optional[timedelta],
    n_simulations: int,
    total_simulations_budget: Optional[int],
    verbose: bool = True,
) -> bool:
    """Check if epsilon, runtime, or budget stopping conditions are met."""
    if minimum_epsilon and epsilon and epsilon < minimum_epsilon:
        if verbose:
            print("Minimum epsilon reached")
        return True

    if max_time and datetime.now() - start_time > max_time:
        if verbose:
            print("Maximum time reached")
        return True

    if total_simulations_budget and n_simulations > total_simulations_budget:
        if verbose:
            print("Total simulations budget reached")
        return True

    return False


def create_results(
    ctx: SamplerContext,
    strategy: str,
    particles_df: pd.DataFrame,
    weights: np.ndarray,
    distances: np.ndarray,
    simulations: List[Dict[str, Any]],
) -> CalibrationResults:
    """Create a CalibrationResults object for generation 0-like outputs."""
    return CalibrationResults(
        calibration_strategy=strategy,
        posterior_distributions={0: particles_df},
        selected_trajectories={0: simulations},
        distances={0: distances},
        weights={0: weights},
        observed_data=ctx.observed_data,
        priors=ctx.priors,
    )
