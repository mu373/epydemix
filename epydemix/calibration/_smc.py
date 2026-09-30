"""ABC-SMC generations and stopping conditions.

The caller supplies the executor; generations use shared particle evaluation.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
    compute_particle_weights,
)
from . import _evaluate
from .calibration_results import CalibrationResults


class SMCRun:
    """Manage SMC generations and retain the RNG on success or failure.

    Generations use shared particle evaluation. Fixed inputs refer to the
    caller's model and observations; workers restore isolated snapshots.
    """

    def __init__(self, particle_inputs, priors, rng, seed_requested):
        self.particle_inputs = particle_inputs
        (
            self.simulation_function,
            self.parameters,
            self.param_names,
            self.observed_data,
            self.distance_function,
        ) = particle_inputs
        self.priors = priors
        self.rng = rng
        self.seed_requested = seed_requested

    def execute(
        self,
        num_particles,
        num_generations,
        epsilon_schedule,
        epsilon_quantile_level,
        minimum_epsilon,
        max_time,
        total_simulations_budget,
        perturbations,
        verbose,
        pool,
        scheduler,
    ) -> CalibrationResults:
        """
        Run the ABC-SMC generations inside an already opened worker pool.

        Args:
            num_particles, num_generations, epsilon_schedule, epsilon_quantile_level,
            minimum_epsilon, max_time, total_simulations_budget, perturbations, verbose:
                See `ABCSampler.run_smc`.
            pool (ProcessPoolExecutor or None): Worker pool, or None for sequential execution.
            scheduler: SequentialScheduler or DynamicScheduler for acceptance sampling.

        Returns:
            CalibrationResults: See `ABCSampler.run_smc`.
        """
        # Initialize perturbations if not provided
        if perturbations is None:
            perturbations = {
                param: (
                    DefaultPerturbationContinuous(param)
                    if hasattr(self.priors[param], "pdf")
                    else DefaultPerturbationDiscrete(param, self.priors[param])
                )
                for param in self.param_names
            }

        if verbose:
            print(
                f"Starting ABC-SMC with {num_particles} particles and {num_generations} generations"
            )

        start_time = datetime.now()
        n_simulations = 0
        results = CalibrationResults(
            calibration_strategy="smc",
            observed_data=self.observed_data,
            priors=self.priors,
        )
        particles = weights = distances = None
        for gen in range(num_generations):
            start_generation_time = datetime.now()

            if epsilon_schedule is not None:
                epsilon = epsilon_schedule[gen]
            elif gen == 0:
                epsilon = float("inf")
            else:
                with np.errstate(invalid="ignore"):
                    epsilon = np.quantile(distances, epsilon_quantile_level)
                if np.isnan(epsilon):
                    raise ValueError(
                        "Undefined distance quantile; use finite distances"
                    )
            if np.isnan(epsilon):
                raise ValueError("epsilon must not be NaN")
            if verbose:
                print(
                    f"\nGeneration {gen + 1}/{num_generations} (epsilon: {epsilon:.6f})"
                )
            if gen > 0:
                for perturbation in perturbations.values():
                    perturbation.update(particles, weights, self.param_names)

            new_gen = self._run_generation(
                particles,
                weights,
                epsilon,
                num_particles,
                perturbations,
                pool,
                scheduler,
                start_time,
                max_time,
                total_simulations_budget,
                n_simulations,
            )
            if new_gen is None:
                if verbose:
                    print(
                        "Maximum time or budget reached during generation 0"
                        if gen == 0
                        else f"Maximum time or budget reached during generation {gen + 1}, keeping last complete generation"
                    )
                break

            n_simulations = new_gen["n_simulations"]
            particles = new_gen["particles"]
            weights = new_gen["weights"]
            distances = new_gen["distances"]
            values = {
                "posterior_distributions": pd.DataFrame(
                    particles, columns=self.param_names, copy=True
                ),
                "weights": weights,
                "distances": distances,
                "selected_trajectories": new_gen["simulations"],
            }
            for name, value in values.items():
                getattr(results, name)[gen] = value

            if verbose:
                # Print generation information
                end_generation_time = datetime.now()
                elapsed_time = end_generation_time - start_generation_time
                formatted_time = f"{elapsed_time.seconds // 3600:02}:{(elapsed_time.seconds % 3600) // 60:02}:{elapsed_time.seconds % 60:02}"
                acceptance_rate = (
                    len(new_gen["particles"]) / new_gen["n_simulations"] * 100
                )
                print(
                    f"\tAccepted {len(new_gen['particles'])}/{new_gen['n_simulations']} (acceptance rate: {acceptance_rate:.2f}%)"
                )
                print(f"\tElapsed time: {formatted_time}")

            # Check stopping conditions between generations
            if _check_stopping_conditions(
                epsilon,
                minimum_epsilon,
                start_time,
                max_time,
                n_simulations,
                total_simulations_budget,
            ):
                break

        return results

    def _run_generation(
        self,
        particles,
        weights,
        epsilon,
        num_particles,
        perturbations,
        pool,
        scheduler,
        start_time,
        max_time,
        total_simulations_budget,
        n_simulations,
    ):
        """
        Run one SMC generation and compute the importance weights.

        Preserve the original boundary rule: generation 0 accepts distance <= epsilon;
        subsequent generations require distance < epsilon.

        Args:
            particles (np.ndarray or None): Previous generation, one row per particle; None for generation 0.
            weights (np.ndarray or None): Weights of the previous generation.
            epsilon (float): Acceptance threshold for this generation.
            num_particles (int): Number of particles to accept.
            perturbations (Dict[str, Perturbation]): Perturbation kernel per parameter.
            pool (ProcessPoolExecutor or None): Worker pool, or None for sequential execution.
            scheduler: SequentialScheduler or DynamicScheduler for acceptance sampling.
            start_time (datetime): Start of the calibration run.
            max_time (timedelta, optional): Time limit measured from start_time.
            total_simulations_budget (int, optional): Maximum number of simulations across the whole run.
            n_simulations (int): Simulations already run in earlier generations.

        Returns:
            Dict[str, Any] or None: A dictionary with keys "particles", "weights", "distances",
                "simulations" and "n_simulations" (cumulative total), or None if a time/budget
                limit stopped the generation before num_particles were accepted.
        """
        # Prepare and evaluate candidates for this generation
        result = _evaluate.run_particle_evaluations(
            self.particle_inputs,
            self.priors,
            self.rng,
            self.seed_requested,
            n_accepted=num_particles,
            epsilon=epsilon,
            pool=pool,
            scheduler=scheduler,
            start_time=start_time,
            max_time=max_time,
            total_simulations_budget=total_simulations_budget,
            n_simulations=n_simulations,
            particles=particles,
            weights=weights,
            perturbations=perturbations,
            inclusive=particles is None,
        )
        accepted = result["accepted_results"]
        # Discard incomplete generations
        if len(accepted) < num_particles:
            return None
        new_particles = np.array([r["params"] for r in accepted])
        # Compute particle weights
        new_weights = compute_particle_weights(
            new_particles,
            particles,
            weights,
            self.priors,
            self.param_names,
            perturbations,
        )
        return {
            "particles": new_particles,
            "weights": new_weights,
            "distances": np.array([r["distance"] for r in accepted]),
            "simulations": [r["simulation"] for r in accepted],
            "n_simulations": n_simulations + result["n_simulations"],
        }


def _check_stopping_conditions(
    epsilon: Optional[float],
    minimum_epsilon: Optional[float],
    start_time: datetime,
    max_time: Optional[timedelta],
    n_simulations: int,
    total_simulations_budget: Optional[int],
    verbose: bool = True,
) -> bool:
    """Check if any stopping condition is met (epsilon convergence, time limit, or budget).

    Use verbose=False from per-simulation calls to avoid repeated output.

    Args:
        epsilon (float, optional): Current epsilon value
        minimum_epsilon (float, optional): Minimum allowable epsilon value
        start_time (datetime): Start time of the calibration run
        max_time (timedelta, optional): Maximum allowed runtime
        n_simulations (int): Number of simulations performed so far
        total_simulations_budget (int, optional): Maximum number of allowed simulations
        verbose (bool): Whether to print a message when a condition is met

    Returns:
        bool: True if any stopping condition is met, False otherwise
    """
    if (
        minimum_epsilon is not None
        and epsilon is not None
        and epsilon < minimum_epsilon
    ):
        if verbose:
            print("Minimum epsilon reached")
        return True
    if max_time is not None and datetime.now() - start_time >= max_time:
        if verbose:
            print("Maximum time reached")
        return True
    if (
        total_simulations_budget is not None
        and n_simulations >= total_simulations_budget
    ):
        if verbose:
            print("Total simulations budget reached")
        return True
    return False
