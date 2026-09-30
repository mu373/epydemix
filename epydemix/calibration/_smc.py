"""ABC-SMC generations and stopping conditions.

The caller supplies the executor; generations use shared particle evaluation.
"""

import logging
from datetime import datetime, timedelta
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd

from .. import _logging
from ..utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
    compute_particle_weights,
)
from . import _evaluate
from .calibration_results import CalibrationResults

_STOP_MESSAGES = {
    "minimum_epsilon": "Minimum epsilon reached",
    "deadline": "Maximum time reached",
    "budget": "Total simulations budget reached",
}


class SMCRun:
    """Manage SMC generations and retain the RNG on success or failure.

    Generations use shared particle evaluation. Fixed inputs refer to the
    caller's model and observations; workers restore isolated snapshots.
    """

    def __init__(self, particle_inputs, priors, rng, seed_requested, *, metrics):
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
        self.metrics = metrics

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
            self.generation = gen
            self.verbose = verbose
            generation_started = perf_counter()
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
                self._record_generation(generation_started)
                self.metrics["stop_reason"] = self.last_generation["stop_reason"]
                if verbose:
                    print(
                        f"\tEvaluated {self.last_generation['simulations']}, "
                        f"Matched {self.last_generation['matched']}, Retained 0 "
                        f"(stopped: {self.last_generation['stop_reason']})"
                    )
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
                    self.last_generation["matched"]
                    / max(self.last_generation["simulations"], 1)
                    * 100
                )
                print(
                    f"\tAccepted {len(new_gen['particles'])}/{self.last_generation['simulations']} (acceptance rate: {acceptance_rate:.2f}%), "
                    f"Matched {self.last_generation['matched']}, Drained {self.last_generation['drained']}"
                )
                print(f"\tElapsed time: {formatted_time}")

            self._record_generation(generation_started)

            # Diagnose the same boundaries without conflating per-generation counts.
            reason = _stopping_reason(
                epsilon,
                minimum_epsilon,
                start_time,
                max_time,
                n_simulations,
                total_simulations_budget,
            )
            if reason is not None:
                self.metrics["stop_reason"] = reason
                if verbose:
                    print(_STOP_MESSAGES[reason])
                break

        return results

    def _record_generation(self, started):
        """Publish completed or discarded generation work after result construction."""
        self.last_generation["generation_seconds"] = perf_counter() - started
        self.metrics["generations"].append(self.last_generation)
        _logging.emit(
            "generation_finished", run_id=self.metrics["run_id"], **self.last_generation
        )

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
        collected_at = perf_counter()
        last_report = 0
        last_reported_at = collected_at

        def progress(completed, matched):
            nonlocal last_report, last_reported_at
            now = perf_counter()
            if completed > last_report and now - last_reported_at >= 1.0:
                last_report = completed
                last_reported_at = now
                retained = min(matched, num_particles)
                if self.verbose:
                    print(
                        f"\tSimulations: {completed}, Matched: {matched}, Collected: {retained}"
                    )
                _logging.emit(
                    "progress",
                    run_id=self.metrics["run_id"],
                    generation=self.generation,
                    simulations=completed,
                    matched=matched,
                    collected=retained,
                )

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
            progress=progress
            if self.verbose or _logging.logger.isEnabledFor(logging.INFO)
            else None,
        )
        accepted = result["accepted_results"]
        self.last_generation = {
            "generation": self.generation,
            "epsilon": float(epsilon),
            "completed": len(accepted) == num_particles,
            "simulations": result["n_simulations"],
            "matched": result["n_matched"],
            "retained": 0,
            "surplus_accepted": max(0, result["n_matched"] - num_particles),
            "drained": result["n_drained"],
            "collection_seconds": perf_counter() - collected_at,
            "weights_seconds": 0.0,
            "stop_reason": result["stop_reason"],
        }
        # Discard incomplete generations
        if len(accepted) < num_particles:
            return None
        new_particles = np.array([r["params"] for r in accepted])
        # Compute particle weights
        weights_started = perf_counter()
        new_weights = compute_particle_weights(
            new_particles,
            particles,
            weights,
            self.priors,
            self.param_names,
            perturbations,
        )
        self.last_generation["weights_seconds"] = perf_counter() - weights_started
        self.last_generation["retained"] = len(new_particles)
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
    reason = _stopping_reason(
        epsilon,
        minimum_epsilon,
        start_time,
        max_time,
        n_simulations,
        total_simulations_budget,
    )
    if reason is not None and verbose:
        print(_STOP_MESSAGES[reason])
    return reason is not None


def _stopping_reason(
    epsilon,
    minimum_epsilon,
    start_time,
    max_time,
    n_simulations,
    total_simulations_budget,
):
    """Describe the first stopping boundary without printing or changing its meaning."""
    if (
        minimum_epsilon is not None
        and epsilon is not None
        and epsilon < minimum_epsilon
    ):
        return "minimum_epsilon"
    if max_time is not None and datetime.now() - start_time >= max_time:
        return "deadline"
    if (
        total_simulations_budget is not None
        and n_simulations >= total_simulations_budget
    ):
        return "budget"
    return None
