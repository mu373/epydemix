"""ABC-SMC run lifecycle: generations, stopping, checkpointing and resume.

The caller supplies the executor; generations use shared particle evaluation. This module
owns SMC state and delegates archive I/O to the checkpoint module.
"""

import copy
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from ..utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
    compute_particle_weights,
)
from . import _checkpoint, _evaluate
from .calibration_results import CalibrationResults


class SMCRun:
    "Manage SMC generations and retain the current RNG on success or failure.\n\nEach generation uses _evaluate.run_particle_evaluations to evaluate candidates.\nFixed inputs are references to the caller's model and data."

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
        checkpoint_path,
        resume,
    ) -> CalibrationResults:
        """
        Run the ABC-SMC generations inside an already opened worker pool.

        Args:
            num_particles, num_generations, epsilon_schedule, epsilon_quantile_level,
            minimum_epsilon, max_time, total_simulations_budget, perturbations, verbose,
            checkpoint_path, resume: See `ABCSampler.run_smc`.
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

        # --- Validate options -------------------------------------------------
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        # --- Checkpoint setup -------------------------------------------------
        # inputs: deep-copied snapshot of everything that defines the run; its
        #   hash guards against resuming with different data or settings.
        # metadata: human-readable manifest, updated after every generation.
        # restored: saved state when resuming, otherwise None.
        inputs = metadata = restored = None
        if checkpoint_path is not None:
            if not self.seed_requested:
                raise ValueError(
                    "Checkpointing requires an explicit rng seed or Generator"
                )
            checkpoint_path = Path(checkpoint_path)
            if not resume and checkpoint_path.exists():
                raise FileExistsError(
                    f"Checkpoint already exists: {checkpoint_path}; use resume=True"
                )
            if not checkpoint_path.parent.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint directory does not exist: {checkpoint_path.parent}"
                )
            inputs = self._snapshot_checkpoint_inputs(
                num_particles,
                epsilon_schedule,
                epsilon_quantile_level,
                minimum_epsilon,
                perturbations,
            )
            fingerprint = _checkpoint.input_hash(inputs)
            # Fail before running simulations if the snapshot cannot be serialized.
            _checkpoint.validate_picklable(inputs)
            if resume:
                # Load the last complete generation and check it is compatible
                # with this call. Settings that only extend the run (more
                # generations, larger budget, other worker count) may change.
                restored, metadata = _checkpoint.read_checkpoint(checkpoint_path)
                if fingerprint != metadata["input_sha256"]:
                    raise ValueError(
                        "Checkpoint inputs or SMC settings do not match this sampler"
                    )
                if num_generations < restored["next_generation"]:
                    raise ValueError("num_generations precedes the saved generation")
                if (
                    total_simulations_budget is not None
                    and total_simulations_budget < restored["n_simulations"]
                ):
                    raise ValueError(
                        "total_simulations_budget is below the saved simulation count"
                    )
                current_environment = _checkpoint.environment()
                if metadata["environment"] != current_environment:
                    warnings.warn(
                        "Checkpoint code or library versions differ; exact replay is not guaranteed.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                metadata["environment"] = current_environment
                # Continue from the saved kernels and RNG position, so a resumed
                # run matches an uninterrupted one.
                inputs = restored["inputs"]
                perturbations = restored["perturbations"]
                self.rng = np.random.Generator(
                    restored["rng_type"](restored["rng_seed_sequence"])
                )
                self.rng.bit_generator.state = restored["rng_state"]
            else:
                metadata = {
                    "strategy": "smc",
                    "run_id": str(uuid4()),
                    "param_names": self.param_names,
                    "num_particles": int(num_particles),
                    "epsilon_quantile_level": float(epsilon_quantile_level),
                    "minimum_epsilon": str(minimum_epsilon)
                    if minimum_epsilon is not None
                    else None,
                    "epsilon_schedule": [str(v) for v in epsilon_schedule]
                    if epsilon_schedule is not None
                    else None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "input_sha256": fingerprint,
                    "environment": _checkpoint.environment(),
                }

        if verbose:
            print(
                f"Starting ABC-SMC with {num_particles} particles and {num_generations} generations"
            )

        # --- Run generations --------------------------------------------------
        # start_time drives max_time, which restarts with every call.
        # call_start + elapsed_before track total run time across resumes.
        start_time = datetime.now()
        call_start = monotonic()
        elapsed_before = restored["elapsed_seconds"] if restored is not None else 0.0
        n_simulations = restored["n_simulations"] if restored is not None else 0
        results = (
            restored["results"]
            if restored is not None
            else CalibrationResults(
                calibration_strategy="smc",
                observed_data=self.observed_data,
                priors=self.priors,
            )
        )
        particles = weights = distances = None
        first_generation = restored["next_generation"] if restored is not None else 0
        if restored is not None:
            # State of the last saved generation, needed to propose the next one.
            particles = results.get_posterior_distribution().to_numpy()
            weights = results.get_weights()
            distances = results.get_distances()
            if _check_stopping_conditions(
                restored["epsilon"],
                minimum_epsilon,
                start_time,
                max_time,
                n_simulations,
                total_simulations_budget,
                verbose=verbose,
            ):
                return results

        for gen in range(first_generation, num_generations):
            start_generation_time = datetime.now()

            if epsilon_schedule is not None:
                epsilon = epsilon_schedule[gen]
            elif gen == 0:
                epsilon = float("inf")
            else:
                epsilon = np.quantile(distances, epsilon_quantile_level)
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

            # Save after every complete generation; an interrupted generation
            # is simply rerun on resume.
            if checkpoint_path is not None:
                elapsed = elapsed_before + monotonic() - call_start
                state = self._build_checkpoint_state(
                    inputs,
                    results,
                    perturbations,
                    next_generation=gen + 1,
                    epsilon=epsilon,
                    n_simulations=n_simulations,
                    elapsed_seconds=elapsed,
                )
                metadata.update(
                    {
                        "completed_generation": gen,
                        "n_simulations": n_simulations,
                        "elapsed_seconds": elapsed,
                        "epsilon": str(epsilon),
                        "num_generations": num_generations,
                        "total_simulations_budget": total_simulations_budget,
                        "max_time_seconds": max_time.total_seconds()
                        if max_time is not None
                        else None,
                        "n_workers": getattr(pool, "_max_workers", None),
                    }
                )
                _checkpoint.write_checkpoint(
                    checkpoint_path,
                    state,
                    metadata,
                    # The first write of a fresh run must not replace a file.
                    overwrite=resume or gen > 0,
                )

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

    def _snapshot_checkpoint_inputs(
        self,
        num_particles,
        epsilon_schedule,
        epsilon_quantile_level,
        minimum_epsilon,
        perturbations,
    ):
        """Copy the inputs defining a run for checkpoint hashing and persistence.

        Excludes the caller's RNG: its current state is saved separately.
        Does not mutate sampler inputs or perform I/O.
        """
        return copy.deepcopy(
            {
                "observed_data": self.observed_data,
                "parameters": {k: v for k, v in self.parameters.items() if k != "rng"},
                "priors": self.priors,
                "param_names": self.param_names,
                "simulation_function": self.simulation_function,
                "distance_function": self.distance_function,
                "num_particles": num_particles,
                "epsilon_schedule": epsilon_schedule,
                "epsilon_quantile_level": epsilon_quantile_level,
                "minimum_epsilon": minimum_epsilon,
                "perturbations": perturbations,
            }
        )

    def _build_checkpoint_state(
        self,
        inputs,
        results,
        perturbations,
        *,
        next_generation,
        epsilon,
        n_simulations,
        elapsed_seconds,
    ):
        """Assemble resumable state without I/O or advancing the sampler RNG.

        Results, input snapshots and kernels are referenced, not copied; the
        caller must serialize the state before continuing calibration.
        """
        return {
            "inputs": inputs,
            "results": results,
            "rng_type": type(self.rng.bit_generator),
            "rng_state": self.rng.bit_generator.state,
            # seed_seq is public only since NumPy 1.25.
            "rng_seed_sequence": getattr(self.rng.bit_generator, "seed_seq", None)
            or self.rng.bit_generator._seed_seq,
            "perturbations": perturbations,
            "next_generation": next_generation,
            "epsilon": epsilon,
            "n_simulations": n_simulations,
            "elapsed_seconds": elapsed_seconds,
        }

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
