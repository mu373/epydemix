"""ABC-SMC generations, stopping and checkpoint lifecycle.

The caller supplies the executor; generations use shared particle evaluation.
SMCRun owns inference state and coarse metrics, delegating archive I/O to _checkpoint.
"""

import copy
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from .. import _logging
from ..utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
    compute_particle_weights,
)
from . import _checkpoint, _evaluate, _history
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
        """
        Initialize the SMC run state and unpack fixed evaluation inputs.

        Args:
            particle_inputs (tuple): Fixed inputs (simulation_function, parameters,
                param_names, observed_data, distance_function).
            priors (dict): Prior distributions for calibrated parameters.
            rng (np.random.Generator): Random number generator for this run.
            seed_requested (bool): Whether the user provided an explicit seed.
            metrics (dict): Mutable dictionary collecting calibration metrics and timings.
        """
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
        checkpoint_path,
        resume,
        history_storage,
    ) -> CalibrationResults:
        """
        Run the ABC-SMC generations inside an already opened worker pool.

        Args:
            num_particles, num_generations, epsilon_schedule, epsilon_quantile_level,
            minimum_epsilon, max_time, total_simulations_budget, perturbations, verbose,
            checkpoint_path, resume, history_storage:
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

        if history_storage not in ("memory", "disk"):
            raise ValueError("history_storage must be 'memory' or 'disk'")
        if history_storage == "disk" and checkpoint_path is None:
            raise ValueError("history_storage='disk' requires checkpoint_path")
        self.history_storage = history_storage
        self.history_directory = (
            Path(str(checkpoint_path) + ".history")
            if history_storage == "disk"
            else None
        )
        self.metrics["history_storage"] = history_storage
        self.metrics["history_io_seconds"] = 0.0
        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        inputs, metadata, restored = self._prepare_checkpoint(
            checkpoint_path,
            resume,
            num_particles,
            num_generations,
            epsilon_schedule,
            epsilon_quantile_level,
            minimum_epsilon,
            perturbations,
            total_simulations_budget,
        )
        if restored is not None:
            perturbations = restored["perturbations"]

        if verbose:
            print(
                f"Starting ABC-SMC with {num_particles} particles and {num_generations} generations"
            )

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
            history_started = perf_counter()
            particles = results.get_posterior_distribution().to_numpy()
            weights = results.get_weights()
            distances = results.get_distances()
            if history_storage == "disk":
                self.metrics["history_io_seconds"] += perf_counter() - history_started
            reason = _stopping_reason(
                restored["epsilon"],
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
                return results

        for gen in range(first_generation, num_generations):
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
            if history_storage == "disk":
                history_started = perf_counter()
                _history.save_generation(
                    results,
                    gen,
                    values,
                    self.history_directory,
                    metadata["environment"],
                )
                elapsed = perf_counter() - history_started
                self.last_generation["history_io_seconds"] = elapsed
                self.metrics["history_io_seconds"] += elapsed
            else:
                for name, value in values.items():
                    getattr(results, name)[gen] = value
            del values

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

            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path,
                    inputs,
                    metadata,
                    results,
                    perturbations,
                    {
                        "completed_generation": gen,
                        "n_simulations": n_simulations,
                        "elapsed_seconds": elapsed_before + monotonic() - call_start,
                        "epsilon": epsilon,
                        "num_generations": num_generations,
                        "total_simulations_budget": total_simulations_budget,
                        "max_time_seconds": max_time.total_seconds()
                        if max_time is not None
                        else None,
                        "n_workers": getattr(pool, "_max_workers", None),
                    },
                    overwrite=resume or gen > 0,
                )

            # Release disk-mode trajectories before requesting the next generation.
            del new_gen
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

    def _prepare_checkpoint(
        self,
        path,
        resume,
        num_particles,
        num_generations,
        epsilon_schedule,
        epsilon_quantile_level,
        minimum_epsilon,
        perturbations,
        total_simulations_budget,
    ):
        """Validate persistence options and snapshot inputs before model evaluation."""
        started = perf_counter()
        self.metrics["checkpoint_io_seconds"] = 0.0
        if resume and path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if path is None:
            self.metrics["checkpoint_setup_seconds"] = perf_counter() - started
            return None, None, None
        if not self.seed_requested:
            raise ValueError("Checkpointing requires an explicit rng seed or Generator")
        if not resume and path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}; use resume=True")
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {path.parent}"
            )
        inputs = self._snapshot_checkpoint_inputs(
            num_particles,
            epsilon_schedule,
            epsilon_quantile_level,
            minimum_epsilon,
            perturbations,
        )
        fingerprint = _checkpoint.input_hash(inputs)
        _checkpoint.validate_picklable(inputs)
        if resume:
            restored, metadata = self._restore_checkpoint(
                path,
                fingerprint,
                num_generations,
                total_simulations_budget,
            )
            inputs = restored["inputs"]
        else:
            restored = None
            metadata = {
                "strategy": "smc",
                "history_storage": self.history_storage,
                "run_id": str(uuid4()),
                "call_run_id": self.metrics["run_id"],
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
        self.metrics["checkpoint_setup_seconds"] = perf_counter() - started
        return inputs, metadata, restored

    def _restore_checkpoint(
        self, path, fingerprint, num_generations, total_simulations_budget
    ):
        """Verify compatibility before restoring kernels/RNG or publishing resume."""
        started = perf_counter()
        restored, metadata = _checkpoint.read_checkpoint(path)
        self.metrics["checkpoint_io_seconds"] += perf_counter() - started
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
        if metadata.get("history_storage", "memory") != self.history_storage:
            raise ValueError("history_storage must match the saved checkpoint")
        if self.history_storage == "disk":
            started = perf_counter()
            _history.bind_history(restored["results"], self.history_directory)
            self.metrics["history_io_seconds"] += perf_counter() - started
        current_environment = _checkpoint.environment()
        if metadata["environment"] != current_environment:
            warnings.warn(
                "Checkpoint code or library versions differ; exact replay is not guaranteed.",
                RuntimeWarning,
                stacklevel=3,
            )
        metadata["environment"] = current_environment
        self.rng = np.random.Generator(
            restored["rng_type"](restored["rng_seed_sequence"])
        )
        self.rng.bit_generator.state = restored["rng_state"]
        self.metrics["resumed_committed_simulations"] = restored["n_simulations"]
        _logging.emit(
            "checkpoint_resumed",
            run_id=self.metrics["run_id"],
            checkpoint_id=metadata["run_id"],
            previous_run_id=metadata.get("call_run_id"),
            checkpoint_path=str(path.resolve()),
            next_generation=restored["next_generation"],
            committed_simulations=restored["n_simulations"],
        )
        return restored, metadata

    def _save_checkpoint(
        self, path, inputs, metadata, results, perturbations, progress, *, overwrite
    ):
        """Persist one committed generation atomically, then publish its identity."""
        state = self._build_checkpoint_state(
            inputs,
            results,
            perturbations,
            next_generation=progress["completed_generation"] + 1,
            epsilon=progress["epsilon"],
            n_simulations=progress["n_simulations"],
            elapsed_seconds=progress["elapsed_seconds"],
        )
        metadata.update({**progress, "epsilon": str(progress["epsilon"])})
        metadata["call_run_id"] = self.metrics["run_id"]
        started = perf_counter()
        _checkpoint.write_checkpoint(path, state, metadata, overwrite=overwrite)
        elapsed = perf_counter() - started
        self.metrics["checkpoint_io_seconds"] += elapsed
        self.last_generation["checkpoint_io_seconds"] = elapsed
        _logging.emit(
            "checkpoint_saved",
            run_id=self.metrics["run_id"],
            checkpoint_id=metadata["run_id"],
            checkpoint_path=str(path.resolve()),
            generation=progress["completed_generation"],
            committed_simulations=progress["n_simulations"],
            io_seconds=elapsed,
        )

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
            "checkpoint_io_seconds": 0.0,
            "history_io_seconds": 0.0,
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


def _stopping_reason(
    epsilon,
    minimum_epsilon,
    start_time,
    max_time,
    n_simulations,
    total_simulations_budget,
):
    """
    Describe the first stopping boundary without printing or changing its meaning.

    Args:
        epsilon (float or None): Current generation distance threshold.
        minimum_epsilon (float or None): Stopping threshold for epsilon.
        start_time (datetime): Timestamp when calibration started.
        max_time (timedelta or None): Maximum allowable calibration duration.
        n_simulations (int): Total number of simulations evaluated so far.
        total_simulations_budget (int or None): Maximum number of allowed simulations.

    Returns:
        str or None: Identifier for the stopping condition ('minimum_epsilon', 'deadline',
            or 'budget'), or None if no stopping condition has been reached.
    """
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
