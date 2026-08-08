from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
)
from ..calibration_results import CalibrationResults
from . import common
from .common import SamplerContext


def build_default_perturbations(ctx: SamplerContext):
    """Build default perturbation kernels based on parameter types."""
    return {
        param: (
            DefaultPerturbationContinuous(param)
            if param in ctx.continuous_params
            else DefaultPerturbationDiscrete(param, ctx.priors[param])
        )
        for param in ctx.param_names
    }


def initialize_particles(
    ctx: SamplerContext,
    num_particles: int,
    epsilon: float,
    start_time: Optional[datetime] = None,
    max_time: Optional[timedelta] = None,
    total_simulations_budget: Optional[int] = None,
    n_simulations: int = 0,
) -> Optional[Dict[str, Any]]:
    """Initialize generation 0 particles by sampling directly from priors."""
    particles, weights, distances, simulations = [], [], [], []

    while len(particles) < num_particles:
        if common.check_stopping_conditions(
            None,
            None,
            start_time,
            max_time,
            n_simulations,
            total_simulations_budget,
            verbose=False,
        ):
            return None

        params = common.sample_params(ctx)
        simulated_data = common.run_simulation(ctx, params)
        dist = ctx.distance_function(data=ctx.observed_data, simulation=simulated_data)
        n_simulations += 1

        if dist <= epsilon:
            particles.append(params)
            weights.append(1.0 / num_particles)
            distances.append(dist)
            simulations.append(simulated_data)

    return {
        "particles": np.array(particles),
        "weights": np.array(weights),
        "distances": np.array(distances),
        "simulations": simulations,
        "n_simulations": n_simulations,
    }


def smc_generation(
    ctx: SamplerContext,
    particles: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
    num_particles: int,
    perturbations: Dict[str, Any],
    start_time: Optional[datetime] = None,
    max_time: Optional[timedelta] = None,
    total_simulations_budget: Optional[int] = None,
    n_simulations: int = 0,
) -> Optional[Dict[str, Any]]:
    """Run one SMC generation from a previous particle population."""
    new_particles, new_weights, new_distances, new_simulations = [], [], [], []

    for _ in range(num_particles):
        while True:
            if common.check_stopping_conditions(
                None,
                None,
                start_time,
                max_time,
                n_simulations,
                total_simulations_budget,
                verbose=False,
            ):
                return None

            index = np.random.choice(len(particles), p=weights / weights.sum())
            candidate_params = particles[index]

            perturbed_params = [
                perturbations[ctx.param_names[i]].propose(candidate_params[i])
                for i in range(len(ctx.param_names))
            ]

            prior_probabilities = [
                ctx.priors[param].pdf(perturbed_params[i])
                if param in ctx.continuous_params
                else ctx.priors[param].pmf(perturbed_params[i])
                for i, param in enumerate(ctx.param_names)
            ]

            if all(prob > 0 for prob in prior_probabilities):
                simulation = common.run_simulation(ctx, perturbed_params)
                distance = ctx.distance_function(ctx.observed_data, simulation)
                n_simulations += 1

                if distance < epsilon:
                    new_particles.append(perturbed_params)
                    weight_numerator = np.prod(
                        [
                            ctx.priors[param].pdf(perturbed_params[i])
                            if param in ctx.continuous_params
                            else ctx.priors[param].pmf(perturbed_params[i])
                            for i, param in enumerate(ctx.param_names)
                        ]
                    )
                    weight_denominator = np.sum(
                        [
                            weights[j]
                            * np.prod(
                                [
                                    perturbations[ctx.param_names[i]].pdf(
                                        perturbed_params[i], particles[j][i]
                                    )
                                    for i in range(len(ctx.param_names))
                                ]
                            )
                            for j in range(len(particles))
                        ]
                    )
                    new_weights.append(weight_numerator / weight_denominator)
                    new_distances.append(distance)
                    new_simulations.append(simulation)
                    break

    new_weights = np.array(new_weights)
    new_weights /= new_weights.sum()

    return {
        "particles": np.array(new_particles),
        "weights": np.array(new_weights),
        "distances": np.array(new_distances),
        "simulations": new_simulations,
        "n_simulations": n_simulations,
    }


def run(
    ctx: SamplerContext,
    num_particles: int = 1000,
    num_generations: int = 10,
    epsilon_schedule: Optional[List[float]] = None,
    epsilon_quantile_level: float = 0.5,
    minimum_epsilon: Optional[float] = None,
    max_time: Optional[timedelta] = None,
    total_simulations_budget: Optional[int] = None,
    perturbations: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> CalibrationResults:
    """Run serial ABC-SMC calibration."""
    if perturbations is None:
        perturbations = build_default_perturbations(ctx)

    if verbose:
        print(
            f"Starting ABC-SMC with {num_particles} particles and {num_generations} generations"
        )

    start_time = datetime.now()
    n_simulations = 0
    results = None

    for gen in range(num_generations):
        start_generation_time = datetime.now()

        if gen == 0:
            epsilon = epsilon_schedule[0] if epsilon_schedule is not None else float("inf")
            if verbose:
                print(
                    f"\nGeneration {gen + 1}/{num_generations} (epsilon: {epsilon:.6f})"
                )

            new_gen = initialize_particles(
                ctx,
                num_particles,
                epsilon,
                start_time,
                max_time,
                total_simulations_budget,
                n_simulations,
            )
            if new_gen is None:
                if verbose:
                    print("Maximum time or budget reached during generation 0")
                break

            n_simulations = new_gen["n_simulations"]
            results = common.create_results(
                ctx,
                "smc",
                pd.DataFrame(
                    data={
                        ctx.param_names[i]: new_gen["particles"][:, i]
                        for i in range(len(ctx.param_names))
                    }
                ),
                new_gen["weights"],
                new_gen["distances"],
                new_gen["simulations"],
            )

            particles = new_gen["particles"]
            weights = new_gen["weights"]
            distances = new_gen["distances"]

        else:
            epsilon = (
                epsilon_schedule[gen]
                if epsilon_schedule is not None
                else np.quantile(distances, epsilon_quantile_level)
            )

            if verbose:
                print(
                    f"\nGeneration {gen + 1}/{num_generations} (epsilon: {epsilon:.6f})"
                )

            for perturbation in perturbations.values():
                perturbation.update(particles, weights, ctx.param_names)

            new_gen = smc_generation(
                ctx,
                particles,
                weights,
                epsilon,
                num_particles,
                perturbations,
                start_time,
                max_time,
                total_simulations_budget,
                n_simulations,
            )
            if new_gen is None:
                if verbose:
                    print(
                        f"Maximum time or budget reached during generation {gen + 1}, keeping last complete generation"
                    )
                break

            n_simulations = new_gen["n_simulations"]
            results.posterior_distributions[gen] = pd.DataFrame(
                data={
                    ctx.param_names[i]: new_gen["particles"][:, i]
                    for i in range(len(ctx.param_names))
                }
            )
            results.distances[gen] = new_gen["distances"]
            results.weights[gen] = new_gen["weights"]
            results.selected_trajectories[gen] = new_gen["simulations"]

            particles = new_gen["particles"]
            weights = new_gen["weights"]
            distances = new_gen["distances"]

        if verbose:
            end_generation_time = datetime.now()
            elapsed_time = end_generation_time - start_generation_time
            formatted_time = (
                f"{elapsed_time.seconds // 3600:02}:"
                f"{(elapsed_time.seconds % 3600) // 60:02}:"
                f"{elapsed_time.seconds % 60:02}"
            )
            acceptance_rate = len(new_gen["particles"]) / new_gen["n_simulations"] * 100
            print(
                f"\tAccepted {len(new_gen['particles'])}/{new_gen['n_simulations']} (acceptance rate: {acceptance_rate:.2f}%)"
            )
            print(f"\tElapsed time: {formatted_time}")

        if common.check_stopping_conditions(
            epsilon,
            minimum_epsilon,
            start_time,
            max_time,
            n_simulations,
            total_simulations_budget,
        ):
            break

    if results is None:
        results = CalibrationResults(
            calibration_strategy="smc",
            observed_data=ctx.observed_data,
            priors=ctx.priors,
        )

    return results
