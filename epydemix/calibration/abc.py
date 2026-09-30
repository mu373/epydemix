import copy
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
)
from . import _evaluate
from ._scheduler import SequentialScheduler
from .calibration_results import CalibrationResults
from .metrics import rmse


class ABCSampler:
    """
    Approximate Bayesian Computation (ABC) class implementing different ABC strategies.
    """

    def __init__(
        self,
        simulation_function: Callable,
        priors: Dict[str, Any],
        parameters: Dict[str, Any],
        observed_data: Any,
        distance_function: Callable = rmse,
        rng: Optional[Any] = None,
    ):
        """Initialize ABC calibration.

        Args:
            rng: Optional seed or ``np.random.Generator`` making calibration
                reproducible. It governs all ABC randomness (prior sampling,
                perturbation kernels, resampling) and, when seeding is requested
                (either here or via an ``"rng"`` key in ``parameters``), is also
                injected into the simulation as an ``rng`` key. If None and
                ``parameters`` has no ``"rng"`` key, a fresh unseeded Generator is
                used and the simulation is not seeded.
        """
        self.simulation_function = simulation_function
        self.priors = priors
        self.parameters = parameters.copy()
        # Whether the user asked for reproducibility (via rng= or parameters["rng"]).
        # Only then do we inject rng into the simulation params.
        self._seed_requested = rng is not None or "rng" in self.parameters
        self.rng = np.random.default_rng(
            rng if rng is not None else self.parameters.get("rng")
        )
        self.observed_data = {"data": observed_data}
        self.distance_function = distance_function
        self.param_names = list(priors.keys())
        self.results = None

        # Separate continuous and discrete parameters
        self.continuous_params = [
            name for name in self.param_names if hasattr(priors[name], "pdf")
        ]
        self.discrete_params = [
            name for name in self.param_names if name not in self.continuous_params
        ]

    def calibrate(self, strategy: str = "smc", **kwargs) -> CalibrationResults:
        """Run calibration using the specified strategy.

        This function allows the user to run Approximate Bayesian Computation (ABC) calibration
        using one of the available strategies: Sequential Monte Carlo (SMC), rejection sampling,
        or top fraction selection. The appropriate method is selected based on the `strategy`
        argument, and additional keyword arguments (`**kwargs`) are passed to the corresponding
        function.

        ### Available Strategies:
        - `"smc"`: Uses Sequential Monte Carlo (ABC-SMC) for calibration.
        - `"rejection"`: Uses ABC rejection sampling.
        - `"top_fraction"`: Selects the best fraction of simulated results.

        ### Arguments:
        - **strategy** (`str`, default: `"smc"`):
        Specifies the calibration strategy. Must be one of `{"smc", "rejection", "top_fraction"}`.
        - **kwargs**: Additional parameters depending on the chosen strategy.

        ### Strategy-Specific Arguments:

        #### `"smc"` (Sequential Monte Carlo)
        - `num_particles` (`int`, default: `1000`): Number of particles (samples) per generation.
        - `num_generations` (`int`, default: `10`): Number of generations for the ABC-SMC process.
        - `epsilon_schedule` (`Optional[List[float]]`, default: `None`): Predefined schedule for epsilon values.
        - `epsilon_quantile_level` (`float`, default: `0.5`): Quantile level to adapt epsilon if no schedule is provided.
        - `minimum_epsilon` (`Optional[float]`, default: `None`): Minimum allowable epsilon value.
        - `max_time` (`Optional[timedelta]`, default: `None`): Time limit checked before each simulation.
        - `total_simulations_budget` (`Optional[int]`, default: `None`): Maximum number of allowed simulations.
        - `perturbations` (`Optional[Dict[str, Any]]`, default: `None`): Perturbation kernels for parameters.
        - `verbose` (`bool`, default: `True`): Whether to print progress updates.

        #### `"rejection"` (ABC Rejection Sampling)
        - `epsilon` (`float`, default: `0.1`): Distance threshold for accepting samples.
        - `num_particles` (`int`, default: `1000`): Number of accepted samples.
        - `max_time` (`Optional[timedelta]`, default: `None`): Time limit checked before each simulation.
        - `total_simulations_budget` (`Optional[int]`, default: `None`): Maximum number of allowed simulations.
        - `verbose` (`bool`, default: `True`): Whether to print progress updates.
        - `progress_update_interval` (`int`, default: `1000`): Interval at which progress updates are printed.

        #### `"top_fraction"` (ABC Top-Fraction Selection)
        - `top_fraction` (`float`, default: `0.05`): Fraction of best-fitting simulations to keep.
        - `Nsim` (`int`, default: `100`): Total number of simulations to run.
        - `verbose` (`bool`, default: `True`): Whether to print progress updates.

        ### Returns:
        - `CalibrationResults`: A deep copy of the results from the chosen calibration strategy.

        ### Raises:
        - `ValueError`: If an unknown strategy is specified.

        Example Usage:
        ```python
        # Run SMC calibration with custom parameters
        results = model.calibrate(strategy="smc", num_particles=500, num_generations=15)

        # Run rejection sampling with a different epsilon threshold
        results = model.calibrate(strategy="rejection", epsilon=0.05, num_particles=2000)

        # Run top fraction selection with a different fraction
        results = model.calibrate(strategy="top_fraction", top_fraction=0.1, Nsim=500)
        ```
        """
        strategies = {
            "smc": self.run_smc,
            "rejection": self.run_rejection,
            "top_fraction": self.run_top_fraction,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of {list(strategies.keys())}"
            )

        self.results = strategies[strategy](**kwargs)
        return copy.deepcopy(self.results)

    def run_smc(
        self,
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
        """
        Run ABC-SMC and retain each complete generation in memory.

        Args:
            num_particles (int, optional): Number of particles per generation. Default is 1000.
            num_generations (int, optional): Number of generations. Default is 10.
            epsilon_schedule (List[float], optional): Epsilon for each generation. If None, epsilon
                is adapted from the previous generation's distances. Default is None.
            epsilon_quantile_level (float, optional): Quantile of the previous distances used as
                epsilon when no schedule is given. Default is 0.5.
            minimum_epsilon (float, optional): Stop once epsilon falls below this value. Default is None.
            max_time (timedelta, optional): Time limit checked before each simulation. Default is None.
            total_simulations_budget (int, optional): Maximum number of simulations across all
                generations. Default is None.
            perturbations (Dict[str, Perturbation], optional): Perturbation kernel per parameter.
                Default is None (default continuous/discrete kernels).
            verbose (bool, optional): Whether to print progress. Default is True.

        Returns:
            CalibrationResults: Results of the last complete generation and its history. Empty if
                generation 0 did not complete.

        """
        return self._execute_smc(
            num_particles,
            num_generations,
            epsilon_schedule,
            epsilon_quantile_level,
            minimum_epsilon,
            max_time,
            total_simulations_budget,
            perturbations,
            verbose,
        )

    def run_rejection(
        self,
        epsilon: float = 0.1,
        num_particles: int = 1000,
        max_time: Optional[timedelta] = None,
        total_simulations_budget: Optional[int] = None,
        verbose: bool = True,
        progress_update_interval: int = 1000,
    ) -> CalibrationResults:
        """
        Run ABC rejection sampling.

        Candidates are drawn from the prior until num_particles have a distance below
        epsilon, or a time/budget limit is reached. Simulations use the sampler RNG
        when a seed was requested. A wall-clock cutoff can change which candidates
        are evaluated.

        Args:
            epsilon (float, optional): Distance threshold for accepting a candidate. Default is 0.1.
            num_particles (int, optional): Number of accepted particles to collect. Default is 1000.
            max_time (timedelta, optional): Time limit checked before each simulation. Default is None.
            total_simulations_budget (int, optional): Maximum number of simulations. Default is None.
            verbose (bool, optional): Whether to print progress. Default is True.
            progress_update_interval (int, optional): Number of simulations between progress
                messages. Default is 1000.

        Returns:
            CalibrationResults: Accepted particles with uniform weights, stored as generation 0.

        Raises:
            ValueError: If progress_update_interval is not positive.
        """
        if progress_update_interval < 1:
            raise ValueError("progress_update_interval must be positive")
        if verbose:
            print(
                f"Starting ABC rejection sampling with {num_particles} particles "
                f"and epsilon threshold {epsilon}"
            )
        last_print = 0

        def progress(completed, accepted):
            # Report progress after candidate evaluations.
            nonlocal last_print
            if verbose and completed - last_print >= progress_update_interval:
                last_print = completed
                print(
                    f"\tSimulations: {completed}, Accepted: {accepted}, "
                    f"Acceptance rate: {accepted / completed * 100:.2f}%"
                )

        result = _evaluate.run_particle_evaluations(
            self._get_particle_inputs(),
            self.priors,
            self.rng,
            self._seed_requested,
            n_accepted=num_particles,
            scheduler=SequentialScheduler(),
            epsilon=epsilon,
            start_time=datetime.now(),
            max_time=max_time,
            total_simulations_budget=total_simulations_budget,
            progress=progress,
        )
        accepted = result["accepted_results"]
        completed = result["n_simulations"]
        if verbose:
            print(
                f"\tFinal: {len(accepted)} particles accepted from {completed} simulations "
                f"({len(accepted) / max(completed, 1) * 100:.2f}% acceptance rate)"
            )
        return self._create_results(
            "rejection",
            pd.DataFrame([r["params"] for r in accepted], columns=self.param_names),
            np.ones(len(accepted)) / max(len(accepted), 1),
            np.array([r["distance"] for r in accepted]),
            [r["simulation"] for r in accepted],
        )

    def run_top_fraction(
        self,
        top_fraction: float = 0.05,
        Nsim: int = 100,
        verbose: bool = True,
    ) -> CalibrationResults:
        """
        Run ABC top fraction selection.

        Runs Nsim simulations from the prior and keeps those whose distance is within
        the top_fraction quantile.

        Args:
            top_fraction (float, optional): Fraction of best-fitting simulations to keep, in (0, 1].
                Default is 0.05.
            Nsim (int, optional): Number of simulations to run. Default is 100.
            verbose (bool, optional): Whether to print progress. Default is True.

        Returns:
            CalibrationResults: Selected particles with uniform weights, stored as generation 0.

        Raises:
            ValueError: If Nsim is not positive or top_fraction is not in (0, 1].
        """
        if Nsim < 1 or not 0 < top_fraction <= 1:
            raise ValueError("Nsim must be positive and top_fraction must be in (0, 1]")
        if verbose:
            print(
                f"Starting ABC top fraction selection with {Nsim} simulations "
                f"and top {top_fraction * 100:.1f}% selected"
            )

        def progress(completed, accepted):
            if completed % max(1, Nsim // 10) == 0 or completed == Nsim:
                print(
                    f"\tProgress: {completed}/{Nsim} simulations completed "
                    f"({completed / Nsim * 100:.1f}%)"
                )

        results = _evaluate.run_particle_evaluations(
            self._get_particle_inputs(),
            self.priors,
            self.rng,
            self._seed_requested,
            n_evaluations=Nsim,
            epsilon=None,
            progress=progress if verbose else None,
        )["accepted_results"]
        distances = np.array([r["distance"] for r in results])
        threshold = np.quantile(distances, top_fraction)
        mask = distances <= threshold
        if verbose:
            print(
                f"\tSelected {sum(mask)} particles (top {top_fraction * 100:.1f}%) "
                f"with distance threshold {threshold:.6f}"
            )
        return self._create_results(
            "top_fraction",
            pd.DataFrame([r["params"] for r in results], columns=self.param_names)[
                mask
            ],
            np.ones(sum(mask)) / sum(mask),
            distances[mask],
            [r["simulation"] for r, keep in zip(results, mask) if keep],
        )

    def _create_results(
        self,
        strategy: str,
        particles: pd.DataFrame,
        weights: np.ndarray,
        distances: np.ndarray,
        simulations: List[Dict],
    ) -> CalibrationResults:
        """
        Create a CalibrationResults object holding a single generation.

        Args:
            strategy (str): Name of the calibration strategy.
            particles (pd.DataFrame): Accepted parameter values, one column per parameter.
            weights (np.ndarray): Particle weights.
            distances (np.ndarray): Particle distances.
            simulations (List[Dict]): Simulation output of each particle.

        Returns:
            CalibrationResults: Results with the data stored as generation 0.
        """
        return CalibrationResults(
            calibration_strategy=strategy,
            posterior_distributions={0: particles},
            selected_trajectories={0: simulations},
            distances={0: distances},
            weights={0: weights},
            observed_data=self.observed_data,
            priors=self.priors,
        )

    def run_projections(
        self,
        parameters: Dict[str, Any],
        iterations: int = 100,
        generation: Optional[int] = None,
        scenario_id: str = "baseline",
        rng: Optional[Any] = None,
    ) -> CalibrationResults:
        """
        Run projections using parameters sampled from the posterior distribution.

        Args:
            parameters: Dictionary of parameters for the projections
            iterations: Number of projection iterations to run. Default is 100.
            generation: Which generation to use for posterior. If None, the last generation is used.
            scenario_id: Identifier for this projection scenario. Default is "baseline".
            rng: Optional seed or ``np.random.Generator`` for this call, seeding both
                the posterior resampling and the simulation of every iteration.
                Results depend only on the seed, so two scenarios given the same seed
                are paired: they draw the same posterior samples and diverge only
                where their scenario-specific parameters differ. If None, the seed is
                taken from an ``"rng"`` key in ``parameters``, otherwise from the
                sampler's own ``rng`` (so seeding the ``ABCSampler`` already makes its
                projections reproducible). Pass ``rng`` explicitly only to override
                this, e.g. to draw an independent ensemble from the same calibration.

        Returns:
            CalibrationResults: A new CalibrationResults object containing the original results plus the new projections
        """

        # Get posterior distribution and weights from specified generation
        posterior = self.results.get_posterior_distribution(generation)
        weights = self.results.get_weights(generation)

        # Determine the seed source and whether to seed the simulation. Precedence:
        # this call's rng= arg, then an "rng" key in the projection parameters, then
        # the sampler's own rng, so seeding the ABCSampler makes calibration and
        # projections reproducible as a set.
        # If none of these was seeded, projections run unseeded and no rng is injected.
        if rng is not None:
            seed_source, inject_rng = rng, True
        elif "rng" in parameters:
            seed_source, inject_rng = parameters["rng"], True
        elif self._seed_requested:
            seed_source, inject_rng = self.rng, True
        else:
            seed_source, inject_rng = None, False

        # Build a fixed child rng per iteration (trajectory) via spawn_key, so paired
        # scenarios (two run_projections calls with the same seed) get identical
        # children. Deriving from the seed's entropy also makes this independent of
        # how far the sampler's rng was advanced during calibration.
        base_bit_generator = np.random.default_rng(seed_source).bit_generator
        # ``bit_generator.seed_seq`` is public only since NumPy 1.25; fall back to the
        # private backing attribute on older NumPy (e.g. the 1.24.x that ships with
        # Python 3.8).
        base_seed_seq = getattr(base_bit_generator, "seed_seq", None)
        if base_seed_seq is None:
            base_seed_seq = base_bit_generator._seed_seq
        child_seed_seqs = [
            np.random.SeedSequence(
                base_seed_seq.entropy,
                spawn_key=base_seed_seq.spawn_key + (i,),
                pool_size=base_seed_seq.pool_size,
            )
            for i in range(iterations)
        ]

        # Run projections and store results
        projections, posterior_samples = [], {}
        for i in range(iterations):
            # Each iteration (trajectory) uses its own child rng
            rng_i = np.random.default_rng(child_seed_seqs[i])

            # Sample from posterior according to weights
            idx = rng_i.choice(len(posterior), p=weights / weights.sum())
            posterior_sample = posterior.iloc[idx]

            for k in posterior_sample.keys():
                if k not in posterior_samples:
                    posterior_samples[k] = []
                posterior_samples[k].append(posterior_sample[k])

            proj_params = parameters.copy()
            proj_params.update(posterior_sample)
            # Set rng last so this iteration's child overrides any "rng" already in
            # parameters.
            if inject_rng:
                proj_params["rng"] = rng_i
            result = _evaluate.simulate_projection(
                self.simulation_function, proj_params
            )
            projections.append(result)

        self.results.projections[scenario_id] = projections
        self.results.projection_parameters[scenario_id] = pd.DataFrame(
            posterior_samples
        )

        return copy.deepcopy(self.results)

    def _execute_smc(
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
    ) -> CalibrationResults:
        """
        Run the ABC-SMC generations sequentially.

        Args:
            num_particles, num_generations, epsilon_schedule, epsilon_quantile_level,
            minimum_epsilon, max_time, total_simulations_budget, perturbations, verbose:
                See `ABCSampler.run_smc`.

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
                epsilon = np.quantile(distances, epsilon_quantile_level)
            if verbose:
                print(
                    f"\nGeneration {gen + 1}/{num_generations} (epsilon: {epsilon:.6f})"
                )
            if gen > 0:
                for perturbation in perturbations.values():
                    perturbation.update(particles, weights, self.param_names)

            new_gen = self._run_smc_generation(
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
            if self._check_stopping_conditions(
                epsilon,
                minimum_epsilon,
                start_time,
                max_time,
                n_simulations,
                total_simulations_budget,
            ):
                break

        return results

    def _run_smc_generation(
        self,
        particles,
        weights,
        epsilon,
        num_particles,
        perturbations,
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
            start_time (datetime): Start of the calibration run.
            max_time (timedelta, optional): Time limit measured from start_time.
            total_simulations_budget (int, optional): Maximum number of simulations across the whole run.
            n_simulations (int): Simulations already run in earlier generations.

        Returns:
            Dict[str, Any] or None: A dictionary with keys "particles", "weights", "distances",
                "simulations" and "n_simulations" (cumulative total), or None if a time/budget
                limit stopped the generation before num_particles were accepted.
        """
        result = _evaluate.run_particle_evaluations(
            self._get_particle_inputs(),
            self.priors,
            self.rng,
            self._seed_requested,
            n_accepted=num_particles,
            scheduler=SequentialScheduler(),
            epsilon=epsilon,
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
        if len(accepted) < num_particles:
            return None
        new_particles = np.array([r["params"] for r in accepted])
        new_weights = self._compute_particle_weights(
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

    def _compute_particle_weights(
        self,
        particles,
        previous_particles,
        previous_weights,
        priors,
        param_names,
        perturbations,
    ):
        """Compute normalized ABC-SMC importance weights without changing inputs.

        Rows of both particle arrays follow ``param_names``. If ``previous_particles``
        is None, the candidates were drawn from the prior and receive uniform weights.
        Otherwise, divide the joint prior density by the previous weighted kernel
        mixture. Kernels must already be updated for the previous generation.
        """
        continuous_params = {
            name for name in param_names if hasattr(priors[name], "pdf")
        }
        # Generation 0 samples the prior, so all weights are equal. Later
        # generations use the ABC-SMC importance weight
        #   w_i = prior(theta_i) / sum_j w_j * K(theta_i | theta_j),
        # where K is the product of the per-parameter perturbation kernels.
        new_weights = np.ones(len(particles))
        if previous_particles is not None:
            for i, params in enumerate(particles):
                numerator = np.prod(
                    [
                        priors[p].pdf(params[k])
                        if p in continuous_params
                        else priors[p].pmf(params[k])
                        for k, p in enumerate(param_names)
                    ]
                )
                denominator = np.sum(
                    [
                        previous_weights[j]
                        * np.prod(
                            [
                                perturbations[p].pdf(
                                    params[k], previous_particles[j, k]
                                )
                                for k, p in enumerate(param_names)
                            ]
                        )
                        for j in range(len(previous_particles))
                    ]
                )
                new_weights[i] = numerator / denominator
        new_weights /= new_weights.sum()
        return new_weights

    def _check_stopping_conditions(
        self,
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

    def _get_particle_inputs(self):
        """Return the fixed inputs shared by all candidate evaluations."""
        return (
            self.simulation_function,
            self.parameters,
            self.param_names,
            self.observed_data,
            self.distance_function,
        )
