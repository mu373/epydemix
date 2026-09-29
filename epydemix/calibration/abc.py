import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from itertools import islice
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.abc_smc_utils import sample_prior
from . import _smc, _worker
from .calibration_results import CalibrationResults
from .metrics import rmse
from .parallel import build_particle_tasks, create_particle_scheduler, executor_context


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
        """
        Initialize ABC calibration.

        Seeded simulations must use parameters["rng"] and avoid shared mutable
        state. With the same seed and numerical environment, results are identical
        across worker counts unless a wall-clock deadline cuts the run short.
        Candidate-specific streams change seeded outputs from earlier versions.

        Args:
            simulation_function (Callable): Function taking a parameter dictionary and returning a
                dictionary of simulated data, compared to the observed data by distance_function.
            priors (Dict[str, Any]): Prior distribution (frozen scipy.stats distribution) for each
                calibrated parameter.
            parameters (Dict[str, Any]): Fixed parameters passed to every simulation.
            observed_data (Any): Observed data to calibrate against.
            distance_function (Callable, optional): Function `(data, simulation) -> float`.
                Default is rmse.
            rng (int or np.random.Generator, optional): Seed or generator making calibration
                reproducible. It governs all ABC randomness (prior sampling, perturbation kernels,
                resampling) and, when seeding is requested (either here or via an "rng" key in
                parameters), is also injected into the simulation as an "rng" key. If None and
                parameters has no "rng" key, a fresh unseeded Generator is used and the simulation
                is not seeded. Default is None.
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

        ### Execution options (all strategies):
        - `n_workers`: Number of worker processes; None runs in the calling process.
        - `executor`: Optional caller-owned ProcessPoolExecutor, taking precedence over n_workers.
        - `parallel_strategy`: Currently only `"dynamic"` is supported.

        Worker counts exceeding detected logical CPU capacity are rejected, including
        caller-owned executors. Linux affinity and visible cgroup quotas are honored;
        fractional quotas are rounded down, with a minimum of one worker. Simulation
        and distance evaluation use one thread per loaded BLAS/OpenMP library in both
        sequential and parallel modes, restoring the previous limits afterward.

        Owned calibration pools receive fixed inputs once at startup. Each candidate
        restores a fresh copy in its worker, preserving isolation of mutable inputs.
        Caller-owned executors retain their initializer and receive full task inputs.

        Set `rng` on ABCSampler to reproduce results across worker counts. Simulation
        functions must use the supplied `parameters["rng"]`; custom perturbations
        must use their supplied rng too. Process workers require picklable functions
        and parameters. Thread executors are not supported because models can mutate
        shared state. A wall-clock `max_time` cutoff stops new submissions and drains
        submitted work; it may change the returned prefix across worker counts.
        A `total_simulations_budget` cutoff is independent of worker count.

        ### Strategy-Specific Arguments:

        #### `"smc"` (Sequential Monte Carlo)
        - `num_particles` (`int`, default: `1000`): Number of particles (samples) per generation.
        - `num_generations` (`int`, default: `10`): Number of generations for the ABC-SMC process.
        - `epsilon_schedule` (`Optional[List[float]]`, default: `None`): Predefined schedule for epsilon values.
        - `epsilon_quantile_level` (`float`, default: `0.5`): Quantile level to adapt epsilon if no schedule is provided.
        - `minimum_epsilon` (`Optional[float]`, default: `None`): Minimum allowable epsilon value.
        - `max_time` (`Optional[timedelta]`, default: `None`): Time limit for submitting work; already-submitted tasks finish.
        - `total_simulations_budget` (`Optional[int]`, default: `None`): Maximum number of allowed simulations.
        - `perturbations` (`Optional[Dict[str, Any]]`, default: `None`): Perturbation kernels for parameters.
        - `verbose` (`bool`, default: `True`): Whether to print progress updates.
        - `checkpoint_path`: Optional file saving inputs and each complete generation.
        - `resume`: Restore a trusted checkpoint; default False. Requires checkpoint_path.
        - `history_storage`: "memory" (default) or "disk" for on-demand generation history.

        #### `"rejection"` (ABC Rejection Sampling)
        - `epsilon` (`float`, default: `0.1`): Distance threshold for accepting samples.
        - `num_particles` (`int`, default: `1000`): Number of accepted samples.
        - `max_time` (`Optional[timedelta]`, default: `None`): Time limit for submitting work; already-submitted tasks finish.
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
        - `TypeError`: If executor is not a ProcessPoolExecutor.

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
        n_workers: Optional[int] = None,
        executor: Optional[ProcessPoolExecutor] = None,
        parallel_strategy: str = "dynamic",
        checkpoint_path: Optional[str] = None,
        resume: bool = False,
        history_storage: str = "memory",
    ) -> CalibrationResults:
        """
        Run ABC-SMC, optionally saving every complete generation to a checkpoint.

        Args:
            num_particles (int, optional): Number of particles per generation. Default is 1000.
            num_generations (int, optional): Number of generations. Default is 10.
            epsilon_schedule (List[float], optional): Epsilon for each generation. If None, epsilon
                is adapted from the previous generation's distances. Default is None.
            epsilon_quantile_level (float, optional): Quantile of the previous distances used as
                epsilon when no schedule is given. Default is 0.5.
            minimum_epsilon (float, optional): Stop once epsilon falls below this value. Default is None.
            max_time (timedelta, optional): Time limit for submitting work; already-submitted
                simulations finish. Restarts on every call, including resumes. Default is None.
            total_simulations_budget (int, optional): Maximum number of simulations across all
                generations. Default is None.
            perturbations (Dict[str, Perturbation], optional): Perturbation kernel per parameter.
                Default is None (default continuous/discrete kernels).
            verbose (bool, optional): Whether to print progress. Default is True.
            n_workers (int, optional): Number of worker processes, at most the detected CPU capacity.
                None runs in the calling process. Default is None.
            executor (ProcessPoolExecutor, optional): Caller-owned worker pool, taking precedence
                over n_workers. Default is None.
            parallel_strategy (str, optional): Parallel scheduling strategy. Only "dynamic" is supported.
                Default is "dynamic".
            checkpoint_path (str or Path, optional): Local snapshot containing inputs, results, RNG and
                kernels. Requires an explicit seed and picklable, hashable input data. Existing files
                are rejected unless resume=True. Only load trusted checkpoints. Default is None.
            resume (bool, optional): Restore the last complete generation from checkpoint_path. The
                sampler must have matching inputs/settings. RNG state comes from the checkpoint;
                workers, target generations and total budget may change. Incomplete generations are
                rerun. Default is False.
            history_storage (str, optional): "memory" or "disk". Disk mode requires a checkpoint and
                stores completed generations in checkpoint_path + ".history". Result mappings read
                individual fields on demand without caching. Keep this directory with the checkpoint;
                use the same mode when resuming. Default is "memory".

        Returns:
            CalibrationResults: Results of the last complete generation and its history. Empty if
                generation 0 did not complete.

        Raises:
            ValueError: If the options are inconsistent, or the checkpoint does not match this sampler.
            FileExistsError: If checkpoint_path exists and resume is False.
            FileNotFoundError: If the checkpoint directory or a history file is missing.
            TypeError: If executor is not a ProcessPoolExecutor, or the inputs cannot be checkpointed.
        """
        scheduler = create_particle_scheduler(parallel_strategy)
        smc_run = _smc.SMCRun(
            self._get_particle_inputs(),
            self.priors,
            self.rng,
            self._seed_requested,
            self._sample_particles,
        )
        try:
            with self._calibration_executor(n_workers, executor) as pool:
                return smc_run.execute(
                    num_particles=num_particles,
                    num_generations=num_generations,
                    epsilon_schedule=epsilon_schedule,
                    epsilon_quantile_level=epsilon_quantile_level,
                    minimum_epsilon=minimum_epsilon,
                    max_time=max_time,
                    total_simulations_budget=total_simulations_budget,
                    perturbations=perturbations,
                    verbose=verbose,
                    pool=pool,
                    scheduler=scheduler,
                    checkpoint_path=checkpoint_path,
                    resume=resume,
                    history_storage=history_storage,
                )
        finally:
            # Resume may replace the generator; preserve it even after a failure.
            self.rng = smc_run.rng

    def run_rejection(
        self,
        epsilon: float = 0.1,
        num_particles: int = 1000,
        max_time: Optional[timedelta] = None,
        total_simulations_budget: Optional[int] = None,
        verbose: bool = True,
        progress_update_interval: int = 1000,
        n_workers: Optional[int] = None,
        executor: Optional[ProcessPoolExecutor] = None,
        parallel_strategy: str = "dynamic",
    ) -> CalibrationResults:
        """
        Run ABC rejection sampling.

        Candidates are drawn from the prior until num_particles have a distance below
        epsilon, or a time/budget limit is reached. With an explicit seed, simulations
        that use the supplied RNG and avoid shared mutable state reproduce results
        across worker counts in the same numerical environment, provided no wall-clock
        cutoff occurs. A max_time cutoff can change which candidates are evaluated.

        Args:
            epsilon (float, optional): Distance threshold for accepting a candidate. Default is 0.1.
            num_particles (int, optional): Number of accepted particles to collect. Default is 1000.
            max_time (timedelta, optional): Time limit for submitting work; already-submitted
                simulations finish. Default is None.
            total_simulations_budget (int, optional): Maximum number of simulations. Default is None.
            verbose (bool, optional): Whether to print progress. Default is True.
            progress_update_interval (int, optional): Number of simulations between progress
                messages. Default is 1000.
            n_workers (int, optional): Number of worker processes, at most the detected CPU capacity.
                None runs in the calling process. Default is None.
            executor (ProcessPoolExecutor, optional): Caller-owned worker pool, taking precedence
                over n_workers. Default is None.
            parallel_strategy (str, optional): Parallel scheduling strategy. Only "dynamic" is supported.
                Default is "dynamic".

        Returns:
            CalibrationResults: Accepted particles with uniform weights, stored as generation 0.

        Raises:
            ValueError: If progress_update_interval is not positive.
        """
        scheduler = create_particle_scheduler(parallel_strategy)
        if progress_update_interval < 1:
            raise ValueError("progress_update_interval must be positive")
        if verbose:
            print(
                f"Starting ABC rejection sampling with {num_particles} particles "
                f"and epsilon threshold {epsilon}"
            )
        last_print = 0

        def progress(completed, accepted):
            # Called by the scheduler after each batch of finished simulations.
            nonlocal last_print
            if verbose and completed - last_print >= progress_update_interval:
                last_print = completed
                print(
                    f"\tSimulations: {completed}, Accepted: {accepted}, "
                    f"Acceptance rate: {accepted / completed * 100:.2f}%"
                )

        with self._calibration_executor(n_workers, executor) as pool:
            result = self._sample_particles(
                num_particles,
                epsilon,
                pool,
                scheduler,
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
        n_workers: Optional[int] = None,
        executor: Optional[ProcessPoolExecutor] = None,
        parallel_strategy: str = "dynamic",
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
            n_workers (int, optional): Number of worker processes, at most the detected CPU capacity.
                None runs in the calling process. Default is None.
            executor (ProcessPoolExecutor, optional): Caller-owned worker pool, taking precedence
                over n_workers. Default is None.
            parallel_strategy (str, optional): Parallel scheduling strategy. Only "dynamic" is supported.
                Default is "dynamic".

        Returns:
            CalibrationResults: Selected particles with uniform weights, stored as generation 0.

        Raises:
            ValueError: If Nsim is not positive or top_fraction is not in (0, 1].
        """
        scheduler = create_particle_scheduler(parallel_strategy)
        if Nsim < 1 or not 0 < top_fraction <= 1:
            raise ValueError("Nsim must be positive and top_fraction must be in (0, 1]")
        if verbose:
            print(
                f"Starting ABC top fraction selection with {Nsim} simulations "
                f"and top {top_fraction * 100:.1f}% selected"
            )
        with self._calibration_executor(n_workers, executor) as pool:
            evaluate, arguments = build_particle_tasks(
                pool, self._get_particle_inputs(), self._generate_candidates()
            )
            results = scheduler.run_batch(pool, evaluate, islice(arguments, Nsim))
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

    def _calibration_executor(self, n_workers, executor):
        """
        Open the worker pool, preloading the inputs shared by every candidate.

        Args:
            n_workers (int, optional): Number of worker processes, or None for sequential execution.
            executor (ProcessPoolExecutor, optional): Caller-owned worker pool.

        Returns:
            ContextManager[Optional[ProcessPoolExecutor]]: See `parallel.executor_context`.
        """
        return executor_context(
            n_workers,
            executor,
            particle_context=self._get_particle_inputs(),
        )

    def _get_particle_inputs(self):
        """Return the fixed inputs shared by all candidate evaluations."""
        return (
            self.simulation_function,
            self.parameters,
            self.param_names,
            self.observed_data,
            self.distance_function,
        )

    def _generate_candidates(
        self,
        epsilon=None,
        particles=None,
        weights=None,
        perturbations=None,
        deadline=None,
        *,
        root_rng=None,
    ):
        """
        Yield candidates with independent RNG streams, in submission order.

        Only one fixed-size draw advances the root RNG per generation/batch. Proposal
        retries and simulation draws affect only that candidate's child stream, so
        candidate k gets the same stream regardless of the number of workers or the
        order in which they finish.

        Args:
            epsilon (float, optional): Acceptance threshold passed to the worker. Default is None (accept all).
            particles (np.ndarray, optional): Previous generation, one row per particle. If None,
                candidates are drawn from the prior; otherwise a particle is resampled by weight and
                perturbed until it has non-zero prior density. Default is None.
            weights (np.ndarray, optional): Weights of the previous generation. Default is None.
            perturbations (Dict[str, Perturbation], optional): Perturbation kernel per parameter.
                Default is None.
            deadline (datetime, optional): Stop yielding after this time. Default is None.
            root_rng (np.random.Generator, optional): Generation seed source. Defaults to
                self.rng; SMC supplies its current generator when resuming.

        Yields:
            tuple: Candidate parameter values, acceptance threshold and simulation RNG.
        """
        root_rng = self.rng if root_rng is None else root_rng
        seeds = np.random.SeedSequence(
            root_rng.integers(0, 2**32, size=4, dtype=np.uint32)
        )
        while deadline is None or datetime.now() < deadline:
            rng = np.random.default_rng(seeds.spawn(1)[0])
            if particles is None:
                params = sample_prior(self.priors, self.param_names, rng)
            else:
                while True:
                    if deadline is not None and datetime.now() >= deadline:
                        return
                    index = rng.choice(len(particles), p=weights / weights.sum())
                    params = [
                        perturbations[p].propose(particles[index, i], rng)
                        for i, p in enumerate(self.param_names)
                    ]
                    if all(
                        (
                            self.priors[p].pdf(params[i])
                            if p in self.continuous_params
                            else self.priors[p].pmf(params[i])
                        )
                        > 0
                        for i, p in enumerate(self.param_names)
                    ):
                        break
            candidate_rng = rng if self._seed_requested else None
            yield params, epsilon, candidate_rng

    def _sample_particles(
        self,
        num_particles,
        epsilon,
        pool,
        scheduler,
        start_time=None,
        max_time=None,
        total_simulations_budget=None,
        n_simulations=0,
        particles=None,
        weights=None,
        perturbations=None,
        progress=None,
        *,
        root_rng=None,
        inclusive=False,
    ):
        """
        Evaluate candidates until num_particles are accepted or a limit is reached.

        Shared by rejection sampling and every SMC generation.

        Args:
            num_particles (int): Number of accepted particles to collect.
            epsilon (float, optional): Acceptance threshold, or None to accept all.
            pool (ProcessPoolExecutor or None): Worker pool, or None for sequential execution.
            scheduler (DynamicParticleScheduler): Scheduler collecting accepted particles.
            start_time (datetime, optional): Start of the calibration run; required with max_time.
                Default is None.
            max_time (timedelta, optional): Time limit measured from start_time. Default is None.
            total_simulations_budget (int, optional): Maximum number of simulations across the whole
                run. Default is None.
            n_simulations (int, optional): Simulations already run in earlier generations. Default is 0.
            particles (np.ndarray, optional): Previous generation to propose from; None samples the
                prior. Default is None.
            weights (np.ndarray, optional): Weights of the previous generation. Default is None.
            perturbations (Dict[str, Perturbation], optional): Perturbation kernel per parameter.
                Default is None.
            progress (Callable[[int, int], None], optional): Progress callback, see
                `DynamicParticleScheduler.run_until_n_accepted`. Default is None.
            root_rng (np.random.Generator, optional): Generation seed source. Defaults to self.rng.
            inclusive (bool, optional): Accept equality with epsilon. Default is False.

        Returns:
            Dict[str, Any]: See `DynamicParticleScheduler.run_until_n_accepted`; "n_simulations" counts only this call.
        """
        deadline = start_time + max_time if max_time is not None else None
        # Budget left for this call, after simulations of earlier generations.
        remaining = (
            total_simulations_budget - n_simulations
            if total_simulations_budget is not None
            else None
        )
        candidates = self._generate_candidates(
            epsilon, particles, weights, perturbations, deadline, root_rng=root_rng
        )
        evaluate, arguments = build_particle_tasks(
            pool, self._get_particle_inputs(), candidates, inclusive=inclusive
        )
        return scheduler.run_until_n_accepted(
            pool,
            evaluate,
            arguments,
            n_target=num_particles,
            max_simulations=remaining,
            deadline=deadline,
            progress=progress,
        )

    def run_projections(
        self,
        parameters: Dict[str, Any],
        iterations: int = 100,
        generation: Optional[int] = None,
        scenario_id: str = "baseline",
        rng: Optional[Any] = None,
        n_workers: Optional[int] = None,
        executor: Optional[ProcessPoolExecutor] = None,
        parallel_strategy: str = "dynamic",
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
            n_workers: Number of parallel workers, at most the detected CPU capacity.
                None for sequential execution. Uses the same CPU/thread limits as calibrate.
            executor: User-provided ProcessPoolExecutor (takes precedence over n_workers).
            parallel_strategy: Parallel scheduling strategy name.

        Returns:
            CalibrationResults: A new CalibrationResults object containing the original results plus the new projections
        """
        scheduler = create_particle_scheduler(parallel_strategy)

        with executor_context(n_workers, executor) as pool:
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
            posterior_samples = {}

            def arguments():
                for i in range(iterations):
                    child_seed = np.random.SeedSequence(
                        base_seed_seq.entropy,
                        spawn_key=base_seed_seq.spawn_key + (i,),
                        pool_size=base_seed_seq.pool_size,
                    )
                    rng_i = np.random.default_rng(child_seed)
                    idx = rng_i.choice(len(posterior), p=weights / weights.sum())
                    posterior_sample = posterior.iloc[idx]
                    for key, value in posterior_sample.items():
                        posterior_samples.setdefault(key, []).append(value)
                    proj_params = {**parameters, **posterior_sample}
                    # Set the trajectory RNG last so it overrides any parameter seed.
                    if inject_rng:
                        proj_params["rng"] = rng_i
                    yield self.simulation_function, proj_params

            projections = scheduler.run_batch(pool, _worker.run_projection, arguments())

            self.results.projections[scenario_id] = projections
            self.results.projection_parameters[scenario_id] = pd.DataFrame(
                posterior_samples
            )

            return copy.deepcopy(self.results)
