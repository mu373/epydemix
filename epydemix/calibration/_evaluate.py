"""Run model simulations for ABC calibration and posterior projections.

For each calibration candidate, combine sampled and fixed parameters, run the
simulation, compute its distance from the observations, and check acceptance.
Projection runs return the simulation output without a distance or acceptance check.

run_particle_evaluations prepares candidates and runs evaluations for all ABC methods.
It uses an acceptance loop for SMC/rejection and map_tasks for fixed-count evaluation.
Owned workers restore fresh inputs through _worker_inputs for each candidate.
"""

from datetime import datetime
from functools import partial
from itertools import islice

import numpy as np

from .._execution import map_tasks, single_threaded
from . import _worker_inputs
from ._proposals import ProposalSequence


def run_particle_evaluations(
    inputs,
    priors,
    root_rng,
    seed_requested,
    *,
    n_accepted=None,
    n_evaluations=None,
    epsilon=None,
    pool=None,
    scheduler=None,
    start_time=None,
    max_time=None,
    total_simulations_budget=None,
    n_simulations=0,
    particles=None,
    weights=None,
    perturbations=None,
    progress=None,
    inclusive=False,
):
    """
    Prepare candidates and run their evaluations for an ABC method.

    Set exactly one of n_accepted (SMC/rejection) or n_evaluations (top fraction).
    Acceptance targets use the supplied SequentialScheduler or DynamicScheduler;
    fixed counts use map_tasks. A time or simulation budget can stop either early.

    Args:
        inputs (tuple): 5-tuple of (simulation_function, fixed_parameters, param_names,
            observed_data, distance_function).
        priors (Dict[str, Any]): Prior distribution for each parameter.
        root_rng (np.random.Generator): Root generator used to seed proposal sequences.
        seed_requested (bool): Whether reproducible seeded execution was requested.
        n_accepted (int, optional): Target number of accepted particles (for SMC/rejection).
        n_evaluations (int, optional): Target number of candidate evaluations (for top fraction).
        epsilon (float, optional): Distance threshold for acceptance. Default is None.
        pool (ProcessPoolExecutor, optional): Worker pool, or None for sequential execution.
        scheduler (SequentialScheduler or DynamicScheduler, optional): Acceptance scheduler.
        start_time (datetime, optional): Start timestamp of the calibration run.
        max_time (timedelta, optional): Time limit measured from start_time.
        total_simulations_budget (int, optional): Overall simulation budget.
        n_simulations (int, optional): Simulations already consumed in earlier generations. Default is 0.
        particles (np.ndarray, optional): Particles from previous SMC generation.
        weights (np.ndarray, optional): Particle weights from previous SMC generation.
        perturbations (Dict[str, Perturbation], optional): Perturbation kernels per parameter.
        progress (Callable, optional): Progress callback (completed, accepted).
        inclusive (bool, optional): Whether distance == epsilon is accepted. Default is False.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "accepted_results" (List[Dict[str, Any]]): Accepted candidates in candidate order.
            - "n_simulations" (int): Number of simulations executed in this call.

    Raises:
        ValueError: If neither or both of n_accepted and n_evaluations are given,
            if the target count is < 1, if scheduler is missing when n_accepted is set,
            or if epsilon is NaN.
    """
    # Select a stopping condition before advancing the RNG or running simulations
    if (n_accepted is None) == (n_evaluations is None):
        raise ValueError("Specify exactly one of n_accepted or n_evaluations")
    target = n_accepted if n_accepted is not None else n_evaluations
    if target < 1:
        raise ValueError("The requested particle count must be positive")
    if n_accepted is not None and scheduler is None:
        raise ValueError("Acceptance sampling requires a scheduler")
    if epsilon is not None and np.isnan(epsilon):
        raise ValueError("epsilon must not be NaN")

    # Calculate the deadline and remaining simulation budget
    deadline = start_time + max_time if max_time is not None else None
    remaining = (
        total_simulations_budget - n_simulations
        if total_simulations_budget is not None
        else None
    )
    if (remaining is not None and remaining <= 0) or (
        deadline is not None and datetime.now() >= deadline
    ):
        return {"accepted_results": [], "n_simulations": 0}

    # Prepare proposals with one root RNG draw per generation or batch
    candidates = ProposalSequence(
        priors,
        inputs[2],  # Parameter names, in the order expected by the evaluator
        root_rng.integers(0, 2**32, size=4, dtype=np.uint32),
        seed_requested,
        epsilon,
        particles,
        weights,
        perturbations,
        deadline,
    )
    # Prepare the evaluator and its inputs
    evaluate, arguments = build_particle_tasks(
        pool, inputs, candidates, inclusive=inclusive
    )

    if n_evaluations is not None:
        # Evaluate a fixed number of candidates and preserve their input order
        count = (
            min(n_evaluations, remaining) if remaining is not None else n_evaluations
        )
        accepted_count = 0

        def report(completed, result):
            nonlocal accepted_count
            accepted_count += int(result["accepted"])
            progress(completed, accepted_count)

        results = map_tasks(
            pool,
            evaluate,
            islice(arguments, count),
            on_completed=report if progress is not None else None,
        )
        accepted = [result for result in results if result["accepted"]]
        return {"accepted_results": accepted, "n_simulations": len(results)}

    # Run the selected acceptance scheduler
    return scheduler.run_until_n_accepted(
        evaluate,
        arguments,
        n_target=n_accepted,
        max_simulations=remaining,
        deadline=deadline,
        progress=progress,
    )


class EvaluationArguments:
    """
    Attach uncached model inputs while preserving random candidate access.

    Args:
        inputs (tuple): 5-tuple of (simulation_function, fixed_parameters, param_names,
            observed_data, distance_function).
        candidates (ProposalSequence): Indexable sequence of proposal candidates.
    """

    def __init__(self, inputs, candidates):
        self.inputs = inputs
        self.candidates = candidates

    def __getitem__(self, index):
        params, epsilon, rng = self.candidates[index]
        function, parameters, names, observed, distance = self.inputs
        return function, parameters, names, params, observed, distance, epsilon, rng

    def __iter__(self):
        """Attach inputs without swallowing IndexError from user callbacks."""
        function, parameters, names, observed, distance = self.inputs
        for params, epsilon, rng in self.candidates:
            yield function, parameters, names, params, observed, distance, epsilon, rng


def build_particle_tasks(executor, inputs, candidates, *, inclusive=False):
    """
    Select an evaluator and lazily encode candidate arguments for the pool.

    Owned calibration pools cache fixed inputs. Sequential evaluation and
    caller-owned pools receive the full inputs for each candidate instead.
    Candidates always supply (parameter values, epsilon, RNG).
    The calibration strategy chooses whether equality with epsilon is accepted.

    Args:
        executor (ProcessPoolExecutor or None): Worker pool or None.
        inputs (tuple): 5-tuple of (simulation_function, fixed_parameters, param_names,
            observed_data, distance_function).
        candidates (ProposalSequence): Proposal sequence supplying candidate arguments.
        inclusive (bool, optional): Whether distance == epsilon is accepted. Default is False.

    Returns:
        Tuple[Callable, Any]: (evaluate_function, arguments) pair for task execution.
    """
    if (
        getattr(executor, "_initializer", None)
        is _worker_inputs.initialize_particle_worker
    ):
        evaluate, arguments = evaluate_particle_with_cached_inputs, candidates
    else:
        evaluate, arguments = evaluate_particle, EvaluationArguments(inputs, candidates)
    return partial(evaluate, inclusive=inclusive), arguments


def evaluate_particle_with_cached_inputs(
    sampled_values, epsilon=None, rng=None, *, inclusive=False
):
    """
    Evaluate a candidate using the inputs cached by `_worker_inputs.initialize_particle_worker`.

    Only the candidate-specific arguments cross the process boundary.

    Args:
        sampled_values (list): Parameter values, in the order of `param_names`.
        epsilon (float, optional): Acceptance threshold. Default is None (accept all).
        rng (np.random.Generator, optional): Candidate-specific generator, injected only for
            seeded calibration. Default is None.
        inclusive (bool, optional): Accept equality with epsilon. Default is False.

    Returns:
        Dict[str, Any]: Same as `evaluate_particle`.
    """
    # Restore fresh model inputs for this candidate
    function, fixed_parameters, names, observed, distance = (
        _worker_inputs.restore_worker_inputs()
    )
    return evaluate_particle(
        function,
        fixed_parameters,
        names,
        sampled_values,
        observed,
        distance,
        epsilon,
        rng,
        inclusive=inclusive,
    )


@single_threaded
def evaluate_particle(
    simulation_function,
    fixed_parameters,
    param_names,
    sampled_values,
    observed_data,
    distance_function,
    epsilon=None,
    rng=None,
    *,
    inclusive=False,
):
    """
    Run a simulation and compute its distance for a single parameter set.

    Args:
        simulation_function (Callable): Function running the simulation model.
        fixed_parameters (Dict[str, Any]): Fixed parameter values passed to the simulation.
        param_names (List[str]): Names of the calibrated parameters.
        sampled_values (list): Parameter values, in the order of `param_names`.
        observed_data (Dict[str, Any]): Observed data used for the distance computation.
        distance_function (Callable): Function `(data, simulation) -> float`.
        epsilon (float, optional): Acceptance threshold. If provided, only accepted results
            include simulation data (saves serialization cost). Default is None.
        rng (np.random.Generator, optional): Candidate-specific generator, injected only for
            seeded calibration. Default is None.
        inclusive (bool, optional): Accept equality with epsilon. Default is False.

    Returns:
        Dict[str, Any]: A dictionary with keys "params", "distance", "simulation" and "accepted".
            "simulation" is None for rejected candidates.

    Raises:
        ValueError: If the simulation does not return a dictionary.
    """
    # Combine fixed parameters with the sampled parameter values
    full_params = {**fixed_parameters, **dict(zip(param_names, sampled_values))}
    # A supplied Generator overrides any seed in fixed_parameters, avoiding a reset
    # to the same integer seed for every simulation. Unseeded calls inject nothing.
    if rng is not None:
        full_params["rng"] = rng

    # Run simulation
    simulation = simulation_function(full_params)

    # Validate simulation output
    if not isinstance(simulation, dict):
        raise ValueError(f"Simulation must return dictionary, got {type(simulation)}")

    # Compute distance
    distance = distance_function(observed_data, simulation)
    if np.isnan(distance):
        raise ValueError("Simulation distance must not be NaN")

    # Check acceptance against the distance threshold
    accepted = epsilon is None or (
        distance <= epsilon if inclusive else distance < epsilon
    )
    return {
        "params": sampled_values,
        "distance": distance,
        "simulation": simulation if accepted else None,
        "accepted": accepted,
    }


@single_threaded
def simulate_projection(simulation_function, proj_params):
    """
    Run a single projection simulation.

    Args:
        simulation_function (Callable): Function running the simulation model.
        proj_params (Dict[str, Any]): Full parameter dictionary for this projection.

    Returns:
        Any: Output of the simulation function.
    """
    return simulation_function(proj_params)
