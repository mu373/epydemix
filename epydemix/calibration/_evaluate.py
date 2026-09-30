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


@single_threaded
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
    """Prepare candidates and run their evaluations for an ABC method.

    Set exactly one of n_accepted (SMC/rejection) or n_evaluations (top fraction).
    Acceptance targets use the supplied SequentialScheduler or DynamicScheduler;
    fixed counts use map_tasks. A time or simulation budget can stop either early.

    inputs contains (simulation_function, parameters, param_names, observed_data,
    distance_function). Previous particles, weights, and perturbations configure
    SMC proposals; without particles, proposals are drawn from the prior.
    max_time requires start_time. n_simulations is the count from earlier generations.

    Return accepted_results in candidate order and this call's n_simulations.
    With epsilon=None, all evaluated candidates are retained for later selection.
    progress(completed, accepted) reports each sequential evaluation, parallel
    scheduler updates, or the completed fixed-count batch.
    """
    # Select a stopping condition before advancing the RNG or running simulations
    if (n_accepted is None) == (n_evaluations is None):
        raise ValueError("Specify exactly one of n_accepted or n_evaluations")
    target = n_accepted if n_accepted is not None else n_evaluations
    if target < 1:
        raise ValueError("The requested particle count must be positive")
    if n_accepted is not None and scheduler is None:
        raise ValueError("Acceptance sampling requires a scheduler")

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
    """Attach uncached model inputs while preserving random candidate access."""

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
    """Select an evaluator and lazily encode candidate arguments for the pool.

    Owned calibration pools cache fixed inputs. Sequential evaluation and
    caller-owned pools receive the full inputs for each candidate instead.
    Candidates always supply (parameter values, epsilon, RNG).
    The calibration strategy chooses whether equality with epsilon is accepted.
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
    params, epsilon=None, rng=None, *, inclusive=False
):
    """
    Evaluate a candidate using the inputs cached by `_worker_inputs.initialize_particle_worker`.

    Only the candidate-specific arguments cross the process boundary.

    Args:
        params (list): Parameter values, in the order of `param_names`.
        epsilon (float, optional): Acceptance threshold. Default is None (accept all).
        rng (np.random.Generator, optional): Candidate-specific generator, injected only for
            seeded calibration. Default is None.
        inclusive (bool, optional): Accept equality with epsilon. Default is False.

    Returns:
        Dict[str, Any]: Same as `evaluate_particle`.
    """
    # Restore fresh model inputs for this candidate
    function, parameters, names, observed, distance = (
        _worker_inputs.restore_worker_inputs()
    )
    return evaluate_particle(
        function,
        parameters,
        names,
        params,
        observed,
        distance,
        epsilon,
        rng,
        inclusive=inclusive,
    )


@single_threaded
def evaluate_particle(
    simulation_function,
    parameters,
    param_names,
    params,
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
        parameters (Dict[str, Any]): Fixed parameters passed to the simulation.
        param_names (List[str]): Names of the calibrated parameters.
        params (list): Parameter values, in the order of `param_names`.
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
    full_params = {**parameters, **dict(zip(param_names, params))}
    # A supplied Generator overrides any seed in parameters, avoiding a reset
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

    # Check acceptance against the distance threshold
    accepted = epsilon is None or (
        distance <= epsilon if inclusive else distance < epsilon
    )
    return {
        "params": params,
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
