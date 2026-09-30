"""Prepare candidate evaluations and run model simulations for ABC."""

from datetime import datetime
from functools import partial
from itertools import islice

import numpy as np

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
    Acceptance targets use the supplied SequentialScheduler;
    fixed counts evaluate candidates sequentially. A time or simulation budget can stop either early.

    Args:
        inputs (tuple): 5-tuple of (simulation_function, fixed_parameters, param_names,
            observed_data, distance_function).
        priors (Dict[str, Any]): Prior distribution for each parameter.
        root_rng (np.random.Generator): Root generator used to seed proposal sequences.
        seed_requested (bool): Whether reproducible seeded execution was requested.
        n_accepted (int, optional): Target number of accepted particles (for SMC/rejection).
        n_evaluations (int, optional): Target number of candidate evaluations (for top fraction).
        epsilon (float, optional): Distance threshold for acceptance. Default is None.
        scheduler (SequentialScheduler, optional): Acceptance scheduler.
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
            if the target count is < 1, or if scheduler is missing when n_accepted is set.
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
    evaluate, arguments = build_particle_tasks(inputs, candidates, inclusive=inclusive)

    if n_evaluations is not None:
        # Evaluate a fixed number of candidates and preserve their input order
        count = (
            min(n_evaluations, remaining) if remaining is not None else n_evaluations
        )
        results, accepted = [], []
        for args in islice(arguments, count):
            result = evaluate(*args)
            results.append(result)
            if result["accepted"]:
                accepted.append(result)
            if progress is not None:
                progress(len(results), len(accepted))
        return {"accepted_results": accepted, "n_simulations": len(results)}

    # Evaluate until enough candidates are accepted or a limit is reached
    return scheduler.run_until_n_accepted(
        evaluate,
        arguments,
        n_target=n_accepted,
        max_simulations=remaining,
        deadline=deadline,
        progress=progress,
    )


def build_particle_tasks(inputs, candidates, *, inclusive=False):
    """Attach fixed model inputs to each candidate for evaluation."""
    function, parameters, names, observed, distance = inputs
    arguments = (
        (function, parameters, names, params, observed, distance, epsilon, rng)
        for params, epsilon, rng in candidates
    )
    return partial(evaluate_particle, inclusive=inclusive), arguments


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
