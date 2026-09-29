"""Picklable worker functions for parallel ABC calibration.

Module-level functions are required for ProcessPoolExecutor serialization
(closures and lambdas cannot be pickled).
"""

import pickle
from functools import wraps

from threadpoolctl import threadpool_limits

# Pickled (simulation_function, parameters, param_names, observed_data,
# distance_function), set once per worker process by the pool initializer.
_serialized_particle_inputs = None


def initialize_particle_worker(path):
    """
    Pool initializer that loads the fixed calibration inputs written by the parent process.

    The inputs are kept as pickled bytes and unpickled for every task, so a simulation
    that mutates its parameters or model cannot leak state into later tasks. This matches
    the isolation of sending the full arguments with every task.

    Args:
        path (Path): Pickle file written by `parallel._particle_executor`.
    """
    global _serialized_particle_inputs  # noqa: PLW0603 - process-local initializer state
    _serialized_particle_inputs = path.read_bytes()


def evaluate_particle_with_cached_inputs(
    params, epsilon=None, rng=None, *, inclusive=False
):
    """
    Evaluate a candidate using the inputs cached by `initialize_particle_worker`.

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
    function, parameters, names, observed, distance = pickle.loads(
        _serialized_particle_inputs
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


def _single_threaded(function):
    """
    Limit supported, already-loaded BLAS/OpenMP thread pools to one thread during a call.

    Previous limits are restored afterward. This reduces nested native parallelism
    and keeps thread settings consistent between sequential and parallel evaluation.

    Args:
        function (Callable): Function to wrap.

    Returns:
        Callable: Wrapped function with the same signature.
    """

    @wraps(function)
    def run(*args, **kwargs):
        # Apply at task entry, including caller-owned pools; restore on failure too.
        # Match sequential numerical reductions to those in process workers.
        with threadpool_limits(limits=1):
            return function(*args, **kwargs)

    return run


@_single_threaded
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
    full_params = {**parameters, **dict(zip(param_names, params))}
    # A supplied Generator overrides any seed in parameters, avoiding a reset
    # to the same integer seed for every simulation. Unseeded calls inject nothing.
    if rng is not None:
        full_params["rng"] = rng
    simulation = simulation_function(full_params)

    if not isinstance(simulation, dict):
        raise ValueError(f"Simulation must return dictionary, got {type(simulation)}")

    distance = distance_function(observed_data, simulation)

    accepted = epsilon is None or (
        distance <= epsilon if inclusive else distance < epsilon
    )
    return {
        "params": params,
        "distance": distance,
        "simulation": simulation if accepted else None,
        "accepted": accepted,
    }


@_single_threaded
def run_projection(simulation_function, proj_params):
    """
    Run a single projection simulation.

    Args:
        simulation_function (Callable): Function running the simulation model.
        proj_params (Dict[str, Any]): Full parameter dictionary for this projection.

    Returns:
        Any: Output of the simulation function.
    """
    return simulation_function(proj_params)
