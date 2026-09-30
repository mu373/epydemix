"""Save calibration inputs once and restore a fresh copy for each candidate.

Owned workers read the saved bytes at startup. Restoring the snapshot for each
evaluation prevents model or parameter mutations from leaking between candidates.
Process-pool creation and ownership are handled by epydemix._execution.
"""

import pickle
from contextlib import contextmanager
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from tempfile import TemporaryDirectory

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
        path (Path): Pickle file written by `save_worker_inputs`.
    """
    global _serialized_particle_inputs  # noqa: PLW0603 - process-local initializer state
    _serialized_particle_inputs = path.read_bytes()


def restore_worker_inputs():
    """Restore fresh model inputs from this worker's cached snapshot for one evaluation."""
    return pickle.loads(_serialized_particle_inputs)


@contextmanager
def worker_input_file():
    """Keep the input file available until the worker pool has shut down."""
    # Large spawn initargs block startup on pipe writes. Pass only a path.
    with TemporaryDirectory(prefix="epydemix-workers-") as directory:
        yield Path(directory) / "inputs.pkl"


def save_worker_inputs(path, inputs):
    """Serialize the shared inputs once, before the first task is submitted."""
    with path.open("wb") as file:
        ForkingPickler(file).dump(inputs)
