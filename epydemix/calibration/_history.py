"""Disk storage for SMC generation history.

Each completed generation writes one checkpoint file per result field (see
``FIELDS``). ``CalibrationResults`` then holds ``DiskHistory`` mappings that
store only file references and load a generation when it is accessed.
"""

from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from . import _checkpoint

FIELDS = ("posterior_distributions", "weights", "distances", "selected_trajectories")


class DiskHistory(Mapping):
    """
    Read-only mapping from generation to one result field, backed by checkpoint files.

    Values are loaded on every access and not cached, so only file references
    are kept in memory.

    Attributes:
        directory (Path): Directory containing the history files.
        entries (Dict[int, Tuple[str, str]]): Generation to (file name, payload SHA-256).
    """

    def __init__(self, directory, entries=None):
        """
        Initialize the mapping.

        Args:
            directory (str or Path): Directory containing the history files.
            entries (Dict[int, Tuple[str, str]], optional): Generation to (file name, payload SHA-256).
                The checksum ties each file to the checkpoint that references it. Default is None (empty).
        """
        self.directory = Path(directory).resolve()
        self.entries = {} if entries is None else entries

    def __iter__(self):
        """Iterate over the stored generation indices."""
        return iter(self.entries)

    def __len__(self):
        """Return the number of stored generations."""
        return len(self.entries)

    def __getitem__(self, generation):
        """
        Load the value saved for one generation.

        Args:
            generation (int): Generation index.

        Returns:
            Any: The saved value (e.g. a DataFrame of particles or an array of weights).

        Raises:
            KeyError: If the generation is missing.
            ValueError: If the archive, metadata or checksums are invalid.
        """
        filename, checksum = self.entries[generation]
        path = self.directory / filename
        state, _ = _checkpoint.read_checkpoint(path, expected_data_sha256=checksum)
        return state["value"]


def save_generation(results, generation, values, directory, environment):
    """
    Write one generation to disk and register it in the results.

    Files get a unique name and are never overwritten. References are added to
    `results` only after every field has been written, so `results` never points at
    a half-saved generation. A failed save may leave unreferenced files, but never
    overwrites older data.

    Args:
        results (CalibrationResults): Results whose fields in `FIELDS` become DiskHistory mappings.
        generation (int): Generation index. Disk history must start at generation zero.
        values (Dict[str, Any]): Value to save for every field name in `FIELDS`.
        directory (str or Path): History directory, created if it does not exist.
        environment (Dict[str, Any]): Output of `_checkpoint.environment()`, recorded in each file.

    Raises:
        ValueError: If results does not already use disk history and generation is not zero.
    """
    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=True)
    identifier = uuid4().hex
    entries = {}
    for name in FIELDS:
        filename = f"{generation:06d}-{identifier}-{name}.checkpoint"
        path = directory / filename
        inputs = {"generation": generation, "field": name}
        metadata = _checkpoint.write_checkpoint(
            path,
            {"inputs": inputs, "value": values[name]},
            {
                "input_sha256": _checkpoint.input_hash(inputs),
                "environment": environment,
            },
            overwrite=False,
        )
        entries[name] = (filename, metadata["data_sha256"])
    # Commit the in-memory references only when all fields have been saved.
    for name in FIELDS:
        field = getattr(results, name)
        if not isinstance(field, DiskHistory):
            if generation != 0:
                raise ValueError("Disk history must start at generation zero")
            field = DiskHistory(directory)
            setattr(results, name, field)
        field.entries[generation] = entries[name]


def bind_history(results, directory):
    """
    Point restored DiskHistory mappings at a directory and check that all files exist.

    This allows a checkpoint and its history directory to be moved together.
    File contents are not loaded here; they are verified when read.

    Args:
        results (CalibrationResults): Results restored from a checkpoint.
        directory (str or Path): History directory next to the checkpoint.

    Raises:
        ValueError: If the results do not use disk history.
        FileNotFoundError: If a referenced history file is missing.
    """
    for name in FIELDS:
        field = getattr(results, name)
        if not isinstance(field, DiskHistory):
            raise ValueError("Checkpoint does not contain disk history")
        field.directory = Path(directory).resolve()
        for filename, _ in field.entries.values():
            if not (field.directory / filename).is_file():
                raise FileNotFoundError(
                    f"Missing generation history: {field.directory / filename}"
                )
