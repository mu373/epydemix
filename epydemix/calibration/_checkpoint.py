"""Trusted, local SMC checkpoints: readable manifest and atomic state snapshots.

A checkpoint is a zip archive with two members:
- ``metadata.json``: human-readable manifest (settings, progress, environment,
  input fingerprint and a SHA-256 of the payload).
- ``state.pkl``: pickled state needed to continue the run.

Pickle can execute code on load: only open checkpoints you created yourself.
"""

import hashlib
import inspect
import json
import os
import pickle
import platform
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ..model.epimodel import EpiModel

FORMAT_VERSION = 1


def _identity(value):
    """Return the class or function name as ``module.qualname``."""
    return f"{value.__module__}.{value.__qualname__}"


def _array_digest(array):
    """
    Hash an array's C-order bytes without copying large arrays.

    Args:
        array (np.ndarray): Array with a non-object dtype.

    Returns:
        str: Hex SHA-256 digest of the array data.
    """
    digest = hashlib.sha256()
    if array.dtype.kind in "fc" and array.dtype.itemsize > (
        8 if array.dtype.kind == "f" else 16
    ):
        # Extended floats can contain padding bytes that change on scalar copies
        # or pickle round trips. Hash round-trip decimal values, retaining precision.
        for value in array.flat:
            parts = (value.real, value.imag) if array.dtype.kind == "c" else (value,)
            for part in parts:
                digest.update(np.format_float_scientific(part, unique=True).encode())
                digest.update(b"\n")
    elif array.size and array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
    elif array.size:
        for chunk in np.nditer(
            array, flags=["external_loop", "buffered"], order="C", buffersize=65536
        ):
            digest.update(chunk.tobytes())
    return digest.hexdigest()


class _HashingWriter:
    """
    File wrapper that hashes bytes as they are written.

    Lets `pickle.dump` stream into the archive while computing its checksum,
    instead of holding the whole serialized state in memory.

    Attributes:
        file (BinaryIO): Underlying writable file.
        digest (hashlib._Hash): Running SHA-256 of everything written.
    """

    def __init__(self, file):
        """
        Initialize the writer.

        Args:
            file (BinaryIO): Underlying writable file.
        """
        self.file = file
        self.digest = hashlib.sha256()

    def write(self, data):
        """
        Hash and write a chunk of data.

        Args:
            data (bytes-like): Data to write.

        Returns:
            int: Number of bytes written to the underlying file.
        """
        data = memoryview(data).cast("B")
        self.digest.update(data)
        return self.file.write(data)


def validate_picklable(value):
    """
    Check that a value can be pickled, discarding the serialized bytes.

    Args:
        value (Any): Value to serialize.

    Raises:
        Exception: Whatever pickle raises for an unpicklable object (e.g. PicklingError,
            TypeError or AttributeError).
    """
    with open(os.devnull, "wb") as file:
        pickle.dump(value, file, protocol=5)


def _verify_member(archive, member, expected):
    """
    Check the SHA-256 of an archive member, reading it in chunks.

    Args:
        archive (zipfile.ZipFile): Open checkpoint archive.
        member (str): Name of the member to verify.
        expected (str): Expected hex SHA-256 digest.

    Raises:
        ValueError: If the digest does not match.
    """
    digest = hashlib.sha256()
    with archive.open(member) as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Checkpoint data checksum mismatch")


def _canonical_pandas_values(value):
    """Hash values without losing categorical metadata or coercing column dtypes.

    Support NumPy-backed values, strings and categoricals. Other extension dtypes must
    be handled explicitly before they can be checkpointed safely.
    """
    if isinstance(value.dtype, pd.CategoricalDtype):
        return [
            "categorical",
            _canonical(value.dtype.categories),
            value.dtype.ordered,
            _canonical(value.array.codes),
        ]
    if isinstance(value.dtype, pd.StringDtype):
        return [
            "string",
            str(value.dtype),
            value.dtype.storage,
            _canonical(value.to_numpy(dtype=object, na_value=None)),
        ]
    if not isinstance(value.dtype, np.dtype):
        raise TypeError(f"Unsupported checkpoint pandas dtype: {value.dtype}")
    return _canonical(value.to_numpy())


def _canonical(value):
    """
    Convert a value to a JSON-serializable form for hashing.

    The result is independent of pickle layout, dict order and object identity.
    Unsupported/cyclic input objects fail before calibration. External files and
    callable globals/closures are deliberately not inspected.

    Args:
        value (Any): Value to convert. Supported: builtins, containers, NumPy arrays, dates,
            paths, pandas objects, frozen scipy distributions, functions and plain objects.

    Returns:
        list: Canonical, JSON-serializable representation tagged with the value's type.

    Raises:
        TypeError: If the value (or a nested value) has an unsupported type.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    # On some platforms longdouble has double precision and NumPy pickle restores
    # it as float64. Canonicalize equivalent 64-bit scalars the same way.
    if isinstance(value, np.floating) and value.dtype.itemsize == 8:
        return ["float", float(value).hex()]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        data = (
            _canonical(array.tolist())
            if array.dtype.hasobject
            else _array_digest(array)
        )
        return ["array", array.dtype.descr, array.shape, data]
    if isinstance(value, dict):
        items = [(_canonical(k), _canonical(v)) for k, v in value.items()]
        return ["dict", sorted(items, key=lambda item: json.dumps(item[0]))]
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_canonical(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=json.dumps)
        return [type(value).__name__, items]
    if isinstance(value, (date, datetime, timedelta, Path)):
        return [_identity(type(value)), str(value)]
    if isinstance(value, pd.DataFrame):
        return [
            "DataFrame",
            _canonical(value.index),
            _canonical(value.columns),
            [_canonical_pandas_values(value.iloc[:, i]) for i in range(value.shape[1])],
        ]
    if isinstance(value, (pd.Series, pd.Index)):
        return [
            _identity(type(value)),
            _canonical(value.name),
            _canonical_pandas_values(value),
            _canonical(value.index) if isinstance(value, pd.Series) else None,
        ]
    if isinstance(value, rv_frozen):
        return [
            "prior",
            _identity(type(value.dist)),
            value.dist.name,
            _canonical(value.args),
            _canonical(value.kwds),
            _canonical(getattr(value.dist, "xk", None)),
            _canonical(getattr(value.dist, "pk", None)),
        ]
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return ["callable", _identity(value)]
    if hasattr(value, "__dict__") and not callable(value):
        attributes = vars(value)
        if isinstance(value, EpiModel):
            # simulate() recomputes these caches on each call.
            attributes = {
                k: v for k, v in attributes.items() if k not in ("Cs", "definitions")
            }
        return [_identity(type(value)), _canonical(attributes)]
    raise TypeError(f"Unsupported checkpoint input type: {type(value).__name__}")


def input_hash(inputs):
    """
    Compute a stable fingerprint of the calibration inputs.

    Used to refuse resuming a checkpoint with different data or settings.

    Args:
        inputs (Dict[str, Any]): Calibration inputs and settings.

    Returns:
        str: Hex SHA-256 digest of the canonical form of the inputs.

    Raises:
        TypeError: If the inputs contain unsupported types or reference cycles.
    """
    try:
        encoded = json.dumps(_canonical(inputs), separators=(",", ":"), allow_nan=False)
    except RecursionError as error:
        raise TypeError(
            "Checkpoint inputs must not contain reference cycles"
        ) from error
    return hashlib.sha256(encoded.encode()).hexdigest()


def environment():
    """
    Describe the software that produced a checkpoint.

    Records library versions and the hashes of the calibration source files, including
    uncommitted edits. A mismatch on resume only warns, since the run can continue but
    may no longer be exactly reproducible.

    Returns:
        Dict[str, Dict[str, Optional[str]]]: A dictionary with keys "versions" (package
            name to version, None if not installed) and "source_sha256" (file name to digest).
    """
    versions = {"python": platform.python_version()}
    for package in ("epydemix", "numpy", "scipy", "pandas", "threadpoolctl", "numba"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    # Record the actual implementation, including uncommitted edits.
    source = {}
    for path in (
        Path(__file__),
        Path(__file__).with_name("abc.py"),
        Path(__file__).with_name("_smc.py"),
        Path(__file__).with_name("_history.py"),
        Path(__file__).with_name("_evaluate.py"),
        Path(__file__).with_name("_worker_inputs.py"),
        Path(__file__).with_name("_proposals.py"),
        Path(__file__).with_name("_scheduler.py"),
        Path(__file__).parents[1] / "utils/abc_smc_utils.py",
        Path(__file__).parents[1] / "_execution.py",
        Path(__file__).parents[1] / "utils/random_utils.py",
    ):
        source[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"versions": versions, "source_sha256": source}


def write_checkpoint(path, state, metadata, *, overwrite):
    """
    Write a checkpoint archive atomically.

    The archive is written and fsynced to a temporary file in the same directory,
    then moved into place, so a crash never leaves a partial checkpoint.

    Args:
        path (str or Path): Destination file.
        state (Dict[str, Any]): Picklable state stored as "state.pkl".
        metadata (Dict[str, Any]): JSON-serializable manifest fields stored as "metadata.json".
        overwrite (bool): Whether to replace an existing file at `path`.

    Returns:
        Dict[str, Any]: Written manifest, including the payload checksum.

    Raises:
        FileExistsError: If overwrite is False and `path` already exists.
    """
    path = Path(path)
    manifest = {
        **metadata,
        "format_version": FORMAT_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            with zipfile.ZipFile(file, "w") as archive:
                with archive.open("state.pkl", "w", force_zip64=True) as payload:
                    writer = _HashingWriter(payload)
                    pickle.dump(state, writer, protocol=5)
                manifest["data_sha256"] = writer.digest.hexdigest()
                archive.writestr(
                    "metadata.json", json.dumps(manifest, indent=2, allow_nan=False)
                )
            file.flush()
            os.fsync(file.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Exclusive publication prevents accidentally replacing an existing run.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def read_checkpoint(path, *, expected_data_sha256=None):
    """
    Load and verify a checkpoint written by `write_checkpoint`.

    Only load trusted checkpoints: the checksum detects damage, not tampering, and
    unpickling can execute arbitrary code. Runtime compatibility is checked by the
    SMC caller when resuming, not when reading stored history.

    Args:
        path (str or Path): Checkpoint file.
        expected_data_sha256 (str, optional): Payload identity recorded by a referring
            checkpoint. Checked before unpickling, in addition to archive integrity.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The saved state and the manifest.

    Raises:
        ValueError: If the file is damaged, has an unsupported format version, or its
            payload/input checksums do not match the manifest.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            if metadata.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported SMC checkpoint format version")
            if (
                expected_data_sha256 is not None
                and metadata.get("data_sha256") != expected_data_sha256
            ):
                raise ValueError(f"Payload does not match checkpoint reference: {path}")
            # Verify the complete payload first; never unpickle unchecked bytes.
            _verify_member(archive, "state.pkl", metadata.get("data_sha256"))
            with archive.open("state.pkl") as payload:
                state = pickle.load(payload)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise ValueError("Invalid or damaged SMC checkpoint") from error
    if input_hash(state["inputs"]) != metadata.get("input_sha256"):
        raise ValueError("Checkpoint input checksum mismatch")
    return state, metadata
