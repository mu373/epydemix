"""Worker input caching: serialization, candidate isolation, and cleanup."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration.abc import ABCSampler
from tests.fixtures.calibration import (
    assert_exact_calibration,
)


class _PickleProbe:
    """Record serialization calls and expose mutable state for isolation checks."""

    def __init__(self, path):
        """Set the serialization log path and initially empty mutable state."""
        self.path = path
        self.values = []

    def __getstate__(self):
        """Log each serialization before returning the instance state."""
        with self.path.open("a") as file:
            file.write("serialized\n")
        return self.__dict__


class _MutatingSimulation:
    """Mutate model and parameter state to detect leaks between candidates."""

    def __init__(self):
        """Start with no simulation calls recorded."""
        self.calls = 0

    def __call__(self, params):
        """Return mutation counters and a random draw, or raise the requested error."""
        if params.get("fail"):
            raise RuntimeError("expected worker failure")
        self.calls += 1
        params["probe"].values.append(params["beta"])
        return {
            "data": np.array(
                [self.calls, len(params["probe"].values), params["rng"].random()]
            )
        }


def _mutating_distance(data, simulation):
    """Check fresh model and input state, then mutate the observations."""
    # Both callback and nested parameter/observation state must reset per candidate.
    np.testing.assert_array_equal(simulation["data"][:2], [1, 1])
    assert data["data"][0] == 0
    data["data"][0] = 99
    return float(simulation["data"][2])


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_fixed_worker_snapshot_serialized_once_and_isolates_mutation(
    tmp_path, monkeypatch, start_method
):
    """Check cache serialization, candidate isolation, pool ownership, and cleanup."""
    import multiprocessing as mp
    import tempfile

    from epydemix.calibration import _worker_inputs

    if start_method not in mp.get_all_start_methods():
        pytest.skip("Start method unavailable")
    initialize = ProcessPoolExecutor.__init__

    def initialize_with_context(pool, *args, **kwargs):
        kwargs["mp_context"] = get_context(start_method)
        initialize(pool, *args, **kwargs)

    monkeypatch.setattr(ProcessPoolExecutor, "__init__", initialize_with_context)
    monkeypatch.setattr(
        _worker_inputs,
        "TemporaryDirectory",
        lambda **kwargs: tempfile.TemporaryDirectory(dir=tmp_path, **kwargs),
    )
    path = tmp_path / "serialization.txt"

    def sampler():
        return ABCSampler(
            _MutatingSimulation(),
            {"beta": stats.uniform()},
            {"probe": _PickleProbe(path)},
            np.zeros(3),
            distance_function=_mutating_distance,
            rng=43,
        )

    options = dict(
        num_particles=6,
        num_generations=2,
        epsilon_schedule=[float("inf")] * 2,
        verbose=False,
    )
    cached = sampler()
    result = cached.calibrate(n_workers=1, **options)
    assert not list(tmp_path.glob("epydemix-workers-*"))
    assert path.read_text().splitlines() == ["serialized"]
    assert cached.parameters["probe"].values == []
    assert cached.simulation_function.calls == 0
    np.testing.assert_array_equal(cached.observed_data["data"], np.zeros(3))

    # Caller-owned pools retain their initializer/lifetime and the original path.
    path.unlink()
    external = sampler()
    with ProcessPoolExecutor(max_workers=1) as pool:
        reference = external.calibrate(executor=pool, **options)
        assert pool.submit(int, "7").result() == 7
    # DYN serializes the caller-owned job once per generation.
    assert len(path.read_text().splitlines()) == 2
    assert_exact_calibration(reference, result)
    assert external.rng.bit_generator.state == cached.rng.bit_generator.state
    cached.parameters["fail"] = True
    with pytest.raises(RuntimeError, match="expected worker failure"):
        cached.calibrate(n_workers=1, **options)
    assert not list(tmp_path.glob("epydemix-workers-*"))


def test_worker_retries_restore_mutable_model_inputs(tmp_path):
    """Reset callbacks, nested inputs and observations on every local retry."""

    def sampler():
        return ABCSampler(
            _MutatingSimulation(),
            {"beta": stats.uniform()},
            {"probe": _PickleProbe(tmp_path / "serialized.txt")},
            np.zeros(3),
            distance_function=_mutating_distance,
            rng=43,
        )

    options = dict(
        strategy="rejection",
        num_particles=6,
        epsilon=0.3,
        verbose=False,
    )
    reference = sampler().calibrate(n_workers=1, **options)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        assert_exact_calibration(
            reference, sampler().calibrate(executor=pool, **options)
        )


def test_initializer_imports_snapshot_before_evaluation_and_keeps_only_bytes(
    tmp_path, monkeypatch
):
    """Warm deserialization imports native model libraries before thread discovery.

    The cached snapshot still restores a fresh mutable object per candidate. Import
    priming must neither evaluate a candidate nor retain a mutated model instance.
    """
    import pickle

    from epydemix import _execution
    from epydemix.calibration import _worker_inputs

    path = tmp_path / "inputs.pkl"
    path.write_bytes(pickle.dumps({"mutable": []}))
    monkeypatch.setattr(_worker_inputs, "_serialized_particle_inputs", None)
    original = _worker_inputs.restore_worker_inputs
    restored = []

    def restore():
        assert not getattr(_execution._thread_limit_scope, "active", False)
        value = original()
        restored.append(value)
        return value

    monkeypatch.setattr(_worker_inputs, "restore_worker_inputs", restore)
    _worker_inputs.initialize_particle_worker(path)
    assert len(restored) == 1
    assert isinstance(_worker_inputs._serialized_particle_inputs, bytes)
    restored[0]["mutable"].append(1)
    assert original() == {"mutable": []}
