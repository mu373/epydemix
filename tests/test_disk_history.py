"""Disk history retains file references, not old generation arrays.

A seeded small SIR model uses eight particles for three generations. In-memory
execution supplies an exact replay oracle; weak references check that trajectories
are released before the next generation. This tests retention and persistence,
not statistical posterior accuracy. Injected failures exercise atomic publication
without memory-size thresholds or reliance on OS timing.
"""

import copy
import json
import shutil
import weakref
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from epydemix.calibration import _checkpoint, _evaluate, _history
from tests.fixtures.calibration import assert_exact_calibration, make_sampler


def run(path, **kwargs):
    return make_sampler("sir").calibrate(
        num_particles=8,
        num_generations=3,
        verbose=False,
        checkpoint_path=path,
        history_storage="disk",
        **kwargs,
    )


def test_arrays_released_and_history_loaded_only_on_access(tmp_path):
    """Weak references must expire before the next generation; deepcopy and checkpoint reads must not eagerly load history."""
    path = tmp_path / "run.checkpoint"
    references = []
    original = _evaluate.evaluate_particle

    def capture(*args, **kwargs):
        if len(references) == 8:
            # Generation 0 arrays must be released before evaluating generation 1.
            assert all(reference() is None for reference in references)
        result = original(*args, **kwargs)
        if result["simulation"] is not None:
            references.append(weakref.ref(result["simulation"]["data"]))
        return result

    with patch.object(_evaluate, "evaluate_particle", side_effect=capture):
        result = run(path)
    assert references and all(reference() is None for reference in references)
    field = result.selected_trajectories
    assert isinstance(field, _history.DiskHistory)
    assert list(field) == [0, 1, 2]
    with patch.object(
        _history.DiskHistory, "__getitem__", side_effect=AssertionError("eager load")
    ):
        copied = copy.deepcopy(result)
        saved, _ = _checkpoint.read_checkpoint(path)
        assert list(saved["results"].weights) == [0, 1, 2]
    selected = copied.get_selected_trajectories(0)
    reference = weakref.ref(selected[0]["data"])
    del selected
    assert reference() is None  # Reading does not populate a hidden cache.
    assert len(copied.get_calibration_quantiles(generation=0)) > 0
    baseline = field[0][0]["data"].copy()
    field[0][0]["data"][:] = -1
    np.testing.assert_array_equal(field[0][0]["data"], baseline)


@pytest.mark.parametrize("failure", ["field", "checkpoint", "simulation"])
def test_failed_append_preserves_previous_checkpoint_and_results(tmp_path, failure):
    """Field, checkpoint and model failures must preserve the previous complete archive and previously returned result mappings."""
    path = tmp_path / "run.checkpoint"
    sampler = make_sampler("sir")
    first = sampler.calibrate(
        num_particles=8,
        num_generations=1,
        verbose=False,
        checkpoint_path=path,
        history_storage="disk",
    )
    snapshot = path.read_bytes()
    write = _checkpoint.write_checkpoint
    evaluate = _evaluate.evaluate_particle

    def fail_save(target, *args, **kwargs):
        should_fail = (
            Path(target) == path
            if failure == "checkpoint"
            else "selected_trajectories" in str(target)
        )
        if failure != "simulation" and should_fail:
            raise OSError("injected write failure")
        return write(target, *args, **kwargs)

    calls = 0

    def fail_simulation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "simulation" and calls == 2:
            raise RuntimeError("injected simulation failure")
        return evaluate(*args, **kwargs)

    with patch.object(_checkpoint, "write_checkpoint", side_effect=fail_save):
        with patch.object(_evaluate, "evaluate_particle", side_effect=fail_simulation):
            with pytest.raises((RuntimeError, OSError), match="injected"):
                run(path, resume=True)
    assert path.read_bytes() == snapshot
    assert list(first.posterior_distributions) == [0]
    baseline = make_sampler("sir").calibrate(
        num_particles=8, num_generations=3, verbose=False
    )
    assert_exact_calibration(baseline, run(path, resume=True))
    # Previously returned results are a snapshot, not a view of future generations.
    assert list(first.posterior_distributions) == [0]
    assert first.get_selected_trajectories(0)


def test_move_bundle_and_reject_missing_generation(tmp_path):
    """A checkpoint/history bundle can move together; missing referenced files must fail before new work starts."""
    path = tmp_path / "run.checkpoint"
    reference = run(path)
    # Materialize the reference before deleting the original location.
    expected = {name: dict(getattr(reference, name)) for name in _history.FIELDS}
    target = tmp_path / "moved.checkpoint"
    path.rename(target)
    shutil.move(str(path) + ".history", str(target) + ".history")
    result = run(target, resume=True)
    for name in _history.FIELDS:
        assert list(getattr(result, name)) == list(expected[name])
    np.testing.assert_array_equal(
        result.get_selected_trajectories(0)[0]["data"],
        expected["selected_trajectories"][0][0]["data"],
    )
    filename, _ = result.selected_trajectories.entries[0]
    (result.selected_trajectories.directory / filename).unlink()
    with pytest.raises(FileNotFoundError, match="Missing generation history"):
        run(target, resume=True)


def test_generation_identity_and_checksum(tmp_path):
    """Substituting a valid different generation must be detected before unpickling, not merely treated as an intact archive."""
    path = tmp_path / "run.checkpoint"
    result = run(path)
    field = result.selected_trajectories
    file0 = field.directory / field.entries[0][0]
    file1 = field.directory / field.entries[1][0]
    shutil.copyfile(file1, file0)
    with patch.object(
        _checkpoint.pickle, "load", side_effect=AssertionError("unpickled")
    ):
        with pytest.raises(ValueError, match="does not match checkpoint reference"):
            field[0]


@pytest.mark.parametrize("storage", ["disk", "invalid"])
def test_storage_validation(storage):
    """Reject unsupported modes and disk history without a destination before simulation work."""
    with pytest.raises(ValueError, match="history_storage"):
        make_sampler("sir").calibrate(history_storage=storage, verbose=False)


def test_storage_mode_mismatch_and_partial_budget(tmp_path):
    """Replay requires the saved storage mode and must exclude an incomplete generation stopped by a physical budget."""
    path = tmp_path / "run.checkpoint"
    partial = run(path, total_simulations_budget=10)
    assert list(partial.posterior_distributions) == [0]
    with pytest.raises(ValueError, match="history_storage must match"):
        make_sampler("sir").calibrate(
            num_particles=8,
            num_generations=3,
            checkpoint_path=path,
            resume=True,
            verbose=False,
        )
    reference = make_sampler("sir").calibrate(
        num_particles=8, num_generations=3, verbose=False
    )
    assert_exact_calibration(reference, run(path, resume=True))


def test_legacy_memory_checkpoint_without_storage_field_can_resume(tmp_path):
    """Version-one memory checkpoints predate the optional storage-mode field.

    Missing mode means memory, so adding disk history must not invalidate those
    archives. Only the manifest is edited; the state and its checksum are intact.
    """
    path = tmp_path / "legacy.checkpoint"
    sampler = make_sampler("deterministic")
    sampler.calibrate(
        num_particles=8, num_generations=1, checkpoint_path=path, verbose=False
    )
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("state.pkl")
        metadata = json.loads(archive.read("metadata.json"))
    metadata.pop("history_storage")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("state.pkl", payload)
        archive.writestr("metadata.json", json.dumps(metadata))
    resumed = make_sampler("deterministic").calibrate(
        num_particles=8,
        num_generations=2,
        checkpoint_path=path,
        resume=True,
        verbose=False,
    )
    reference = make_sampler("deterministic").calibrate(
        num_particles=8, num_generations=2, verbose=False
    )
    assert_exact_calibration(reference, resumed)
