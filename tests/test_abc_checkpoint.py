"""SMC checkpoint replay, input matching, integrity, and atomic publication."""

import pickle
import subprocess
import sys
import warnings
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from epydemix.calibration import _checkpoint, _worker
from epydemix.calibration.parallel import _get_available_cpu_count
from epydemix.utils.abc_smc_utils import DefaultPerturbationContinuous
from tests.fixtures.calibration import (
    assert_exact_calibration,
    make_sampler,
    make_seeded_sampler,
)


class StatefulKernel(DefaultPerturbationContinuous):
    def __init__(self, param_name):
        super().__init__(param_name)
        self.updates = 0

    def update(self, particles, weights, param_names):
        super().update(particles, weights, param_names)
        self.updates += 1
        self.std *= 1 + 0.05 * self.updates


def _run(sampler, **kwargs):
    return sampler.calibrate(
        num_particles=8, num_generations=3, verbose=False, **kwargs
    )


@pytest.mark.parametrize("model", ["deterministic", "sir"])
def test_replay_in_fresh_spawn_process_with_changed_workers_and_projections(
    tmp_path, model
):
    checkpoint = tmp_path / "run.checkpoint"
    output = tmp_path / "resumed.pkl"
    reference_sampler = make_sampler(model)
    reference = _run(reference_sampler)
    reference_projection = reference_sampler.run_projections(
        reference_sampler.parameters, iterations=4
    )
    partial = make_sampler(model)
    partial.calibrate(
        num_particles=8,
        num_generations=1,
        checkpoint_path=checkpoint,
        verbose=False,
    )
    saved, meta = _checkpoint.read_checkpoint(checkpoint)
    assert saved["next_generation"] == 1
    assert saved["n_simulations"] == 8
    np.testing.assert_array_equal(
        saved["inputs"]["observed_data"]["data"], partial.observed_data["data"]
    )
    assert (
        saved["inputs"]["priors"]["transmission_rate"].kwds
        == partial.priors["transmission_rate"].kwds
    )
    if model == "sir":
        np.testing.assert_array_equal(
            saved["inputs"]["parameters"]["epimodel"].population.Nk, [10000]
        )
        assert saved["inputs"]["parameters"]["epimodel"].definitions == {}
    # An actual new interpreter, not just a new object; spawn workers inside it.
    script = """
import multiprocessing as mp
import pickle
import sys
from tests.fixtures.calibration import make_sampler
if __name__ == "__main__":
    mp.set_start_method("spawn")
    sampler = make_sampler(sys.argv[1], seed=999)
    result = sampler.calibrate(num_particles=8, num_generations=3, checkpoint_path=sys.argv[2], resume=True, n_workers=int(sys.argv[4]), verbose=False)
    projection = sampler.run_projections(sampler.parameters, iterations=4, n_workers=int(sys.argv[4]))
    with open(sys.argv[3], "wb") as f:
        pickle.dump((result, projection, sampler.rng.bit_generator.state), f)
"""
    # -c uses importable installed/workspace modules without modifying sys.path.
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            model,
            str(checkpoint),
            str(output),
            str(min(2, _get_available_cpu_count())),
        ],
        check=True,
        timeout=60,
    )
    with output.open("rb") as file:
        result, projection, rng_state = pickle.load(file)
    assert_exact_calibration(reference, result)
    np.testing.assert_array_equal(
        reference_projection.get_projection_trajectories()["data"],
        projection.get_projection_trajectories()["data"],
    )
    assert reference_sampler.rng.bit_generator.state == rng_state
    state, final_meta = _checkpoint.read_checkpoint(checkpoint)
    assert state["next_generation"] == 3
    assert final_meta["run_id"] == meta["run_id"]
    assert final_meta["elapsed_seconds"] >= meta["elapsed_seconds"]


def test_stateful_kernel_and_nondefault_rng_replay(tmp_path):
    def sampler():
        value = make_seeded_sampler(43, "argument-int")
        value.rng = np.random.Generator(np.random.Philox(43))
        return value

    reference = _run(
        sampler(),
        perturbations={"beta": StatefulKernel("beta"), "count": _discrete_kernel()},
    )
    path = tmp_path / "stateful.checkpoint"
    partial = sampler()
    partial.calibrate(
        num_particles=8,
        num_generations=2,
        perturbations={"beta": StatefulKernel("beta"), "count": _discrete_kernel()},
        checkpoint_path=path,
        verbose=False,
    )
    result = _run(
        sampler(),
        perturbations={"beta": StatefulKernel("beta"), "count": _discrete_kernel()},
        checkpoint_path=path,
        resume=True,
    )
    assert_exact_calibration(reference, result)
    state, _ = _checkpoint.read_checkpoint(path)
    assert state["perturbations"]["beta"].updates == 2


def _discrete_kernel():
    from epydemix.utils.abc_smc_utils import DefaultPerturbationDiscrete

    return DefaultPerturbationDiscrete("count", stats.randint(1, 6))


@pytest.mark.parametrize(
    "change", ["observed", "parameters", "prior", "particles", "epsilon"]
)
def test_mismatched_inputs_rejected_without_rng_change(tmp_path, change):
    path = tmp_path / "run.checkpoint"
    _run(make_sampler("deterministic"), checkpoint_path=path)
    original = path.read_bytes()
    sampler = make_sampler("deterministic")
    kwargs = dict(num_particles=8, num_generations=4, verbose=False)
    if change == "observed":
        sampler.observed_data["data"][0] += 1
    elif change == "parameters":
        sampler.parameters["extra"] = 1
    elif change == "prior":
        sampler.priors["transmission_rate"] = stats.uniform(0.2, 0.4)
    elif change == "particles":
        kwargs["num_particles"] = 9
    else:
        kwargs["epsilon_quantile_level"] = 0.7
    state = sampler.rng.bit_generator.state
    with pytest.raises(ValueError, match="inputs or SMC settings"):
        sampler.calibrate(checkpoint_path=path, resume=True, **kwargs)
    assert sampler.rng.bit_generator.state == state
    assert path.read_bytes() == original


def test_input_hash_is_order_independent_and_array_sensitive():
    left = {"a": np.arange(4, dtype=np.int32), "b": stats.uniform(0.1, 0.4)}
    right = {"b": stats.uniform(0.1, 0.4), "a": np.arange(4, dtype=np.int32)}
    assert _checkpoint.input_hash(left) == _checkpoint.input_hash(right)
    for value in [
        np.arange(4, dtype=np.int64),
        np.arange(4, dtype=np.int32).reshape(2, 2),
        np.arange(4, dtype=np.int32) + 1,
    ]:
        assert _checkpoint.input_hash(left) != _checkpoint.input_hash(
            {**right, "a": value}
        )


def test_payload_checksum_checked_before_unpickling(tmp_path):
    path = tmp_path / "run.checkpoint"
    _run(make_sampler("deterministic"), checkpoint_path=path)
    with zipfile.ZipFile(path) as archive:
        metadata = archive.read("metadata.json")
        data = archive.read("state.pkl")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", metadata)
        archive.writestr("state.pkl", data[:-1] + b"!")
    with patch.object(_checkpoint.pickle, "load") as loads:
        with pytest.raises(ValueError, match="data checksum"):
            _checkpoint.read_checkpoint(path)
        loads.assert_not_called()


def test_failed_save_preserves_previous_generation(tmp_path):
    path = tmp_path / "run.checkpoint"
    make_sampler("sir").calibrate(
        num_particles=8, num_generations=1, checkpoint_path=path, verbose=False
    )
    previous = path.read_bytes()
    with patch.object(_checkpoint.os, "replace", side_effect=OSError("disk error")):
        with pytest.raises(OSError, match="disk error"):
            _run(make_sampler("sir"), checkpoint_path=path, resume=True)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    assert_exact_calibration(
        _run(make_sampler("sir")),
        _run(make_sampler("sir"), checkpoint_path=path, resume=True),
    )


def test_incomplete_generation_and_budget_resume(tmp_path):
    path = tmp_path / "run.checkpoint"
    partial = _run(
        make_sampler("sir"), checkpoint_path=path, total_simulations_budget=10
    )
    assert len(partial.posterior_distributions) == 1
    state, _ = _checkpoint.read_checkpoint(path)
    assert state["n_simulations"] == 8
    assert_exact_calibration(
        _run(make_sampler("sir")),
        _run(
            make_sampler("sir"),
            checkpoint_path=path,
            resume=True,
            total_simulations_budget=1000,
        ),
    )


def test_existing_file_and_no_complete_generation(tmp_path):
    path = tmp_path / "run.checkpoint"
    _run(make_sampler("sir"), checkpoint_path=path, total_simulations_budget=0)
    assert not path.exists()
    _run(make_sampler("sir"), checkpoint_path=path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _run(make_sampler("sir"), checkpoint_path=path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="requires checkpoint_path"):
        _run(make_sampler("sir"), resume=True)


def test_simulation_failure_leaves_replayable_complete_generation(tmp_path):
    from epydemix.calibration import _worker

    path = tmp_path / "run.checkpoint"
    original = _worker.evaluate_particle
    count = 0

    def fail_mid_generation(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 10:
            raise RuntimeError("interrupted simulation")
        return original(*args, **kwargs)

    with patch.object(_worker, "evaluate_particle", side_effect=fail_mid_generation):
        with pytest.raises(RuntimeError, match="interrupted simulation"):
            _run(make_sampler("sir"), checkpoint_path=path)
    state, _ = _checkpoint.read_checkpoint(path)
    assert state["next_generation"] == 1
    assert_exact_calibration(
        _run(make_sampler("sir")),
        _run(make_sampler("sir"), checkpoint_path=path, resume=True),
    )


def test_resuming_finished_run_and_fresh_time_allowance(tmp_path):
    from datetime import timedelta

    path = tmp_path / "run.checkpoint"
    reference = _run(make_sampler("sir"), checkpoint_path=path)
    saved = path.read_bytes()
    assert_exact_calibration(
        reference, _run(make_sampler("sir"), checkpoint_path=path, resume=True)
    )
    assert path.read_bytes() == saved
    result = make_sampler("sir").calibrate(
        num_particles=8,
        num_generations=4,
        checkpoint_path=path,
        resume=True,
        max_time=timedelta(0),
        verbose=False,
    )
    assert_exact_calibration(reference, result)
    assert path.read_bytes() == saved
    result = make_sampler("sir").calibrate(
        num_particles=8,
        num_generations=4,
        checkpoint_path=path,
        resume=True,
        max_time=timedelta(seconds=30),
        verbose=False,
    )
    assert len(result.posterior_distributions) == 4


@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_streamed_checkpoint_without_whole_payload_buffers(tmp_path, order):
    data = np.arange(1024 * 1024, dtype=np.float64).reshape(1024, 1024)
    if order == "F":
        data = np.asfortranarray(data)
    elif order == "strided":
        data = data[:, ::2]
    state = {"inputs": {"data": data}, "results": data}
    metadata = {
        "input_sha256": _checkpoint.input_hash(state["inputs"]),
        "environment": _checkpoint.environment(),
    }
    path = tmp_path / "stream.checkpoint"
    read = zipfile.ZipFile.read

    def metadata_only(archive, name, *args, **kwargs):
        assert name == "metadata.json"
        return read(archive, name, *args, **kwargs)

    with patch.object(
        _checkpoint.pickle, "dumps", side_effect=AssertionError("whole payload")
    ):
        _checkpoint.validate_picklable(state)
        _checkpoint.write_checkpoint(path, state, metadata, overwrite=False)
    with patch.object(zipfile.ZipFile, "read", metadata_only):
        restored, _ = _checkpoint.read_checkpoint(path)
    np.testing.assert_array_equal(restored["results"], data)
    assert _checkpoint._array_digest(data) == _checkpoint._array_digest(
        np.array(data, order="C")
    )


def test_checkpoint_manifest_and_expected_identity(tmp_path):
    state = {"inputs": {"x": 1}, "results": [1, 2]}
    metadata = {
        "input_sha256": _checkpoint.input_hash(state["inputs"]),
        "environment": _checkpoint.environment(),
    }
    path = tmp_path / "identity.checkpoint"
    manifest = _checkpoint.write_checkpoint(path, state, metadata, overwrite=False)
    restored, loaded_manifest = _checkpoint.read_checkpoint(
        path, expected_data_sha256=manifest["data_sha256"]
    )
    assert restored == state
    assert loaded_manifest == manifest
    assert "data_sha256" not in metadata
    with patch.object(
        _checkpoint.pickle, "load", side_effect=AssertionError("unpickled")
    ):
        with pytest.raises(ValueError, match="does not match checkpoint reference"):
            _checkpoint.read_checkpoint(path, expected_data_sha256="0" * 64)


@pytest.mark.parametrize("outcome", ["complete", "stopped", "error"])
def test_resumed_rng_returns_to_sampler_on_every_exit(tmp_path, outcome):
    path = tmp_path / "rng.checkpoint"
    partial = make_sampler("deterministic")
    partial.rng = np.random.Generator(np.random.Philox(43))
    partial.calibrate(
        num_particles=8, num_generations=1, checkpoint_path=path, verbose=False
    )
    saved, _ = _checkpoint.read_checkpoint(path)
    expected_rng = np.random.Generator(np.random.Philox(saved["rng_seed_sequence"]))
    expected_rng.bit_generator.state = saved["rng_state"]
    resumed = make_sampler("deterministic", seed=999)  # Starts with PCG64.
    options = dict(
        num_particles=8,
        num_generations=3,
        checkpoint_path=path,
        resume=True,
        verbose=False,
    )
    if outcome == "complete":
        reference = make_sampler("deterministic")
        reference.rng = np.random.Generator(np.random.Philox(43))
        expected = _run(reference)
        actual = resumed.calibrate(**options)
        assert_exact_calibration(expected, actual)
        expected_rng = reference.rng
    elif outcome == "stopped":
        actual = resumed.calibrate(**options, total_simulations_budget=8)
        assert list(actual.posterior_distributions) == [0]
    else:
        original = path.read_bytes()
        with patch.object(
            _worker, "evaluate_particle", side_effect=RuntimeError("failed")
        ):
            with pytest.raises(RuntimeError, match="failed"):
                resumed.calibrate(**options)
        # A generation consumes one root draw before the first candidate evaluation.
        expected_rng.integers(0, 2**32, size=4, dtype=np.uint32)
        assert path.read_bytes() == original
    assert isinstance(resumed.rng.bit_generator, np.random.Philox)
    np.testing.assert_equal(
        resumed.rng.bit_generator.state, expected_rng.bit_generator.state
    )


@pytest.mark.parametrize("kind", ["series", "frame", "index"])
@pytest.mark.parametrize("change", ["order", "categories", "ordered"])
def test_pandas_hash_preserves_categorical_metadata(kind, change):
    def make(categories, ordered):
        values = pd.Categorical(["a", "b"], categories=categories, ordered=ordered)
        if kind == "index":
            return pd.CategoricalIndex(values)
        series = pd.Series(values)
        return series.to_frame() if kind == "frame" else series

    baseline = make(["a", "b"], True)
    changed = make(
        ["b", "a"]
        if change == "order"
        else ["a", "b", "c"]
        if change == "categories"
        else ["a", "b"],
        change != "ordered",
    )
    fingerprint = _checkpoint.input_hash(baseline)
    assert fingerprint != _checkpoint.input_hash(changed)
    assert fingerprint == _checkpoint.input_hash(pickle.loads(pickle.dumps(baseline)))


def test_dataframe_hash_preserves_large_integers_in_mixed_columns():
    left = pd.DataFrame({"integer": [2**53], "float": [0.5]})
    right = pd.DataFrame({"integer": [2**53 + 1], "float": [0.5]})
    np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())
    assert _checkpoint.input_hash(left) != _checkpoint.input_hash(right)


@pytest.mark.parametrize("kind", ["series", "frame", "index"])
def test_pandas_string_hash_preserves_values_and_missing_data(kind):
    def make(values):
        series = pd.Series(values, dtype=pd.StringDtype(storage="python"))
        if kind == "index":
            return pd.Index(series)
        return series.to_frame() if kind == "frame" else series

    original = make(["a", None, "b"])
    fingerprint = _checkpoint.input_hash(original)
    assert fingerprint == _checkpoint.input_hash(pickle.loads(pickle.dumps(original)))
    assert fingerprint != _checkpoint.input_hash(make(["a", None, "c"]))
    assert fingerprint != _checkpoint.input_hash(make(["a", "", "b"]))


def test_unsupported_pandas_extension_dtype_fails_explicitly():
    with pytest.raises(TypeError, match="Unsupported checkpoint pandas dtype"):
        _checkpoint.input_hash(pd.Series([1, 2], dtype="Int64"))


def test_resume_rejects_changed_categories(tmp_path):
    path = tmp_path / "categorical.checkpoint"
    sampler = make_sampler("deterministic")
    sampler.parameters["labels"] = pd.Series(pd.Categorical(["a", "b"], ordered=True))
    sampler.calibrate(
        num_particles=3, num_generations=1, checkpoint_path=path, verbose=False
    )
    sampler.parameters["labels"] = sampler.parameters["labels"].cat.reorder_categories(
        ["b", "a"]
    )
    with pytest.raises(ValueError, match="inputs or SMC settings"):
        sampler.calibrate(
            num_particles=3,
            num_generations=2,
            checkpoint_path=path,
            resume=True,
            verbose=False,
        )


def test_environment_checked_only_when_resuming(tmp_path):
    path = tmp_path / "environment.checkpoint"
    options = dict(
        num_particles=3,
        num_generations=1,
        checkpoint_path=path,
        verbose=False,
    )
    result = make_sampler("deterministic").calibrate(**options)
    with patch.object(
        _checkpoint, "environment", return_value={"different": "runtime"}
    ) as inspect_environment:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            _checkpoint.read_checkpoint(path)
            np.testing.assert_array_equal(result.get_weights(), np.full(3, 1 / 3))
        assert not emitted
        inspect_environment.assert_not_called()
        with pytest.warns(
            RuntimeWarning, match="exact replay is not guaranteed"
        ) as emitted:
            make_sampler("deterministic").calibrate(**options, resume=True)
        assert len(emitted) == 1
        inspect_environment.assert_called_once()


def test_unsupported_checkpoint_format_is_rejected(tmp_path):
    path = tmp_path / "unsupported.checkpoint"
    with patch.object(_checkpoint, "FORMAT_VERSION", 999):
        make_sampler("deterministic").calibrate(
            num_particles=3, num_generations=1, checkpoint_path=path, verbose=False
        )
    with pytest.raises(ValueError, match="Unsupported SMC checkpoint format"):
        _checkpoint.read_checkpoint(path)
