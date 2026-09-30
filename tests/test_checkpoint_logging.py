"""Checkpoint identity and invocation metrics complement exact replay checks.

The seeded mixed continuous/discrete toy model comes from calibration fixtures.
Infinite explicit thresholds make four candidates a complete generation. Resume
can then prove that prior committed work is distinct from this call's physical
work without Monte Carlo tolerances or wall-time assertions.
"""

import logging

import numpy as np
import pytest

from epydemix.calibration import _checkpoint
from tests.fixtures.calibration import make_seeded_sampler


def _run(path, **options):
    sampler = make_seeded_sampler(43, "argument-int")
    return sampler.run_smc(
        num_particles=4,
        epsilon_schedule=[float("inf")] * 3,
        checkpoint_path=path,
        verbose=False,
        **options,
    )


def test_resume_links_invocations_without_reusing_old_metrics(tmp_path, caplog):
    """One saved generation plus two resumed generations means 4 old and 8 new evaluations."""
    path = tmp_path / "run.checkpoint"
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        first = _run(path, num_generations=1)
    first_id = first.calibration_params["execution_metrics"]["run_id"]
    _, first_metadata = _checkpoint.read_checkpoint(path)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        resumed = _run(path, num_generations=3, resume=True)
    metrics = resumed.calibration_params["execution_metrics"]
    assert metrics["run_id"] != first_id
    assert metrics["resumed_committed_simulations"] == 4
    assert metrics["totals"]["simulations"] == metrics["totals"]["retained"] == 8
    assert [generation["generation"] for generation in metrics["generations"]] == [1, 2]
    assert sum(len(p) for p in resumed.posterior_distributions.values()) == 12
    assert metrics["checkpoint_io_seconds"] >= sum(
        g["checkpoint_io_seconds"] for g in metrics["generations"]
    )
    events = [record.epydemix for record in caplog.records]
    resume = next(event for event in events if event["event"] == "checkpoint_resumed")
    assert resume["previous_run_id"] == first_id
    assert resume["checkpoint_id"] == first_metadata["run_id"]
    state, metadata = _checkpoint.read_checkpoint(path)
    assert state["n_simulations"] == 12
    assert metadata["call_run_id"] == metrics["run_id"]
    assert metadata["run_id"] == first_metadata["run_id"]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        finished = _run(path, num_generations=3, resume=True)
    assert (
        finished.calibration_params["execution_metrics"]["totals"]["simulations"] == 0
    )
    assert not [
        record
        for record in caplog.records
        if record.epydemix["event"] == "generation_finished"
    ]


def test_saved_threshold_preserves_scalar_precision(tmp_path):
    """Manifest strings must not round the scalar used for resume's stopping boundary."""
    path = tmp_path / "threshold.checkpoint"
    epsilon = np.longdouble("2.0000000000000000002")
    make_seeded_sampler(43, "argument-int").run_smc(
        num_particles=4,
        num_generations=1,
        epsilon_schedule=[epsilon],
        checkpoint_path=path,
        verbose=False,
    )
    state, _ = _checkpoint.read_checkpoint(path)
    assert isinstance(state["epsilon"], np.longdouble)
    assert state["epsilon"] == epsilon


def test_failed_publication_has_no_save_event_and_preserves_archive(
    tmp_path, caplog, monkeypatch
):
    """A failed resumed write must preserve the last snapshot and propagate its original error."""
    path = tmp_path / "run.checkpoint"
    _run(path, num_generations=1)
    original = path.read_bytes()
    failure = OSError("expected publication failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(_checkpoint, "write_checkpoint", fail)
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        with pytest.raises(OSError) as error:
            _run(path, num_generations=2, resume=True)
    assert error.value is failure
    assert path.read_bytes() == original
    events = [record.epydemix["event"] for record in caplog.records]
    assert events.count("run_failed") == 1
    assert "checkpoint_saved" not in events and "run_finished" not in events
