"""Check observable ABC work independently of posterior/RNG correctness.

Small injected-RNG normal models exercise all entry points. An explicit budget
leaves a second SMC generation incomplete; its physical work must be observable
even though its particles are discarded. No timing threshold is used.
"""

import io
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, get_context

import numpy as np
import pytest
from scipy import stats

from epydemix import _logging
from epydemix.calibration.abc import ABCSampler
from tests.fixtures.calibration import assert_exact_calibration


def _normal_observation(parameters):
    return {"data": np.array([parameters["rng"].normal(parameters["mu"], 1)])}


def _synchronized_observation(parameters):
    parameters["barrier"].wait(timeout=10)
    return {"data": np.array([1.0])}


def _sampler():
    return ABCSampler(_normal_observation, {"mu": stats.norm()}, {}, np.ones(1), rng=43)


@pytest.mark.parametrize(
    "strategy,options",
    [
        (
            "smc",
            {
                "num_particles": 4,
                "num_generations": 2,
                "epsilon_schedule": [float("inf")] * 2,
            },
        ),
        (
            "rejection",
            {
                "num_particles": 4,
                "epsilon": float("inf"),
                "progress_update_interval": 2,
            },
        ),
        ("top_fraction", {"Nsim": 8, "top_fraction": 0.5}),
    ],
)
def test_logging_is_independent_of_verbose_and_preserves_rng(
    caplog, capsys, strategy, options
):
    """INFO capture with verbose=False preserves the same full mathematical output."""
    reference_sampler, sampler = _sampler(), _sampler()
    with caplog.at_level(logging.WARNING, logger="epydemix.calibration"):
        reference = reference_sampler.calibrate(
            strategy=strategy, verbose=False, **options
        )
    assert not [record for record in caplog.records if hasattr(record, "epydemix")]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        result = sampler.calibrate(strategy=strategy, verbose=False, **options)
    assert_exact_calibration(reference, result)
    assert sampler.rng.bit_generator.state == reference_sampler.rng.bit_generator.state
    assert capsys.readouterr().out == ""
    events = [
        record.epydemix for record in caplog.records if hasattr(record, "epydemix")
    ]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    assert len({event["run_id"] for event in events}) == 1
    metrics = result.calibration_params["execution_metrics"]
    assert (
        metrics["run_id"] != reference.calibration_params["execution_metrics"]["run_id"]
    )
    expected_simulations = 4 if strategy == "rejection" else 8
    assert metrics["totals"]["simulations"] == expected_simulations
    assert metrics["totals"]["retained"] == sum(
        len(p) for p in result.posterior_distributions.values()
    )
    formatter = _logging.JSONFormatter()
    decoded = [json.loads(formatter.format(record)) for record in caplog.records]
    assert all(event["timestamp"].endswith("+00:00") for event in decoded)
    assert not any(
        key in event
        for event in decoded
        for key in ("parameters", "data", "trajectories")
    )
    if strategy != "top_fraction":
        generations = [
            event for event in decoded if event["event"] == "generation_finished"
        ]
        assert generations[0]["epsilon"] == "Infinity"


def test_incomplete_generation_work_is_counted_but_not_retained(caplog):
    """A budget of six completes four particles, then discards two next-gen matches."""
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        result = _sampler().run_smc(
            num_particles=4,
            num_generations=2,
            epsilon_schedule=[float("inf")] * 2,
            total_simulations_budget=6,
            verbose=False,
        )
    assert list(result.posterior_distributions) == [0]
    metrics = result.calibration_params["execution_metrics"]
    assert metrics["totals"] == {
        "simulations": 6,
        "matched": 6,
        "retained": 4,
        "surplus_accepted": 0,
        "drained": 0,
    }
    assert metrics["stop_reason"] == "budget"
    first, second = metrics["generations"]
    assert first["simulations"] == 4 and first["completed"] is True
    assert (
        second["simulations"] == 2
        and second["retained"] == 0
        and second["completed"] is False
    )
    terminal = caplog.records[-1].epydemix
    assert terminal["totals"]["simulations"] == 6


def test_many_cheap_rejections_do_not_emit_per_candidate_smc_progress(
    caplog, monkeypatch
):
    """200 evaluations within a frozen display clock produce one final generation event.

    Cheap models can reject many times per particle. Particle-count-based intervals
    must not turn their progress into a log/print per evaluation. The final counts
    remain observable even when all intermediate reports are suppressed.
    """
    from epydemix.calibration import _smc

    monkeypatch.setattr(_smc, "perf_counter", lambda: 0.0)
    sampler = _sampler()
    calls = 0

    def rare_match(parameters):
        nonlocal calls
        calls += 1
        return {"data": np.array([1.0 if calls % 100 == 0 else 0.0])}

    sampler.simulation_function = rare_match
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        result = sampler.run_smc(
            num_particles=2, num_generations=1, epsilon_schedule=[0.5], verbose=False
        )
    events = [record.epydemix for record in caplog.records]
    assert [event["event"] for event in events] == [
        "run_started",
        "generation_finished",
        "run_finished",
    ]
    assert (
        result.calibration_params["execution_metrics"]["totals"]["simulations"]
        == calls
        == 200
    )


def test_parallel_logs_separate_matches_retention_and_drain(caplog):
    """A barrier forces two completed matches for one retained rejection particle.

    Reporting actual matches and post-cutoff completions avoids treating all
    rejected proposals as surplus, or hiding physical work behind the target.
    """
    with Manager() as manager:
        sampler = ABCSampler(
            _synchronized_observation,
            {"mu": stats.norm()},
            {"barrier": manager.Barrier(2)},
            np.ones(1),
            rng=43,
        )
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as pool:
            with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
                result = sampler.run_rejection(
                    num_particles=1,
                    epsilon=1.0,
                    executor=pool,
                    progress_update_interval=1,
                    verbose=False,
                )
    metrics = result.calibration_params["execution_metrics"]
    assert metrics["workers"] == 2
    assert metrics["totals"] == {
        "simulations": 2,
        "matched": 2,
        "retained": 1,
        "surplus_accepted": 1,
        "drained": 1,
    }
    progress = [
        record.epydemix
        for record in caplog.records
        if record.epydemix["event"] == "progress"
    ]
    assert progress[-1]["matched"] == 2 and progress[-1]["collected"] == 1
    assert [event["simulations"] for event in progress] == sorted(
        event["simulations"] for event in progress
    )


def test_failure_emits_once_and_preserves_original_exception(caplog):
    """A user callback failure propagates unchanged and omits potentially private values."""
    failure = RuntimeError("private model inputs must not enter logs")
    sampler = _sampler()

    def fail(parameters):
        raise failure

    sampler.simulation_function = fail
    with caplog.at_level(logging.INFO, logger="epydemix.calibration"):
        with pytest.raises(RuntimeError) as error:
            sampler.calibrate(strategy="rejection", num_particles=1, verbose=False)
    assert error.value is failure
    events = [record.epydemix for record in caplog.records]
    assert [event["event"] for event in events] == ["run_started", "run_failed"]
    assert events[-1]["error_type"] == "RuntimeError"
    assert "private" not in json.dumps(events)


def test_json_handler_writes_one_line_without_root_reconfiguration():
    """Applications can install a stream handler; the library supplies only formatting."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_logging.JSONFormatter())
    logger = _logging.logger
    previous = logger.level
    root_handlers = list(logging.getLogger().handlers)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        _logging.emit("example", run_id="test", epsilon=float("inf"))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert logging.getLogger().handlers == root_handlers
    assert len(stream.getvalue().splitlines()) == 1
    assert json.loads(stream.getvalue())["epsilon"] == "Infinity"
