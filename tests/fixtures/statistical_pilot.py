"""Run from a source checkout to audit independent seeds and negative controls."""

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path.cwd()))

import numpy as np

from tests.fixtures.abc_reference import environment
from tests.fixtures.statistical_models import (
    DISCRETE_TARGET,
    NORMAL_SCHEDULE,
    PARTICLES,
    make_statistical_sampler,
    normal_reference,
    posterior_summary,
)
from tests.test_abc_statistical import assert_statistical_accuracy


def main(destination):
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment": environment(),
        "strategy": "smc",
        "particles": PARTICLES,
        "seeds": [101, 211, 307],
        "models": {},
    }
    for model in ("normal", "discrete"):
        schedule = NORMAL_SCHEDULE if model == "normal" else (0.5,) * 3
        targets = np.array(
            [
                normal_reference(e) if model == "normal" else DISCRETE_TARGET
                for e in schedule
            ]
        )
        bounds = (
            np.array([0.10, 0.16, 0.08, 0.08, 0.08])
            if model == "normal"
            else np.full(3, 0.06)
        )
        weighted, unweighted = [], []
        start = perf_counter()
        for seed in report["seeds"]:
            result = make_statistical_sampler(model, seed).calibrate(
                num_particles=PARTICLES,
                num_generations=3,
                epsilon_schedule=schedule,
                verbose=False,
            )
            weighted.append([posterior_summary(result, model, g) for g in range(3)])
            unweighted.append(
                [posterior_summary(result, model, g, weighted=False) for g in range(3)]
            )
            print(model, seed, round(perf_counter() - start, 2), flush=True)
        assert_statistical_accuracy(weighted, targets, bounds)
        try:
            assert_statistical_accuracy(unweighted, targets, bounds)
        except AssertionError:
            detected = True
        else:
            detected = False
        prior = (
            np.array([0.0, 1.0, 0.5, 0.6914624612740131, 0.8413447460685429])
            if model == "normal"
            else np.array([0.2, 0.5, 0.3])
        )
        try:
            assert_statistical_accuracy(np.tile(prior, (3, 3, 1)), targets, bounds)
        except AssertionError:
            prior_detected = True
        else:
            prior_detected = False
        report["models"][model] = {
            "targets": targets.tolist(),
            "bounds": bounds.tolist(),
            "rms_errors": np.sqrt(
                np.mean((np.array(weighted) - targets) ** 2, axis=0)
            ).tolist(),
            "max_errors": np.max(abs(np.array(weighted) - targets), axis=0).tolist(),
            "ignored_weights_rms": np.sqrt(
                np.mean((np.array(unweighted) - targets) ** 2, axis=0)
            ).tolist(),
            "ignored_weights_detected": detected,
            "prior_detected": prior_detected,
            "elapsed_seconds": perf_counter() - start,
        }
        destination.write_text(json.dumps(report, indent=2) + "\n")
    print("Pilot saved:", destination, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
