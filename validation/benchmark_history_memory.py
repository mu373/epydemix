"""Compare SMC history retention with synthetic 16 MiB-per-generation outputs."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from scipy import stats

from epydemix._logging import _json_values
from epydemix.calibration.abc import ABCSampler


def simulate(params):
    return {
        "data": np.array([params["beta"]]),
        "payload": np.ones(int(params["payload_mib"] * 2**20 / 8)),
    }


def child(storage, generations, path, particles, payload_mib):
    sampler = ABCSampler(
        simulate,
        {"beta": stats.uniform(0.1, 0.4)},
        {"payload_mib": payload_mib},
        np.zeros(1),
        rng=43,
    )
    process = psutil.Process()
    before = process.memory_info().rss / 2**20
    started = perf_counter()
    result = sampler.calibrate(
        num_particles=particles,
        num_generations=generations,
        epsilon_schedule=[float("inf")] * generations,
        checkpoint_path=path,
        history_storage=storage,
        verbose=False,
    )
    elapsed = perf_counter() - started
    after = process.memory_info().rss / 2**20
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = max(before, after, maxrss / (2**20 if sys.platform == "darwin" else 1024))
    assert len(result.posterior_distributions) == generations
    fingerprint_started = perf_counter()
    digest = hashlib.sha256()
    for generation in result.posterior_distributions:
        for array in (
            result.get_posterior_distribution(generation).to_numpy(),
            result.get_weights(generation),
            result.get_distances(generation),
        ):
            digest.update(np.asarray(array).tobytes())
        for trajectory in result.get_selected_trajectories(generation):
            for array in trajectory.values():
                digest.update(np.asarray(array).tobytes())
    fingerprint_seconds = perf_counter() - fingerprint_started
    return {
        "storage": storage,
        "generations": generations,
        "particles": particles,
        "payload_mib": payload_mib,
        "seed": 43,
        "trajectory_data_mib": particles
        * int(payload_mib * 2**20 / 8)
        * 8
        * generations
        / 2**20,
        "elapsed_seconds": elapsed,
        "fingerprint": digest.hexdigest(),
        "fingerprint_seconds": fingerprint_seconds,
        "execution_metrics": _json_values(
            result.calibration_params["execution_metrics"]
        ),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "rss_before_mib": before,
        "rss_after_mib": after,
        "peak_rss_mib": peak,
        "checkpoint_mib": path.stat().st_size / 2**20,
        "history_mib": sum(
            file.stat().st_size for file in Path(str(path) + ".history").glob("*")
        )
        / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=["memory", "disk"])
    parser.add_argument("--generations", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--payload-mib", type=float, default=2)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.child:
        print(
            json.dumps(
                child(
                    args.child,
                    args.generations[0],
                    args.path,
                    args.particles,
                    args.payload_mib,
                ),
                allow_nan=False,
            )
        )
        return
    with tempfile.TemporaryDirectory() as directory:
        for repeat in range(args.repeat):
            for generations in args.generations:
                for storage in ("memory", "disk"):
                    path = str(
                        Path(directory) / f"{storage}-{generations}-{repeat}.checkpoint"
                    )
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "validation.benchmark_history_memory",
                            "--child",
                            storage,
                            "--generations",
                            str(generations),
                            "--particles",
                            str(args.particles),
                            "--payload-mib",
                            str(args.payload_mib),
                            "--path",
                            path,
                        ],
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=60,
                    )
                    report = json.loads(completed.stdout)
                    print(json.dumps(dict(report, repeat=repeat)), flush=True)


if __name__ == "__main__":
    main()
