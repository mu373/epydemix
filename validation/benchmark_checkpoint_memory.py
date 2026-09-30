"""Measure checkpoint I/O in fresh processes using synthetic numeric trajectories."""

import argparse
import copy
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

from epydemix.calibration import _checkpoint
from epydemix.calibration.calibration_results import CalibrationResults


def child(operation, path, size_mib):
    if size_mib < 1:
        raise ValueError("size_mib must be positive")
    inputs = {"observed_data": np.zeros(16), "parameters": {"example": True}}
    if operation != "load":
        result = CalibrationResults(
            selected_trajectories={
                generation: [{"data": np.ones(size_mib * 2**20 // 8 // 8)}]
                for generation in range(8)
            }
        )
        state = {"inputs": inputs, "results": result}
    metadata = {
        "input_sha256": _checkpoint.input_hash(inputs),
        "environment": _checkpoint.environment(),
    }
    process = psutil.Process()
    before = process.memory_info().rss / 2**20
    started = perf_counter()
    if operation == "save":
        _checkpoint.write_checkpoint(path, state, metadata, overwrite=False)
    elif operation == "load":
        state, metadata = _checkpoint.read_checkpoint(path)
    else:
        duplicate = copy.deepcopy(result)
        assert (
            duplicate.selected_trajectories.keys()
            == result.selected_trajectories.keys()
        )
        state["results"] = duplicate
    elapsed = perf_counter() - started
    after = process.memory_info().rss / 2**20
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = max(before, after, maxrss / (2**20 if sys.platform == "darwin" else 1024))
    digest = hashlib.sha256()
    for generation in state["results"].selected_trajectories.values():
        for trajectory in generation:
            digest.update(trajectory["data"].tobytes())
    return {
        "operation": operation,
        "elapsed_seconds": elapsed,
        "fingerprint": digest.hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "checkpoint": _checkpoint.environment(),
        },
        "requested_size_mib": size_mib,
        "model": "synthetic constant float64 trajectories",
        "seed": None,
        "generations": 8,
        "trajectory_data_mib": sum(
            trajectory["data"].nbytes
            for generation in state["results"].selected_trajectories.values()
            for trajectory in generation
        )
        / 2**20,
        "rss_before_mib": before,
        "rss_after_mib": after,
        "peak_rss_mib": peak,
        "checkpoint_mib": path.stat().st_size / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=["save", "load", "copy"])
    parser.add_argument("--path", type=Path)
    parser.add_argument("--size-mib", type=int, default=64)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(args.child, args.path, args.size_mib)))
        return
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "memory.checkpoint")
        for operation in ("save", "load", "copy"):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "validation.benchmark_checkpoint_memory",
                    "--child",
                    operation,
                    "--path",
                    path,
                    "--size-mib",
                    str(args.size_mib),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            print(completed.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
