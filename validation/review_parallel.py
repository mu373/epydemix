"""Bounded DYN fault probes: python -m validation.review_parallel.

These are diagnostic subprocess checks, not speed or statistical tests. Deadlines
stop new candidates and drain running callbacks; arbitrary hung model code cannot
be preempted. The intentional hang is killed as a process group by this launcher.
Run on POSIX; standard cross-platform executor tests live in tests/ instead.
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
from datetime import timedelta
from time import perf_counter, sleep

import numpy as np
from scipy import stats

from epydemix.calibration import ABCSampler


def fault_model(parameters):
    """A scalar simulation isolates scheduler exception and drain behavior."""
    case = parameters["case"]
    if case == "error":
        raise RuntimeError("intentional validation exception")
    if case == "crash":
        os._exit(12)
    if case == "hang":
        while True:
            sleep(1)
    if case == "deadline":
        sleep(0.1)
    if case == "isolation":
        parameters["mutable"].append(parameters["beta"])
        return {"data": np.array([len(parameters["mutable"])])}
    return {"data": np.array([1.0])}


def probe(case):
    mp.set_start_method("spawn", force=True)
    function = (
        (lambda parameters: {"data": np.array([1.0])})
        if case == "pickle"
        else fault_model
    )
    sampler = ABCSampler(
        function,
        {"beta": stats.uniform()},
        {"case": case, "mutable": []},
        np.array([0.0]),
        rng=43,
    )
    started = perf_counter()
    try:
        result = sampler.calibrate(
            strategy="top_fraction" if case == "isolation" else "rejection",
            n_workers=2,
            verbose=False,
            **(
                {"Nsim": 12, "top_fraction": 1.0}
                if case == "isolation"
                else {
                    "num_particles": 4,
                    "epsilon": 0.5,
                    "max_time": timedelta(seconds=5)
                    if case in ("hang", "deadline")
                    else None,
                }
            ),
        )
    except Exception as error:
        expected = {
            "error": (RuntimeError,),
            "crash": (Exception,),
            "pickle": (AttributeError,),
        }
        assert case in expected and isinstance(error, expected[case]), (case, error)
        if case == "crash":
            assert type(error).__name__ == "BrokenProcessPool"
        outcome = {"exception": type(error).__name__}
    else:
        metrics = result.calibration_params["execution_metrics"]
        if case == "isolation":
            np.testing.assert_array_equal(
                result.get_calibration_trajectories()["data"], np.ones((12, 1))
            )
            assert not sampler.parameters["mutable"]
        else:
            assert case == "deadline" and metrics["stop_reason"] == "deadline"
        outcome = {"stop_reason": metrics["stop_reason"], "totals": metrics["totals"]}
    assert not mp.active_children(), "Owned processes must be joined on every exit"
    return dict(outcome, seconds=perf_counter() - started, live_children=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = ["error", "crash", "pickle", "deadline", "isolation", "hang"]
    parser.add_argument("--cases", nargs="+", choices=cases, default=cases)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--child", choices=cases)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(probe(args.child)), flush=True)
        return
    if os.name != "posix":
        parser.error("The bounded hang launcher requires POSIX process groups")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for case in args.cases:
        process = subprocess.Popen(
            [sys.executable, "-m", "validation.review_parallel", "--child", case],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            assert case == "hang", (case, "unexpected timeout", stderr)
            outcome = {
                "bounded_timeout": True,
                "limitation": "deadline cannot preempt a hung callback",
            }
        else:
            assert case != "hang" and process.returncode == 0, (
                case,
                process.returncode,
                stderr,
            )
            outcome = json.loads(stdout)
        print(
            json.dumps(
                dict(
                    outcome,
                    case=case,
                    passed=True,
                    source_commit=commit,
                    timeout_seconds=args.timeout,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
