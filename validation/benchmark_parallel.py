"""Benchmark current ABC execution in fresh processes; psutil is audit-only.

Normal timing and optional detailed profiling are separate runs. DYN dispatches a
worker loop per task: task count is never interpreted as a simulation count. Run
from the source checkout whose implementation is being measured, using either
python -m validation.benchmark_parallel or this file's absolute path.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from pathlib import Path
from threading import Thread
from time import perf_counter, sleep

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import psutil
import scipy
from scipy import stats

from epydemix.calibration import _evaluate
from epydemix.calibration.abc import ABCSampler
from epydemix.model import simulate
from epydemix.model.predefined_models import create_sir
from epydemix.population import Population


def statistical_simulation(parameters):
    """Normal observation; an optional fixed array isolates model-input transport."""
    return {"data": np.array([parameters["rng"].normal(parameters["mu"], 1)])}


def sir_simulation(parameters):
    """Retain only incidence for a small network-free epidemic workload."""
    return {"data": simulate(**parameters).transitions["Susceptible_to_Infected_total"]}


def make_sampler(model, payload_mib, seed):
    if model == "normal":
        fixed = {"payload": np.zeros(int(payload_mib * 1024**2 / 8))}
        return ABCSampler(
            statistical_simulation,
            {"mu": stats.norm()},
            fixed,
            np.array([1.0]),
            rng=seed,
        )
    epidemic = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    population = Population()
    population.add_population([10000])
    population.add_contact_matrix(np.array([[1.0]]))
    epidemic.set_population(population)
    fixed = dict(
        epimodel=epidemic,
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict={
            "Susceptible": np.array([9900]),
            "Infected": np.array([100]),
            "Recovered": np.array([0]),
        },
    )
    observed = sir_simulation(dict(fixed, rng=np.random.default_rng(0)))["data"]
    return ABCSampler(
        sir_simulation,
        {"transmission_rate": stats.uniform(0.1, 0.5)},
        fixed,
        observed,
        rng=seed,
    )


def _profile_task(function, arguments):
    """Time one process task, which may contain many DYN evaluations."""
    start = perf_counter()
    result = function(*arguments)
    return result, {"pid": os.getpid(), "started": start, "finished": perf_counter()}


def install_dispatch_profile():
    """Patch only this disposable audit process; preserve the library result API."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import _CallItem
    from multiprocessing.reduction import ForkingPickler

    report = {"tasks": [], "serialization": []}
    submit, dumps = ProcessPoolExecutor.submit, ForkingPickler.dumps

    def measured_submit(pool, function, *arguments, **kwargs):
        if kwargs:
            raise ValueError("Audit expects positional task arguments")
        submitted = perf_counter()
        future = submit(pool, _profile_task, function, arguments)
        original_result = future.result
        recorded = False

        def result(*args, **kwargs):
            nonlocal recorded
            output, timing = original_result(*args, **kwargs)
            if not recorded:
                report["tasks"].append(dict(timing, submitted=submitted))
                recorded = True
            return output

        future.result = result
        return future

    def measured_dumps(value, protocol=None):
        start = perf_counter()
        payload = dumps(value, protocol)
        # Other manager messages are deliberately excluded from dispatch totals.
        if isinstance(value, _CallItem) or (
            isinstance(value, tuple) and len(value) == 2 and callable(value[0])
        ):
            report["serialization"].append(
                {
                    "kind": "process_task"
                    if isinstance(value, _CallItem)
                    else "dynamic_job",
                    "seconds": perf_counter() - start,
                    "bytes": len(payload),
                }
            )
        return payload

    ProcessPoolExecutor.submit = measured_submit
    ForkingPickler.dumps = staticmethod(measured_dumps)
    return report


def fingerprint(result):
    """Hash retained mathematical output, excluding timings and surplus counts."""
    digest = hashlib.sha256()
    for generation in result.posterior_distributions:
        values = [
            result.get_posterior_distribution(generation).to_numpy(),
            np.asarray(result.get_weights(generation)),
            np.asarray(result.get_distances(generation)),
        ]
        values.extend(result.get_calibration_trajectories(generation).values())
        for value in values:
            digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def child(args):
    mp.set_start_method(args.start_method, force=True)
    profile = install_dispatch_profile() if args.profile else None
    phases = []
    original = _evaluate.run_particle_evaluations

    def measured(*positional, **keywords):
        start = perf_counter()
        output = original(*positional, **keywords)
        phases.append(
            {
                "seconds": perf_counter() - start,
                "simulations": output["n_simulations"],
                "retained": len(output["accepted_results"]),
            }
        )
        return output

    # Batch-level adapter for stages before native generation metrics exist.
    _evaluate.run_particle_evaluations = measured
    sampler = make_sampler(args.model, args.payload_mib, args.seed)
    started = perf_counter()
    if args.strategy == "top_fraction":
        options = {"Nsim": args.particles, "top_fraction": 0.5}
    elif args.strategy == "smc":
        options = {
            "num_generations": args.generations,
            "epsilon_schedule": [float("inf")] * args.generations,
            "num_particles": args.particles,
        }
    else:
        options = {"epsilon": float("inf"), "num_particles": args.particles}
    result = sampler.calibrate(
        strategy=args.strategy,
        n_workers=args.worker or None,
        verbose=False,
        **options,
    )
    elapsed = perf_counter() - started
    fingerprint_started = perf_counter()
    output_hash = fingerprint(result)
    print(
        json.dumps(
            {
                "model": args.model,
                "strategy": args.strategy,
                "workers": args.worker,
                "particles": args.particles,
                "generations": args.generations,
                "seed": args.seed,
                "payload_mib": args.payload_mib,
                "start_method": args.start_method,
                "profile": args.profile,
                "elapsed_seconds": elapsed,
                "calibration_started": started,
                "calibration_finished": started + elapsed,
                "epsilon": None if args.strategy == "top_fraction" else "Infinity",
                "top_fraction": 0.5 if args.strategy == "top_fraction" else None,
                "candidate_phases": phases,
                "weight_and_other_seconds": elapsed - sum(p["seconds"] for p in phases),
                "fingerprint": output_hash,
                "fingerprint_seconds": perf_counter() - fingerprint_started,
                "dispatch": profile,
                "benchmark_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "dirty_diff": subprocess.check_output(
                    ["git", "diff", "HEAD"], text=True
                ),
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                },
            }
        ),
        flush=True,
    )


def main(args):
    for worker in args.workers:
        for repeat in range(args.repeat):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--worker",
                str(worker),
                "--model",
                args.model,
                "--strategy",
                args.strategy,
                "--particles",
                str(args.particles),
                "--generations",
                str(args.generations),
                "--seed",
                str(args.seed),
                "--payload-mib",
                str(args.payload_mib),
                "--start-method",
                args.start_method,
            ]
            if args.profile:
                command.append("--profile")
            samples, cpu_totals = [], {}
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, text=True
            ) as process:
                root = psutil.Process(process.pid)

                def sample():
                    while process.poll() is None:
                        memory = 0
                        try:
                            descendants = [root, *root.children(recursive=True)]
                        except psutil.NoSuchProcess:
                            descendants = []
                        for child_process in descendants:
                            try:
                                memory += child_process.memory_info().rss
                                times = child_process.cpu_times()
                                identity = (
                                    child_process.pid,
                                    child_process.create_time(),
                                )
                                cpu_totals[identity] = max(
                                    cpu_totals.get(identity, 0),
                                    times.user + times.system,
                                )
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        samples.append((perf_counter(), memory))
                        sleep(0.02)

                monitor = Thread(target=sample)
                monitor.start()
                # Drain stdout while monitoring: a large dirty diff can fill the
                # pipe and otherwise prevent the child from exiting.
                output, _ = process.communicate()
                monitor.join()
                if process.returncode:
                    raise RuntimeError(f"Benchmark child failed: {process.returncode}")
            report = json.loads(output)
            calibration_memory = [
                memory
                for timestamp, memory in samples
                if report["calibration_started"]
                <= timestamp
                <= report["calibration_finished"]
            ]
            report.update(
                repeat=repeat,
                peak_process_tree_rss_bytes=max(calibration_memory, default=None),
                sampled_process_tree_lifetime_cpu_seconds=sum(cpu_totals.values()),
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["normal", "sir"], default="normal")
    parser.add_argument(
        "--strategy", choices=["smc", "rejection", "top_fraction"], default="smc"
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--payload-mib", type=float, default=0)
    parser.add_argument(
        "--start-method", choices=mp.get_all_start_methods(), default="spawn"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=int, default=0, help=argparse.SUPPRESS)
    options = parser.parse_args()
    child(options) if options.child else main(options)
