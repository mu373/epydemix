"""Benchmark current ABC execution in fresh processes; psutil is audit-only.

Normal timing and optional detailed profiling are separate runs. DYN dispatches a
worker loop per task: task count is never interpreted as a simulation count. Run
from the source checkout whose implementation is being measured, using either
python -m validation.benchmark_parallel or this file's absolute path.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter, sleep
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import psutil
import scipy
from scipy import stats

from epydemix._logging import JSONFormatter, _json_values
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


def _trace(directory, kind, started, **values):
    """One file per PID avoids profiler queues/locks changing DYN scheduling."""
    with (Path(directory) / f"{os.getpid()}.jsonl").open("a") as file:
        file.write(
            json.dumps(
                dict(
                    kind=kind,
                    pid=os.getpid(),
                    started=started,
                    finished=perf_counter(),
                    **values,
                )
            )
            + "\n"
        )


class ProfileSimulation:
    """Measure model callback time only in --profile; forward RNG unchanged."""

    def __init__(self, function, directory):
        self.function, self.directory = function, directory

    def __call__(self, parameters):
        started = perf_counter()
        try:
            return self.function(parameters)
        finally:
            _trace(self.directory, "simulation", started)


def _profile_initializer(directory, initializer, initargs):
    """Time real initialization and fresh-input restores in a disposable worker."""
    from epydemix.calibration import _scheduler, _worker_inputs

    started = perf_counter()
    restore, loads = _worker_inputs.restore_worker_inputs, _scheduler.pickle.loads

    def measured_restore():
        start = perf_counter()
        result = restore()
        _trace(directory, "input_restore", start)
        return result

    def measured_loads(value):
        start = perf_counter()
        result = loads(value)
        _trace(directory, "job_restore", start, bytes=len(value))
        return result

    _worker_inputs.restore_worker_inputs = measured_restore
    # Limit the hook to this module's job restores, excluding unrelated pickle IPC.
    _scheduler.pickle = SimpleNamespace(loads=measured_loads)
    if initializer is not None:
        initializer(*initargs)
    _trace(directory, "initializer", started)


def _profile_task(directory, function, arguments):
    """Time one process task, which may contain many DYN evaluations."""
    start = perf_counter()
    if getattr(function, "__name__", None) == "_evaluate_in_worker":
        state = arguments[1]
        claim = state.report_and_claim

        def measured_claim(*args):
            started = perf_counter()
            result = claim(*args)
            _trace(directory, "scheduler_rpc", started)
            return result

        state.report_and_claim = measured_claim
    result = function(*arguments)
    return result, {"pid": os.getpid(), "started": start, "finished": perf_counter()}


def install_dispatch_profile(directory):
    """Patch only this disposable audit process; preserve the library result API."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import _CallItem
    from multiprocessing.reduction import ForkingPickler

    from epydemix.calibration import _worker_inputs

    report = {"tasks": [], "serialization": []}
    submit, dumps = ProcessPoolExecutor.submit, ForkingPickler.dumps
    initialize = ProcessPoolExecutor.__init__
    save_inputs = _worker_inputs.save_worker_inputs

    def measured_save_inputs(path, inputs):
        started = perf_counter()
        save_inputs(path, inputs)
        report["serialization"].append(
            {
                "kind": "input_snapshot",
                "seconds": perf_counter() - started,
                "bytes": path.stat().st_size,
            }
        )

    def measured_initialize(pool, *args, initializer=None, initargs=(), **kwargs):
        initialize(
            pool,
            *args,
            initializer=partial(_profile_initializer, directory, initializer, initargs),
            **kwargs,
        )

    def measured_submit(pool, function, *arguments, **kwargs):
        if kwargs:
            raise ValueError("Audit expects positional task arguments")
        submitted = perf_counter()
        future = submit(pool, _profile_task, directory, function, arguments)
        original_result = future.result
        recorded = False

        def result(*args, **kwargs):
            nonlocal recorded
            output, timing = original_result(*args, **kwargs)
            if not recorded:
                report["tasks"].append(
                    dict(
                        timing,
                        submitted=submitted,
                        retrieved=perf_counter(),
                        function=getattr(function, "__name__", type(function).__name__),
                    )
                )
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
    ProcessPoolExecutor.__init__ = measured_initialize
    _worker_inputs.save_worker_inputs = measured_save_inputs
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
    if args.log_json:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger = logging.getLogger("epydemix.calibration")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    trace_directory = TemporaryDirectory(prefix="epydemix-profile-")
    profile = install_dispatch_profile(trace_directory.name) if args.profile else None
    sampler = make_sampler(args.model, args.payload_mib, args.seed)
    if args.profile:
        sampler.simulation_function = ProfileSimulation(
            sampler.simulation_function, trace_directory.name
        )
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
    pool = ProcessPoolExecutor(max_workers=args.worker) if args.reuse_pool else None
    warmup_seconds = None
    if pool is not None:
        warmup_started = perf_counter()
        warm = make_sampler(args.model, args.payload_mib, args.seed).calibrate(
            strategy=args.strategy, executor=pool, verbose=False, **options
        )
        warm_fingerprint = fingerprint(warm)
        del warm
        warmup_seconds = perf_counter() - warmup_started
    started = perf_counter()
    try:
        result = sampler.calibrate(
            strategy=args.strategy,
            n_workers=args.worker or None,
            executor=pool,
            verbose=False,
            **options,
        )
    finally:
        # Caller-owned shutdown is outside the measured calibration interval.
        finished = perf_counter()
        if pool is not None:
            pool.shutdown()
    elapsed = finished - started
    metrics = result.calibration_params["execution_metrics"]
    phases = [
        {
            "seconds": generation["collection_seconds"],
            "simulations": generation["simulations"],
            "retained": generation["retained"],
        }
        for generation in metrics["generations"]
    ]
    fingerprint_started = perf_counter()
    output_hash = fingerprint(result)
    fingerprint_seconds = perf_counter() - fingerprint_started
    if pool is not None:
        assert warm_fingerprint == output_hash
    if profile is not None:
        events = [
            json.loads(line)
            for path in Path(trace_directory.name).glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        profile["worker_events"] = events
        measured_simulations = [e for e in events if e["kind"] == "simulation"]
        assert len(measured_simulations) == metrics["totals"]["simulations"]
        profile["simulation_seconds"] = sum(
            e["finished"] - e["started"] for e in measured_simulations
        )
        # Task wall time includes proposals, distance, restores and RPC, not pure wait.
        profile["worker_task_non_simulation_seconds"] = (
            (
                sum(
                    task["finished"] - task["started"]
                    for task in profile["tasks"]
                    if task["started"] >= started
                )
                - profile["simulation_seconds"]
            )
            if args.worker
            else None
        )
        submissions = [task["submitted"] for task in profile["tasks"]]
        profile["worker_ready_after_first_submit_seconds"] = (
            [
                event["finished"] - min(submissions)
                for event in events
                if event["kind"] == "initializer"
            ]
            if submissions
            else []
        )
        loops = sorted(
            (
                task
                for task in profile["tasks"]
                if task["function"] == "_evaluate_in_worker"
                and task["started"] >= started
            ),
            key=lambda task: task["submitted"],
        )
        profile["dynamic_batches_finish_spread_seconds"] = [
            max(task["finished"] for task in batch)
            - min(task["finished"] for task in batch)
            for index in range(0, len(loops), args.worker or 1)
            for batch in [loops[index : index + (args.worker or 1)]]
        ]
    trace_directory.cleanup()
    print(
        json.dumps(
            _json_values(
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
                    "log_json": args.log_json,
                    "pool_mode": "reused_external" if args.reuse_pool else "new_owned",
                    "warmup_seconds": warmup_seconds,
                    "elapsed_seconds": elapsed,
                    "calibration_started": started,
                    "calibration_finished": started + elapsed,
                    "epsilon": None if args.strategy == "top_fraction" else "Infinity",
                    "top_fraction": 0.5 if args.strategy == "top_fraction" else None,
                    "execution_metrics": metrics,
                    "candidate_phases": phases,
                    "weight_and_other_seconds": elapsed
                    - sum(p["seconds"] for p in phases),
                    "fingerprint": output_hash,
                    "fingerprint_seconds": fingerprint_seconds,
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
            allow_nan=False,
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
            if args.log_json:
                command.append("--log-json")
            if args.reuse_pool:
                command.append("--reuse-pool")
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
    parser.add_argument(
        "--reuse-pool",
        action="store_true",
        help="Warm an external pool with an identical untimed calibration",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Send application-configured JSON events to stderr",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=int, default=0, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.reuse_pool and (
        options.worker == 0 if options.child else 0 in options.workers
    ):
        parser.error("--reuse-pool requires positive worker counts")
    child(options) if options.child else main(options)
