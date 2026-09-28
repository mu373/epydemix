"""Process-pool scheduling for ABC calibration.

Provides:
- Worker-count limits based on detected logical CPU capacity, Linux affinity
  and visible cgroup quotas. These limits do not reserve CPUs across concurrent runs.
- ``DynamicParticleScheduler``, which keeps a bounded number of tasks in flight and returns
  results in submission order.
- ``executor_context``, which validates, creates and tears down worker pools.
"""

import operator
import os
import re
import sys
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path

from . import _worker


def _cgroup_cpu_limits():
    """
    Yield the CPU quota of every visible cgroup ancestor of this process.

    Supports both cgroup v1 (`cpu.cfs_quota_us` / `cpu.cfs_period_us`) and v2 (`cpu.max`).
    A quota can be set at any level of the hierarchy, so the walk goes from this process's
    own cgroup up to the mount point. Levels without a quota ("max" or -1) are skipped.

    Yields:
        int: CPU quota in whole CPUs, rounded down with a minimum of one.

    Raises:
        OSError: If `/proc/self/cgroup` or `/proc/self/mountinfo` cannot be read.
        ValueError: If a cgroup or quota file cannot be parsed.
    """
    # /proc/self/cgroup lines look like "ID:controllers:path". v2 has an empty
    # controller list; v1 lists the controllers bound to that hierarchy.
    groups = {}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers or "cpu" in controllers.split(","):
            groups["cgroup2" if not controllers else "cgroup"] = Path(path)
    # Find where each hierarchy is mounted. In mountinfo, fields 4 and 5 are
    # the mounted root inside the hierarchy and the mount point on this host;
    # the filesystem type and super options follow the " - " separator.
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, filesystem = line.split(" - ")
        kind, _, options = filesystem.split()
        if kind not in groups or (kind == "cgroup" and "cpu" not in options.split(",")):
            continue
        # mountinfo escapes spaces etc. as octal ("\040"); decode them.
        root, mount = (
            Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value))
            for value in fields.split()[3:5]
        )
        try:
            relative = groups[kind].relative_to(root)
        except ValueError:
            # A container may expose only its namespace root at this mount.
            relative = Path(".")
        leaf = mount / relative if ".." not in relative.parts else mount
        # Walk upward from our own cgroup, stopping at the mount point.
        for directory in (leaf, *leaf.parents):
            try:
                if kind == "cgroup2":
                    quota, period = (directory / "cpu.max").read_text().split()
                else:
                    quota = (directory / "cpu.cfs_quota_us").read_text().strip()
                    period = (directory / "cpu.cfs_period_us").read_text().strip()
            except FileNotFoundError:
                pass  # CPU controller may be disabled at this level.
            else:
                if quota != "max" and int(quota) > 0:
                    yield max(1, int(quota) // int(period))
            if directory == mount:
                break


def _available_cpu_count():
    """
    Return the number of logical CPUs this process may use.

    Takes the minimum of `os.cpu_count`, `os.process_cpu_count` (Python 3.13+), the scheduler
    affinity mask and any cgroup quota. This is a capacity limit, not a measure of currently
    idle CPUs. If the limits cannot be read, a RuntimeWarning is issued and the
    worker-count limit falls back to one.

    Returns:
        int: Number of usable logical CPUs (at least one).
    """
    count = min(
        os.cpu_count() or 1, getattr(os, "process_cpu_count", os.cpu_count)() or 1
    )
    try:
        if hasattr(os, "sched_getaffinity"):
            count = min(count, len(os.sched_getaffinity(0)))
        if sys.platform == "linux":
            count = min([count, *_cgroup_cpu_limits()])
    except (OSError, ValueError, ZeroDivisionError) as error:
        warnings.warn(
            f"Cannot determine CPU limits ({error}); allowing one worker only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1
    return max(1, count)


class DynamicParticleScheduler:
    """
    Parallel scheduler that refills workers dynamically and preserves submission order.

    Pending tasks count as potential acceptances. This avoids speculative work
    beyond the target. With repeatable candidate results and no wall-clock cutoff,
    the evaluated prefix and simulation count match sequential execution regardless
    of completion order.

    Attributes:
        queue_factor (int): Number of tasks kept in flight per worker.
    """

    def __init__(self, queue_factor=2):
        """
        Initialize the scheduler.

        Args:
            queue_factor (int, optional): Number of tasks kept in flight per worker. Values above one
                keep workers busy while the main process handles finished results. Default is 2.

        Raises:
            ValueError: If queue_factor is less than 1.
        """
        if queue_factor < 1:
            raise ValueError("queue_factor must be positive")
        self.queue_factor = queue_factor

    def collect_particles(
        self,
        executor,
        fn,
        args_list,
        n_target,
        *,
        max_simulations=None,
        deadline=None,
        progress=None,
    ):
        """
        Submit candidates until `n_target` are accepted or a cutoff is reached.

        Candidates are numbered in submission order and accepted results are returned in
        that order. A budget or deadline cutoff stops new submissions, but tasks already
        submitted still run to completion and are counted. A deadline can change which
        candidates are evaluated across worker counts or completion orders.

        Args:
            executor (ProcessPoolExecutor or None): Worker pool, or None when tasks run in the calling process.
            fn (Callable): Evaluate one candidate from its positional arguments.
            args_list (Iterable[tuple]): Candidate arguments, consumed lazily in submission order.
            n_target (int): Number of accepted results to collect.
            max_simulations (int, optional): Maximum number of tasks to submit. Default is None (no limit).
            deadline (datetime, optional): Time after which no new tasks are submitted. Default is None.
            progress (Callable[[int, int], None], optional): Called as `progress(n_completed, n_accepted)`
                after each batch of finished tasks. Default is None.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - "accepted_results": accepted task results, in submission order.
                - "n_simulations": number of completed simulations, including rejected ones.

        Raises:
            ValueError: If n_target is less than 1.
        """
        if n_target < 1:
            raise ValueError("n_target must be positive")
        arguments = iter(args_list)
        queue_size = (
            executor._max_workers * self.queue_factor if executor is not None else 1
        )
        futures, accepted = {}, {}
        submitted = completed = 0
        exhausted = False
        try:
            while True:
                # Refill the queue. Only as many tasks as could still be needed
                # are in flight, assuming every pending task will be accepted.
                # ponytail: taper concurrency near the target to avoid discarded
                # simulations; speculative work would need separate budget semantics.
                capacity = min(queue_size, n_target - len(accepted))
                while not exhausted and len(futures) < capacity:
                    if max_simulations is not None and submitted >= max_simulations:
                        break
                    if deadline is not None and datetime.now() >= deadline:
                        break
                    try:
                        args = next(arguments)
                    except StopIteration:
                        exhausted = True
                        break
                    if executor is None:
                        future = Future()
                        future.set_result(fn(*args))
                    else:
                        future = executor.submit(fn, *args)
                    futures[future] = submitted
                    submitted += 1
                if not futures:
                    break  # target reached, or cutoff hit and all work drained
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    result = future.result()
                    completed += 1
                    if result["accepted"]:
                        accepted[index] = result
                if progress is not None:
                    progress(completed, len(accepted))
        finally:
            # Only reached with pending futures on error (e.g. a task raised).
            for future in futures:
                future.cancel()

        return {
            "accepted_results": [accepted[i] for i in sorted(accepted)],
            "n_simulations": completed,
        }

    def run_batch(self, executor, fn, args_list):
        """
        Run `fn(*args)` for every item of `args_list` and return the results in input order.

        Unlike `executor.map`, only a bounded number of tasks is submitted at a time,
        so a large or lazy `args_list` is not materialised up front.

        Args:
            executor (ProcessPoolExecutor or None): Worker pool, or None to run in the calling process.
            fn (Callable): Picklable module-level function to run.
            args_list (Iterable[tuple]): Argument tuples for `fn`, consumed lazily.

        Returns:
            List[Any]: One result per argument tuple, in input order.
        """
        if executor is None:
            return [fn(*args) for args in args_list]
        arguments = iter(args_list)
        capacity = executor._max_workers * self.queue_factor
        futures, results = {}, []
        try:
            while True:
                while len(futures) < capacity:
                    try:
                        args = next(arguments)
                    except StopIteration:
                        break
                    futures[executor.submit(fn, *args)] = len(results)
                    results.append(None)
                if not futures:
                    return results
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    results[futures.pop(future)] = future.result()
        finally:
            for future in futures:
                future.cancel()


def build_particle_tasks(executor, inputs, candidates, *, inclusive=False):
    """Lazily encode candidate evaluations for sequential or process execution."""
    function, parameters, names, observed, distance = inputs
    arguments = (
        (function, parameters, names, params, observed, distance, epsilon, rng)
        for params, epsilon, rng in candidates
    )
    return partial(_worker.evaluate_particle, inclusive=inclusive), arguments


def executor_context(n_workers=None, executor=None):
    """
    Validate the requested parallelism and return a context manager for the worker pool.

    Pools created here are shut down (and their workers joined) on exit; a
    caller-provided executor is validated and left open.

    Args:
        n_workers (int, optional): Number of worker processes to start, or None to run
            sequentially. Ignored when `executor` is given. Default is None.
        executor (ProcessPoolExecutor, optional): Caller-owned worker pool. Default is None.

    Returns:
        ContextManager[Optional[ProcessPoolExecutor]]: Yields the pool, or None for sequential execution.

    Raises:
        TypeError: If executor is not a ProcessPoolExecutor.
        ValueError: If the worker count is not a positive integer or exceeds the available CPU capacity.
    """
    if executor is not None and not isinstance(executor, ProcessPoolExecutor):
        raise TypeError(
            "executor must be a ProcessPoolExecutor; thread executors are not supported "
            "because simulation models may mutate shared state"
        )
    requested = executor._max_workers if executor is not None else n_workers
    if requested is None:
        return nullcontext(None)
    if isinstance(requested, bool):
        raise ValueError("n_workers must be a positive integer")
    requested = operator.index(requested)
    if requested < 1:
        raise ValueError("n_workers must be a positive integer")
    available = _available_cpu_count()
    if requested > available:
        raise ValueError(
            f"Requested {requested} workers, but only {available} logical CPUs are "
            "available under the current CPU affinity and quota. "
            "Reduce n_workers or the executor's max_workers."
        )
    if executor is not None:
        return nullcontext(executor)
    return ProcessPoolExecutor(max_workers=requested)


def create_particle_scheduler(strategy="dyn", **kwargs):
    """
    Return the parallel scheduler for the given strategy.

    Args:
        strategy (str, optional): Scheduling strategy. Only "dyn" is supported. Default is "dyn".
        **kwargs: Passed to the scheduler constructor.

    Returns:
        DynamicParticleScheduler: The scheduler instance.

    Raises:
        ValueError: If the strategy is unknown.
    """
    if strategy != "dyn":
        raise ValueError(f"Unknown parallel strategy: {strategy}. Must be 'dyn'")
    return DynamicParticleScheduler(**kwargs)
