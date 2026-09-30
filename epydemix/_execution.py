"""Manage process pools and run a finite collection of independent tasks.

executor_context validates CPU capacity and manages pool ownership. map_tasks
submits a bounded number of tasks and returns results in input order. Neither
function knows about calibration candidates or simulation stopping conditions.
"""

import operator
import os
import re
import sys
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from threading import local

from threadpoolctl import threadpool_limits

_thread_limit_scope = local()


def _iter_cgroup_cpu_quotas():
    """
    Yield visible cgroup CPU quotas as whole-CPU limits, from this process upward.

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


def _get_available_cpu_count():
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
            count = min([count, *_iter_cgroup_cpu_quotas()])
    except (OSError, ValueError, ZeroDivisionError) as error:
        warnings.warn(
            f"Cannot determine CPU limits ({error}); allowing one worker only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1
    return max(1, count)


def executor_context(n_workers=None, executor=None, *, initializer=None, initargs=()):
    """
    Validate the requested parallelism and return a context manager for the worker pool.

    Pools created here are shut down (and their workers joined) on exit; a
    caller-provided executor is validated and left open.

    Caller-owned pools keep their initializer.

    Args:
        n_workers (int, optional): Number of worker processes, or None to run sequentially.
            Negative values use max(1, available_cpus + 1 + n_workers): -1 uses all
            available CPUs, -2 uses all but one, etc. Zero is invalid. Ignored when
            `executor` is given. Default is None.
        executor (ProcessPoolExecutor, optional): Caller-owned worker pool. Default is None.
        initializer (Callable, optional): Function called when each owned worker starts.
        initargs (tuple): Arguments for the initializer. Ignored for caller-owned pools.

    Returns:
        ContextManager[Optional[ProcessPoolExecutor]]: Yields the pool, or None for sequential execution.

    Raises:
        TypeError: If n_workers is not an integer or executor is not a ProcessPoolExecutor.
        ValueError: If n_workers is zero or Boolean, or exceeds the available CPU capacity.
    """
    if executor is not None and not isinstance(executor, ProcessPoolExecutor):
        raise TypeError(
            "executor must be a ProcessPoolExecutor; thread executors are not supported "
            "because simulation models may mutate shared state"
        )
    # Determine and validate the requested worker count
    requested = executor._max_workers if executor is not None else n_workers
    if requested is None:
        return nullcontext(None)
    if isinstance(requested, bool):
        raise ValueError("n_workers must be a non-zero integer")
    requested = operator.index(requested)
    if requested == 0:
        raise ValueError("n_workers must be a non-zero integer")
    # Resolve negative counts relative to CPU capacity, keeping at least one worker
    available = _get_available_cpu_count()
    if requested < 0:
        requested = max(1, available + 1 + requested)
    # Check explicit counts against the available CPU capacity
    if requested > available:
        raise ValueError(
            f"Requested {requested} workers, but only {available} logical CPUs are "
            "available under the current CPU affinity and quota. "
            "Reduce n_workers or the executor's max_workers."
        )
    # Reuse a caller-owned pool without closing it
    if executor is not None:
        return nullcontext(executor)
    return ProcessPoolExecutor(
        max_workers=requested, initializer=initializer, initargs=initargs
    )


def single_threaded(function):
    """
    Limit supported, already-loaded BLAS/OpenMP thread pools to one thread during a call.

    Previous limits are restored afterward. Nested calls reuse the outer limit,
    avoiding native-library discovery for every candidate in a serial batch or
    worker loop. A new outer call discovers libraries again, including libraries
    imported between batches. Libraries must be loaded before entering the scope.

    Args:
        function (Callable): Function to wrap.

    Returns:
        Callable: Wrapped function with the same signature.
    """

    @wraps(function)
    def run(*args, **kwargs):
        pid = os.getpid()
        # A fork inherits thread-local state, but its worker initializer may set
        # different native limits. Never inherit the parent's active-scope flag.
        if getattr(_thread_limit_scope, "pid", None) == pid and getattr(
            _thread_limit_scope, "active", False
        ):
            return function(*args, **kwargs)
        with threadpool_limits(limits=1):
            _thread_limit_scope.pid = pid
            _thread_limit_scope.active = True
            try:
                return function(*args, **kwargs)
            finally:
                _thread_limit_scope.active = False

    return run


def map_tasks(executor, fn, args_list, *, on_completed=None):
    """
    Run `fn(*args)` for every item of `args_list` and return the results in input order.

    Unlike `executor.map`, only a bounded number of tasks is submitted at a time,
    so a large or lazy `args_list` is not materialised up front.

    Args:
        executor (ProcessPoolExecutor or None): Worker pool, or None to run in the calling process.
        fn (Callable): Picklable module-level function to run.
        args_list (Iterable[tuple]): Argument tuples for `fn`, consumed lazily.
        on_completed (Callable, optional): Called in the parent with the completed
            count and result, in completion order. Returned results retain input order.

    Returns:
        List[Any]: One result per argument tuple, in input order.
    """
    if executor is None:
        return _map_sequential(fn, args_list, on_completed)
    arguments = iter(args_list)
    capacity = executor._max_workers * 2
    futures, results = {}, []
    completed = 0
    try:
        while True:
            # Submit tasks up to the queue limit
            while len(futures) < capacity:
                try:
                    args = next(arguments)
                except StopIteration:
                    break
                futures[executor.submit(fn, *args)] = len(results)
                results.append(None)
            if not futures:
                return results
            # Store completed results in their original input order
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results[futures.pop(future)] = result
                completed += 1
                if on_completed is not None:
                    on_completed(completed, result)
    finally:
        # Cancel any remaining futures on error or exit
        for future in futures:
            future.cancel()


@single_threaded
def _map_sequential(fn, args_list, on_completed):
    """Apply native limits once to a batch executed in the calling process."""
    results = []
    for args in args_list:
        result = fn(*args)
        results.append(result)
        if on_completed is not None:
            on_completed(len(results), result)
    return results
