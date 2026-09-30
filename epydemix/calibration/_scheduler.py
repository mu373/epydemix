"""Distribute candidate evaluations and collect accepted results.

DynamicScheduler assigns candidate IDs, stops new evaluations at the acceptance
or budget limit, drains running evaluations, and returns results in candidate
order. Proposal generation lives in _proposals; simulation and acceptance checks
live in _evaluate. Sequential calibration uses the same stopping conditions.
Process creation and ownership live in epydemix._execution.
"""

import pickle
from concurrent.futures import FIRST_COMPLETED, wait
from datetime import datetime
from multiprocessing.managers import BaseManager
from multiprocessing.reduction import ForkingPickler
from threading import Lock

from ._proposals import ProposalDeadline


def validate_parallel_strategy(strategy):
    """Check an acceptance-scheduler name without constructing a scheduler."""
    # Currently only supports dynamic scheduling, but may be extended
    if strategy != "dynamic":
        raise ValueError(f"Unknown parallel strategy: {strategy}. Must be 'dynamic'")


def create_particle_scheduler(strategy="dynamic"):
    """Return the acceptance scheduler selected for SMC or rejection sampling."""
    validate_parallel_strategy(strategy)
    return DynamicScheduler()


class DynamicScheduler:
    """Keep one evaluation loop per worker until the shared acceptance target.

    Drain running evaluations and retain the earliest accepted candidate IDs,
    preserving seeded populations without binding time or physical-budget cutoffs.
    Surplus evaluations count toward the physical simulation budget.
    See the DYN strategy: https://doi.org/10.1371/journal.pone.0294015
    """

    def run_until_n_accepted(
        self,
        executor,
        evaluate,
        arguments,
        n_target,
        *,
        max_simulations=None,
        deadline=None,
        progress=None,
    ):
        """Return accepted_results in candidate order and physical n_simulations.

        Requires an executor and picklable, indexable arguments. Each worker
        generates its own candidates using shared IDs. A deadline stops new work and drains running
        evaluations. progress(completed, accepted) reports physical completions
        and the retained acceptance count. Binding cumulative budgets or wall-clock
        cutoffs can change later SMC populations across worker counts.
        """
        # Validations
        if n_target < 1:
            raise ValueError("n_target must be positive")
        if executor is None:
            raise ValueError("DynamicScheduler requires an executor")
        if not hasattr(arguments, "__getitem__"):
            raise TypeError("DYN requires indexable candidate arguments")
        if (max_simulations is not None and max_simulations <= 0) or (
            deadline is not None and datetime.now() >= deadline
        ):
            return {"accepted_results": [], "n_simulations": 0}

        # Serialize once; passing bytes avoids repeatedly pickling the full context.
        serialized_job = bytes(ForkingPickler.dumps((evaluate, arguments)))

        accepted = []
        with _SchedulerManager(ctx=executor._mp_context) as manager:
            # Share candidate IDs, counts, and stopping conditions across workers
            state = manager.SchedulerState(n_target, max_simulations, deadline)
            pending = set()
            try:
                # Start one evaluation loop per worker
                for _ in range(executor._max_workers):
                    pending.add(
                        executor.submit(_evaluate_in_worker, serialized_job, state)
                    )
                # Collect results as worker loops finish
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=0.1 if progress else None,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        accepted.extend(future.result())
                    if progress is not None:
                        _, completed, n_accepted, _ = state.snapshot()
                        progress(completed, min(n_accepted, n_target))
            finally:
                # Keep the manager alive while any running job drains after a failure.
                state.abort()
                for future in pending:
                    future.cancel()
                wait(pending)
            _, completed, _, _ = state.snapshot()
        # Keep the earliest accepted candidates, regardless of completion order
        accepted.sort(key=lambda item: item[0])
        return {
            "accepted_results": [result for _, result in accepted[:n_target]],
            "n_simulations": completed,
        }


def run_sequential_until_n_accepted(
    fn,
    args_list,
    n_target,
    *,
    max_simulations=None,
    deadline=None,
    progress=None,
):
    """Collect accepted candidates sequentially in the calling process.

    Stop at n_target acceptances, max_simulations, deadline, or exhausted inputs.
    Return accepted_results in candidate order; n_simulations counts all evaluations.
    Call progress(completed, accepted) after each evaluation, when provided.
    """
    if n_target < 1:
        raise ValueError("n_target must be positive")
    accepted = []
    completed = 0
    arguments = iter(args_list)
    while len(accepted) < n_target:
        # Check limits before requesting another candidate
        if max_simulations is not None and completed >= max_simulations:
            break
        if deadline is not None and datetime.now() >= deadline:
            break
        # Get the next candidate
        try:
            args = next(arguments)
        except StopIteration:
            break
        # Evaluate the candidate
        result = fn(*args)
        completed += 1
        # Collect accepted candidates and report progress
        if result["accepted"]:
            accepted.append(result)
        if progress is not None:
            progress(completed, len(accepted))
    return {"accepted_results": accepted, "n_simulations": completed}


class _SchedulerState:
    """Track candidate IDs, evaluation counts, acceptances, and stopping conditions."""

    def __init__(self, target, budget, deadline):
        self.target = target
        self.budget = budget
        self.deadline = deadline
        self.submitted = self.completed = self.accepted = 0
        self.stopped = False
        self.lock = Lock()

    def report_and_claim(self, accepted=None):
        """Count the previous evaluation and claim another ID in one locked call.

        The initial call has no completed evaluation. Each worker continues
        until the shared target is reached.
        Even after stopping, running evaluations must report their physical work.
        """
        with self.lock:
            # Count the previous evaluation, even after stopping
            if accepted is not None:
                self.completed += 1
                self.accepted += int(accepted)

            # Stop issuing candidates when a limit is reached
            if (
                self.stopped
                or self.accepted >= self.target
                or (self.budget is not None and self.submitted >= self.budget)
                or (self.deadline is not None and datetime.now() >= self.deadline)
            ):
                return None
            # Assign the next candidate ID
            index = self.submitted
            self.submitted += 1
            return index

    def abort(self):
        """Stop further claims; already claimed evaluations may finish."""
        with self.lock:
            self.stopped = True

    def snapshot(self):
        """Read physical counters and the stopping flag atomically."""
        with self.lock:
            return self.submitted, self.completed, self.accepted, self.stopped


class _SchedulerManager(BaseManager):
    """Share scheduler counters and stopping state between process workers."""


_SchedulerManager.register("SchedulerState", _SchedulerState)


def _evaluate_in_worker(serialized_job, state):
    """Evaluate candidates in one worker until the shared stopping condition.

    Restore proposal inputs for every candidate in both owned and external pools.
    Owned pools restore their cached model snapshot in the evaluator; external
    pools include the model in serialized_job. No mutable task state is reused.
    """
    accepted = []
    try:
        # Request the first candidate ID
        index = state.report_and_claim()
        while index is not None:
            evaluate, arguments = pickle.loads(serialized_job)
            if hasattr(arguments, "__len__") and index >= len(arguments):
                state.abort()
                return accepted
            try:
                # Generate the parameter proposal and prepare evaluation inputs
                args = arguments[index]
            except ProposalDeadline:
                state.abort()
                return accepted
            # Run simulation, compute distance, and check acceptance
            result = evaluate(*args)
            del args, evaluate, arguments
            if result["accepted"]:
                accepted.append((index, result))
            # Report the result and request the next candidate ID
            index = state.report_and_claim(bool(result["accepted"]))
        return accepted
    except BaseException:
        state.abort()
        raise
