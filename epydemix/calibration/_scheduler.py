"""Evaluate candidates sequentially until the acceptance target or a limit."""

from datetime import datetime

from .._execution import single_threaded


class SequentialScheduler:
    """Evaluate one candidate at a time until the acceptance target or a limit."""

    @single_threaded
    def run_until_n_accepted(
        self,
        evaluate,
        arguments,
        n_target,
        *,
        max_simulations=None,
        deadline=None,
        progress=None,
    ):
        """
        Collect accepted candidates sequentially in the calling process.

        Stop at n_target acceptances, max_simulations, deadline, or exhausted inputs.
        Return accepted_results in candidate order; n_simulations counts all evaluations.

        Args:
            evaluate (Callable): Particle evaluation function.
            arguments (Iterable): Candidate argument iterable.
            n_target (int): Target number of accepted candidates.
            max_simulations (int, optional): Simulation budget cutoff. Default is None.
            deadline (datetime, optional): Wall-clock cutoff datetime. Default is None.
            progress (Callable, optional): Progress callback (completed, accepted).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "accepted_results" (List[Dict[str, Any]]): Accepted candidates in candidate order.
                - "n_simulations" (int): Total simulations completed.

        Raises:
            ValueError: If n_target < 1.
        """
        if n_target < 1:
            raise ValueError("n_target must be positive")
        accepted = []
        completed = 0
        arguments = iter(arguments)
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
            result = evaluate(*args)
            completed += 1
            # Collect accepted candidates and report progress
            if result["accepted"]:
                accepted.append(result)
            if progress is not None:
                progress(completed, len(accepted))
        return {"accepted_results": accepted, "n_simulations": completed}
