"""Scheduler queue bounds, ordering, partial results, and failure cleanup."""

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from itertools import count
from unittest.mock import patch

import numpy as np
import pytest

from epydemix.calibration.parallel import (
    DynamicParticleScheduler,
)


class TestDynamicOrdering:
    """Check accepted-result ordering and the requested acceptance count."""

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dynamic_ordering_respects_submission_order(self, parallel):
        """Return accepted results in submission order for either execution mode."""
        sampler = DynamicParticleScheduler(queue_factor=2)

        def evaluate(idx):
            # All tasks accepted
            return {
                "params": [idx],
                "distance": 0.5,
                "simulation": {"data": np.array([idx])},
                "accepted": True,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.run_until_n_accepted(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=5,
            )

        accepted = result["accepted_results"]
        assert len(accepted) == 5
        # Verify results are in submission order
        indices = [r["params"][0] for r in accepted]
        assert indices == sorted(indices)

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dynamic_stops_at_n_target(self, parallel):
        """Stop at the requested count while skipping rejected candidates."""
        sampler = DynamicParticleScheduler(queue_factor=1)

        def evaluate(idx):
            # Alternate accepted/rejected
            accepted = idx % 2 == 0
            return {
                "params": [idx],
                "distance": 0.5 if accepted else 999.0,
                "simulation": {"data": np.array([idx])} if accepted else None,
                "accepted": accepted,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.run_until_n_accepted(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=3,
            )

        assert len(result["accepted_results"]) == 3


class TestDynamicPartialResults:
    """Retain accepted results when the simulation budget cuts off sampling."""

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dynamic_returns_partial_on_budget(self, parallel):
        """When max_simulations < n_target, accepted particles should not be lost."""
        sampler = DynamicParticleScheduler(queue_factor=1)

        def evaluate(idx):
            return {
                "params": [idx],
                "distance": 0.5,
                "simulation": {"data": np.array([idx])},
                "accepted": True,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.run_until_n_accepted(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=100,
                max_simulations=5,
            )

        accepted = result["accepted_results"]
        # Budget only allows 5 sims, but all are accepted; should get them back
        assert len(accepted) > 0
        assert len(accepted) <= 5

    @pytest.mark.parametrize("parallel", [False, True])
    def test_dynamic_returns_partial_on_mixed_results(self, parallel):
        """Budget cutoff with some rejected particles still returns accepted ones."""
        sampler = DynamicParticleScheduler(queue_factor=1)

        def evaluate(idx):
            accepted = idx % 2 == 0  # Every other particle rejected
            return {
                "params": [idx],
                "distance": 0.5 if accepted else 999.0,
                "simulation": {"data": np.array([idx])} if accepted else None,
                "accepted": accepted,
            }

        with (
            ThreadPoolExecutor(max_workers=2) if parallel else nullcontext()
        ) as executor:
            result = sampler.run_until_n_accepted(
                executor,
                evaluate,
                ((index,) for index in count()),
                n_target=100,
                max_simulations=6,
            )

        accepted = result["accepted_results"]
        # 6 sims, indices 0-5, accepted are 0,2,4 = 3 particles
        assert len(accepted) > 0
        assert all(r["accepted"] for r in accepted)


def test_batch_bounds_lazy_input_and_preserves_order():
    """Check lazy inputs stay within the queue bound and results keep input order."""
    consumed = 0
    retrieved = 0

    def arguments():
        nonlocal consumed
        for index in range(25):
            consumed += 1
            assert consumed - retrieved <= 4
            yield (index,)

    def submit(fn, index):
        future = Future()
        future.set_result(fn(index))
        original_result = future.result

        def result():
            nonlocal retrieved
            retrieved += 1
            return original_result()

        future.result = result
        return future

    # Completed futures make the queue bound deterministic, with no timing sleeps.
    with ProcessPoolExecutor(max_workers=2) as executor:
        with patch.object(executor, "submit", side_effect=submit):
            assert DynamicParticleScheduler().run_batch(executor, str, arguments()) == [
                str(i) for i in range(25)
            ]
    assert consumed == retrieved == 25


@pytest.mark.parametrize("failure", ["worker", "input"])
def test_batch_cancels_pending_work_on_failure(failure):
    """Check pending tasks are cancelled after worker or input-iterator failure."""
    submitted = []

    def arguments():
        for index in range(20):
            if failure == "input" and index == 2:
                raise RuntimeError("bad input")
            yield (index,)

    def submit(*args):
        future = Future()
        if failure == "worker" and not submitted:
            future.set_exception(RuntimeError("bad worker"))
        submitted.append(future)
        return future

    with ProcessPoolExecutor(max_workers=2) as executor:
        with patch.object(executor, "submit", side_effect=submit):
            with pytest.raises(RuntimeError, match="bad " + failure):
                DynamicParticleScheduler().run_batch(executor, str, arguments())
    assert len(submitted) == (4 if failure == "worker" else 2)
    assert all(
        future.cancelled() for future in submitted[1 if failure == "worker" else 0 :]
    )
