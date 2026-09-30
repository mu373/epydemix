"""Proposal contracts for indexed RNGs and process scheduling.

Support retries belong to proposal generation, not simulation accounting. These
tests use a Uniform(0,1) prior and deterministic kernels to isolate that boundary.
Deadlines use a fake clock, never a sleep. Weighted parent selection is tested with
a degenerate distribution so the assertion is exact rather than a frequency test.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration import _proposals
from epydemix.calibration._evaluate import run_particle_evaluations
from epydemix.calibration._scheduler import SequentialScheduler


class RetryKernel:
    """Return two out-of-support proposals, then a valid one, without RNG draws."""

    def __init__(self):
        self.calls = 0

    def propose(self, parent, rng):
        self.calls += 1
        return -1 if self.calls < 3 else 0.5


def test_outside_support_retries_do_not_consume_simulation_budget():
    """Only the third, valid proposal invokes simulation and consumes budget=1."""
    kernel = RetryKernel()
    calls = []

    def simulate(parameters):
        calls.append(parameters["theta"])
        return {"data": 0.0}

    result = run_particle_evaluations(
        (simulate, {}, ["theta"], None, lambda data, simulation: 0),
        {"theta": stats.uniform()},
        np.random.default_rng(43),
        True,
        n_accepted=1,
        epsilon=1,
        scheduler=SequentialScheduler(),
        total_simulations_budget=1,
        particles=np.array([[0.25]]),
        weights=np.array([1.0]),
        perturbations={"theta": kernel},
    )
    assert kernel.calls == 3
    assert calls == [0.5]
    assert result["n_simulations"] == 1


def test_support_retry_respects_deadline(monkeypatch):
    """An impossible proposal cannot loop forever while no simulations are counted."""
    start = datetime(2026, 1, 1)

    class Clock:
        ticks = 0

        @classmethod
        def now(cls):
            cls.ticks += 1
            return start + timedelta(seconds=cls.ticks)

    class OutsideKernel:
        def propose(self, parent, rng):
            return -1

    monkeypatch.setattr(_proposals, "datetime", Clock)
    candidates = _proposals.ProposalSequence(
        {"theta": stats.uniform()},
        ["theta"],
        43,
        True,
        particles=np.array([[0.25]]),
        weights=np.array([1.0]),
        perturbations={"theta": OutsideKernel()},
        deadline=start + timedelta(seconds=3),
    )
    assert list(candidates) == []
    assert Clock.ticks <= 4


@pytest.mark.parametrize("weights, expected", [([0, 1, 0], 0.5), ([1, 0, 0], 0.25)])
def test_zero_weight_parents_are_never_selected(weights, expected):
    """A single positive mass forces one parent, exposing zero-weight resampling."""

    class IdentityKernel:
        def propose(self, parent, rng):
            return parent

    candidates = _proposals.ProposalSequence(
        {"theta": stats.uniform()},
        ["theta"],
        43,
        True,
        particles=np.array([[0.25], [0.5], [0.75]]),
        weights=np.array(weights),
        perturbations={"theta": IdentityKernel()},
    )
    for _ in range(10):
        assert candidates[_][0] == [expected]
