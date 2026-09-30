"""Indexed proposals preserve the candidate RNG mapping."""

import numpy as np
from scipy import stats

from epydemix.calibration._proposals import ProposalSequence


def test_indexed_candidates_preserve_previous_spawn_streams():
    """Random access must preserve the old sequential SeedSequence.spawn mapping."""
    entropy = [12, 34, 56, 78]
    candidates = ProposalSequence({"x": stats.norm()}, ["x"], entropy, True)
    children = np.random.SeedSequence(entropy).spawn(21)
    for index in (20, 0, 3):
        reference = np.random.default_rng(children[index])
        params, _, rng = candidates[index]
        assert params == [stats.norm().rvs(random_state=reference)]
        np.testing.assert_array_equal(rng.random(10), reference.random(10))
