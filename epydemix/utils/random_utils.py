"""Derive an independent random stream from a seed and a task index."""

import numpy as np


def rng_for_index(seed_sequence, index):
    """Return a child RNG without advancing the parent SeedSequence.

    The index identifies a candidate or simulation, independently of its worker
    or execution order. Preserve the parent's spawn key and entropy pool size.
    """
    child = np.random.SeedSequence(
        seed_sequence.entropy,
        spawn_key=seed_sequence.spawn_key + (index,),
        pool_size=seed_sequence.pool_size,
    )
    return np.random.default_rng(child)
