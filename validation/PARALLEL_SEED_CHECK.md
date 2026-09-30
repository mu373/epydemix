# Exact seeded calibration / projection checks

```sh
python -m validation.check_parallel_seed --workers 0 1 2
```

The current audit (`cf3e89c`, spawn, NumPy 1.26.4) passed all 48 conditions:
two models × two seed sources × four strategies × three worker settings.
Every condition repeats exactly and matches sequential execution; seed 44 is a
negative control against seed 43. Results include every generation's posterior,
weights, distances and trajectories plus parent RNG state. Projections start from
the same calibrated posterior. Raw results are results/seed-check-spawn.jsonl.

The deterministic linear model isolates proposals; the offline single-group SIR
also checks RNG injection into stochastic simulation. Small counts (8 particles,
2 SMC generations or 24 top-fraction evaluations) keep exact contracts cheap.
Time and physical simulation budgets do not bind: their cutoffs are not an
execution-mode equivalence guarantee. DYN surplus work is not included in the
mathematical output comparison. Independent finite-epsilon statistical targets
are verified by tests/test_abc_statistical.py, rather than this same-code comparison.

The CLI supports subsets of models, strategies, seed sources and start methods.
Default worker settings adapt to CPU capacity; an explicit unavailable count fails.
Detailed CPU quota/native-thread restoration and owned/external executor contracts
are standard tests, independent of validation/. Prior reports are in historical/.
