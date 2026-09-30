# ABC compatibility reference

`abc_reference.json` is an immutable snapshot generated at `7fc22d1` using Python
3.12.11, NumPy 1.26.4, SciPy 1.16.3 and pandas 2.3.3.
It contains six repeated calibrations, every generation, projections and root RNG
state. The workload and its rationale live in `tests/fixtures/abc_reference.py`.

Reproduce in the source checkout, with these numerical versions installed:

```sh
python -m tests.fixtures.abc_reference tests/data/abc_reference.json --source-commit 7fc22d1
python -m pytest tests/test_abc_reference.py --no-cov
```

The generator is an explicit maintenance command, never called by pytest. A pure
refactor must retain this snapshot. An intentional RNG change must first fail the
old reference, then update only affected outputs and document the source commit.
The mathematical/public-API contract tests do not require this pinned environment.

The pinned versions do not pin CPU implementations of PDF arithmetic. The CI run
36708373748 differed by one float64 ULP in one computed SMC weight, with all other
fields identical. Weight values alone permit four ULPs; their dtype/shape and all
particles, distances, trajectories, projections and RNG states remain exact. The
baseline fixture is unchanged. Independent weight-formula tests remain enabled.
## Candidate RNG redesign

The old golden failed at the indexed-RNG change, as intended. The active reference
was explicitly regenerated from `9b084b37ab398d63e6e55618556ad79b9c038c1a`; the unchanged baseline is
preserved as `abc_reference_baseline.json`. Model, seeds, numerical environment,
mathematical target tests and threshold contracts were not changed. Each batch now
consumes one four-word root seed and derives candidate streams by index, changing
particle values, weights, distances, trajectories, projections and parent RNG state.

To regenerate the active snapshot, run the recipe above using this source commit.
