# Reproducible parallel calibration

Based on upstream `7fc22d1` (v1.3.2). The earlier audit found that the draft parallel
paths ignored the sampler's RNG. Those paths have now been replaced by shared
sequential/parallel candidate generation, evaluation, and result assembly.

## Usage

```python
sampler = ABCSampler(simulation_function, priors, parameters, observed_data, rng=42)
results = sampler.calibrate(strategy="smc", n_workers=4)
```

Omit `n_workers` for execution in the calling process; `n_workers=1` uses a single
worker process. Construct a fresh sampler with the same integer seed (or a fresh,
identically seeded Generator) when comparing runs. Reusing one sampler advances
its RNG between calibrations, identically across worker counts. A caller-owned
`ProcessPoolExecutor` passed as `executor` takes precedence and is left open.
Thread executors are rejected before generating candidates or advancing the RNG: the
model mutates shared state during simulation.

## CPU and native thread limits

Explicit worker counts (including a caller-owned executor's `max_workers`) must
not exceed the detected logical CPU capacity. Excess counts raise `ValueError`
before candidates are generated, RNG state advances, or tasks are submitted.
`n_workers=None` still means sequential execution; an external executor still takes
precedence over `n_workers` and is never resized or shut down by this guard.

The bound is the minimum of system/process CPU count, CPU affinity where available,
and, on Linux, CPU quotas in the process's visible cgroup v1/v2 ancestry. Quotas are
rounded down, with a minimum of one worker for fractional allocations below one CPU.
Unreadable or malformed quota metadata produces a warning and a one-worker limit.
This measures configured capacity, not currently idle CPUs or reserved capacity.
It does not coordinate multiple simultaneous calibration calls or detect hidden
ancestor quotas outside the mounted cgroup namespace; OS scheduling/quota enforcement
still applies. Limits are checked at call entry, not continuously during a run.

`threadpoolctl>=3.1.0` is now a declared runtime dependency. Each simulation and
distance evaluation uses one thread in supported, already-loaded BLAS/OpenMP
libraries. Sequential evaluation uses the same limit to preserve numeric equality.
Previous settings are restored even on exceptions, including in caller-owned pools.
The control is process-wide during a task. Custom callbacks that create additional
pools/threads, import new native libraries during the task, or override thread limits
must manage that extra parallelism themselves. The built-in Numba kernel is serial.

Verification includes v1/v2 parent quotas, fractional quotas, subtree/namespace
mounts, all four public paths rejecting oversized pools, and native thread-limit
restoration after success/failure in a real spawn pool. A fresh process restricted
to one actual affinity CPU rejects two workers; the calibration tests also run with
that affinity (46 passed, two cases requiring more CPUs skipped).

References: [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html),
[v1 quota hierarchy](https://docs.kernel.org/scheduler/sched-bwc.html),
[threadpoolctl scope and limitations](https://github.com/joblib/threadpoolctl).

SMC now supports generation-boundary checkpoints with input data and integrity
hashes. See [SMC_CHECKPOINT.md](SMC_CHECKPOINT.md) for usage and replay semantics.

## Implementation and guarantees

- Each generation/batch consumes a fixed draw from the sampler's RNG to create a
  `SeedSequence`. Each candidate receives its own child Generator, used for prior
  draws or perturbations and, when seeding was requested, for its simulation.
- Sequential and parallel modes evaluate the same worker function. Candidates are
  generated in the parent process and results are ordered by submission index.
- Pending tasks count as potential acceptances. Concurrency tapers near the target,
  avoiding discarded simulations and giving every worker count the same evaluated
  prefix. Simulation budgets count all actual simulations, including rejections.
- Projections share upstream's per-trajectory child RNG setup in both modes.
- Top-fraction and projection batches hold at most twice the worker count in
  submitted, uncollected tasks. Arguments and projection RNGs are generated lazily;
  returned results retain input order. Stored outputs still grow with batch size.
- Owned process pools are joined on exit. Repeated fork-based audit runs now finish
  normally; the previous draft left executor threads/processes running on return.

Seeded outputs change from the upstream sequential implementation because the
random streams are now split per candidate. Exact equality applies within the
same numerical environment, with the same inputs and RNG state. Simulation
functions must use `parameters["rng"]`, custom perturbations must use their passed
RNG, and callbacks must avoid shared mutable state or other unseeded randomness.
Process workers require picklable functions and parameters; use a main guard when
launching scripts on platforms that use spawn.

A wall-clock `max_time` cutoff can return different prefixes on different worker
counts. It stops new submissions and waits for submitted work; it does not kill a
running simulation. Fixed `total_simulations_budget` cutoffs remain reproducible,
including partial rejection results and the last complete SMC generation.

## Verification

Python 3.12.11, NumPy 1.26.4, SciPy 1.16.3, Linux.

| Same-seed comparison | Exact equality |
|---|---|
| Sequential repeated | Yes |
| Parallel repeated with 1, 2, or 4 workers | Yes |
| Sequential versus 1 or 2 workers | Yes |
| 2 versus 4 workers | Yes |
| Different seed (negative control) | No, as expected |

All comparisons passed for SMC, rejection, top-fraction calibration, and projections,
using both a deterministic model and the actual stochastic SIR model. Seeds were
supplied via either `ABCSampler(rng=43)` or `parameters["rng"] = 43`. Both `fork` and
`spawn` completed all 16 comparison rows. The audit compares ordered parameter
values, distances, weights, and trajectories in every generation with no tolerance.

The related pytest suite passed **147 tests**, including new checks for integer and
Generator seeds, continuous and discrete priors, variable simulation draw counts,
repeated calibrations, projection seed precedence, rejection of thread executors,
caller-owned spawn pools, bounded lazy batches, cancellation after worker/input
errors, out-of-order completion, actual simulation budgets, and partial/zero-budget
results. Pytest also
compares posterior DataFrame indices and dtypes. Changed Python files pass Ruff.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_abc.py tests/test_abc_parallel.py tests/test_abc_checkpoint.py tests/test_parallel_resources.py tests/test_seed_reproducibility.py tests/test_abc_smc_utils.py tests/test_calibration_results.py -o addopts='-q'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.check_parallel_seed --start-method fork
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.check_parallel_seed --start-method spawn
```

The audit now asserts all comparisons and exits unsuccessfully on a mismatch.

## Completed spawn audit output

```text
NumPy 1.26.4; process start method: spawn
Exact numeric equality across all generations; no tolerance.
S = sequential (n_workers=None); Pn = n_workers=n.
| Model / seed source / strategy | S repeat | P1 repeat | P2 repeat | P4 repeat | S=P1 | S=P2 | P2=P4 |
|---|---|---|---|---|---|---|---|
| deterministic / rng / smc | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| deterministic / rng / rejection | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| deterministic / rng / top_fraction | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| deterministic / rng / projections | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| deterministic / parameters / smc | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| deterministic / parameters / rejection | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| deterministic / parameters / top_fraction | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| deterministic / parameters / projections | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / rng / smc | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / rng / rejection | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / rng / top_fraction | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / rng / projections | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / parameters / smc | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / parameters / rejection | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / parameters / top_fraction | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
| sir / parameters / projections | yes | yes | yes | yes | yes | yes | yes |
  S vs P2 differing fields: none
```
