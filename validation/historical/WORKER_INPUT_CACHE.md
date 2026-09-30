# Worker input caching: implementation and measurements

Measured 2026-09-27 after the [baseline scaling audit](CALIBRATION_SCALING.md).
Production source hashes are recorded in `dispatch_comparison.jsonl`.

## Implementation

Calibration pools created through `n_workers` now serialize the fixed simulation
function, model/parameters, parameter names, observations and distance function
**once per calibration call**. A private temporary directory holds the serialized
snapshot. Each worker receives its path, reads the snapshot once at initialization,
and retains the serialized bytes. Candidate tasks send only sampled parameters,
epsilon and the candidate RNG.

Each candidate restores its own inputs from the serialized snapshot. Thus mutable
models, nested parameters, callable objects and observed arrays cannot contaminate
later candidates in the worker. Deserialization still happens per candidate; this
optimization removes repeated parent-side serialization and task transfer.
Each worker retains one extra serialized input snapshot in RAM, in addition to its
active restored inputs. The system temporary directory must be writable.

The temporary input file remains available until the pool closes, and the directory
is removed after normal completion or exceptions. External executors retain their
own initializer/lifetime and use the original full-argument dispatch path. The
optimization applies to SMC, rejection and top-fraction calibration; projection
execution uses its existing path.

The first attempt passed snapshot bytes directly as initializer arguments. On this
host, spawn with eight workers regressed to 11.32–11.48 s. CPython's POSIX spawn
launcher writes those arguments into a pipe during process startup; a large payload
can block that write while the child imports its main module. Passing only a small
file path removed this regression. These exploratory records are retained as
`dispatch_inline_initializer_*.jsonl`; the partial comparison there was stopped and
is not included in the final results below.

## Paired timing comparison

Same workload as the baseline: stochastic SIR, synthetic observations, 16 groups,
180 days, 96 particles × 3 generations, full returned trajectories, seeds 43/2026.
Ryzen 7 5825U, 8 physical cores / 16 logical CPUs; native thread limits 1.
Checkpoint disabled. Pool startup and shutdown are included; coordinator/model
setup is excluded. Memory is sampled peak aggregate process-tree PSS.

Each condition ran twice in a fresh process. For each worker count, old/new runs
were adjacent; their order and worker order were reversed for the second repetition.
Tables show medians. Timing runs did not use the profiling hooks. Sequential baselines
were measured in each start-method batch. This is a small local comparison, not a
confidence interval or production-data performance guarantee.

| Method | Workers | Old seconds | New seconds | Improvement over old | New speedup vs sequential | Old PSS MiB | New PSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fork | 4 | 5.760 | 5.425 | 1.06× | 3.24× | 280.6 | 278.0 |
| fork | 8 | 4.197 | 3.742 | 1.12× | 4.70× | 319.7 | 314.0 |
| fork | 16 | 3.921 | 3.541 | 1.11× | 4.97× | 393.0 | 382.9 |
| spawn | 4 | 6.936 | 6.861 | 1.01× | 2.60× | 795.6 | 789.3 |
| spawn | 8 | 6.115 | 5.410 | 1.13× | 3.29× | 1304.7 | 1295.5 |
| spawn | 16 | 6.940 | 6.796 | 1.02× | 2.62× | 2308.8 | 2294.4 |

At eight workers the improvement is about **1.12× for fork and 1.13× for spawn**.
The new sequential-relative speedups are about **4.70× and 3.29×** respectively.
Spawn improvements at four and sixteen workers are only about 1–2%; reducing task
transfer does not remove worker startup/import cost or add physical cores.

## Separate dispatch/occupancy profile

Eight workers, same workload, two runs per condition. The task wrapper records
worker-call start/end times and parent submission times. The parent also records
`ForkingPickler.dumps` wall duration and serialized task size. Result fingerprints
match the uninstrumented benchmark.

| Method | Dispatch | Task payload total MiB | Parent serialization seconds | Mean submit-to-start ms | Underfilled worker capacity |
|---|---|---:|---:|---:|---:|
| fork | full | 392.356 | 1.753 | 54.1 | 4.5% |
| fork | cached | 0.259 | 0.045 | 45.3 | 5.0% |
| spawn | full | 392.356 | 1.760 | 95.8 | 3.6% |
| spawn | cached | 0.259 | 0.048 | 90.0 | 3.6% |

- The 504 serialized task envelopes shrink from **392.36 MiB to 0.26 MiB** (about
  99.93% less). These figures include the measurement wrapper and exclude the
  one-time input file/read, result transfers and IPC framing. Raw candidate argument
  tuples shrink from approximately 797 KiB to **339 bytes**.
- Serialization duration is accumulated wall time in the parent feeder thread.
  It overlaps simulation, so its reduction cannot simply be subtracted from total
  runtime. Submit-to-start includes queueing, startup, transfer and input decoding;
  pure pipe-transfer time was not isolated.
- Underfilled capacity is a **lower bound** on unused worker slots: during collector
  waits, integrate `max(workers - pending_futures, 0) × wait_duration`, then divide
  by `workers × total_collection_time`. Here this reflects the end-of-generation
  submission taper. Startup and other idle intervals are not fully counted.
  About 3–5% of available worker-slot time remains unused by this measure.
- Raw files also contain time spent inside the worker callable and tail intervals.
  This is not CPU utilization: the new cached callable includes snapshot restoration,
  while the old task is decoded before that timer starts. Do not interpret the raw
  callable-occupancy difference as a precise reduction in idle time.

The measured improvement supports retaining this transfer optimization. It does
not establish which remaining cost dominates overall scaling.

## Verification

- The related test set passed **163 tests** (102 parallel/resource/checkpoint/history
  checks plus 61 calibration/RNG/results checks). After changing initialization to
  use the temporary file, all 102 affected checks passed again.
- New fork/spawn tests verify one parent serialization per calibration; mutable
  callable, nested input and observation isolation; exact equivalence to an external
  executor using the original path; final RNG state; caller pool reuse; and temporary
  file cleanup after success and worker exceptions. Changed inputs on a later call
  are seen by the newly created pool.
- All **28 paired timing runs and 8 profiling runs** completed. Ordered posterior
  parameters, weights, distances and all trajectory arrays match the historical
  baseline exactly. Every run performs **504 candidate simulations**.
- Pytest reported the existing Python 3.12 multithreaded-fork warnings and expected
  NaN test warnings. Standalone measurements do not establish notebook fork safety.

## Reproduce individual conditions

From the repository root, with `psutil` installed for measurement:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# Original full-argument dispatch, selected only inside this benchmark harness:
python -m validation.benchmark_parallel --cases large --particles 96 --full-trajectory --workers 4 8 16 --repeats 2 --start-method fork --uncached --output validation/recheck_full_fork.jsonl
# Optimized production path:
python -m validation.benchmark_parallel --cases large --particles 96 --full-trajectory --workers 4 8 16 --repeats 2 --start-method fork --output validation/recheck_cached_fork.jsonl
```

Use `--start-method spawn` for the other start method. For serialization/occupancy
profiling add `--profile --workers 8`. The paired audit alternated the individual
old/new conditions as described above. `--uncached` is an audit-only switch;
calibration users receive the optimization automatically with `n_workers`.

Raw final records: `dispatch_comparison.jsonl` and `dispatch_profile_{full,cached}_{fork,spawn}.jsonl`.
The earlier `dispatch_before_*.jsonl` files record the initial diagnosis.

## Follow-up: why eight workers do not reach eightfold speedup

A diagnostic run measured `perf_counter` wall time and `process_time` CPU time
around each candidate evaluation. Same 96-particle/3-generation SIR workload,
504 candidates, fork, one run per condition, no process-tree memory sampler.
Process-worker timers include restoration of the fixed input snapshot. The
sequential timer covers evaluation directly. All numerical hashes matched.

| Workers | Total seconds | Mean candidate wall ms | Mean candidate CPU ms | CPU / wall |
|---|---:|---:|---:|---:|
| Sequential | 16.890 | 32.63 | 32.62 | 99.97% |
| 1 | 17.252 | 33.27 | 33.25 | 99.96% |
| 4 | 5.117 | 36.92 | 36.90 | 99.94% |
| 8 | 3.615 | 48.49 | 48.45 | 99.92% |

With the same process-worker path, going from one to eight workers increases
evaluation CPU time per candidate from 33.25 to 48.45 ms (about 1.46×). CPU time
closely tracks evaluation wall time, so this increase is not primarily time spent
waiting for CPU scheduling or IPC inside that measured interval. Memory/cache
stalls still count as CPU time. Lower all-core clock speed and shared cache/memory
contention are plausible contributors; neither frequencies nor hardware counters
were measured, so their individual contributions remain unproven.

This per-candidate slowdown alone reduces ideal eight-worker throughput to roughly
8 / 1.46 = 5.5 times the one-worker path, before coordinator work and tail underfill.
The earlier profile places about 92% of fork collection worker-slot time inside
evaluation, while the paired benchmark has about 0.27 s outside collection. These
measurements explain much of the gap without assuming a large scheduling backlog.
The figures come from separate diagnostic runs, not an exact additive decomposition
of a single timing run. Raw records: `worker_cpu_probe.jsonl`.
