# Parallel implementation review and measurements

Original measurement scope: working tree based on upstream `7fc22d1`, after
candidate-specific RNG unification. The timing tables and source hashes below
record that version, before the follow-up fixes.

## Follow-up fixes

- All calibration strategies and projections now require a `ProcessPoolExecutor`
  for caller-provided executors. Thread executors fail before simulation or RNG
  consumption. This prevents the shared-model race through the ABC API; direct
  threaded calls to the model itself remain unsafe. Caller-owned process pools
  remain open after calls.
- Batch submissions now use a window of twice the worker count, retrieving
  completed futures while preserving input order. Projection arguments and child
  RNGs are generated lazily. Pending futures are canceled on worker/input errors.
  Already-running work is still joined for owned pools; stored outputs still grow
  with the number of results.
- The updated gate probe measured **4 outstanding tasks for 2 workers**, with all
  256 results returned in input order (previously all 256 were submitted upfront).
- The related pytest suite passed **109 tests**, including caller-owned spawn
  pools, thread rejection, bounded lazy input, and cancellation on failure.

`max_time` retains its documented soft submission deadline. Hard worker termination was not added. SMC checkpoint/restart was subsequently
implemented; see `SMC_CHECKPOINT.md`. The original findings below are retained as
review evidence; findings 1 and 3 are addressed at the ABC boundary and scheduler.

## CPU-capacity follow-up

Worker counts now have an entry-time upper bound based on detected logical CPU
capacity, Linux affinity and visible cgroup v1/v2 quotas (including parent quotas).
The same check applies to caller-owned process pools. Simulation/distance tasks use
one thread per supported loaded BLAS/OpenMP library and restore settings afterward;
`threadpoolctl` is now a declared runtime dependency. See `PARALLEL_SEED_CHECK.md`
for quota rounding, scope, and verification. Related tests including checkpoint/restart now total 147 passes.
The original timing tables below predate these guards and per-task thread limiting.

## Original findings, in priority order

### 1. Shared EpiModel with ThreadPoolExecutor can silently corrupt results

**Conditional correctness issue.** `simulate()` assigns `epimodel.definitions`,
then reads that shared attribute again for simulation. The worker only shallow-copies
its parameter dictionary. Two threads using the same model can overwrite each
other's parameter definitions.

The probe forces a valid interleaving with a barrier immediately before initial
conditions are applied. With transmission rates 0.15 and 0.45 and the same RNG
seed, the two threaded trajectories became identical; one differed from its serial
reference. This is a deterministic reproduction of the race, not an estimate of
how often normal scheduling triggers it. Default process workers passed all
numerical comparisons.

Locations: `epydemix/model/epimodel.py:909` and
`epydemix/calibration/_worker.py:37`.

Recommended action: explicitly constrain EpiModel execution to processes or give
each concurrent simulation its own model. Generic thread executors remain suitable
only for callbacks without shared mutable state.

### 2. max_time does not bound return time, including on failure

**Documented operational limitation, not a new RNG defect.** The collector waits
without a timeout, and owned process pools join running work at exit. A worker
that never returns also prevents the calibration from returning. Canceling its
Future cannot stop running code. There is no checkpoint/restart support for
completed generations inside a failed calibration call.

Locations: `epydemix/calibration/parallel.py:62`, `:94` and
`epydemix/calibration/abc.py:349`.

Recommended action: decide whether a hard execution deadline is required. If it is,
use a killable outer job boundary plus saved complete generations; simply adding
`Future.result(timeout=...)` would still leave the pool's exit waiting for workers.
For the current soft deadline, retain explicit documentation and operational
supervision of long-running jobs.

### 3. Batch submissions are unbounded

**Scaling limitation.** `run_batch()` creates all Futures before retrieving any
results. A two-thread probe submitted all 256 tasks while none had finished. Both
top-fraction calibration and projections use this path; SMC/rejection already
bound their pending work. Large runs can accumulate tasks, parameter payloads and
completed trajectories in memory. This probe confirms the queue behavior; it does
not measure an out-of-memory threshold.

Location: `epydemix/calibration/parallel.py:80`.

Recommended action: apply a bounded submission window here before large batch
runs. Final stored trajectories will still scale with the number of results.
The ordered `.result()` loop also delays noticing a later task's error if an earlier
one is slow.

## Measurements

Real stochastic SIR, 48 accepted particles per generation, 3 SMC generations,
continuous priors on transmission/recovery, adaptive epsilon, calibration seed 43
and synthetic observation seed 2026. No time or simulation-budget cutoff.

- Small: 1 demographic group, 10 daily steps; 307 candidate simulations per run.
- Large: 16 demographic groups, 180 daily steps; 282 candidate simulations per run.
- Both return aggregate incidence as the calibration target, not all model outputs.
- Each condition ran 3 times in fresh subprocesses, with execution order rotated.
  The fork and spawn batches ran separately, with no other audit workload running.
- Timings include process pool startup, worker imports, calibration and shutdown.
  Coordinator imports, model construction and observation generation are excluded;
  the coordinator's simulator/Numba path is warmed before timing. Spawn workers
  import their modules inside the timed interval. Pools are reused across the three
  SMC generations, but not across separate calibrations.
- Memory is the median of sampled peak **total process-tree PSS** (MiB), including
  coordinator, workers, and any resource tracker. PSS apportions shared pages;
  summing RSS double-counts them. Sampling is roughly every 25 ms plus inspection
  overhead, so short peaks can be missed. Raw files also contain aggregate RSS.

### Small SIR

| Execution | fork seconds | spawn seconds | fork PSS MiB | spawn PSS MiB |
|---|---:|---:|---:|---:|
| Sequential | 0.526 | 0.527 | 192.9 | 193.0 |
| 1 worker(s) | 0.530 | 1.671 | 201.0 | 346.0 |
| 2 worker(s) | 0.334 | 1.544 | 206.2 | 479.2 |
| 4 worker(s) | 0.317 | 1.688 | 215.5 | 734.3 |

### Large SIR

| Execution | fork seconds | spawn seconds | fork PSS MiB | spawn PSS MiB |
|---|---:|---:|---:|---:|
| Sequential | 9.175 | 9.224 | 195.7 | 196.9 |
| 1 worker(s) | 9.498 | 10.901 | 210.2 | 354.7 |
| 2 worker(s) | 5.139 | 6.396 | 217.0 | 491.0 |
| 4 worker(s) | 2.919 | 4.376 | 231.7 | 752.4 |

Large-model four-worker speedups were **3.14x with fork** and **2.11x with spawn**.
For small jobs, spawn startup outweighs parallel execution: 0.527 s sequential
versus 1.688 s with four spawn workers. These are local measurements, not a general
speedup guarantee. Fork sharing reduces memory considerably, but the calibration
currently inherits the platform's process start method; these single-threaded
benchmark coordinators do not establish fork safety inside a threaded notebook or
server. The earlier pytest environment emitted a multithreaded-fork warning.

The large case's first serialized task was about **797 KiB**, compared with **3.5 KiB**
for the small case. Model and observed data are sent with each candidate. Multiplying
that sample size by 282 suggests roughly 219 MiB of task payloads, but actual IPC byte
traffic was not instrumented. Initialize fixed worker data once only if larger-model
profiling establishes that repeated transfer is a bottleneck.

Time outside particle collection was about 0.05–0.06 s in the fork measurements,
including weights, result assembly and shutdown. The quadratic importance-weight
loop is therefore not the dominant cost at 48 particles. Candidate proposal work
also happens in the coordinator but is included inside collection time, so this
metric is not the total coordinator CPU time. Larger particle counts need their
own profile before weight optimization.

**All 48 runs matched exactly** in the hash of ordered parameter arrays, weights,
distances and trajectories for all generations, and in total simulation counts.
Each case's hash also matched between fork and spawn.

## Fault and API probes

| Probe | Observed outcome |
|---|---|
| Simulation raises RuntimeError while another worker sleeps 0.5 s | RuntimeError returned after 0.534 s; no live child processes remained |
| Worker exits abruptly with status 7 | BrokenProcessPool after 0.022 s; no live child processes remained |
| Non-returning simulation, max_time=0.05 s | Still blocked after 5.142 s; the probe supervisor killed its isolated process group |
| Three 0.25 s simulations, 2 workers, max_time=0.05 s | Returned 3 accepted particles after 0.526 s, draining submitted work |
| Local lambda as simulation function | AttributeError during serialization; no live child processes remained |
| Same EpiModel shared between two threads | Forced race changed one trajectory; both returned the same trajectory despite different transmission rates |
| 256 batch tasks, 2 threads, tasks blocked at a gate (rerun after fix) | 4 submitted before first completion; peak window 4; all 256 outputs ordered |

These diagnostic probes assert the characterized behavior and are intentionally
outside pytest's test discovery. The batch probe was updated and rerun after the
fix; other rows retain the original measurements. The shared-model probe calls
the simulator directly, bypassing the new ABC executor guard. The deliberate hang
has a supervisor timeout and process-group cleanup.

## Environment and reproduction

AMD Ryzen 7 5825U, 8 physical cores / 16 logical CPUs available, Linux, no cgroup
CPU or memory limit on the benchmark's cgroup ancestry. The machine is shared;
CPU clocks and other activity were not controlled. There are only 3 samples per
condition. BLAS, OpenMP and Numba thread limits were all set to 1 to avoid nested
CPU parallelism. No new runtime package dependency was added; the memory harness
uses the already-installed psutil.

- numpy: 1.26.4
- scipy: 1.16.3
- numba: 0.64.0
- psutil: 7.2.2

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m validation.benchmark_parallel --start-method fork --output validation/parallel_benchmark_fork.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m validation.benchmark_parallel --start-method spawn --output validation/parallel_benchmark_spawn.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m validation.review_parallel
```

Artifacts: `parallel_benchmark_fork.jsonl`, `parallel_benchmark_spawn.jsonl`, and
`parallel_review_probes.json`. The benchmark files include all individual timings,
sampled memory peaks, simulation counts and numerical-output hashes.

Source hashes at measurement (SHA-256):

- `epydemix/calibration/abc.py`: `597d37209464935eb2914c534a8a72e2e664fff2c3beeebde443456677805148`
- `epydemix/calibration/parallel.py`: `30da6c7522757b2875ccc4d2033edfaefbcab905e806847dafe6ff3424ae633e`
- `epydemix/calibration/_worker.py`: `670dd410d3cb4e4d75ff73c1f6a7ee0b9b7602870c819c8dc39ad5e1576d1802`
