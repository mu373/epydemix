# Detailed dispatch profile (separate from timing)

Run --profile in a fresh disposable audit process. The hooks live only in the
validation harness. The returned calibration API, candidate RNGs and core code are
unchanged; profiles are not performance comparisons because every model call writes
a trace record. Both owned and reused pools and fixed-count tasks are checked.

```sh
python -m validation.benchmark_parallel --workers 2 --particles 64 --generations 3 --repeat 1 --profile
python -m validation.benchmark_parallel --model sir_large --workers 2 --particles 64 --generations 3 --repeat 1 --profile
python -m validation.benchmark_parallel --workers 2 --particles 12 --generations 2 --repeat 1 --profile --reuse-pool
python -m validation.benchmark_parallel --strategy top_fraction --workers 2 --particles 12 --payload-mib 4 --repeat 1 --profile
```

- input_snapshot serialization records bytes/time for the owned input file;
  process_task and dynamic_job records measure parent dispatch pickle time/bytes.
- initializer traces measure actual initializer execution. Worker readiness is its
  completion minus first task submission, including spawn/imports. Constructor
  time alone is not worker startup. External-pool warm-up precedes measurement.
- input_restore/job_restore trace fresh deserialization. scheduler_rpc includes
  manager IPC, lock/counter work and waiting; it is not pure transport latency.
- simulation events measure the callback, excluding distance/proposal/restore/RPC.
  Their count is asserted equal to native physical simulation counts, including
  surplus. DYN process task count is worker-loop count, not simulation count.
- worker_task_non_simulation_seconds includes proposals, distance, deserialization,
  RPC, thread discovery and trace overhead. It is not a pure idle-time estimate.
- dynamic_batches_finish_spread_seconds reports the earliest-to-latest loop finish
  within each measured DYN batch; raw task times also expose dispatch/return delay.
- manager_start records actual DYN Manager.start duration per generation. For reused
  pools, worker_events and manager_start include warm-up: restrict timestamps to
  calibration_started/finished when analyzing the measured call.

The traces preserve per-PID time intervals for inspection, rather than claiming
that residual wall time uniquely identifies IPC or idle time. The native parent
collection/weight clocks and sampled RSS/CPU have the scopes documented in
CALIBRATION_SCALING.md. Raw normal/larger-SIR profiles and small fixed/reused smoke
checks are in results/. Each smoke's posterior fingerprint matches ordinary
execution for its workload. All profiled physical model-call counts match native
counts; even a surplus task is recorded. No wall-time threshold is in unit tests.

The corrected harness (`5386c28`) preserves initializer identity during the
pure evaluator-selection step, before task submission, then restores the profile
initializer. cache_selected is asserted against the production rule and records
true for owned caches / false for serial or reused external pools. Owned-cache
input_restore counts equal physical simulations plus initializer warm restores.
Earlier uncached-path profiles are archived with an explicit label. Corrected
profile output hashes match those earlier mathematical results and ordinary runs.
