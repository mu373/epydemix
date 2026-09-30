# Current parallel fault / isolation probes

```sh
python -m validation.review_parallel
```

All six current probes passed using clean source `cf3e89c`, with a 15-second
launcher timeout. Raw results are results/parallel-fault-probes.jsonl.

| Probe | Observed contract |
| --- | --- |
| Model exception | Original RuntimeError propagated; owned children joined |
| Worker process crash | BrokenProcessPool propagated; owned children joined |
| Unpicklable lambda | AttributeError surfaced before running work |
| Deadline | No new candidates after cutoff; 51 completed / one drained, zero matches |
| Mutable fixed input | All 12 fixed-count evaluations received an empty list; parent unchanged |
| Deliberate hang | Launcher killed the process group after timeout |

Deadline counts/durations are diagnostic observations, not stable timing or count
assertions. They depend on startup and host load. An arbitrary model callback that
never returns cannot be interrupted by a scheduler deadline; the deadline stops
new work and drains running callbacks. The launcher bounds this deliberate probe
and is POSIX-only. Standard executor tests cover other supported operating systems,
bounded fixed-count submission, ordered results, external ownership and native
thread restoration without depending on these performance observations.

ThreadPoolExecutor is rejected because mutable simulation models can share state.
The historical direct-thread shared-model race and old scheduler audit remain in
historical/; they are not reports of the updated public API. Normal timing and
profile evidence are in CALIBRATION_SCALING.md and DISPATCH_PROFILE.md.
