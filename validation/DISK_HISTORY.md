# Disk-backed SMC history

history_storage="disk" requires checkpoint_path and stores lazy references to
independently checksummed generation/field archives. Result deepcopy does not
read all trajectories. Accessing a field loads that generation without a persistent
object cache, and consumed generation arrays are released before the next one.
Missing fields, identity mismatch and partial publication are standard test cases.

Fresh-process comparison, clean source `d0dac42`, Linux / Python 3.12.11 /
NumPy 1.26.4, seed 43, eight particles, 1 MiB constant payload per particle.
Explicit infinite epsilon accepts every proposal. Each condition ran three times.
Raw metadata/fingerprints are in results/history-memory-repeat.jsonl.

| Generations | Storage | Median s | Min–max s | Median peak RSS MiB | Peak min–max MiB | Simulations |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 3 | memory | 0.1404 | 0.1386–0.1464 | 316.0 | 315.9–316.2 | 24 |
| 3 | disk | 0.1065 | 0.1037–0.1075 | 275.1 | 275.1–275.1 | 24 |
| 6 | memory | 0.3710 | 0.3662–0.3717 | 364.0 | 364.0–364.3 | 48 |
| 6 | disk | 0.1914 | 0.1853–0.1931 | 275.0 | 275.0–275.2 | 48 |

```sh
python -m validation.benchmark_history_memory --generations 3 6 --payload-mib 1 --repeat 3
```

Memory/disk output fingerprints match for each generation count. Measurements
include calibration, checkpoint/history writes and calibrate deepcopy; later
fingerprint/readback is excluded and reported separately. Whole-process peak RSS
includes imports/baseline. The synthetic scalar model measures retained trajectory
storage, not epidemic simulation cost or a universal speedup/memory bound. Memory
retains all generations; disk keeps live data bounded by the current generation.
OS page cache and later getter loads are outside this per-call metric scope.
Initial single-observation smoke remains in results/history-memory-smoke.jsonl.

Keep the checkpoint and its .history directory together when relocating them.
The loader rebinds moved references. Legacy v1 checkpoints lacking history_storage
remain readable as memory mode; changing saved storage mode is rejected.
history_io_seconds counts writes and resume field reads during this call. User
getter calls after calibration do not change completed-run metrics. Checkpoint
chain IDs/current call IDs and discarded-work counters retain their distinct scopes.

```python
result = sampler.calibrate(strategy="smc", num_particles=100, num_generations=10,
                           checkpoint_path="run.checkpoint", history_storage="disk")
# Only generation zero's trajectory field is loaded here.
trajectories = result.get_calibration_trajectories(generation=0)
```

Standard tests and docstrings in tests/test_disk_history.py and
tests/test_checkpoint_logging.py verify exact output, failed publication safety,
legacy resume and weak-reference release without RSS thresholds.
