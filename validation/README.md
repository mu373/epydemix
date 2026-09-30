# Validation

Run from the checkout being measured. Unit tests do not depend on this directory.
The scripts are research and performance checks; detailed profiling is separate
from normal timing. Optional audit dependencies include `psutil`.

```sh
python -m validation.benchmark_parallel --model normal --workers 0 1 2 --particles 100 --generations 3 --repeat 3
python -m validation.benchmark_parallel --model sir --workers 0 2 --particles 40 --generations 3 --repeat 3
python -m validation.benchmark_parallel --model normal --workers 2 --particles 40 --generations 2 --profile
```

Worker value 0 requests the sequential path. Each repetition uses a fresh process
and reports source commit, tracked dirty diff, environment, seed, all workload
arguments, mathematical output fingerprint, simulation counts and timings as JSON
Lines. See [WORKER_INPUT_CACHE.md](WORKER_INPUT_CACHE.md) for paired measurements.
The benchmark reads native per-call metrics. To collect application-configured
JSON events on stderr, add `--log-json`; compare logging-enabled timing separately
from ordinary timing. See [logging definitions](../docs/CALIBRATION_LOGGING.md).
Further lifecycle, checkpoint/history, and notebook validation is updated with the
corresponding PRs.

Checkpoint streaming audit (optional `psutil`, Unix/macOS `resource`):

```sh
python -m validation.benchmark_checkpoint_memory --size-mib 8
```

Save, load and deepcopy run in fresh interpreters. Output records the actual
synthetic payload size, whole-process peak RSS, operation duration, source and
environment, and matching trajectory fingerprints. Peak RSS includes the process
baseline and imports; it is not an incremental allocation measurement. This
synthetic audit complements exact SMC resume and atomic-write tests. The 8 MiB
smoke report is in `results/checkpoint-streaming-smoke.jsonl`; its save/load/copy
fingerprints match. It records a pre-commit source plus the exact dirty diff and
benchmark SHA, and does not establish performance scaling.
