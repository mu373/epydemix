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
