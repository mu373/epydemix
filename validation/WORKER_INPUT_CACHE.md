# Worker input cache

The cache remains because it reduces repeated transport of large fixed inputs in
fixed-count parallel calibration. DYN sends a worker loop once per generation, so
the same advantage must not be assumed for DYN.

Measured on Linux, Python 3.12.11, NumPy 1.26.4, SciPy 1.16.3, 16 logical CPUs;
two workers, spawn, seed 43, three fresh-process repetitions. The pre-cache source
is `fd9cda3`. Post-cache measurements were taken before committing; each JSONL
records the exact tracked source diff and benchmark SHA256. All paired output
fingerprints match. Timings exclude imports and fingerprint readback.

| Workload | Before median (range), seconds | Cache median (range), seconds |
| --- | --- | --- |
| Normal, DYN SMC, 100 particles, 3 generations, 8 MiB fixed payload | 5.618 (5.519–5.632) | 5.500 (5.451–5.544) |
| Normal, top-fraction, 100 evaluations, 8 MiB fixed payload | 2.581 (2.563–2.630) | 1.488 (1.486–1.529) |
| SIR, DYN SMC, 40 particles, 3 generations, no added payload | 5.251 (5.192–5.384) | 5.270 (5.255–5.271) |

The fixed-count transport case improves by about 42%. The DYN changes are small
and do not establish a reliable improvement; the SIR ranges overlap. These cases
use infinite SMC thresholds to isolate execution overhead, not posterior accuracy.
Statistical correctness is tested separately against finite-epsilon targets.

Raw measurements are in `results/cache-*.jsonl`. To reproduce on each source
checkout, invoke the same benchmark file by absolute path from that checkout:

```sh
python /path/to/validation/benchmark_parallel.py --model normal --workers 2 --particles 100 --generations 3 --payload-mib 8 --repeat 3
python /path/to/validation/benchmark_parallel.py --model normal --strategy top_fraction --workers 2 --particles 100 --payload-mib 8 --repeat 3
python /path/to/validation/benchmark_parallel.py --model sir --workers 2 --particles 40 --generations 3 --repeat 3
```

Owned pools serialize fixed inputs once to a temporary file, load immutable bytes
at worker startup, and restore a fresh object per candidate. An initial disposable
deserialization imports simulator modules before discovering native thread pools.
Caller-owned pools keep their initializer and lifetime and use the uncached path.
Isolation, error cleanup, spawn/fork, and external-pool parity have runnable tests
in `tests/test_worker_input_cache.py`.

## Benchmark limits

`psutil` is a validation dependency, not a library dependency. Sampled process-tree
RSS sums shared pages repeatedly and is not unique memory. CPU time is sampled
over child process lifetimes and includes imports. Historical cache measurements used a batch timing adapter. The current benchmark
reads core execution metrics directly; use the recorded benchmark SHA to reproduce
the historical instrument. With `--profile`, task
spans and serialization are measured in a separate disposable process; a DYN task
can evaluate many candidates, so task counts are not simulation counts. Profile
results must not be mixed with ordinary wall-time comparisons.
