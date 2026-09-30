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
Current timing/scaling results and scopes are in
[CALIBRATION_SCALING.md](CALIBRATION_SCALING.md); detailed profile definitions are
in [DISPATCH_PROFILE.md](DISPATCH_PROFILE.md). Add `--reuse-pool` with positive
worker counts to warm a caller-owned pool with an identical calibration before
measurement. `--model sir_large` uses 16 groups / 181 daily points. Infinite
epsilon benchmarks measure execution cost, not posterior accuracy.

Seed and bounded fault checks:

```sh
python -m validation.check_parallel_seed --workers 0 1 2
python -m validation.review_parallel
```

The seed checker compares exact retained output and parent RNG state for every
ABC strategy and projections, with repeat and different-seed controls. Defaults
adapt worker counts to detected CPU capacity. Fault probes require POSIX process
groups: the intentional hanging callback is killed by the launcher. Deadlines
stop new work and drain callbacks; they cannot interrupt arbitrary user code.
See [PARALLEL_SEED_CHECK.md](PARALLEL_SEED_CHECK.md) and
[PARALLEL_REVIEW.md](PARALLEL_REVIEW.md).

Optional pyabc comparison (not a correctness oracle or core dependency):

```sh
python -m pip install pyabc nbclient nbformat ipykernel
python -m ipykernel install --user --name epydemix-validation
python -m validation.run_notebooks compare_pyabc.ipynb --kernel epydemix-validation
```

The kernel needs this checkout's dependencies. The runner executes from validation/
and replaces outputs only after every cell succeeds. The notebook covers continuous
and mixed discrete/continuous priors, explicit RNG injection, weighted distributions,
and matching epsilon schedules. It was executed with pyabc 0.13.0. The independent
ABC/SMC theoretical target tests live in tests/ and need none of these dependencies.

The two epidemic comparison notebooks and six reference models are updated in the
separate simulation integration PR based on the common executor; use its
`validation/MODEL_COMPARISONS.md` for offline model/ensemble validation. The combined
series executes all three notebooks. Standard tests also run from an export without
validation/.

Historical reports/results are archived unchanged under historical/. Their manifest
records file fingerprints and the snapshot checkout; missing generation commits
are explicitly unknown. Current results in results/ have their own provenance.
Checkpoint/history validation is added by the corresponding feature PRs.
