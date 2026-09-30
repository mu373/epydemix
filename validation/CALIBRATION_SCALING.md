# Calibration scaling: revised DYN execution

Fresh-process measurements on AMD Ryzen 7 5825U (16 logical CPUs), Linux /
Python 3.12.11, NumPy 1.26.4, SciPy 1.16.3.
Every timing condition has three repetitions, seed 43, explicit infinite epsilon
and no binding time or simulation budget. Infinite epsilon isolates execution cost;
these measurements do not test posterior accuracy. Raw JSONL records commit,
benchmark SHA, clean source state, environment, workload, native phase/counter metrics,
and output fingerprints. The original harness commit is `5b50721`; the larger SIR
workload and manager-start profiling use `cf3e89c`.

| Model | Particles | Generations | Workers | Pool / logging | Median s | Min–max s | Median weight s |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| normal | 64 | 3 | 0 | new_owned / False | 0.0821 | 0.0814–0.0821 | 0.0423 |
| normal | 64 | 3 | 0 | new_owned / True | 0.0858 | 0.0855–0.0882 | 0.0448 |
| normal | 64 | 3 | 1 | new_owned / False | 5.2436 | 5.1370–5.2740 | 0.0478 |
| normal | 64 | 3 | 2 | new_owned / False | 5.2654 | 5.2430–5.3173 | 0.0446 |
| normal | 64 | 3 | 2 | new_owned / True | 5.3268 | 5.3009–5.3354 | 0.0473 |
| normal | 64 | 3 | 2 | reused_external / False | 3.9084 | 3.8938–3.9937 | 0.0484 |
| normal | 64 | 3 | 4 | new_owned / False | 5.4938 | 5.4403–5.6475 | 0.0461 |
| normal | 64 | 3 | 4 | reused_external / False | 3.9092 | 3.8628–3.9749 | 0.0450 |
| normal | 64 | 5 | 0 | new_owned / False | 0.1523 | 0.1508–0.1536 | 0.0875 |
| normal | 64 | 5 | 2 | new_owned / False | 7.9534 | 7.9530–8.0226 | 0.0896 |
| normal | 128 | 3 | 0 | new_owned / False | 0.2296 | 0.2209–0.2311 | 0.1597 |
| normal | 128 | 3 | 2 | new_owned / False | 5.4621 | 5.4402–5.5103 | 0.1584 |
| sir | 32 | 3 | 0 | new_owned / False | 0.1596 | 0.1565–0.1756 | 0.0137 |
| sir | 32 | 3 | 2 | new_owned / False | 5.3407 | 5.2661–5.3905 | 0.0146 |
| sir | 32 | 3 | 4 | new_owned / False | 5.4793 | 5.3871–5.5363 | 0.0145 |
| sir_large | 256 | 3 | 0 | new_owned / False | 14.9266 | 14.6109–14.9913 | 0.5669 |
| sir_large | 256 | 3 | 2 | new_owned / False | 13.9685 | 13.8725–14.6237 | 0.5966 |
| sir_large | 256 | 3 | 4 | new_owned / False | 10.5221 | 10.3984–10.5296 | 0.5946 |

Worker 0 is sequential. Normal is one Normal(mu,1) draw; SIR uses one synthetic
group of 10,000 people over 10 daily points. sir_large uses 16 such groups over
181 points, with contact matrix diagonal 1 and off-diagonal .2. Both calibrate only
transmission rate; only total incidence is retained. Reused pools receive an
identical full calibration as warm-up, outside the measured interval; warm-up time
is reported separately. This external-pool path sends full inputs, so it is also
distinct from the owned-pool fixed-input cache path. Pool shutdown is included for
owned pools and excluded for caller-owned pools. Final calibrate deepcopy is included.

All retained posterior/weight/distance/trajectory fingerprints agree within each
model/particle/generation condition, including logging and pool-mode variants.
DYN physical simulations can differ because of surplus work. Compare the native
simulation counts separately. The larger SIR example benefits from four workers;
cheap normal and small SIR examples are dominated by startup/coordination overhead.
These medians and ranges are local observations, not confidence intervals or a
general speedup promise. Increasing particles exposes quadratic importance-weight
cost; adding generations also creates further DYN manager lifetimes.

Peak memory is sampled aggregate process-tree RSS during calibration; shared pages
can be counted repeatedly and short peaks can be missed. Lifetime CPU samples also
include child imports. Fingerprint/readback is outside the timed memory window.
Profiles are separate from timing rows; per-worker trace writes and hooks add cost.

```sh
python -m validation.benchmark_parallel --model normal --workers 0 1 2 4 --particles 64 --generations 3 --repeat 3
python -m validation.benchmark_parallel --model normal --workers 0 2 --particles 128 --generations 3 --repeat 3
python -m validation.benchmark_parallel --model normal --workers 0 2 --particles 64 --generations 5 --repeat 3
python -m validation.benchmark_parallel --model sir --workers 0 2 4 --particles 32 --generations 3 --repeat 3
python -m validation.benchmark_parallel --model sir_large --workers 0 2 4 --particles 256 --generations 3 --repeat 3
python -m validation.benchmark_parallel --workers 2 4 --particles 64 --generations 3 --repeat 3 --reuse-pool
python -m validation.benchmark_parallel --workers 0 2 --particles 64 --generations 3 --repeat 3 --log-json
```

For cache before/after evidence, use [WORKER_INPUT_CACHE.md](WORKER_INPUT_CACHE.md).
Prior-series reports/results are preserved unchanged under historical/, with a
snapshot manifest and explicit unknown provenance where the old files lacked it.
