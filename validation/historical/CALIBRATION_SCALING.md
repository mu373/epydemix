# Calibration scaling measurements

Historical baseline before worker input caching. See [WORKER_INPUT_CACHE.md](WORKER_INPUT_CACHE.md) for the subsequent optimization. The commands below use `--uncached` with the current harness to select this original dispatch path.

Measured 2026-09-27 against calibration code at `8a41de63c23021f1fc42de1cbc4ec7fdc08138ec`.
Only the benchmark harness changed for this audit; calibration implementation and defaults were unchanged.

## Conditions

- AMD Ryzen 7 5825U: 8 physical cores, 16 logical CPUs, detected worker capacity 16.
- Linux, Python 3.12.11; BLAS/OpenMP/Numba thread environment limits set to 1.
- Real stochastic SIR simulator, **synthetic observations**, 16 demographic groups, 180 daily steps, two calibrated parameters.
- Calibration seed 43; observation seed 2026. Adaptive epsilon uses the previous generation's median distance. No calibration time or simulation budget cutoff.
- Each accepted simulation retains **all compartment and transition arrays**, plus aggregate incidence used as the target. Roughly 120 KiB of numeric output per particle.
- Each condition ran **twice** in fresh subprocesses; tables show medians. Worker order rotated between repetitions. The 36 runs were executed sequentially, without overlapping benchmark jobs.
- Time includes pool startup/imports, calibration, checkpoint writes where enabled, return-value copying and pool shutdown. Coordinator imports, model construction, observation generation and simulator warmup are excluded.
- Memory is sampled peak **aggregate process-tree PSS**, including coordinator and workers. PSS apportions shared pages. Sampling interval is about 25 ms plus inspection overhead; brief peaks may be missed. OS file cache is not included.
- Numerical readback/hashing occurs after calibration and outside its memory sampling window. Disk readback uses the normal OS page cache; this is not a cold-storage test. Checkpoints use local NVMe/ext4 storage under `/tmp` and are removed after measurement.
- These are exploratory local measurements, not confidence intervals or a Jupyter end-to-end benchmark. Start method, model cost, returned data and host load affect scaling.

## Worker scaling: 96 particles × 3 generations

Checkpoint disabled; history retained in memory. Every run evaluates exactly **504 candidates**.

| Workers | fork seconds | fork speedup | fork PSS MiB | spawn seconds | spawn speedup | spawn PSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 17.70 | 1.00× | 262.8 | 17.52 | 1.00× | 263.5 |
| 1 | 18.32 | 0.97× | 269.0 | 19.42 | 0.90× | 396.1 |
| 2 | 9.73 | 1.82× | 269.8 | 10.96 | 1.60× | 533.5 |
| 4 | 5.63 | 3.14× | 281.7 | 6.99 | 2.51× | 795.5 |
| 8 | 4.26 | 4.16× | 317.6 | 5.99 | 2.92× | 1304.2 |
| 16 | 3.97 | 4.46× | 384.2 | 7.04 | 2.49× | 2307.4 |

Four workers give about 3.14× speedup with fork and 2.51× with spawn. Eight workers give 4.16× and 2.92× respectively.
For fork, doubling from 8 to 16 workers cuts time by only 7%; for spawn it **increases time by 17%**, with PSS rising from 1.27 to 2.25 GiB.
Both repetitions show the same ordering: spawn 8 workers takes 5.89–6.10 s versus 7.01–7.07 s for 16.
For this workload, 4–8 workers is a practical starting range. Choosing the maximum logical CPU count does not guarantee the best performance.
These standalone measurements do not establish fork safety inside a threaded notebook.

## Particle scaling: 3 generations, fork, no checkpoint

| Particles | Candidates | Sequential seconds | 4-worker seconds | 4-worker PSS MiB |
|---|---:|---:|---:|---:|
| 96 | 504 | 17.70 | 5.63 | 281.7 |
| 192 | 1044 | 36.44 | 11.92 | 357.0 |

Doubling particles increases sequential time 2.06× and four-worker time 2.12×. Candidate count rises 2.07×; adaptive epsilon means it need not scale exactly with particle count.
At 192 particles, time outside particle collection is about 0.81 s of the 11.92 s four-worker total. This includes weight calculation, result assembly/copying and shutdown, so it is not a dedicated weight profile.
The first serialized candidate argument is about 797 KiB; fixed model/data arguments are sent per candidate. IPC and coordinator work remain possible limits at larger scales, but this audit does not isolate their contributions.

## History storage and generation count

96 particles, 4 fork workers, checkpoint every completed generation. Both storage modes retain the same numerical results.
`memory` keeps history in RAM and snapshots it into the checkpoint; `disk` keeps completed-generation history in sibling files.

| Generations | Storage | Seconds | Peak tree PSS MiB | Final bundle MiB | Read + hash seconds |
|---|---|---:|---:|---:|---:|
| 3 | memory | 6.15 | 337.8 | 35.83 | 0.110 |
| 3 | disk | 5.95 | 263.1 | 35.85 | 0.412 |
| 6 | memory | 22.12 | 397.7 | 70.87 | 0.226 |
| 6 | disk | 21.01 | 307.1 | 70.90 | 0.802 |

At six generations, disk history reduces measured peak process memory by about **23%**, without a timing penalty in these runs. Its main checkpoint is about 0.80 MiB and generation files total 70.11 MiB. Bundle size is the final on-disk footprint, not cumulative bytes written.
Disk readback is slower than reading existing arrays, although the complete six-generation read/hash still takes about 0.80 s here.

**Disk mode does not make total process memory constant:** measured PSS rises from 263 to 307 MiB between 3 and 6 generations. Inputs, current generation, workers, allocator state and reference metadata still occupy memory; the cause of this increase was not profiled. This short experiment does not establish an upper memory bound for long runs.

Generation count also does not imply linear runtime. Candidate counts per generation are **96, 156, 252, 277, 415, 711**. Three generations require 504 simulations; six require 1,907 (3.78×). The last generation accepts only 96/711 = 13.5% of candidates. Tighter adaptive epsilon is an important part of the runtime growth.

## Correctness and scope

All **36 runs completed**. Within each identical `(particles, generations)` condition, hashes of ordered posterior parameters, weights, distances and every returned trajectory array matched exactly across repetitions, worker counts, start methods and storage modes wherever measured. Candidate counts also matched.
Groups checked: 96×3 (28 runs), 192×3 (4 runs), 96×6 (4 runs).
Different particle/generation counts are different experiments and are not claimed to produce identical outputs.

No production-data calibration, very large particle population, long-run memory bound, distributed execution or notebook UI/kernel timing is established by these results. The current practical takeaway is to start at 4–8 workers and use disk history when retaining many generations or large trajectories; no new optimization was added.

## Reproduce

Run from the repository root. `psutil` is needed by the audit harness.
The harness asserts matching numerical hashes and simulation counts within each invocation.
The fingerprint now includes all returned trajectory fields; its digest format differs from the older parallel benchmark reports.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for method in fork spawn; do
  python -m validation.benchmark_parallel --uncached --cases large --workers 0 1 2 4 8 16 --particles 96 --full-trajectory --repeats 2 --start-method "$method" --output "validation/scale_workers_${method}.jsonl"
done
python -m validation.benchmark_parallel --uncached --cases large --workers 0 4 --particles 192 --full-trajectory --repeats 2 --start-method fork --output validation/scale_particles.jsonl
for generations in 3 6; do
  for storage in memory disk; do
    python -m validation.benchmark_parallel --uncached --cases large --workers 4 --particles 96 --generations "$generations" --storage "$storage" --full-trajectory --repeats 2 --start-method fork --output "validation/scale_storage_${storage}_${generations}.jsonl"
  done
done
```

Cross-file numerical check:

```python
import json
from pathlib import Path

rows = [json.loads(line)
        for path in Path("validation").glob("scale_*.jsonl")
        for line in path.read_text().splitlines()[1:]]
assert len(rows) == 36
for key in {(r["particles"], r["generations"]) for r in rows}:
    group = [r for r in rows if (r["particles"], r["generations"]) == key]
    assert len({(r["fingerprint"], r["simulations"]) for r in group}) == 1
```
