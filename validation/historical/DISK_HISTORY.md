# Disk-backed SMC history

Use `history_storage="disk"` with `checkpoint_path` to keep completed-generation
arrays on disk. The default remains `"memory"`; rejection, top-fraction and
projection storage are unchanged.

```python
results = sampler.calibrate(
    strategy="smc", num_particles=100, num_generations=10,
    checkpoint_path="run.checkpoint", history_storage="disk", n_workers=2,
)

# Load just generation 0's selected trajectories; other generations stay on disk.
trajectories = results.get_selected_trajectories(generation=0)
posterior = results.get_posterior_distribution()  # latest generation only

# Reconstruct a sampler with the same inputs/settings and resume with the same mode.
results = sampler.calibrate(
    strategy="smc", num_particles=100, num_generations=10,
    checkpoint_path="run.checkpoint", history_storage="disk", resume=True,
)
```

## Storage and memory

The bundle consists of `run.checkpoint` and its sibling directory
`run.checkpoint.history/`. Each generation creates four immutable files: particles,
weights, distances, and trajectories. Each field uses the existing checkpoint
writer/reader, input and payload checksums, and atomic file publication.

The main checkpoint retains the actual input snapshot, RNG/kernel state and
progress, plus filenames and SHA-256 references for history. Saving it does not
reread or rewrite older generation arrays. SMC retains the current particles,
weights and distances for proposal generation; trajectories are released after
saving. Memory for trajectory history no longer grows with generation count.
File-reference metadata still grows with the number of generations.

Existing result getters work. Per-generation attributes such as
`selected_trajectories` are now **read-only Mapping objects** in disk mode. Indexing
loads one field of one generation without caching. Mutating a returned array or
DataFrame changes that loaded object, not the saved file. Holding many loaded
values, calling `dict(mapping)`, or stacking trajectories can still use substantial
memory. Loading trajectories fetches all variables for that selected generation.

`deepcopy(results)` copies file references and ordinary metadata without loading
history. Thus the existing return-value deepcopy does not duplicate all historical
trajectory arrays in this mode. Inputs, current-generation calculation, workers,
and user-retained outputs still consume memory. Projections remain in memory.

## Failure and relocation

New immutable field files are published first. Only after all four are complete
are their references added to the next main checkpoint. A failed field save,
main-checkpoint replacement, or simulation leaves the previous checkpoint and its
referenced generations intact. Retrying uses new filenames, preserving previously
returned results as snapshots of their completed generations.

Failures can leave unreferenced files; automatic garbage collection is deliberately
not provided because other saved result objects may still refer to older files.
Use one writer per checkpoint path. Keep the bundle for as long as its result
objects are needed. Deleting it invalidates those objects' history access.

To move or rename a run, move the main file and rename its sibling history directory
to `<new checkpoint path>.history`. `resume=True` rebinds references to the new
location and checks for missing generation files before restoring RNG state.
Field identity and data checksums are verified when fields are loaded. Copying only
the main checkpoint or pickling only the result object does not copy history.

Resume requires the same storage mode. Migration/compatibility with earlier draft
formats is not a supported contract. Only load trusted bundles: generation files,
like the main checkpoint, contain pickle data.

## Verification and measured memory

Tests compare disk results with normal in-memory SIR calibration, including a
fresh interpreter resuming with spawn workers and subsequent projections. They
also check array release using weak references, no eager reads during deepcopy or
checkpoint load, no hidden read cache, old-generation getters/quantiles, failed
field/main saves, interrupted simulation, partial-budget replay, storage mismatch,
relocation, missing files, and mismatched generation-file identity.

The related suite passes **161 tests** (20 checkpoint tests plus 9 disk-history
checks, alongside the existing calibration/parallel/resource tests).

Synthetic SMC workload: 8 particles, 2 MiB of numeric payload per simulation,
16 MiB of accepted trajectories per generation, always-accept epsilon schedule,
sequential coordinator to isolate storage effects. Each row runs in a fresh process.

| Generations | Total trajectory data | Memory mode peak RSS | Disk mode peak RSS |
|---|---:|---:|---:|
| 4 | 64 MiB | 395.2 MiB | 281.9 MiB |
| 16 | 256 MiB | 780.3 MiB | 283.1 MiB |

Memory mode includes the normal return-value deepcopy. Disk mode's RSS after return
was about 267 MiB in both cases; its main checkpoint was only 15–22 KiB, with the
64/256 MiB of trajectory history on disk. Fixed inputs/models in real workloads can
make that main checkpoint substantially larger.

One local run per condition, Linux/Python 3.12.11, BLAS/OpenMP threads=1. Peak is the
maximum of before/after RSS and Linux ru_maxrss; these counters differ slightly.
This is a retention measurement with synthetic payloads, not a timing or complete
process-tree memory benchmark. Raw observations: `history_memory.json`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.benchmark_history_memory
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_disk_history.py tests/test_abc_checkpoint.py -o addopts='-q'
```

This feature uses the pre-existing checkpoint I/O interface and can be applied
without the separate streaming-I/O change. Without streaming I/O, per-generation
serialization has a higher temporary peak, but older generations still stay on disk.
