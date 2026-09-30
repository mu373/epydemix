# Checkpoint memory measurements

## Streaming I/O

A synthetic CalibrationResults contains 64 MiB of float64 trajectories: 8 generations,
16 trajectories per generation, each 512 x 128. Each operation runs in a fresh Python
process. Save/copy start with the result already resident; load starts without it.
These measurements isolate coordinator I/O/copy behavior, not simulation or workers.

| Operation | Before: peak RSS MiB | Streaming: peak RSS MiB |
|---|---:|---:|
| Save | 458.2 | 331.1 |
| Load | 394.6 | 331.3 |
| Return-style deepcopy (unchanged control) | 394.3 | 394.2 |

The already-resident state before saving occupied about 331 MiB. The previous writer
added about 127 MiB at peak; streaming had no visible comparable increase. On load,
the roughly 64 MiB of restored trajectory data remain resident as expected.

One local run per operation, Linux/Python 3.12.11/NumPy 1.26.4, BLAS/OpenMP threads=1.
Peak is the maximum of psutil RSS before/after and Linux ru_maxrss (the counters can
differ slightly); sub-MiB differences should not be interpreted as exact allocation
counts. Data layout, object type and runtime affect buffering and peaks. This change
does not remove all-generation result retention or the normal return-value deepcopy.

Raw observations: `checkpoint_memory_streaming.json`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.benchmark_checkpoint_memory
```
