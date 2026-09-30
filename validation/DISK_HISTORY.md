# Disk history audit

The current implementation stores references to independently checksummed field
archives and loads a generation only when accessed. Run metrics include history
writes and resume reads. Returned result mappings and deepcopy do not eagerly load
trajectory arrays. Missing fields, wrong payload identity and partial publication
are covered by `tests/test_disk_history.py`.

Fresh-process smoke, Linux/Python 3.12.11/NumPy 1.26.4, seed 43, eight particles,
1 MiB constant payload per particle, infinite explicit thresholds:

| Generations | Storage | Calibration seconds | Whole-process peak RSS MiB | Physical simulations |
| --- | --- | ---: | ---: | ---: |
| 3 | memory | 0.142 | 315.8 | 24 |
| 3 | disk | 0.113 | 274.8 | 24 |
| 6 | memory | 0.391 | 363.7 | 48 |
| 6 | disk | 0.177 | 275.5 | 48 |

Memory/disk mathematical output fingerprints match at each generation count.
These are single observations, not repeated speed estimates. Peak RSS includes
imports and the process baseline; fingerprint readback occurs after measurement.
Disk history keeps the measured peak nearly unchanged here when generation count
doubles, while memory retains every generation. This small synthetic workload does
not establish a universal memory bound or speedup.

Raw records in `results/history-memory-smoke.jsonl` include source commit, the
pre-commit tracked diff, environment, workload and benchmark SHA. Reproduce:

```sh
python -m validation.benchmark_history_memory --generations 3 6 --payload-mib 1
```

Keep the checkpoint and its `.history` directory together. Legacy version-one
memory checkpoints without `history_storage` remain readable as memory mode;
switching an existing saved run between memory and disk is rejected.
