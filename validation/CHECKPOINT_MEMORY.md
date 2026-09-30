# Checkpoint streaming measurements

Clean source `480ae0e`, Linux / Python 3.12.11 / NumPy 1.26.4. Synthetic constant
float64 trajectories total exactly 32 MiB in eight generations. Save/load/copy
each run three times in fresh interpreters; metadata records benchmark SHA,
actual payload/checkpoint size, source, environment and matching output hashes.
Operation time and RSS measurements exclude later fingerprint computation.

| Operation | Median s | Min–max s | Median peak RSS MiB | Peak RSS min–max MiB |
| --- | ---: | --- | ---: | --- |
| save | 0.0515 | 0.0487–0.0548 | 299.8 | 299.7–300.0 |
| load | 0.0944 | 0.0936–0.0944 | 306.6 | 306.5–306.7 |
| copy | 0.0188 | 0.0144–0.0240 | 331.9 | 331.7–331.9 |

```sh
python -m validation.benchmark_checkpoint_memory --size-mib 32 --repeat 3
```

Peak RSS includes imports/process baseline; it is not incremental allocation.
Load verifies the archive before unpickling; save uses streaming pickle and atomic
publication. Copy intentionally duplicates the in-memory state. This synthetic
measurement checks transport/streaming cost, not full epidemic calibration or a
speedup against a previous implementation. Standard tests verify corruption and
publication failure semantics separately. Initial 8 MiB smoke remains recorded in
results/checkpoint-streaming-smoke.jsonl; old-series results are in historical/.
Trust boundaries and resume semantics are in SMC_CHECKPOINT.md.
