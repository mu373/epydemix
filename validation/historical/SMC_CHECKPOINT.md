# SMC checkpoints

`ABCSampler.calibrate(strategy="smc", checkpoint_path=..., resume=...)` saves one
local checkpoint after each complete generation. Omitting `checkpoint_path`
retains the existing behavior. An explicit RNG seed or Generator is required.

For optional disk-backed generation history, see [DISK_HISTORY.md](DISK_HISTORY.md).
The single-file/full-history description below applies to the default memory mode.

```python
# Define simulation_function at module scope; use parameters["rng"] inside it.
# Keep this code inside a main guard when starting process workers with spawn.
sampler = ABCSampler(simulation_function, priors, parameters, observed_data, rng=42)
result = sampler.calibrate(
    strategy="smc", num_particles=100, num_generations=10,
    checkpoint_path="run.checkpoint", n_workers=2,
)

# After an interruption, construct a sampler with the same inputs and settings.
# The saved RNG state overrides the seed supplied to this fresh sampler.
sampler = ABCSampler(simulation_function, priors, parameters, observed_data, rng=42)
result = sampler.calibrate(
    strategy="smc", num_particles=100, num_generations=10,
    checkpoint_path="run.checkpoint", resume=True, n_workers=4,
)
```

Worker counts must fit the detected CPU capacity. The parent directory must exist.
A fresh run rejects an existing checkpoint instead of overwriting it. Reusing a
path with `resume=True` loads that file and replaces it after each new generation.
Use one writer per path. There is no background checkpoint writer or worker-state
snapshot.

## Contents

The file is a ZIP archive containing:

- `metadata.json`: format version, run ID, creation/update timestamps, SMC settings,
  completed generation (zero-based), simulation count, elapsed time, runtime
  limits/worker count, Python/package versions, calibration source SHA-256 hashes,
  input SHA-256 and serialized-state SHA-256. Source hashes include local edits.
- `state.pkl`: **actual input data**, not just identifiers: observed data, fixed
  parameters (including the EpiModel/population when supplied), priors, parameter
  order, settings, and initial kernel configuration. Also includes all completed
  generations' particles, weights, distances and simulation trajectories, current
  perturbation-kernel state, next generation, cumulative simulation count,
  elapsed time and the sampler's RNG state/type/SeedSequence.

The top-level `parameters["rng"]` is represented by the separately saved sampler
RNG, which controls seeded simulations. Input data are copied before computation,
so sequential simulations cannot mutate the saved initial input snapshot.
EpiModel's derived `definitions` and `Cs` caches are excluded from the input hash:
`simulate()` rebuilds them. The actual snapshot still contains the supplied model.

The SeedSequence is saved separately because pickling a NumPy Generator preserves
its draw state but does not necessarily preserve its original seed entropy. Both
are needed to reproduce calibration **and subsequent paired projections**.

Metadata can be inspected without unpickling:

```python
import json
from zipfile import ZipFile

with ZipFile("run.checkpoint") as archive:
    metadata = json.loads(archive.read("metadata.json"))
print(metadata)
```

## Matching and integrity

Resume compares input/configuration SHA-256 before changing sampler RNG state or
submitting simulations. Dictionary order is normalized. Arrays include dtype,
shape and C-order values. Supported inputs include normal Python containers,
NumPy arrays, pandas tables, SciPy frozen priors, EpiModel/Population, and plain
objects with serializable attribute dictionaries. Unsupported or cyclic inputs
fail before calculation. Functions must be pickleable/importable; closures and
external files/global state are not captured or inspected.

Particle count, prior definitions, observations, fixed parameters, parameter order,
epsilon policy, minimum epsilon and initial kernel configuration must match.
Target generation count, total simulation budget, per-call time limit, verbosity
and worker count may change. Generation count and budget cannot be below progress
already saved. Changing the input seed is allowed because saved RNG state is used.
Code/package differences emit a warning; exact replay is only promised in the same
numerical environment with RNG-respecting, independent simulation callbacks.

The serialized-state checksum is checked **before unpickling**. The restored input
snapshot is also checked against its input checksum. These are corruption checks,
not authentication: only load checkpoints you created or otherwise trust. They
contain pickle data, which can execute code when loaded.

## Interruption semantics and limits

- An incomplete generation is discarded and rerun from the last saved boundary.
  RNG and kernel state restore to that boundary. The saved simulation budget/count
  covers completed generations; wasted work after the last save is not counted on
  resume. Physical total simulations can therefore exceed the saved counter.
- `max_time` is a fresh soft submission deadline for each call. It drains submitted
  tasks; checkpointing does not terminate hung workers. A checkpoint is written
  after a generation completes, even if saving extends beyond the deadline.
- Elapsed time records accumulated computation time up to each save. It excludes
  loading/validation, the current save, and lost work after the previous save.
- No checkpoint is produced until generation zero completes. Resuming a finished
  run returns saved results without generating new candidates. Minimum-epsilon
  convergence and exhausted budgets still stop a resumed call.
- A sibling temporary file is flushed and fsynced before atomic publication.
  Serialization/write/replace failures preserve the previous checkpoint. An abrupt
  kill may leave an unused temporary file, but the published checkpoint stays whole.
- Each save still rewrites all completed results, but protocol-5 pickle data are
  streamed into the archive instead of creating a whole serialized copy in memory.
  Loading verifies the payload in 1 MiB chunks before opening it again for streaming
  unpickling. Individual objects
  may still need serialization buffers; result history itself stays in memory.
  Incremental storage/compression and retention policies are separate changes. There is no guarantee against storage-device
  loss, and no multi-writer locking.

## Verification

Related suite: **150 passed** (including 18 checkpoint tests).

The checkpoint tests run an actual SIR calibration against synthetic observations,
then resume in a fresh Python interpreter with spawn workers and a different worker
count. Ordered particles, weights, distances, all trajectories, final RNG state,
and subsequent projections match uninterrupted execution exactly. Additional tests
cover Philox, a stateful custom kernel, immutable input snapshots, configuration
mismatches, array dtype/shape/value hashes, corrupt payload detection before
unpickling, failed atomic replacement, simulation failure, partial-budget replay,
finished runs and fresh time allowances.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_abc_checkpoint.py -o addopts='-q'
```
