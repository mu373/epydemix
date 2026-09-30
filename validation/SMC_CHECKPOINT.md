# SMC checkpoints

`ABCSampler.calibrate(strategy="smc", checkpoint_path=..., resume=...)` saves one
local checkpoint after each complete generation. Omitting `checkpoint_path`
retains the existing behavior. An explicit RNG seed or Generator is required.

This PR adds single-file memory checkpoints. Optional disk-backed history follows
in its own PR; it is not needed for parallel ABC-SMC.

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

## Per-call logging and measurements

Preparation, restore and atomic save are separate private helpers. Run metrics
report checkpoint setup/compatibility work separately from checkpoint read/write
I/O. Generation metrics contain that generation's save time. Source hash/payload
verification remains part of actual restore; no asynchronous writer is introduced.

The legacy metadata run_id identifies the checkpoint chain. Structured logging's
run_id is new for every call; checkpoint_id refers to the legacy chain ID.
checkpoint_resumed includes previous_run_id; a successful save records the current
call_run_id. The saved cumulative count covers committed generations, whereas
current-call execution_metrics counts all current work, including discarded partial
generations. Returning an already-complete saved run has zero new simulations.
Legacy cumulative elapsed time is distinct from this call's measured duration.

## Verification

Standard checkpoint tests use small deterministic/SIR models, compare every retained
field and parent RNG state, and cover cross-process resume, projections, non-default
bit generators, custom kernels, mismatches, corruption, failed publication and
partial-generation replay. Added logging tests check current-call work, stable
checkpoint identity/new call identity, complete-run zero work and failed-save events.
A longdouble epsilon case checks that helper extraction does not round saved state.

```sh
python -m pytest tests/test_abc_checkpoint.py tests/test_checkpoint_logging.py --no-cov
python -m validation.benchmark_checkpoint_memory --size-mib 32 --repeat 3
```

See CHECKPOINT_MEMORY.md for synthetic streaming measurements and their limits.
All standard tests run without validation/; detailed docstrings explain the model
and regression contract. Historical reports are preserved under historical/.
