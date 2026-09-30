# Calibration metrics and structured logging

Every calibration call records coarse execution metrics at
`results.calibration_params["execution_metrics"]`. These are available with
logging disabled and `verbose=False`. The existing human display remains
controlled by `verbose`; application logging levels are independent.

Configure JSON Lines in the application's launcher, before calibration:

```python
import logging
import sys
from epydemix._logging import JSONFormatter

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("epydemix.calibration")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Set verbose=False when stdout must contain only machine-readable JSON.
results = sampler.calibrate(num_particles=100, num_generations=3, verbose=False)
```

Cloud services can collect the stream through their normal stdout capture. The
library does not configure root handlers or require a cloud SDK. Configure a
handler once per application; installing it repeatedly duplicates output.

Events are `run_started`, throttled `progress`, `generation_finished`, and exactly
one `run_finished` or `run_failed`. They originate in the calling process.
Timestamps use UTC; durations use a monotonic clock. A new UUID identifies each
invocation without advancing the model RNG. Infinite thresholds are JSON strings
`"Infinity"` or `"-Infinity"`. Model parameters, observations and trajectories are
never event fields. Failures log the exception type, then propagate the original
exception; their physical count may be unknown and is not invented.

## Count definitions

Generation and batch entries are under `generations`, using zero-based generation
IDs. `totals` sum this invocation's entries, including an incomplete SMC generation:

| Field | Meaning |
| --- | --- |
| `simulations` | All physical completed evaluations, including rejection and drain |
| `matched` | Threshold matches; for top-fraction, candidates passing the final quantile |
| `retained` | Particles committed to this call's results; zero for an incomplete SMC generation |
| `surplus_accepted` | Matching DYN candidates beyond the requested particle count |
| `drained` | Completions reported after the scheduler first observes a cutoff |

Rejected candidates are not automatically surplus. Drained candidates can displace
later candidate IDs and therefore still be retained. `retained` totals sum all
completed generations, not just the final posterior size. A progress event's
`collected` count is provisional until a generation is committed. Threshold
acceptance rate is `matched / simulations`; retained throughput can instead use
`retained / elapsed_seconds` with an explicit time scope.

`stop_reason` distinguishes `completed`, `target`, `fixed_count`, `minimum_epsilon`,
`budget`, `deadline`, or exhausted inputs. The run's worker count is populated
once the pool is validated (`None` for sequential execution); it can be unknown
in an early start/failure event. `parallel_strategy` remains `dynamic` for SMC and
rejection, and is `fixed` for top-fraction execution.

## Time scopes

- `elapsed_seconds` includes the public run method's setup and owned pool lifetime;
  `calibrate()`'s final deepcopy and subsequent projections are outside that scope.
- `generation_seconds` covers generation preparation, collection and result
  construction; collection and importance-weight time are also reported separately.
- `collection_seconds` includes candidate generation and evaluation. SMC creates
  its pool once before the generations; rejection/top-fraction collection includes
  their owned pool setup and teardown.
- `weights_seconds` is the SMC importance-weight calculation, zero for strategies
  with uniform weights. Constructor duration is not reported as worker startup time.

Timing and physical surplus can vary with logging and OS scheduling. Retained
posterior and RNG parity are verified without binding wall-time/budget cutoffs.
Detailed transport/worker/memory profiles remain validation-only measurements.
