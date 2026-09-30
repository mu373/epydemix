# Statistical calibration checks

The models and finite-epsilon target derivations are documented in
`tests/fixtures/statistical_models.py`; the test's purpose and acceptance criteria
are in `tests/test_abc_statistical.py`. These checks complement exact golden data
and serial/parallel equality: the oracle does not call the calibration algorithm.

## Design and limits

Each run retains 512 particles. SMC uses three fixed generations; rejection uses
the final epsilon. Seeds 11, 29 and 47 were fixed before measuring results. The
independent pilot used seeds 101, 211 and 307, also without selecting successful
seeds or retrying failures. Normal schedules are [1,.5,.25]; discrete schedules
are [.5,.5,.5]. Observation is always 1, with an injected simulation RNG.

For N=512, an IID mean standard error for the normal target is about .03-.034,
and a central empirical CDF standard error is at most .022. SMC particles share
ancestors and unequal weights, so these IID values explain scale only. RMS bounds
across three independent runs are .10 (mean), .16 (variance), .08 (CDF), and .06
(state probabilities). Every run must also stay below twice these bounds. Errors
are squared before aggregation, so opposing errors cannot cancel. Variance uses
the second central moment of the normalized empirical measure.

The bounds allow dependence and multiple metrics/generations. They detect material
bias, not arbitrarily small distribution changes, and are not a formal equivalence
test. More generations do not make a population of 512 particles equivalent to
thousands of independent draws. Fast tests separately cover deterministic arithmetic,
thresholds, seed plumbing, ordering, and execution-mode equality.

## Baseline pilot evidence

Production source: `7fc22d1`; environment and full generation-level errors are in
`statistical_pilot.json`. The ordinary slow suite passed four tests on this source.
The largest RMS errors across the three SMC generations in the independent pilot:

| Model/metric | Observed RMS maximum | Bound |
| --- | ---: | ---: |
| Normal mean | .04688 | .10 |
| Normal variance | .03508 | .16 |
| Normal CDF, three fixed points | .03481 | .08 |
| Discrete probabilities | .04051 | .06 |

Negative controls reuse those populations while discarding their importance weights.
The normal second-generation mean RMS error rose to .31559; discrete probability
errors rose above .12. Both violate the unchanged bounds. Returning the prior
also violates the bounds for both models. A small deterministic simulation-stream
test covers a frozen/reseeded RNG, which need not be diagnosed statistically.

Reproduce the pilot, including its negative controls, from the recorded source:

```sh
python -m tests.fixtures.statistical_pilot tests/data/statistical_pilot.json
python -m pytest tests/test_abc_statistical.py --no-cov
```

Later RNG redesigns change empirical results but not the mathematical targets or
the preselected seeds. Record new evidence separately; do not erase baseline data
or relax bounds merely to pass a changed implementation.
