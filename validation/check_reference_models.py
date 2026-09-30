"""Check all six binomial SIR/SIS/SEIR references: python -m validation.check_reference_models.

Why: global RNG calls, mutated initial arrays and inconsistent extinction grids
make model comparisons misleading. Small one/two-group positive-population models
have exact conservation, nonnegativity and seed reproducibility invariants. These
checks test simulation contracts, not approximate equivalence to another engine.
"""

import json
import subprocess

import numpy as np

from validation.models.stochastic_seir import StochasticSEIR
from validation.models.stochastic_seir_population import StochasticSEIRAgeGroups
from validation.models.stochastic_sir import StochasticSIR
from validation.models.stochastic_sir_population import StochasticSIRAgeGroups
from validation.models.stochastic_sis import StochasticSIS
from validation.models.stochastic_sis_population import StochasticSISAgeGroups


def models():
    """Use 20 steps, population 1,000 per group and positive rates to exercise draws."""
    scalar = dict(S0=970, I0=30, beta=0.3, gamma=0.1, population=1000, time_steps=20)
    grouped = {
        **scalar,
        "S0": np.array([970, 980]),
        "I0": np.array([30, 20]),
        "population": np.array([1000, 1000]),
        "contact_matrix": np.array([[1.0, 0.2], [0.2, 1.0]]),
    }
    yield StochasticSIS(**scalar)
    yield StochasticSIR(**scalar, R0=0)
    yield StochasticSEIR(**{**scalar, "S0": 960}, E0=10, R0=0, sigma=0.2)
    yield StochasticSISAgeGroups(**grouped)
    yield StochasticSIRAgeGroups(**grouped, R0=np.zeros(2, dtype=int))
    yield StochasticSEIRAgeGroups(
        **{**grouped, "S0": np.array([960, 970])},
        E0=np.full(2, 10),
        R0=np.zeros(2, dtype=int),
        sigma=0.2,
    )


def main():
    def fail_global(*args, **kwargs):
        raise AssertionError("A reference model used global NumPy randomness")

    original = np.random.binomial
    np.random.binomial = fail_global
    try:
        for model in models():
            initial = {
                name: np.array(getattr(model, name), copy=True)
                for name in ("S0", "E0", "I0", "R0")
                if hasattr(model, name)
            }
            generator = np.random.default_rng(43)
            before = generator.bit_generator.state
            actual, expected = model.simulate(rng=generator), model.simulate(rng=43)
            assert before != generator.bit_generator.state
            total = np.zeros_like(np.asarray(actual["S"]))
            for field in actual:
                values = np.asarray(actual[field])
                np.testing.assert_array_equal(values, expected[field])
                assert values.shape[0] == 20 and np.all(values >= 0)
                total += values
            np.testing.assert_array_equal(total, np.broadcast_to(model.N, total.shape))
            for name, value in initial.items():
                np.testing.assert_array_equal(getattr(model, name), value)
            left, right = (
                model.run_simulations(4, rng=29),
                model.run_simulations(4, rng=29),
            )
            for field in left:
                for quantile in left[field]:
                    np.testing.assert_array_equal(
                        left[field][quantile], right[field][quantile]
                    )
                    assert left[field][quantile].shape[0] == 20
            # Extinction must keep the full grid rather than shortening all trials.
            model.I0 = model.I0 * 0
            if hasattr(model, "E0"):
                model.E0 = model.E0 * 0
            model.S0 = model.N - getattr(model, "R0", 0)
            assert len(model.simulate(rng=47)["S"]) == 20
            print(
                json.dumps(
                    {
                        "model": type(model).__name__,
                        "steps": 20,
                        "seeds": [43, 29, 47],
                        "passed": True,
                        "numpy": np.__version__,
                        "source_commit": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], text=True
                        ).strip(),
                    }
                )
            )
    finally:
        np.random.binomial = original


if __name__ == "__main__":
    main()
