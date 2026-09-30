"""Immutable pre-refactor reference, complementing same-implementation parity.

See tests.fixtures.abc_reference for the model and explicit regeneration recipe.
Comparison is intentionally restricted to its recorded numerical environment;
the mathematical/semantic tests run across all supported versions. Six calls need
only six particles each: this tests random-stream consumption and data alignment,
not Monte Carlo accuracy. No time or simulation limit affects this workload.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.abc_reference import environment, reference_snapshot


def test_calibration_matches_committed_reference():
    """Detect shared serial/parallel regressions against independently saved output.

    Compare every generation, trajectory, projection and RNG state exactly, except
    computed weights allow four float64 ULPs for CPU-dependent PDF arithmetic.
    CI found a one-ULP difference with identical particles and RNG states. An RNG
    redesign must fail here first; update only its intentionally affected fields and
    record that change. Successful serial/parallel comparison alone cannot replace it.
    """
    reference = json.loads(
        (Path(__file__).parent / "data/abc_reference.json").read_text()
    )
    if reference["environment"] != environment():
        pytest.skip("Exact golden requires the versions in tests/data/README.md")
    actual = reference_snapshot()
    expected = reference["results"]
    for name, result in actual.items():
        for generation, weights in result["weights"].items():
            reference_weights = expected[name]["weights"][generation]
            np.testing.assert_array_max_ulp(
                weights["data"], reference_weights["data"], maxulp=4
            )
            # The remaining comparison still checks dtype, shape and every RNG draw.
            weights["data"] = reference_weights["data"]
    assert actual == expected
