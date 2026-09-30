"""Immutable pre-refactor reference, complementing same-implementation parity.

See tests.fixtures.abc_reference for the model and explicit regeneration recipe.
Exact comparison is intentionally restricted to its recorded numerical environment;
the mathematical/semantic tests run across all supported versions. Six calls need
only six particles each: this tests random-stream consumption and data alignment,
not Monte Carlo accuracy. No time or simulation limit affects this workload.
"""

import json
from pathlib import Path

import pytest

from tests.fixtures.abc_reference import environment, reference_snapshot


def test_calibration_matches_committed_reference():
    """Detect shared serial/parallel regressions against independently saved output.

    Compare every generation, trajectory, projection and RNG state exactly. An RNG
    redesign must fail here first; update only its intentionally affected fields and
    record that change. Successful serial/parallel comparison alone cannot replace it.
    """
    reference = json.loads(
        (Path(__file__).parent / "data/abc_reference.json").read_text()
    )
    if reference["environment"] != environment():
        pytest.skip("Exact golden requires the versions in tests/data/README.md")
    assert reference_snapshot() == reference["results"]
