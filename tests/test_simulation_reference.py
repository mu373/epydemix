"""Saved pre-change ensembles detect shared simulation and random-stream regressions."""

import json
from pathlib import Path

import pytest

from tests.fixtures.simulation_reference import environment, reference_snapshot


def test_simulation_matches_committed_reference():
    """Compare every trajectory field and parent RNG after two successive ensembles.

    This is a compatibility check, not a Monte Carlo accuracy test. An intentional
    trial-stream redesign must fail here first and preserve the original fixture
    when updating affected arrays/RNG states. The model is documented in the fixture.
    """
    reference = json.loads(
        (Path(__file__).parent / "data/simulation_reference.json").read_text()
    )
    if environment() != reference["environment"]:
        pytest.skip(
            "Exact simulation reference requires the pinned numerical environment"
        )
    assert reference_snapshot() == reference["results"]
