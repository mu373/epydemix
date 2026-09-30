"""Generate a tiny pre-integration SIR ensemble reference with explicit provenance.

Why: serial/process parity can share a wrong simulation or RNG implementation.
Model: one group of 10,000 people, SIR rates beta=.3/gamma=.1, contact matrix [1],
initial S=9,900/I=100/R=0, six daily steps, three trajectories per call, two calls
using the same Generator seeded 43. No external population dataset is loaded.
Reference: immutable full trajectory arrays, dates, parameter definitions and parent
RNG state, generated before independent trial streams are introduced. Bitwise checks
use the pinned numerical environment. Parent model cache mutations are excluded:
isolation of trial-local model state is intentionally introduced by integration.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from epydemix.model.predefined_models import create_sir
from epydemix.population import Population
from tests.fixtures.abc_reference import as_json, environment


def reference_snapshot():
    """Capture two complete ensembles, preserving sequential caller RNG advancement."""
    model = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    population = Population()
    population.add_population([10000])
    population.add_contact_matrix(np.array([[1.0]]))
    model.set_population(population)
    rng = np.random.default_rng(43)
    output = []
    for _ in range(2):
        result = model.run_simulations(
            start_date="2023-01-01",
            end_date="2023-01-06",
            Nsim=3,
            rng=rng,
            initial_conditions_dict={
                "Susceptible": np.array([9900]),
                "Infected": np.array([100]),
                "Recovered": np.array([0]),
            },
        )
        trajectories = [
            {
                "dates": [str(date) for date in trajectory.dates],
                "compartment_idx": trajectory.compartment_idx,
                "transitions_idx": trajectory.transitions_idx,
                **{
                    field: getattr(trajectory, field)
                    for field in ("compartments", "transitions", "parameters")
                },
            }
            for trajectory in result.trajectories
        ]
        output.append(
            as_json(
                {
                    "trajectories": trajectories,
                    "parameters": result.parameters,
                    "rng": rng.bit_generator.state,
                }
            )
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    expected = subprocess.check_output(
        ["git", "rev-parse", args.source_commit], text=True
    ).strip()
    if commit != expected:
        parser.error("Generate from the explicitly requested source commit")
    args.output.write_text(
        json.dumps(
            {
                "source_commit": commit,
                "environment": environment(),
                "results": reference_snapshot(),
            },
            indent=2,
        )
        + "\n"
    )
