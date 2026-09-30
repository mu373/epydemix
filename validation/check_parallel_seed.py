"""Audit exact seed reproducibility: python -m validation.check_parallel_seed.

Asserts exact equality across worker counts and a different-seed negative control.
Defaults to spawn to exercise serialization as well as the numerical outputs.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess

import numpy as np
from scipy import stats

from epydemix._execution import _get_available_cpu_count
from epydemix.calibration.abc import ABCSampler
from epydemix.model import simulate
from epydemix.model.predefined_models import create_sir
from epydemix.population import Population


def deterministic(parameters):
    return {"data": parameters["transmission_rate"] * np.arange(10)}


def stochastic(parameters):
    return {"data": simulate(**parameters).transitions["Susceptible_to_Infected_total"]}


def make_sampler(model, seed=43, seed_source="rng"):
    parameters = {}
    if model == "sir":
        epimodel = create_sir(transmission_rate=0.3, recovery_rate=0.1)
        population = Population()
        population.add_population([10000])
        population.add_contact_matrix(np.array([[1.0]]))
        epimodel.set_population(population)
        parameters = dict(
            epimodel=epimodel,
            start_date="2023-01-01",
            end_date="2023-01-10",
            initial_conditions_dict={
                "Susceptible": np.array([9900]),
                "Infected": np.array([100]),
                "Recovered": np.array([0]),
            },
        )
    if seed_source == "parameters":
        parameters["rng"] = seed
    return ABCSampler(
        simulation_function=stochastic if model == "sir" else deterministic,
        priors={"transmission_rate": stats.uniform(0.1, 0.4)},
        parameters=parameters,
        observed_data=np.arange(10, dtype=float),
        rng=seed if seed_source == "rng" else None,
    )


def run(model, strategy, workers, seed=43, seed_source="rng"):
    sampler = make_sampler(model, seed, seed_source)
    if strategy == "projections":
        # Every execution mode starts from exactly the same calibrated posterior.
        sampler.calibrate(
            strategy="rejection", epsilon=1000, num_particles=8, verbose=False
        )
        result = sampler.run_projections(
            sampler.parameters, iterations=8, rng=seed, n_workers=workers
        )
        return {
            "parameters": result.projection_parameters["baseline"].to_numpy(),
            "trajectories": result.get_projection_trajectories()["data"],
            "rng_state": np.asarray(
                json.dumps(sampler.rng.bit_generator.state, sort_keys=True)
            ),
        }
    options = {
        "smc": dict(num_particles=8, num_generations=2),
        "rejection": dict(num_particles=8, epsilon=40 if model == "sir" else 4),
        "top_fraction": dict(Nsim=24, top_fraction=0.5),
    }
    result = sampler.calibrate(
        strategy=strategy, n_workers=workers, verbose=False, **options[strategy]
    )
    assert len(result.posterior_distributions) == (2 if strategy == "smc" else 1)
    outputs = {
        f"{generation}/{field}": np.asarray(values)
        for generation in result.posterior_distributions
        for field, values in {
            "parameters": result.posterior_distributions[generation].to_numpy(),
            "distances": result.distances[generation],
            "weights": result.weights[generation],
            "trajectories": result.get_calibration_trajectories(generation)["data"],
        }.items()
    }
    outputs["rng_state"] = np.asarray(
        json.dumps(sampler.rng.bit_generator.state, sort_keys=True)
    )
    return outputs


def differences(left, right):
    assert left.keys() == right.keys()
    return [key for key in left if not np.array_equal(left[key], right[key])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-method", choices=mp.get_all_start_methods(), default="spawn"
    )
    capacity = _get_available_cpu_count()
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[0, *[n for n in (1, 2, 4) if n <= capacity]],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["deterministic", "sir"],
        default=["deterministic", "sir"],
    )
    parser.add_argument(
        "--seed-sources",
        nargs="+",
        choices=["rng", "parameters"],
        default=["rng", "parameters"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["smc", "rejection", "top_fraction", "projections"],
        default=["smc", "rejection", "top_fraction", "projections"],
    )
    args = parser.parse_args()
    mp.set_start_method(args.start_method)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    for model in args.models:
        for source in args.seed_sources:
            for strategy in args.strategies:
                reference = run(model, strategy, None, seed_source=source)
                assert differences(
                    reference, run(model, strategy, None, seed=44, seed_source=source)
                ), (model, source, strategy, "different seed")
                for worker in args.workers:
                    first = run(model, strategy, worker or None, seed_source=source)
                    second = run(model, strategy, worker or None, seed_source=source)
                    assert not differences(first, second), (
                        model,
                        source,
                        strategy,
                        worker,
                        "repeat",
                    )
                    assert not differences(reference, first), (
                        model,
                        source,
                        strategy,
                        worker,
                        "sequential",
                    )
                    digest = hashlib.sha256()
                    for key, value in first.items():
                        digest.update(key.encode())
                        digest.update(value.tobytes())
                    print(
                        json.dumps(
                            dict(
                                model=model,
                                seed_source=source,
                                strategy=strategy,
                                workers=worker,
                                seed=43,
                                negative_control_seed=44,
                                exact_repeat=True,
                                exact_sequential=True,
                                includes_parent_rng=True,
                                fingerprint=digest.hexdigest(),
                                start_method=args.start_method,
                                numpy=np.__version__,
                                source_commit=source_commit,
                            )
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
