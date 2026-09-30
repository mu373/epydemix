"""Propose prior or perturbed parameters using the shared sequential RNG."""

from datetime import datetime

from ..utils.abc_smc_utils import sample_prior


def generate_candidates(
    priors,
    names,
    root_rng,
    seed_requested,
    epsilon=None,
    particles=None,
    weights=None,
    perturbations=None,
    deadline=None,
):
    """Yield proposals and their RNG without changing the existing sequential stream."""
    continuous_params = [name for name in names if hasattr(priors[name], "pdf")]
    while deadline is None or datetime.now() <= deadline:
        rng = root_rng
        if particles is None:
            params = sample_prior(priors, names, rng)
        else:
            while True:
                if deadline is not None and datetime.now() > deadline:
                    return
                index = rng.choice(len(particles), p=weights / weights.sum())
                params = [
                    perturbations[p].propose(particles[index, i], rng)
                    for i, p in enumerate(names)
                ]
                if all(
                    (
                        priors[p].pdf(params[i])
                        if p in continuous_params
                        else priors[p].pmf(params[i])
                    )
                    > 0
                    for i, p in enumerate(names)
                ):
                    break
        candidate_rng = rng if seed_requested else None
        yield params, epsilon, candidate_rng
