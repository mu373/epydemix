"""Propose parameter values using a reproducible RNG stream per candidate.

Initial candidates draw from the prior. Later SMC generations resample a previous
particle by weight and perturb it until the proposal has positive prior density.
ProposalSequence returns parameter values, the acceptance threshold, and the RNG
for simulation; model execution and distance calculation live in _evaluate.
"""

from datetime import datetime
from itertools import count

import numpy as np

from ..utils.abc_smc_utils import sample_prior
from ..utils.random_utils import rng_for_index


class ProposalDeadline(Exception):
    """Stop proposing after a wall-clock cutoff without hiding callback errors."""


class ProposalSequence:
    """An indexable, picklable proposal population for one SMC generation.

    Candidate j uses the same child SeedSequence as the j-th spawn from a fresh
    generation seed. Workers can therefore propose independently without owning
    mutable RNG streams. Priors and kernels must use the supplied RNG and have
    no hidden state that changes between proposals.
    """

    def __init__(
        self,
        priors,
        names,
        entropy,
        seeded,
        epsilon=None,
        particles=None,
        weights=None,
        perturbations=None,
        deadline=None,
    ):
        self.priors = priors
        self.names = names
        self.seed_sequence = np.random.SeedSequence(entropy)
        self.seeded = seeded
        self.epsilon = epsilon
        self.particles = particles
        self.weights = weights
        self.perturbations = perturbations
        self.deadline = deadline

    def __getitem__(self, index):
        """Return parameter values, epsilon and simulation RNG for candidate index.

        ProposalDeadline marks a cutoff; callback exceptions propagate unchanged.
        Proposal retries belong entirely to this candidate's RNG stream.
        """
        # Create the RNG for this candidate ID
        rng = rng_for_index(self.seed_sequence, index)
        while self.deadline is None or datetime.now() < self.deadline:
            if self.particles is None:
                # Sample parameters from the prior
                params = sample_prior(self.priors, self.names, rng)
                return params, self.epsilon, rng if self.seeded else None

            # Select a previous particle using its weight
            parent = rng.choice(
                len(self.particles), p=self.weights / self.weights.sum()
            )

            # Perturb the selected particle
            params = [
                self.perturbations[name].propose(self.particles[parent, i], rng)
                for i, name in enumerate(self.names)
            ]

            # Retry proposals outside the prior support
            if all(
                (
                    self.priors[name].pdf(params[i])
                    if hasattr(self.priors[name], "pdf")
                    else self.priors[name].pmf(params[i])
                )
                > 0
                for i, name in enumerate(self.names)
            ):
                return params, self.epsilon, rng if self.seeded else None
        raise ProposalDeadline("Proposal deadline reached")

    def __iter__(self):
        """Lazily enumerate candidates until their deadline."""
        for index in count():
            try:
                yield self[index]
            except ProposalDeadline:
                return
