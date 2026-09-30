"""Cheap stochastic models with independent finite-epsilon ABC target distributions.

Normal model: mu ~ N(0,1), Y|mu ~ N(mu,1), observed Y=1, distance |Y-1|.
Acceptance probability is Phi(1+epsilon-mu)-Phi(1-epsilon-mu), so the ABC density
is its product with phi(mu), normalized by the N(0,2) probability of that interval.
Quadrature below uses this identity, never the sampler's proposals or weights.

Discrete model: k in {0,1,2}, prior [.2,.5,.3], Y|k ~ Bernoulli([.1,.5,.9][k]),
observed Y=1, distance |Y-1|. Epsilon=.5 accepts exactly Y=1, giving posterior
[1/27,25/54,1/2]. A fixed schedule avoids adaptive epsilon=0 when all distances=0.

Simulation uses only the supplied Generator. Both observations are fixed constants,
not new noisy synthetic data in each test. Module-level functions support spawn.
"""

import numpy as np
from scipy import integrate, stats

from epydemix.calibration.abc import ABCSampler

SEEDS = (11, 29, 47)
PARTICLES = 512
NORMAL_SCHEDULE = (1.0, 0.5, 0.25)
DISCRETE_TARGET = np.array([1 / 27, 25 / 54, 1 / 2])
CDF_POINTS = np.array([0.0, 0.5, 1.0])


def simulate_normal(parameters):
    """One Normal(mu,1) draw per candidate from the injected RNG."""
    return {"data": np.array([parameters["rng"].normal(parameters["mu"], 1.0)])}


def simulate_discrete(parameters):
    """One Bernoulli draw; the latent state indexes three observation probabilities."""
    probability = (0.1, 0.5, 0.9)[int(parameters["k"])]
    return {"data": np.array([parameters["rng"].binomial(1, probability)])}


def absolute_distance(data, simulation):
    """Names also support the baseline initial-SMC keyword-callback convention."""
    return float(abs(simulation["data"][0] - data["data"][0]))


def make_statistical_sampler(model, seed):
    """Use explicit sampler RNG so the simulation always receives a Generator."""
    if model == "normal":
        simulate, prior = simulate_normal, {"mu": stats.norm()}
    else:
        simulate = simulate_discrete
        prior = {"k": stats.rv_discrete(values=([0, 1, 2], [0.2, 0.5, 0.3]))}
    return ABCSampler(simulate, prior, {}, np.array([1.0]), absolute_distance, rng=seed)


def normal_reference(epsilon):
    """Mean, variance and fixed-point CDF of the finite-epsilon ABC density.

    Integration tolerance 1e-11 is many orders below Monte Carlo test tolerances.
    The epsilon->0 posterior mean=.5, variance=.5 is not the finite-epsilon target.
    """
    scale = np.sqrt(2)
    normalizer = stats.norm.cdf((1 + epsilon) / scale) - stats.norm.cdf(
        (1 - epsilon) / scale
    )

    def density(mu):
        acceptance = stats.norm.cdf(1 + epsilon - mu) - stats.norm.cdf(1 - epsilon - mu)
        return stats.norm.pdf(mu) * acceptance / normalizer

    def integral(function, upper=np.inf):
        return integrate.quad(function, -np.inf, upper, epsabs=1e-11, epsrel=1e-11)[0]

    mean = integral(lambda mu: mu * density(mu))
    variance = integral(lambda mu: (mu - mean) ** 2 * density(mu))
    cdf = np.array([integral(density, upper) for upper in CDF_POINTS])
    return np.concatenate(([mean, variance], cdf))


def posterior_summary(result, model, generation, *, weighted=True):
    """Summarize the empirical measure, keeping SMC importance weights explicit."""
    values = result.get_posterior_distribution(generation).iloc[:, 0].to_numpy()
    weights = np.asarray(result.get_weights(generation))
    if not weighted:
        weights = np.ones(len(values))
    weights = weights / weights.sum()
    if model == "discrete":
        return np.array([weights[values == k].sum() for k in range(3)])
    mean = weights @ values
    variance = weights @ ((values - mean) ** 2)
    cdf = np.array([weights[values <= point].sum() for point in CDF_POINTS])
    return np.concatenate(([mean, variance], cdf))
