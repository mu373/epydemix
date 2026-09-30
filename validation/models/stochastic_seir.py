"""Independent binomial epidemic reference for validation comparisons.

Use the supplied seed/Generator; global NumPy random state is never consumed.
Trajectories cover the full configured grid, including extinction. Ensemble
trials use independent child streams so their identity does not depend on Nsim.
These are small comparison models, not calibration correctness oracles.
"""

import matplotlib.pyplot as plt
import numpy as np


class StochasticSEIR:
    def __init__(self, S0, E0, I0, R0, beta, sigma, gamma, population, time_steps):
        """
        Initializes the stochastic SEIR model.

        Args:
            S0 (int): Initial number of susceptible individuals.
            E0 (int): Initial number of exposed individuals.
            I0 (int): Initial number of infected individuals.
            R0 (int): Initial number of recovered individuals.
            beta (float): Transmission rate.
            sigma (float): Incubation rate (1/incubation period).
            gamma (float): Recovery rate.
            population (int): Total population (N = S + E + I + R).
            time_steps (int): Number of time steps for the simulation.
        """
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = population
        self.time_steps = time_steps

    def simulate(self, rng=None):
        """
        Runs a single simulation of the SEIR model for the defined number of time steps.

        Args:
            rng (int, SeedSequence or Generator, optional): Source of all random draws.

        Returns:
            results (dict): A dictionary with keys 'S', 'E', 'I', 'R' containing lists of population sizes
                            at each time step for susceptible, exposed, infected, and recovered compartments.
        """
        # Initialize the compartments
        rng = np.random.default_rng(rng)
        S = self.S0
        E = self.E0
        I = self.I0
        R = self.R0

        results = {"S": [], "E": [], "I": [], "R": []}

        for t in range(self.time_steps):
            results["S"].append(S)
            results["E"].append(E)
            results["I"].append(I)
            results["R"].append(R)

            # Probabilities for new infections, new exposures, and recoveries
            infection_prob = 1 - np.exp(-self.beta * I / self.N)
            incubation_prob = 1 - np.exp(-self.sigma)
            recovery_prob = 1 - np.exp(-self.gamma)

            # New exposures, new infections, and recoveries based on binomial distributions
            new_exposures = rng.binomial(S, infection_prob)
            new_infections = rng.binomial(E, incubation_prob)
            new_recoveries = rng.binomial(I, recovery_prob)

            # Update compartments
            S -= new_exposures
            E += new_exposures - new_infections
            I += new_infections - new_recoveries
            R += new_recoveries

        return results

    def run_simulations(self, Nsim, quantiles=(0.25, 0.5, 0.75), rng=None):
        """
        Runs the SEIR model simulation Nsim times and computes the specified quantiles.

        Args:
            Nsim (int): The number of simulations to run.
            rng (int or Generator, optional): Seed for independent child trial streams.
            quantiles (list of float): A list of quantiles to compute (e.g., [0.25, 0.5, 0.75]).

        Returns:
            quantile_results (dict of dict): A dictionary of dictionaries where the outer key is the compartment ('S', 'E', 'I', 'R')
                                             and the inner key is the quantile, each containing an array of shape (time_steps).
        """
        # Initialize lists to store all simulation results
        if Nsim < 1:
            raise ValueError("Nsim must be positive")
        rng = np.random.default_rng(rng)
        children = np.random.SeedSequence(
            rng.integers(0, 2**32, size=4, dtype=np.uint32)
        ).spawn(Nsim)
        all_S = []
        all_E = []
        all_I = []
        all_R = []

        # Run Nsim simulations and collect results
        for child in children:
            results = self.simulate(rng=child)
            all_S.append(results["S"])
            all_E.append(results["E"])
            all_I.append(results["I"])
            all_R.append(results["R"])

        # Convert lists to arrays to compute the quantiles
        all_S = np.array(all_S)
        all_E = np.array(all_E)
        all_I = np.array(all_I)
        all_R = np.array(all_R)

        # Compute the quantiles across the simulations and store them in a dict of dicts
        quantile_results = {
            "S": {q: np.quantile(all_S, q, axis=0) for q in quantiles},
            "E": {q: np.quantile(all_E, q, axis=0) for q in quantiles},
            "I": {q: np.quantile(all_I, q, axis=0) for q in quantiles},
            "R": {q: np.quantile(all_R, q, axis=0) for q in quantiles},
        }

        return quantile_results

    def plot(self, quantile_results, quantiles=(0.25, 0.5, 0.75)):
        """
        Plots the quantile trajectories of the SEIR model.

        Args:
            quantile_results (dict of dict): The quantile results of the simulations with keys 'S', 'E', 'I', 'R',
                                             and inner keys as the quantiles (e.g., 0.25, 0.5, 0.75).
            quantiles (list of float): The quantiles to plot (e.g., [0.25, 0.5, 0.75]).
        """
        time_range = range(len(quantile_results["S"][quantiles[0]]))

        for q in quantiles:
            plt.plot(
                time_range,
                quantile_results["S"][q],
                label=f"Susceptible ({q * 100:.0f}%)",
            )
            plt.plot(
                time_range, quantile_results["E"][q], label=f"Exposed ({q * 100:.0f}%)"
            )
            plt.plot(
                time_range, quantile_results["I"][q], label=f"Infected ({q * 100:.0f}%)"
            )
            plt.plot(
                time_range,
                quantile_results["R"][q],
                label=f"Recovered ({q * 100:.0f}%)",
            )

        plt.xlabel("Time Steps")
        plt.ylabel("Population")
        plt.legend()
        plt.title("Stochastic SEIR Model - Quantiles over Multiple Simulations")
        plt.show()
