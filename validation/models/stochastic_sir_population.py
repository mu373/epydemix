"""Independent binomial epidemic reference for validation comparisons.

Use the supplied seed/Generator; global NumPy random state is never consumed.
Trajectories cover the full configured grid, including extinction. Ensemble
trials use independent child streams so their identity does not depend on Nsim.
These are small comparison models, not calibration correctness oracles.
"""

import matplotlib.pyplot as plt
import numpy as np


class StochasticSIRAgeGroups:
    def __init__(self, S0, I0, R0, beta, gamma, contact_matrix, population, time_steps):
        """
        Initializes the stochastic SIR model with age groups and a contact matrix.

        Args:
            S0 (np.ndarray): Initial number of susceptible individuals in each age group.
            I0 (np.ndarray): Initial number of infected individuals in each age group.
            R0 (np.ndarray): Initial number of recovered individuals in each age group.
            beta (float): Transmission rate.
            gamma (float): Recovery rate.
            contact_matrix (np.ndarray): Contact matrix representing contact rates between age groups.
            population (np.ndarray): Total population in each age group.
            time_steps (int): Number of time steps for the simulation.
        """
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.beta = beta
        self.gamma = gamma
        self.contact_matrix = contact_matrix
        self.N = population
        self.time_steps = time_steps
        self.num_groups = len(S0)  # Number of age groups

    def simulate(self, rng=None):
        """
        Runs a single simulation of the SIR model for the defined number of time steps with age groups.

        Args:
            rng (int, SeedSequence or Generator, optional): Source of all random draws.

        Returns:
            results (dict): A dictionary with keys 'S', 'I', 'R' containing lists of population sizes
                            at each time step for each compartment and age group.
        """
        # Initialize the compartments for each age group
        rng = np.random.default_rng(rng)
        S = self.S0.copy()
        I = self.I0.copy()
        R = self.R0.copy()

        results = {"S": [], "I": [], "R": []}

        for t in range(self.time_steps):
            results["S"].append(S.copy())
            results["I"].append(I.copy())
            results["R"].append(R.copy())

            # Compute new infections for each age group
            new_infections = np.zeros(self.num_groups)
            for i in range(self.num_groups):
                # The force of infection for group i depends on contacts with all other groups
                force_of_infection = np.sum(self.contact_matrix[i, :] * I / self.N)
                infection_prob = 1 - np.exp(-self.beta * force_of_infection)
                new_infections[i] = rng.binomial(S[i], infection_prob)

            # Compute recoveries
            new_recoveries = rng.binomial(I, 1 - np.exp(-self.gamma))

            # Update compartments with casting to int
            S -= new_infections.astype(int)
            I += (new_infections - new_recoveries).astype(int)
            R += new_recoveries.astype(int)

        return results

    def run_simulations(self, Nsim, quantiles=(0.25, 0.5, 0.75), rng=None):
        """
        Runs the SIR model simulation Nsim times and computes the specified quantiles.

        Args:
            Nsim (int): The number of simulations to run.
            rng (int or Generator, optional): Seed for independent child trial streams.
            quantiles (list of float): A list of quantiles to compute (e.g., [0.25, 0.5, 0.75]).

        Returns:
            quantile_results (dict of dict): A dictionary of dictionaries where the outer key is the compartment ('S', 'I', 'R')
                                             and the inner key is the quantile, each containing an array of shape (time_steps, num_groups).
        """
        # Initialize lists to store all simulation results
        if Nsim < 1:
            raise ValueError("Nsim must be positive")
        rng = np.random.default_rng(rng)
        children = np.random.SeedSequence(
            rng.integers(0, 2**32, size=4, dtype=np.uint32)
        ).spawn(Nsim)
        all_S = []
        all_I = []
        all_R = []

        # Run Nsim simulations and collect results
        for child in children:
            results = self.simulate(rng=child)
            all_S.append(np.array(results["S"]))
            all_I.append(np.array(results["I"]))
            all_R.append(np.array(results["R"]))

        # Convert lists to arrays to compute the quantiles
        all_S = np.array(all_S)
        all_I = np.array(all_I)
        all_R = np.array(all_R)

        # Compute the quantiles across the simulations and store them in a dict of dicts
        quantile_results = {
            "S": {q: np.quantile(all_S, q, axis=0) for q in quantiles},
            "I": {q: np.quantile(all_I, q, axis=0) for q in quantiles},
            "R": {q: np.quantile(all_R, q, axis=0) for q in quantiles},
        }

        return quantile_results

    def plot(self, quantile_results, quantiles=(0.25, 0.5, 0.75), age_group_idx=0):
        """
        Plots the quantile trajectories of the SIR model for a specific age group.

        Args:
            quantile_results (dict of dict): The quantile results of the simulations with keys 'S', 'I', 'R',
                                             and inner keys as the quantiles (e.g., 0.25, 0.5, 0.75).
            quantiles (list of float): The quantiles to plot (e.g., [0.25, 0.5, 0.75]).
            age_group_idx (int): The index of the age group to plot.
        """
        time_range = range(quantile_results["S"][quantiles[0]].shape[0])

        for q in quantiles:
            plt.plot(
                time_range,
                quantile_results["S"][q][:, age_group_idx],
                label=f"Susceptible ({q * 100:.0f}%)",
            )
            plt.plot(
                time_range,
                quantile_results["I"][q][:, age_group_idx],
                label=f"Infected ({q * 100:.0f}%)",
            )
            plt.plot(
                time_range,
                quantile_results["R"][q][:, age_group_idx],
                label=f"Recovered ({q * 100:.0f}%)",
            )

        plt.xlabel("Time Steps")
        plt.ylabel("Population")
        plt.legend()
        plt.title(f"Stochastic SIR Model (Age Group {age_group_idx})")
        plt.show()
