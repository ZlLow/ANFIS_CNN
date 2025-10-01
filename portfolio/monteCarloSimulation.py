# monteCarloSimulation.py

import numpy as np

class MonteCarloSimulation:
    def __init__(self, returns, initial_investment=1, weights=None):
        """
        Initializes the Monte Carlo Simulation.
        :param returns: A pandas DataFrame of asset returns.
        :param initial_investment: The starting value of the portfolio.
        :param weights: A list or numpy array of portfolio weights.
        """
        self.returns = returns
        # Handle both DataFrame and Series for returns input
        if isinstance(returns, np.ndarray) or hasattr(returns, 'mean'):
            self.mean = returns.mean()
        else:
            raise TypeError("Returns must be a pandas DataFrame or Series.")

        # Handle portfolio vs single asset case
        if hasattr(returns, 'cov'):
            self.covariance = returns.cov()
            num_assets = len(self.mean)
            if weights is None:
                self.weights = np.ones(num_assets) / num_assets
            else:
                self.weights = np.array(weights)
        else: # This is for a single series of returns (our strategy output)
            self.covariance = np.array([[returns.std()**2]]) # Covariance is just variance
            self.weights = np.array([1.0])


    def run_simulation(self, num_simulations, time_horizon):
        """
        Runs the Monte Carlo simulation.
        :param num_simulations: The number of simulation paths to generate.
        :param time_horizon: The number of future time steps (e.g., days) to project.
        :return: A tuple of (all_cumulative_returns, final_portfolio_values).
        """
        all_cumulative_returns = np.zeros((time_horizon, num_simulations))
        final_portfolio_values = np.zeros(num_simulations)

        for sim in range(num_simulations):
            # Generate random returns from the multivariate normal distribution
            simulated_returns = np.random.multivariate_normal(
                self.mean, self.covariance, time_horizon
            )

            # Calculate cumulative returns for each simulation path
            # (1 + r1) * (1 + r2) * ...
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)

            # Apply portfolio weights
            if len(self.weights) > 1:
                portfolio_cumulative_returns = cumulative_returns.dot(self.weights)
            else: # Single asset/strategy case
                portfolio_cumulative_returns = cumulative_returns.flatten()


            # Store the results
            all_cumulative_returns[:, sim] = portfolio_cumulative_returns * self.initial_investment
            final_portfolio_values[sim] = portfolio_cumulative_returns[-1] * self.initial_investment

        return all_cumulative_returns, final_portfolio_values