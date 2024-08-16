import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set time for number of years
years = 15

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365*years)

# List of tickers
tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VLT']

# Download daily adjusted close price for the tickers
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=startDate, end=endDate)
    adj_close_df[ticker] = data['Adj Close']

# Calculate daily log returns and drop any NAs
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

# Create a function to calculate expected returns
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252  # Annualized return

# Create function that will be used to calculate portfolio standard deviation
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance) * np.sqrt(252)  # Annualized standard deviation

# Create covariance matrix for all the securities
cov_matrix = log_returns.cov()

# Create equally weighted portfolio and find total portfolio expected return
portfolio_value = 100000
weights = np.array([1/len(tickers)]*len(tickers))
portfolio_expected_return = expected_return(weights, log_returns)
portfolio_std_dev = standard_deviation(weights, cov_matrix)

# Create a function that gives a random Z-score based on normal distribution
def random_z_score():
    return np.random.normal(0, 1)

# Create function to calculate scenarioGainLoss
days = 25

def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * (portfolio_expected_return * days/252) + portfolio_value * portfolio_std_dev * z_score * np.sqrt(days/252)

# Run 10000 simulations
simulations = 10000
scenario_return = []

for i in range(simulations):
    z_score = random_z_score()
    scenario_return.append(scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days))

# Specify a confidence interval and calculate the value at risk
confidence_interval = 0.95
VaR = -np.percentile(scenario_return, 100 * (1 - confidence_interval))
print(f"Value at Risk (VaR) at {confidence_interval*100}% confidence level: ${VaR:.2f}")

# Plot the result of all 10000 scenarios
plt.hist(scenario_return, bins=50, density=True)
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')
plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()

