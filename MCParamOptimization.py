#a lot will be the same as montecarlov1.py, but we will be optimizing the parameters of the model
#there will be fewer comments on the repetitive parts of the code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import time


#variables to change

historical_data_length = 1000  # Number of days of historical data to use
historical_end_date = pd.to_datetime("2025-02-28")  # Example fixed end date
M=1000 #number of simulations
stock = "AAPL"  # Stock ticker symbol

#repeated code section, see MonteCarlov1.py for more comments
dt=1
historical_start_date = historical_end_date - timedelta(days=historical_data_length)
#Uncomment the following lines to see the historical start and end dates
#print(f"Historical Start Date: {historical_start_date.date()}")
#print(f"Historical End Date: {historical_end_date.date()}")
StockCloseData = yf.download(stock, start=historical_start_date, end=historical_end_date, progress=False).loc[:, "Close"]
StockCloseData = StockCloseData.reset_index()
StockCloseData['Date'] = pd.to_datetime(StockCloseData['Date'])
StockCloseData['Log Return'] = np.log(StockCloseData[stock] / StockCloseData[stock].shift(1))
mean = StockCloseData['Log Return'].mean()
volatility = StockCloseData['Log Return'].std()
S_0 = StockCloseData[stock].iloc[-1]  # Current stock price

#Compare predicted stock price to actual stock price
#our goal here is to backtest our model while varying M and historical data length and attempt to optimize the parameters in the model
#this will have to be a rough estimate and will be unlikely to be generalizable, probably a range, as the variance in the market is pretty significant
#we want to predict over a large number of days to give our model more of a sample size to compare to
prediction_start = historical_end_date + timedelta(days=1)
prediction_end = pd.to_datetime("2025-03-15")
actual_prices_df = yf.download(stock, start=prediction_start, end=prediction_end, progress=False)[['Close']]
actual_prices_df = actual_prices_df.reset_index()  # Convert index to column
actual_prices_df['Date'] = pd.to_datetime(actual_prices_df['Date'])  # Ensure Date is datetime
actual_prices = actual_prices_df['Close'].values
T = len(actual_prices)
#now we actually have to predict the stock price for each day of our prediction time period
S_results = np.zeros((T, M))
S_results[0] = S_0  # Set initial stock price for all simulations
for t in range(1, T):
    S_results[t] = S_results[t-1] * np.exp((mean - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.standard_normal(M))
predicted_prices = np.mean(S_results, axis=1)
mae = mean_absolute_error(actual_prices, predicted_prices)
#print(f"Mean Absolute Error (MAE) for results period: ${mae:.2f}")
"""
#plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(S_results, alpha=0.2)  # Plot all simulation paths with transparency
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"Monte Carlo Simulation of {M} Paths for {stock} over {T} Days")
plt.show()

#now we have to actually optimize the parameters
#we aren't just looking for the lowest MAE, but also computational efficiency and time efficiency
#we will be optimizing M and historical_data_length against these three metrics
M_values = M_values = np.linspace(100, 50000, num=500, dtype=int) 
historical_lengths = np.linspace(25, 1000, num=100, dtype=int) # Different historical data windows
mae_results_M = []
mae_results_H = []

for M in M_values:
    S_results = np.zeros((T, M))
    S_results[0] = S_0  # Initialize starting price

    for t in range(1, T):
        S_results[t] = S_results[t-1] * np.exp(
            (mean - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.standard_normal(M)
        )

    predicted_prices = np.mean(S_results, axis=1)  # Average predictions

    # Ensure consistent lengths
    min_length = min(len(actual_prices), len(predicted_prices))
    mae = mean_absolute_error(actual_prices[:min_length], predicted_prices[:min_length])

    mae_results_M.append(mae)  # Store MAE for this M

for historical_length in historical_lengths:
    #fix start date
    historical_start_date = historical_end_date - timedelta(days=int(historical_length))

    #refetch historical data
    StockCloseData = yf.download("AAPL", start=historical_start_date, end=historical_end_date, progress=False)
    StockCloseData = StockCloseData[['Close']].reset_index()
    StockCloseData['Log Return'] = np.log(StockCloseData['Close'] / StockCloseData['Close'].shift(1))

    mean = StockCloseData['Log Return'].mean()
    volatility = StockCloseData['Log Return'].std()
    S_0 = StockCloseData['Close'].iloc[-1]  # Current stock price

    # Run Monte Carlo with this historical window
    S_results = np.zeros((T, M))
    S_results[0] = S_0

    for t in range(1, T):
        S_results[t] = S_results[t-1] * np.exp(
            (mean - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.standard_normal(M)
        )

    predicted_prices = np.mean(S_results, axis=1)

    # Ensure consistent lengths
    min_length = min(len(actual_prices), len(predicted_prices))
    mae = mean_absolute_error(actual_prices[:min_length], predicted_prices[:min_length])

    mae_results_H.append(mae)  # Store MAE for this historical length"""

"""# Plot MAE vs M
plt.figure(figsize=(8, 5))
plt.plot(M_values, mae_results_M, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Simulations (M)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE vs Number of Simulations")
plt.grid(True)
plt.show()"""

"""# Plot MAE vs Historical Data Length
plt.figure(figsize=(8, 5))
plt.plot(historical_lengths, mae_results_H, marker='o', linestyle='-', color='r')
plt.xlabel("Historical Data Length (Days)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE vs Historical Data Length")
plt.grid(True)
plt.show()"""

num_trials = 200  # Number of runs for averaging

# Define ranges for M and H
M_values = np.linspace(100, 5000, num=20, dtype=int)  # Number of simulations
H_values = np.linspace(100, 10000, num=50, dtype=int)  # Historical data length

time_results_M = []
time_results_H = []

# Function to process historical stock data
def historical_data_processing(H):
    historical_start_date = historical_end_date - pd.Timedelta(days=int(H))
    df = yf.download(stock, start=historical_start_date, end=historical_end_date, progress=False)
    df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df['Log Return'].mean(), df['Log Return'].std(), df['Close'].iloc[-1]

"""# Function to perform Monte Carlo simulation
def monte_carlo_simulation(M, mean, volatility, S_0):
    dt = 1
    S = np.zeros((T, M))
    S[0] = S_0  # Set initial stock price
    Wt = np.random.standard_normal((T, M))  # Brownian motion component

    for t in range(1, T):
        S[t] = S[t-1] * np.exp((mean - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Wt[t])
    return S
"""
# Function to get average execution time over multiple runs
def average_execution_time(func, *args):
    times = []
    for _ in range(num_trials):
        start_time = time.time()
        func(*args)  # Run function with arguments
        times.append(time.time() - start_time)
    return np.mean(times)

"""# Benchmarking execution time vs M
for M in M_values:
    mean, volatility, S_0 = historical_data_processing(500)  # Fixed historical window for M test
    avg_time = average_execution_time(monte_carlo_simulation, M, mean, volatility, S_0)
    time_results_M.append(avg_time)"""

# Benchmarking execution time vs H
for H in H_values:
    avg_time = average_execution_time(historical_data_processing, H)
    time_results_H.append(avg_time)

"""# Plot execution time vs M
plt.figure(figsize=(8, 5))
plt.plot(M_values, time_results_M, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Simulations (M)")
plt.ylabel("Average Computation Time (s)")
plt.title("Computation Time vs Number of Simulations (M)")
plt.grid(True)
plt.show()"""

# Plot execution time vs H
plt.figure(figsize=(8, 5))
plt.plot(H_values, time_results_H, marker='o', linestyle='-', color='r')
plt.xlabel("Historical Data Length (H)")
plt.ylabel("Average Computation Time (s)")
plt.title("Computation Time vs Historical Data Length (H)")
plt.grid(True)
plt.show()
