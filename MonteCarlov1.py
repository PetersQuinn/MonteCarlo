import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error

#get historical stock data and store it, choosing apple because it is a well known company
StockCloseData = yf.download("AAPL", start="2025-01-01", end="2025-03-15").loc[:, "Close"]
StockCloseData = StockCloseData.reset_index()
StockCloseData['Date'] = pd.to_datetime(StockCloseData['Date'])
#calculate the log returns
StockCloseData['Log Return'] = np.log(StockCloseData['AAPL'] / StockCloseData['AAPL'].shift(1))
mean = StockCloseData['Log Return'].mean()
volatility = StockCloseData['Log Return'].std()
#actually simulate stock price with monte carlo simulation
S_0 = StockCloseData['AAPL'].iloc[-1]  #current stock price
print(S_0) 
T=252 #number of future days to simulate
N=1 #number of steps per day (really focused on close prices so, 1 step per day)
M=1000 #number of simulations
dt=1/N #time step
Wt = np.random.standard_normal((T, M))  # Matrix of random numbers
S = np.zeros((T, M)) #initialize matrix of stock prices
S[0] = S_0 # Set initial stock price
print(S[0])
for t in range(1, T):
    S[t] = S[t-1] * np.exp((mean - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Wt[t])

#plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(S, alpha=0.2)  # Plot all simulation paths with transparency
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"Monte Carlo Simulation of {M} Paths for AAPL over {T} Days")
plt.show()
#analyze the results
expected_price = np.mean(S[-1, :])  #average stock price on the last simulated day
print(f"Expected Future Stock Price: ${expected_price:.2f}")
percentile_5 = np.percentile(S[-1, :], 5)  # 5th percentile
percentile_95 = np.percentile(S[-1, :], 95)  # 95th percentile
print(f"5% Confidence Interval: ${percentile_5:.2f}")
print(f"95% Confidence Interval: ${percentile_95:.2f}")
