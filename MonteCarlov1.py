import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

#get historical stock data and store it, choosing apple because it is a well known company
AppleCloseData = yf.download("AAPL", start="2025-01-01", end="2025-03-14").loc[:, "Close"]
AppleCloseData = AppleCloseData.reset_index()
AppleCloseData['Date'] = pd.to_datetime(AppleCloseData['Date'])
#calculate the log returns
AppleCloseData['Log Return'] = np.log(AppleCloseData['AAPL'] / AppleCloseData['AAPL'].shift(1))
print(AppleCloseData.head())
#Find mean and volatility of stock
mean = AppleCloseData['Log Return'].mean()
volatility = AppleCloseData['Log Return'].std()
#actually simulate stock price with monte carlo simulation
S_0 = AppleCloseData['AAPL'].iloc[-1]  # Current stock price
print(S_0)  # Outputs an array, so use close_price[0] if needed
T=252 #number of future days to simulate
N=10 #number of steps per day
M=1000 #number of simulations
dt=1/N #time step
Wt = np.random.standard_normal((T, M))  # Matrix of random numbers
S = np.zeros((T, M)) #initialize matrix of stock prices
S[0] = S_0 # Set initial stock price
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
historical_mean = np.mean(AppleCloseData["AAPL"])
historical_volatility = np.std(AppleCloseData["AAPL"])
print(f"Historical Mean Price: ${historical_mean:.2f}")
print(f"Historical Volatility: {historical_volatility:.2f}")
plt.figure(figsize=(10, 5))
plt.plot(S, alpha=0.2, color="blue")  
plt.plot(range(len(AppleCloseData)), AppleCloseData["AAPL"], color="red", label="Historical Data", linewidth=2)
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"Monte Carlo vs. Historical Data for AAPL")
plt.legend()
plt.show()


