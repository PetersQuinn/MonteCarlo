#a lot will be the same as montecarlov1.py, but we will be optimizing the parameters of the model
#there will be fewer comments on the repetitive parts of the code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

#repeated code section, see MonteCarlov1.py for more comments
#I want the length of our historical data to be a parameter that we can optimize, as this is an important question (how far back do we go?)
historical_data_length = 1000
# Set fixed end date for historical data
historical_end_date = pd.to_datetime("2025-02-28")  # Example fixed end date
historical_start_date = historical_end_date - timedelta(days=historical_data_length)

print(f"Historical Start Date: {historical_start_date.date()}")
print(f"Historical End Date: {historical_end_date.date()}")

AppleCloseData = yf.download("AAPL", start=historical_start_date, end=historical_end_date).loc[:, "Close"]
AppleCloseData = AppleCloseData.reset_index()
AppleCloseData['Date'] = pd.to_datetime(AppleCloseData['Date'])
print(AppleCloseData.head())
"""#Compare predicted stock price to actual stock price
#our goal here is to backtest our model while varying T, M, and historical data length and attempt to optimize the parameters in the model
#this will have to be a rough estimate and will be unlikely to be generalizable, probably a range, as the variance in the market is pretty significant
#we want to predict over a large number of days to give our model more of a sample size to compare to
prediction_start = pd.to_datetime("2025-03-01")
prediction_end = pd.to_datetime("2025-03-15") 

actual_prices = AppleCloseData.loc[
    (AppleCloseData['Date'] >= prediction_start) & 
    (AppleCloseData['Date'] <= prediction_end),
    'AAPL'
].values

T = len(actual_prices)
#now we actually have to predict the stock price for each day of our prediction time period
#we will use the same parameters as before (should we? will revisit later. The parameters were somewhat dependent on the true value from our prediction period)
S_march = np.zeros((T, M))
S_march[0] = S_0 #will have to ensure this makes sense. What S_0 should we use so this is generalizable over various prediction periods?

for t in range(1, T):
    S_march[t] = S_march[t-1] * np.exp((mean - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.standard_normal(M))

# Compute expected predicted prices
predicted_prices = np.mean(S_march, axis=1)

# Calculate Prediction Error (Mean Absolute Error)
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"Mean Absolute Error (MAE) for March 2025: ${mae:.2f}")
"""