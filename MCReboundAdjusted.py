# Imports
import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime
from sklearn.preprocessing import StandardScaler
from pandas_datareader.data import DataReader

'''Initially, we have to determine which factors tend to affect our test stock, especially over recent times. 
This will allow us to build the rebound model for testing.'''

# Set up the date range
start_date = '2020-01-01'
end_date = '2025-03-15'
stock = 'AAPL'  # Apple Inc.

# Downloading AAPL stock data (Daily)
stock_data = yf.download(stock, start=start_date, end=end_date)['Close']

# Downloading VIX data (Daily)
vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']

# Downloading Economic Data from FRED (Non-daily data)
gdp_data = DataReader('GDP', 'fred', start_date, end_date)
fed_rate = DataReader('FEDFUNDS', 'fred', start_date, end_date)
cpi_data = DataReader('CPIAUCSL', 'fred', start_date, end_date)
unemployment_data = DataReader('UNRATE', 'fred', start_date, end_date)

# Resample all non-daily data to daily frequency
gdp_data = gdp_data.resample('D').ffill()
fed_rate = fed_rate.resample('D').ffill()
cpi_data = cpi_data.resample('D').ffill()
unemployment_data = unemployment_data.resample('D').ffill()

# Combine all datasets into one DataFrame
data = pd.concat([stock_data, vix_data, gdp_data, fed_rate, cpi_data, unemployment_data], axis=1)
data.columns = ['AAPL_Close', 'VIX', 'GDP', 'FedRate', 'CPI', 'Unemployment']

# Forward-fill remaining missing values and drop rows with missing data
data = data.ffill().dropna()

# Save the aligned dataset
data.to_csv('economic_indicators_aligned.csv')

# Feature Engineering
# Rolling Averages (Smoothing)
data['VIX_Rolling_30'] = data['VIX'].rolling(window=30).mean()
data['CPI_Rolling_90'] = data['CPI'].rolling(window=90).mean()

# Rate of Change (Momentum)
data['GDP_Change'] = data['GDP'].pct_change() * 100
data['CPI_Change'] = data['CPI'].pct_change() * 100
data['VIX_Change'] = data['VIX'].pct_change() * 100

# Interaction Terms
data['VIX_FedRate_Interaction'] = data['VIX'] * data['FedRate']

# Lag Features (Capturing Delayed Effects)
data['GDP_Lag_1Q'] = data['GDP'].shift(90)      # Lagging GDP by 1 Quarter (90 days)
data['CPI_Lag_1M'] = data['CPI'].shift(30)      # Lagging CPI by 1 Month (30 days)
data['FedRate_Lag_1M'] = data['FedRate'].shift(30)  # Lagging Federal Funds Rate by 1 Month (30 days)
data['Unemployment_Lag_1M'] = data['Unemployment'].shift(30)  # Lagging Unemployment Rate by 1 Month (30 days)

# Drop rows with NaN values introduced by feature engineering
data.dropna(inplace=True)

# Normalization & Scaling
features_to_normalize = ['GDP', 'FedRate', 'CPI', 'Unemployment', 'VIX', 
                         'VIX_Rolling_30', 'CPI_Rolling_90', 
                         'GDP_Change', 'CPI_Change', 'VIX_Change', 
                         'VIX_FedRate_Interaction', 'GDP_Lag_1Q', 'CPI_Lag_1M',
                         'FedRate_Lag_1M', 'Unemployment_Lag_1M']

scaler = StandardScaler()
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Save your dataset with features
data.to_csv('economic_indicators_with_features.csv')

# Check the data structure
print(data.head())

# Check the individual datasets before merging
print("AAPL Stock Data")
print(stock_data.head())
print(stock_data.tail())

print("\nVIX Data")
print(vix_data.head())
print(vix_data.tail())

print("\nGDP Data")
print(gdp_data.head())
print(gdp_data.tail())

print("\nFederal Funds Rate Data")
print(fed_rate.head())
print(fed_rate.tail())

print("\nCPI Data")
print(cpi_data.head())
print(cpi_data.tail())

print("\nUnemployment Rate Data")
print(unemployment_data.head())
print(unemployment_data.tail())

# Checking the last available date for each dataset
print(f"AAPL Data Ends On: {stock_data.index[-1]}")
print(f"VIX Data Ends On: {vix_data.index[-1]}")
print(f"GDP Data Ends On: {gdp_data.index[-1]}")
print(f"Fed Rate Data Ends On: {fed_rate.index[-1]}")
print(f"CPI Data Ends On: {cpi_data.index[-1]}")
print(f"Unemployment Data Ends On: {unemployment_data.index[-1]}")

# Checking for NaN values in your merged DataFrame
nan_counts = data.isna().sum()
print(nan_counts[nan_counts > 0])

import matplotlib.pyplot as plt

# Plot each dataset's availability over time
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, np.ones(len(stock_data)), label='AAPL', marker='o', linestyle='', alpha=0.5)
plt.plot(vix_data.index, np.ones(len(vix_data)) * 1.1, label='VIX', marker='o', linestyle='', alpha=0.5)
plt.plot(gdp_data.index, np.ones(len(gdp_data)) * 1.2, label='GDP', marker='o', linestyle='', alpha=0.5)
plt.plot(fed_rate.index, np.ones(len(fed_rate)) * 1.3, label='Fed Rate', marker='o', linestyle='', alpha=0.5)
plt.plot(cpi_data.index, np.ones(len(cpi_data)) * 1.4, label='CPI', marker='o', linestyle='', alpha=0.5)
plt.plot(unemployment_data.index, np.ones(len(unemployment_data)) * 1.5, label='Unemployment', marker='o', linestyle='', alpha=0.5)

plt.yticks([1, 1.1, 1.2, 1.3, 1.4, 1.5], ['AAPL', 'VIX', 'GDP', 'Fed Rate', 'CPI', 'Unemployment'])
plt.title("Data Availability Over Time")
plt.xlabel("Date")
plt.ylabel("Datasets")
plt.legend()
plt.show()
