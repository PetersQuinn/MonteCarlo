#imports
import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime

'''Initially, we have to determine which factors tend to affect our test stock, especially over recent times. 
This will allow us to build the rebound model for testing.'''

# Set up the date range
start_date = '2020-01-01'
end_date = '2025-03-15'
stock = 'AAPL'  # Apple Inc.
# Downloading AAPL stock data
stock_data = yf.download(stock, start=start_date, end=end_date)['Close']

# Downloading VIX data (Volatility Index)
vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']

# Downloading Economic Data from FRED
from pandas_datareader.data import DataReader

# GDP Growth Rate (Quarterly)
gdp_data = DataReader('GDP', 'fred', start_date, end_date)

# Federal Funds Rate (Monthly)
fed_rate = DataReader('FEDFUNDS', 'fred', start_date, end_date)

# Inflation Rate (CPI) (Monthly)
cpi_data = DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Unemployment Rate (Monthly)
unemployment_data = DataReader('UNRATE', 'fred', start_date, end_date)

# Combine all datasets into one DataFrame
data = pd.concat([stock_data, vix_data, gdp_data, fed_rate, cpi_data, unemployment_data], axis=1)
data.columns = ['AAPL_Close', 'VIX', 'GDP', 'FedRate', 'CPI', 'Unemployment']
print(data.head())

# Forward-fill missing data to match daily frequency
data = data.ffill().dropna()

print(data.tail())

data.to_csv('economic_indicators.csv')

