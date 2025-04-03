#imports
import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime
from sklearn.preprocessing import StandardScaler

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
# Forward-fill missing data to match daily frequency
data = data.ffill().dropna()
print(data.tail())
data.to_csv('economic_indicators.csv')
# Load the cleaned dataset
data = pd.read_csv('economic_indicators.csv', index_col=0, parse_dates=True)
# Check the data structure
print(data.head())

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
data['GDP_Lag_1'] = data['GDP'].shift(1)
data['CPI_Lag_1'] = data['CPI'].shift(1)
# Drop rows with NaN values introduced by feature engineering
data.dropna(inplace=True)
# Normalization & Scaling
features_to_normalize = ['GDP', 'FedRate', 'CPI', 'Unemployment', 'VIX', 
                         'VIX_Rolling_30', 'CPI_Rolling_90', 
                         'GDP_Change', 'CPI_Change', 'VIX_Change', 
                         'VIX_FedRate_Interaction', 'GDP_Lag_1', 'CPI_Lag_1']

scaler = StandardScaler()
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Save your dataset with features
data.to_csv('economic_indicators_with_features.csv')

# Check the data structure
print(data.head())
