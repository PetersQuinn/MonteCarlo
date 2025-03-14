import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

#get historical stock data and store it, choosing apple because it is a well known company
AppleCloseData = yf.download("AAPL", start="2025-01-01", end="2025-03-14").loc[:, "Close"]
print(AppleCloseData)
#calculate the log returns

#actually simulate stock price with monte carlo simulation

#plot the results

#analyze the results

