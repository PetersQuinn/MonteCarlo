#imports
import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime

'''Initially, we have to determine which factors tend to affect our test stock, especially over recent times. 
This will allow us to build the rebound model for testing.'''