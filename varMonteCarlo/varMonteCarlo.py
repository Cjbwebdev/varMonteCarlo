import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


#set time for number of years
years = 15

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365*years)

#List of tickers
tickers = ['SPY','BND','GLD','QQQ','VLT']

#download daily adjusted close price for the tickers