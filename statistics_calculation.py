import numpy as np
import pandas as pd
from pandas import DataFrame


# Volatility
def calc_volatility(df:DataFrame) -> DataFrame:
    pass

# Moving Averege
def calc_ma(df:DataFrame)->DataFrame:
    
    """Calculate moving averages for the last 50 and 200 datapoints."""

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    return df

# RSI
def calc_rsi(df:DataFrame) -> DataFrame:
    pass

# Bollinger Bends
def calc_bollinger_bends(df:DataFrame) -> DataFrame:
    df["STD"] = df["Close"].rolling(20).std()

    df["MB"] = df["Close"].rolling(20).mean()
    df["UB"] = df["MB"] + 2 * df["STD"]
    df["LB"] = df["MB"] - 2 * df["STD"]

    return df

# Fibonacci retracement
def calc_fibonacci_retracement(df:DataFrame) -> DataFrame:
    pass

# OBV
def calc_obv(df:DataFrame) -> DataFrame:
    pass

#MACD
def calc_macd(df:DataFrame) -> DataFrame:
    pass