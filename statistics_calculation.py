import numpy as np
import pandas as pd
from pandas import DataFrame


# Volatility
def calc_volatility(df:DataFrame, periods:int = 20) -> DataFrame:

    """Calculate volatility."""

    log_returns = np.log(df["Close"]/df["Close"].shift())
    volatility = log_returns.rolling(periods).std()*periods**.5

    df["Volatility"] = volatility

    return df


# Moving Averege
def calc_ma(df:DataFrame)->DataFrame:
    
    """Calculate moving averages for the last 50 and 200 datapoints."""

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    return df

# RSI    
def calc_rsi(df:DataFrame, periods:int = 20, ema:bool = True) -> DataFrame:
    
    """Calculate relative strength index."""

    close_delta = df['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    df["RSI"] = rsi

    return df

# Bollinger Bends
def calc_bollinger_bends(df:DataFrame, periods:int = 20) -> DataFrame:

    """Claculate Bollinger Bends."""

    df["STD"] = df["Close"].rolling(periods).std()

    df["MB"] = df["Close"].rolling(periods).mean()
    df["UB"] = df["MB"] + 2 * df["STD"]
    df["LB"] = df["MB"] - 2 * df["STD"]

    return df

# Fibonacci retracement
def calc_fibonacci_retracement(df:DataFrame, periods:int=20) -> DataFrame:

    """Calculate Fibonacci retracement."""

    max_value = df["Close"].rolling(periods).max()
    min_value = df["Close"].rolling(periods).min()
    difference = max_value - min_value
    df["FR-0.236"] = max_value - difference * 0.236
    df["FR-0.382"] = max_value - difference * 0.382
    df["FR-0.5"] = max_value - difference * 0.5
    df["FR-0.618"] = max_value - difference * 0.618

    return df


# OBV
def calc_obv(df:DataFrame) -> DataFrame:

    """Calculate On-balance volume."""
    df["OBV"] = (np.sign(df["Close"].diff())* df["Volume"]).fillna(0).cumsum()

    return df

# MACD
def calc_macd(df:DataFrame, slow:float = 24, fast:float = 16, smooth:float = 10) -> DataFrame:

    """Calculate moving average convergence/divergence."""

    exp1 = df["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["Close"].ewm(span=slow, adjust=False).mean()

    macd = pd.DataFrame(exp1-exp2).rename(columns={'Close':'MACD'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'MACD':'Signal'})

    df["MACD"] = macd["MACD"]
    df["Signal"] = signal["Signal"]

    return df

# Calculate all statistics
def calc_statistics(df:DataFrame, periods:int=20, slow:float = 24, fast:float = 16, smooth:float = 10) -> DataFrame:

    """Calculate all necessery statistics."""

    df = calc_volatility(df, periods)
    df = calc_ma(df)
    df = calc_rsi(df, periods)
    df = calc_bollinger_bends(df, periods)
    df = calc_fibonacci_retracement(df, periods)
    df = calc_macd(df, slow, fast, smooth)
    df = calc_obv(df)
    
    return df