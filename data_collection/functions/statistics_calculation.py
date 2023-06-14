import numpy as np
import pandas as pd
from pandas import DataFrame


# Volatility
def calc_volatility(df:DataFrame, periods:int = 20) -> DataFrame:

    """Calculate volatility."""

    log_returns = np.log(df["close"]/df["close"].shift()) # calculate log returns
    volatility = log_returns.rolling(periods).std()*periods**.5 # calculate volatility

    df["volatility"] = volatility # add volatility to the dataframe

    return df

# Growth
def calc_growth(df:DataFrame) -> DataFrame:

    """Calculate growth."""

    growth = df["close"] - df["open"] # calculate the growth of asset price during the day
    df["growth"] = growth # add growth to the dataframe
    
    return df

# Moving Averege
def calc_ma(df:DataFrame)->DataFrame:
    
    """Calculate moving averages for the last 7, 25 and 99 datapoints."""

    df["ma7"] = df["close"].rolling(7).mean() # calculate the moving average for the last 7 datapoints
    df["ma25"] = df["close"].rolling(25).mean() # calculate the moving average for the last 25 datapoints
    df["ma99"] = df["close"].rolling(99).mean() # calculate the moving average for the last 99 datapoints

    return df

# RSI    
def calc_rsi(df:DataFrame, periods:int = 20, ema:bool = True) -> DataFrame:
    
    """Calculate relative strength index."""

    close_delta = df['close'].diff() # calculate the difference between the closing prices
    up = close_delta.clip(lower=0) # calculate the positive difference
    down = -1 * close_delta.clip(upper=0) # calculate the negative difference

    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
    
    rs = ma_up / ma_down # calculate the relative strength
    rsi = 100 - (100/(1 + rs)) # calculate the relative strength index
    df["rsi"] = rsi # add the relative strength index to the dataframe

    return df

# Bollinger Bends
def calc_bollinger_bends(df:DataFrame, periods:int = 20) -> DataFrame:

    """Claculate Bollinger Bends."""

    df["std"] = df["close"].rolling(periods).std() # calculate the standard deviation

    df["bb-mb"] = df["close"].rolling(periods).mean() # calculate the middle band
    df["bb-ub"] = df["bb-mb"] + 2 * df["std"] # calculate the upper band
    df["bb-lb"] = df["bb-mb"] - 2 * df["std"] # calculate the lower band

    return df

# Fibonacci retracement
def calc_fibonacci_retracement(df:DataFrame, periods:int=20) -> DataFrame:

    """Calculate Fibonacci retracement."""

    max_value = df["close"].rolling(periods).max() # calculate the maximum value
    min_value = df["close"].rolling(periods).min() # calculate the minimum value
    difference = max_value - min_value # calculate the difference between the maximum and minimum value
    df["fr-0.236"] = max_value - difference * 0.236 # calculate the Fibonacci retracement at 23.6%
    df["fr-0.382"] = max_value - difference * 0.382 # calculate the Fibonacci retracement at 38.2%
    df["fr-0.5"] = max_value - difference * 0.5 # calculate the Fibonacci retracement at 50%
    df["fr-0.618"] = max_value - difference * 0.618 # calculate the Fibonacci retracement at 61.8%

    return df


# OBV
def calc_obv(df:DataFrame) -> DataFrame:

    """Calculate On-balance volume."""

    df["obv"] = (np.sign(df["close"].diff())* df["volume"]).fillna(0).cumsum() # calculate the on-balance volume ans add it to the dataframe

    return df

# MACD
def calc_macd(df:DataFrame, slow:float = 24, fast:float = 16, smooth:float = 10) -> DataFrame:

    """Calculate moving average convergence/divergence."""

    exp1 = df["close"].ewm(span=fast, adjust=False).mean() # calculate the exponential moving average for the fast period
    exp2 = df["close"].ewm(span=slow, adjust=False).mean() # calculate the exponential moving average for the slow period

    macd = pd.DataFrame(exp1-exp2).rename(columns={'close':'macd'}) # calculate the moving average convergence/divergence
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'}) # calculate the signal line

    df["macd"] = macd["macd"] # add the moving average convergence/divergence to the dataframe
    df["signal"] = signal["signal"] # add the signal line to the dataframe

    return df

# Calculate all statistics
def calc_statistics(df:DataFrame, periods:int=20, slow:float = 24, fast:float = 16, smooth:float = 10) -> DataFrame:

    """Calculate all necessery statistics."""

    df = calc_volatility(df, periods) # calculate volatility
    df = calc_growth(df) # calculate growth
    df = calc_ma(df) # calculate moving averages
    df = calc_rsi(df, periods) # calculate relative strength index
    df = calc_bollinger_bends(df, periods) # calculate Bollinger Bends
    df = calc_fibonacci_retracement(df, periods) # calculate Fibonacci retracement
    df = calc_macd(df, slow, fast, smooth) # calculate moving average convergence/divergence
    df = calc_obv(df) # calculate on-balance volume
    
    return df