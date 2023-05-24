import numpy as np
import pandas as pd
from pandas import DataFrame


# Volatility
def calc_volatility(df:DataFrame, periods:int = 20) -> DataFrame:

    """Calculate volatility."""

    log_returns = np.log(df["close"]/df["close"].shift())
    volatility = log_returns.rolling(periods).std()*periods**.5

    df["volatility"] = volatility

    return df

# Growth
def calc_growth(df:DataFrame) -> DataFrame:
    growth = df["close"] - df["open"]
    df["growth"] = growth
    
    return df

# Moving Averege
def calc_ma(df:DataFrame)->DataFrame:
    
    """Calculate moving averages for the last 7, 25 and 99 datapoints."""
    print(df.info())
    print(df["close"])

    df["ma7"] = df["close"].rolling(7).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    df["ma99"] = df["close"].rolling(99).mean()

    return df

# RSI    
def calc_rsi(df:DataFrame, periods:int = 20, ema:bool = True) -> DataFrame:
    
    """Calculate relative strength index."""

    close_delta = df['close'].diff()
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
    df["rsi"] = rsi

    return df

# Bollinger Bends
def calc_bollinger_bends(df:DataFrame, periods:int = 20) -> DataFrame:

    """Claculate Bollinger Bends."""

    df["std"] = df["close"].rolling(periods).std()

    df["bb-mb"] = df["close"].rolling(periods).mean()
    df["bb-ub"] = df["bb-mb"] + 2 * df["std"]
    df["bb-lb"] = df["bb-mb"] - 2 * df["std"]

    return df

# Fibonacci retracement
def calc_fibonacci_retracement(df:DataFrame, periods:int=20) -> DataFrame:

    """Calculate Fibonacci retracement."""

    max_value = df["close"].rolling(periods).max()
    min_value = df["close"].rolling(periods).min()
    difference = max_value - min_value
    df["fr-0.236"] = max_value - difference * 0.236
    df["fr-0.382"] = max_value - difference * 0.382
    df["fr-0.5"] = max_value - difference * 0.5
    df["fr-0.618"] = max_value - difference * 0.618

    return df


# OBV
def calc_obv(df:DataFrame) -> DataFrame:

    """Calculate On-balance volume."""
    df["obv"] = (np.sign(df["close"].diff())* df["volume"]).fillna(0).cumsum()

    return df

# MACD
def calc_macd(df:DataFrame, slow:float = 24, fast:float = 16, smooth:float = 10) -> DataFrame:

    """Calculate moving average convergence/divergence."""

    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()

    macd = pd.DataFrame(exp1-exp2).rename(columns={'close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})

    df["macd"] = macd["macd"]
    df["signal"] = signal["signal"]

    return df

# Calculate all statistics
def calc_statistics(df:DataFrame, periods:int=20, slow:float = 24, fast:float = 16, smooth:float = 10) -> DataFrame:

    """Calculate all necessery statistics."""

    df = calc_volatility(df, periods)
    df = calc_growth(df)
    print(df)
    df = calc_ma(df)
    df = calc_rsi(df, periods)
    df = calc_bollinger_bends(df, periods)
    df = calc_fibonacci_retracement(df, periods)
    df = calc_macd(df, slow, fast, smooth)
    df = calc_obv(df)
    
    return df

def calculate_statistics_by_day(df: pd.DataFrame):
    ...