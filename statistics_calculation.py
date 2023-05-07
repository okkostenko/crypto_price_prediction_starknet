import numpy as np
import pandas as pd


def calc_ma(df:pd.DataFrame)->pd.DataFrame:
    
    """Calculate moving averages for the last 50 and 200 datapoints."""

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    return df