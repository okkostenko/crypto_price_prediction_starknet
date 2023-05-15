import time
import pandas as pd
from functions.utils import get_data, visualize_data
from functions.statistics_calculation import calc_statistics

btc = get_data("BTCUSDT", "1h", save=True)

print(f"\n\nBitcoin historical price, volume and market cap data:\n")
print(btc)

print("\n\nBitcoin dataframe info:\n")
print(btc.info())

print(f"\n\nBitcoin descriptive statistics:\n")
print(btc.describe())

btc = calc_statistics(btc, periods=20)

print("\n\nBitcoin historical data with calculated statistics:\n")
print(btc)
visualize_data(btc)

time.sleep(30)