import pandas as pd

from data_utils import get_data, visualize_data
from statistics_calculation import calc_statistics

btc = get_data('BTC', start='2023-04-06', end='2023-05-06')

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