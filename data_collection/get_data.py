import os
import pandas as pd
from functions.data import get_data_full, visualize_data, add_sentiments
from functions.statistics_calculation import calc_statistics, calculate_statistics_by_day
from constants import BTC_TOKEN, ETH_TOKEN, TIMEFRAME, PERIODS

# We are making a prediction for a time point in a week 
# So we have to include at least the last 2 weeks data, etc. last 14 rows of data, if the timeframe is set to 1d
# There for periods have to be set to 14

# We will predict what close price will be on t+7 date

def get_full_data(token:str, timeframe:str, periods:int) -> None:

    """Gets historical data, calculates all nessecery statistics and saves all this information as .csv file."""

    df = get_data_full(token, timeframe, save=True)
    df = calc_statistics(df=df, periods=periods)

    df["label"] = df["growth"].shift(-1)

    # df.dropna(subset=["label"], inplace=True)
    # df.fillna(0, inplace=True)
    df.dropna(inplace=True)
    
    df.to_csv(f'./data_collection/datasets/full/data_with_statistics_{token}_{TIMEFRAME}_full.csv')
    print("\n\nBitcoin historical data with calculated statistics:\n")
    visualize_data(df)

def statistics_growth(filename:str, token:str) -> None:
    df = pd.read_csv(filename)
    
    df = calc_statistics(df=df, periods=PERIODS)

    df["label"] = df["growth"].shift(-1)

    # df.dropna(subset=["label"], inplace=True)
    # df.fillna(0, inplace=True)
    df.dropna(inplace=True)

    new_filepath = f'./data_collection/datasets/full/data_with_statistics_{token+"USDT"}_{TIMEFRAME}_full.csv'
    os.makedirs("./data_collection/datasets/full", exist_ok=True)
    
    df.to_csv(new_filepath)
    print("\n\nBitcoin historical data with calculated statistics:\n")
    visualize_data(df)

if __name__=="__main__":

    # print("Collecting data for Ethirium-Tether...")
    # get_full_data(token=ETH_TOKEN, timeframe=TIMEFRAME, periods=PERIODS)
    # print("Data for Ethirium-Tether is collected\n\n")

    add_sentiments()