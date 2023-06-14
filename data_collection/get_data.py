import os
import pandas as pd
from functions.data import get_data_full, visualize_data, add_sentiments
from functions.statistics_calculation import calc_statistics
from constants import ETH_TOKEN, TIMEFRAME, PERIODS


def get_full_data(token:str, timeframe:str, periods:int) -> None:

    """Gets historical data, calculates all nessecery statistics and saves all this information as .csv file."""

    df = get_data_full(token, timeframe, save=True) # get the historical data
    df = calc_statistics(df=df, periods=periods) # calculate the statistics

    df["label"] = df["growth"].shift(-1) # shift the label column by one row

    df.dropna(inplace=True) # drop the NaN values
    
    df.to_csv(f'./data_collection/datasets/full/data_with_statistics_{token}_{TIMEFRAME}_full.csv') # save the data
    print("\n\nBitcoin historical data with calculated statistics:\n") 
    visualize_data(df) # visualize the data

def statistics_growth(filename:str, token:str) -> None:

    """Calculates the statistics for the growth of the token."""

    df = pd.read_csv(filename) # read the data
    
    df = calc_statistics(df=df, periods=PERIODS) # calculate the statistics

    df["label"] = df["growth"].shift(-1) # shift the label column by one row

    df.dropna(inplace=True)

    new_filepath = f'./data_collection/datasets/full/data_with_statistics_{token+"USDT"}_{TIMEFRAME}_full.csv' # set the path where the data will be saved
    os.makedirs("./data_collection/datasets/full", exist_ok=True) # create a folder for the data
    
    df.to_csv(new_filepath) # save the data
    print("\n\Ethereum historical data with calculated statistics:\n")
    visualize_data(df)

if __name__=="__main__":

    print("Collecting data for Ethirium-Tether...")
    get_full_data(token=ETH_TOKEN, timeframe=TIMEFRAME, periods=PERIODS)
    print("Data for Ethirium-Tether is collected\n\n")
    add_sentiments()