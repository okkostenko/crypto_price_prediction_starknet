import pandas as pd
from functions.data import get_data_full, visualize_data, update_data
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
    
    df.to_csv(f"/{token.lower()}/data/datasets/data_with_statistics_{token}_{timeframe}_full.csv")
    print("\n\nBitcoin historical data with calculated statistics:\n")
    print(df)
    visualize_data(df)

# def update_historical_data(token:str, filename:str) -> None:
#     df = pd.read_csv(filename)
#     df = update_data(token, df)
#     df = calculate_statistics_by_day(df)

if __name__=="__main__":

    print("Collecting data for Bitcoin-Tether...")
    get_full_data(token=BTC_TOKEN, timeframe=TIMEFRAME, periods=PERIODS)
    print("Data for Bitcoin-Tether is collected\n\n")

    print("Collecting data for Ethirium-Tether...")
    get_full_data(token=ETH_TOKEN, timeframe=TIMEFRAME, periods=PERIODS)
    print("Data for Ethirium-Tether is collected\n\n")