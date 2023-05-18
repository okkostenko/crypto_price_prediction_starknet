from functions.data import get_data, visualize_data
from functions.statistics_calculation import calc_statistics
from constants import BTC_TOKEN, ETH_TOKEN, TIMEFRAME, PERIODS

# We are making a prediction for a time point in a week 
# So we have to include at least the last 2 weeks data, etc. last 14 rows of data, if the timeframe is set to 1d
# There for periods have to be set to 14

# We will predict what close price will be on t+7 date

def get_full_data(token:str, timeframe:str, periods:int) -> None:

    """Gets historical data, calculates all nessecery statistics and saves all this information as .csv file."""

    df = get_data(token, timeframe, save=True)

    print(f"\n\nBitcoin historical price, volume and market cap data:\n")
    print(df)

    print("\n\nBitcoin dataframe info:\n")
    print(df.info())

    print(f"\n\nBitcoin descriptive statistics:\n")
    print(df.describe())

    df = calc_statistics(token, periods=periods)
    df.to_csv(f"/{token.lower()}/data/datasets/data_with_statistics_{token}_{timeframe}_full.csv")
    print("\n\nBitcoin historical data with calculated statistics:\n")
    print(df)
    visualize_data(df)

if __name__=="__main__":

    print("Collecting data for Bitcoin-Tether...")
    get_full_data(token=BTC_TOKEN, timeframe=TIMEFRAME, periods=PERIODS)
    print("Data for Bitcoin-Tether is collected\n\n")

    print("Collecting data for Ethirium-Tether...")
    get_full_data(token=ETH_TOKEN, timeframe=TIMEFRAME, periods=PERIODS)
    print("Data for Ethirium-Tether is collected\n\n")