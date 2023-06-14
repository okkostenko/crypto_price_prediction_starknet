import numpy as np
import pandas as pd
import plotly.graph_objects as go
from cryptocmd import CmcScraper
from typing import Union
import pandas as pd
import math
import os.path
import time
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook

from constants import TIMEFRAME
from functions.f_and_g_index import get_gf_index

from authkeys import BINANCE_API_KEY, BINANCE_API_SECRET

binance_api_key = BINANCE_API_KEY
binance_api_secret = BINANCE_API_SECRET

binsizes = {Client.KLINE_INTERVAL_15MINUTE: 15, 
            Client.KLINE_INTERVAL_30MINUTE: 30, 
            Client.KLINE_INTERVAL_1HOUR: 60, 
            Client.KLINE_INTERVAL_12HOUR: 720, 
            Client.KLINE_INTERVAL_1DAY: 1440}

binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

def minutes_of_new_data(symbol:str, timeframe:str, data:pd.DataFrame, source:str) -> Union[pd.Timestamp, datetime]:

    """Calculate ms from the last timepoint to the current time."""

    if len(data) > 0:  
        old = parser.parse(data["timestamp"].iloc[-1]) # old is the last timepoint we have in the data
    elif source == "binance": 
        old = datetime.strptime('1 Jan 2018', '%d %b %Y') # if we don't have any timepoints in the data, set the old timepoint to 1 Jan 2017

    if source == "binance": 
        new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=timeframe)[-1][0], unit='ms') # new is the current timepoint

    return old, new


def get_historical_data(symbol:str, filepath:str, timeframe:str|None = Client.KLINE_INTERVAL_12HOUR) -> Union[pd.DataFrame, datetime, datetime]:

    """Get historical data."""

    if os.path.isfile(filepath): # if the file exists, read the data from the CSV
        df = pd.read_csv(filepath)
    else: 
        df = pd.DataFrame()

    # Calculate the oldest and the newest time point of the available data.
    oldest_point, newest_point = minutes_of_new_data(symbol, timeframe, df, source = "binance") # get the last timepoint we have in the data
    print(type(newest_point))
    print(type(oldest_point))
    delta_min = (newest_point - oldest_point).total_seconds()/60 # calculate the difference between the last timepoint and the current timepoint in minutes

    # Available new data
    available_data = math.ceil(delta_min/binsizes[timeframe]) # calculate the number of new timepoints available

    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): 
        print(f'Downloading all available {timeframe} data for {symbol}..')
    else: 
        print(f'Downloading {available_data} instances of {timeframe} new data available for {symbol}..')

    # Download the data
    klines = binance_client.get_historical_klines(symbol, 
                                                  timeframe, 
                                                  oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                  newest_point.strftime("%d %b %Y %H:%M:%S")) # download the data from the last timepoint to the current timepoint

    # Create a dataframe 
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']) # create a dataframe from the downloaded data

    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']] # keep only the columns we need

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    print(data['timestamp'])

    # if we already have some of the data, append the new data to the existing dataframe
    if len(df) > 0: 
        temp_df = pd.DataFrame(data)
        df = pd.concat([df, temp_df], axis = 1)
    else: 
        df = data

    print('Historical price data is downloaded!')

    return df, oldest_point, newest_point


def get_market_cap(ticker:str, start_date:str|None=None, end_date:str|None=None) -> pd.DataFrame:

    """Get daily historical market capitalization data by ticker."""

    scraper = CmcScraper(ticker, start_date=start_date, end_date=end_date) # initialise scraper
    market_cap = scraper.get_dataframe() # get market cap data
    market_cap = market_cap[["Date", "Market Cap"]].rename(columns={"Date":"date", "Market Cap":"market_cap"}) 

    return market_cap


def merge_history_data_and_market_cap(symbol:str, df:pd.DataFrame) -> pd.DataFrame:

    """Merges historical data and market dataframes."""

    tiker = symbol.replace("USDT", "")

    # GET HISTORICAL MARKET CAP DATA
    df["date"] = pd.to_datetime(df["timestamp"].dt.date) # create a new column with date only

    market_cap = get_market_cap(tiker) # get market cap data
    df=df.merge(market_cap, on="date", how="left") # merge historical data and market cap data

    df.drop(["date"], axis="columns", inplace=True) # drop date column

    df.set_index("timestamp", inplace=True) # set timestamp as index
    
    df = convert_dtypes(df) # convert data types, so that we can use the data for modelling

    return df


def get_data_full(symbol:str, timeframe:str|None = TIMEFRAME, save:bool|None = False) -> pd.DataFrame:

    """Get historical price, volume and market capitalization data by ticker."""

    filepath = f'./data_collection/datasets/full/historical_{symbol}_{timeframe}_full.csv' # set the path where the data will be saved
    os.makedirs("./data_collection/datasets/full", exist_ok=True) # create a folder for the data
    
    # Get historical data
    df, _, _ = get_historical_data(symbol=symbol, filepath=filepath, timeframe=timeframe)

    # Merge historical data and market capitalization
    df = merge_history_data_and_market_cap(symbol=symbol, df=df)

    if save: 
        df.to_csv(filepath)
    
    return df


def visualize_data(df:pd.DataFrame) -> None:

    """Vizualize data."""

    fig = go.Figure() 

    #Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'], name = 'market data'))

    # Add titles
    fig.update_layout(
        title='Ethereum live share price evolution',
        yaxis_title='Ethereum Price (kUS Dollars)')

    # X-Axes
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    #Show
    fig.show()


def convert_dtypes(df:pd.DataFrame) -> pd.DataFrame:

    """Convert the type of the object type columns to float."""

    object_columns = [column_name for column_name in df.columns if df[column_name].dtypes == "object"] # get the names of the columns with object type
    df[object_columns] = df[object_columns].astype("float") # convert the type of the object type columns to float

    return df


def add_sentiments() -> pd.DataFrame:

    """Add sentiments to the data."""

    df = pd.read_csv("data_collection/datasets/full/data_with_statistics_ETHUSDT_1d_full.csv") # read the historical data with statistics calculated
    news_df = pd.read_csv("data_collection/datasets/sentiments/news_with_financial_summary_and_sentiment.csv") # read the news data with sentiments
    gf_index = get_gf_index() # det the greed and fear index

    news_df["sentiment"] = news_df["sentiment"].apply(lambda x: 1 if x == "Bearish" else 2 if x == "Bullish" else 0) # convert the sentiment to categorical variable
    sentiments = news_df["sentiment"].groupby(news_df["date"]).agg(pd.Series.mode).reset_index() # get the most frequent sentiment for each day
    
    df["date"] = pd.to_datetime(df["timestamp"]) # convert the timestamp to date
    del df["timestamp"]

    sentiments.reset_index(inplace=True) 
    sentiments["date"] = pd.to_datetime(sentiments["date"]) # convert the timestamp to date

    gf_index = gf_index.reset_index()
    gf_index["date"] = pd.to_datetime(gf_index["timestamp"])
    gf_index["gf-index"] = gf_index["gf-index"].astype("int") # convert the G&F-index to int
    del gf_index["timestamp"]

    sentiments_gf = pd.merge(sentiments, gf_index, on="date", how="left") # merge the sentiments and G&F-index
    df = pd.merge(df, sentiments_gf, on="date", how="left") # merge the historical data with sentiments and G&F-index 
    del df["index"]
    df.dropna(inplace=True) # drop the rows with NaN values 
    df.to_csv("data_collection/datasets/sentiments/data_with_sentiments.csv") # save the data
    return df

if __name__ == "__main__":
    add_sentiments()