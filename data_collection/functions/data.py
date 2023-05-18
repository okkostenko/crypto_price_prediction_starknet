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

from data_collection.authkeys import BINANCE_API_KEY, BINANCE_API_SECRET

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
        old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": 
        old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    if source == "binance": 
        new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=timeframe)[-1][0], unit='ms')

    return old, new

def get_historical_data(symbol:str, filepath:str, timeframe:str|None = Client.KLINE_INTERVAL_12HOUR) -> pd.DataFrame:

    """Get historical data."""

    if os.path.isfile(filepath): 
        df = pd.read_csv(filepath)
    else: 
        df = pd.DataFrame()

    # Calculate the oldest and the newest time point of the available data.
    oldest_point, newest_point = minutes_of_new_data(symbol, timeframe, df, source = "binance")
    print(type(newest_point))
    print(type(oldest_point))
    delta_min = (newest_point - oldest_point).total_seconds()/60

    # Available new data
    available_data = math.ceil(delta_min/binsizes[timeframe])

    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): 
        print(f'Downloading all available {timeframe} data for {symbol}..')
    else: 
        print(f'Downloading {available_data} instances of {timeframe} new data available for {symbol}..')

    # Download the data
    klines = binance_client.get_historical_klines(symbol, timeframe, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))

    # Create a dataframe 
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    if len(df) > 0:
        temp_df = pd.DataFrame(data)
        df = pd.concat([df, temp_df], axis = 1)
    else: 
        df = data

    print(df)
    print('Historical price data is downloaded!')

    return df

def get_market_cap(ticker:str) -> pd.DataFrame:

    """Get daily historical market capitalization data by ticker."""

    scraper = CmcScraper(ticker)
    market_cap = scraper.get_dataframe()
    market_cap = market_cap[["Date", "Market Cap"]].rename(columns={"Date":"date", "Market Cap":"market_cap"})

    return market_cap

def merge_history_data_and_market_cap(symbol:str, df:pd.DataFrame) -> pd.DataFrame:

    """Merges historical data and market dataframes."""

    tiker = symbol.replace("USDT", "")

    # GET HISTORICAL MARKET CAP DATA
    df["date"] = pd.to_datetime(df["timestamp"].dt.date)

    market_cap = get_market_cap(tiker)
    df=df.merge(market_cap, on="date", how="left")

    df.drop(["date"], axis="columns", inplace=True)

    df.set_index("timestamp", inplace=True)
    
    df = convert_dtypes(df)

    return df

def get_data(symbol:str, timeframe:str|None = TIMEFRAME, save:bool|None = False) -> pd.DataFrame:

    """Get historical price, volume and market capitalization data by ticker."""

    filepath = f'/{symbol.lower()}/data/datasets/historical_{symbol}_{timeframe}_full.csv'
    
    # Get historical data
    df = get_historical_data(symbol=symbol, filepath=filepath, timeframe=timeframe)

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
        title='Bitcoin live share price evolution',
        yaxis_title='Bitcoin Price (kUS Dollars)')

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

    object_columns = [column_name for column_name in df.columns if df[column_name].dtypes == "object"]
    df[object_columns] = df[object_columns].astype("float")

    return df