import numpy as np
import pandas as pd
from cryptocmd import CmcScraper
from binance.client import Client
from datetime import timedelta, datetime
from typing import Union
from data_collection.functions.data import convert_dtypes, get_market_cap, get_data_full

from data_collection.constants import TIMEFRAME
from data_collection.authkeys import BINANCE_API_KEY, BINANCE_API_SECRET

binance_api_key = BINANCE_API_KEY
binance_api_secret = BINANCE_API_SECRET

binsizes = {Client.KLINE_INTERVAL_15MINUTE: 15, 
            Client.KLINE_INTERVAL_30MINUTE: 30, 
            Client.KLINE_INTERVAL_1HOUR: 60, 
            Client.KLINE_INTERVAL_12HOUR: 720, 
            Client.KLINE_INTERVAL_1DAY: 1440}

binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

def get_last_historical_data(symbol:str, timeframe:str, period:int) -> Union[pd.DataFrame, datetime, datetime]:
    
    today = datetime.now()
    last_days = today - timedelta(days=period+1)

    klines = binance_client.get_historical_klines(symbol=symbol, interval=timeframe, start_str=last_days.strftime("%d %b %Y %H:%M:%S"), end_str=today.strftime("%d %b %Y %H:%M:%S"))

    # Create a dataframe 
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    return data, last_days, today


def merge_last_market_cap(symbol:str, historical_df:pd.DataFrame, start_date:datetime, end_date:datetime) -> pd.DataFrame:

    tiker = symbol.replace("USDT", "")

    # GET HISTORICAL MARKET CAP DATA
    historical_df["date"] = pd.to_datetime(historical_df["timestamp"].dt.date)

    market_cap = get_market_cap(tiker, start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y"))
    historical_df=historical_df.merge(market_cap, on="date", how="left")

    historical_df.drop(["date"], axis="columns", inplace=True)

    historical_df.set_index("timestamp", inplace=True)
    
    historical_df = convert_dtypes(historical_df)

    return historical_df

def calculate_last_statistics():
    ...

def get_last_36_days_data():
    ...