import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from cryptocmd import CmcScraper


def get_data(ticker:str, period:str|None='max', start:str|None=None, end:str|None=None) -> pd.DataFrame:
    
    """Get historical price, volume and market capitalization data by ticker."""

    df = yf.download(tickers=ticker+'-USD', start=start, end=end, interval='15m').reset_index()
    df["Date"] = pd.to_datetime(df["Datetime"].dt.date)

    market_cap = get_market_cap(ticker)
    df=df.merge(market_cap, on="Date", how="left")

    df.drop(["Date"], axis="columns", inplace=True)
    df.set_index("Datetime", inplace=True)

    return df

def get_market_cap(ticker:str) -> pd.DataFrame:

    """Get daily historical market capitalization data by ticker."""

    scraper = CmcScraper("BTC")
    market_cap = scraper.get_dataframe()
    market_cap = market_cap[["Date", "Market Cap"]]

    return market_cap

def visualize_data(df:pd.DataFrame) -> None:

    """Vizualize data."""

    fig = go.Figure()

    #Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name = 'market data'))

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