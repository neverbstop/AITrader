# data_pipeline.py
import yfinance as yf
import pandas as pd


def load_data_from_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Reset index to keep Date as a column
    df = df.reset_index()

    # Ensure columns exist (Adj Close may not exist depending on auto_adjust)
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        keep_cols.append("Adj Close")

    df = df[keep_cols]
    return df


def moving_average_crossover(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Add moving averages and generate trading signals based on crossover strategy.
    """
    df["SMA_Short"] = df["Close"].rolling(window=short_window, min_periods=1).mean()
    df["SMA_Long"] = df["Close"].rolling(window=long_window, min_periods=1).mean()

    # BUY when short MA crosses above long MA, SELL when short MA crosses below
    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1   # BUY
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1  # SELL

    # Correct for lookahead bias by shifting the signal.
    # A signal generated on day T is used for a trade on day T+1.
    df["Signal"] = df["Signal"].shift(1).fillna(0)

    return df