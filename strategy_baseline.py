# strategy_baseline.py
import pandas as pd

def moving_average_crossover(df, short_window=20, long_window=50):
    """
    Adds trading signals (BUY/SELL/HOLD) based on moving average crossover.
    """
    df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()

    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1   # BUY
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1  # SELL

    return df
