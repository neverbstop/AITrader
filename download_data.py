# download_data.py
import yfinance as yf
import pandas as pd
import os

def download_ohlcv(ticker="AAPL", start="2020-01-01", end="2024-12-31", out_path="data/raw/ohlcv.csv"):
    os.makedirs("data/raw", exist_ok=True)
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.to_csv(out_path, index=False)
    print(f"✅ OHLCV data saved to {out_path}")

if __name__ == "__main__":
    download_ohlcv("AAPL")  # you can replace with RELIANCE.NS or any ticker
