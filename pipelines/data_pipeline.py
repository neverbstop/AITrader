# pipelines/data_pipeline.py
import polars as pl
import yfinance as yf
from pathlib import Path

class DataPipeline:
    """
    Handles stock market data collection, cleaning, and feature engineering.
    If the stock data file does not exist, it will be downloaded automatically.
    """

    def __init__(self, stock_file: str, ticker: str):
        self.stock_file = Path(stock_file)
        self.ticker = ticker
        self.data = None

    def _download_data(self):
        """Downloads historical stock data using yfinance if the local file is missing."""
        print(f"Stock file not found at '{self.stock_file}'.")
        print(f"Downloading data for ticker: {self.ticker}...")

        # Ensure the data directory exists
        self.stock_file.parent.mkdir(parents=True, exist_ok=True)

        # Download 5 years of historical data, which is a good range for backtesting
        stock_data = yf.download(self.ticker, period="5y", auto_adjust=True)

        if stock_data.empty:
            raise ConnectionError(
                f"Failed to download data for '{self.ticker}'. "
                "Check the ticker symbol or your internet connection."
            )

        # Save to CSV
        stock_data.to_csv(self.stock_file)
        print(f"Data successfully saved to '{self.stock_file}'")

    def load_data(self):
        """Load historical stock data. If the file doesn't exist, download it."""
        if not self.stock_file.exists():
            self._download_data()

        # yfinance saves the date as an index, which becomes the first column.
        # try_parse_dates will correctly handle the 'Date' column.
        self.data = pl.read_csv(self.stock_file, try_parse_dates=True)
        return self.data

    def add_features(self):
        """Add technical indicators (basic SMA now, expand later)"""
        if self.data is None:
            raise ValueError("Data not loaded yet!")
        df = self.data.with_columns([
            pl.col("Close").rolling_mean(5).alias("SMA5"),
            pl.col("Close").rolling_mean(20).alias("SMA20")
        ])
        self.data = df
        return self.data
    def run(self):
        """Full pipeline: load → feature engineering → return dataframe"""
        self.load_data()
        self.add_features()
        return self.data
