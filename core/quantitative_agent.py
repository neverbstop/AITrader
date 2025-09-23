import yfinance as yf
import pandas as pd
from config import QUANT_SHORT_WINDOW, QUANT_LONG_WINDOW
from datetime import datetime, timedelta

class QuantitativeAgent:
    def __init__(self, ticker):
        self.ticker = ticker
        self.short_window = QUANT_SHORT_WINDOW
        self.long_window = QUANT_LONG_WINDOW
        self.data = None

    def fetch_data(self, days_to_fetch=100):
        """
        Fetches historical data from Yahoo Finance for a specified period.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_fetch)
        
        try:
            df = yf.download(self.ticker, start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                print(f"⚠️ No data downloaded for {self.ticker}.")
                return None
            self.data = df
            return self.data
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            return None

    def moving_average_crossover(self):
        """
        Generates trading signals based on the moving average crossover strategy.
        """
        if self.data is None or len(self.data) < self.long_window:
            return "Insufficient data for moving average calculation."

        # Calculate short and long moving averages
        self.data["SMA_Short"] = self.data["Close"].rolling(window=self.short_window).mean()
        self.data["SMA_Long"] = self.data["Close"].rolling(window=self.long_window).mean()

        # Generate signals based on crossover
        self.data["Signal"] = 0
        # Buy signal when short MA crosses above long MA
        self.data.loc[self.data["SMA_Short"] > self.data["SMA_Long"], "Signal"] = 1
        # Sell signal when short MA crosses below long MA
        self.data.loc[self.data["SMA_Short"] < self.data["SMA_Long"], "Signal"] = -1
        
        # We correct for lookahead bias by shifting the signal one day back.
        # A signal on day T is a decision for day T+1.
        self.data["Signal"] = self.data["Signal"].shift(1)

        # Determine if a crossover has occurred in the most recent data point
        recent_signal = self.data["Signal"].iloc[-1]
        previous_signal = self.data["Signal"].iloc[-2]

        if recent_signal == 1 and previous_signal <= 0:
            return "QUANT_BUY"
        elif recent_signal == -1 and previous_signal >= 0:
            return "QUANT_SELL"
        else:
            return "QUANT_HOLD"

    def check_for_signals(self):
        """
        Main method to be called by the orchestrator.
        Fetches fresh data and runs the strategy.
        """
        self.fetch_data()
        return self.moving_average_crossover()

if __name__ == "__main__":
    # This block is for testing the QuantitativeAgent independently
    quant_agent = QuantitativeAgent(ticker="RELIANCE.NS")
    print("Quantitative Agent initialized.")
    signal = quant_agent.check_for_signals()
    print(f"Latest signal from Quantitative Agent: {signal}")
