import time
import pandas as pd
from datetime import datetime
import os
import yfinance as yf
from news_agent import NewsAgent
from quantitative_agent import QuantitativeAgent
from autonomous_agent import AutonomousAgent
from config import TICKER, CAPITAL, CHECK_INTERVAL_SECONDS
from dashboard import run_dashboard # Assuming we will update this later

def main():
    """
    The main orchestrator for the AI trading bot.
    Initializes all agents and runs the main trading loop.
    """
    print("🤖 Initializing AI Trading Bot...")

    # Initialize agents
    news_agent = NewsAgent(ticker=TICKER)
    quant_agent = QuantitativeAgent(ticker=TICKER)
    autonomous_agent = AutonomousAgent(ticker=TICKER)

    # State variables for the main loop
    trades_summary = pd.DataFrame(columns=['date', 'action', 'price', 'shares', 'pnl'])
    performance_summary = pd.DataFrame()
    last_price = 0.0

    print("✅ All agents are initialized.")
    print(f"Starting live trading simulation for {TICKER}...")
    
    # Let's assume the autonomous agent starts in disabled mode
    # autonomous_agent.disable() 

    # We need to get the current price to start the loop
    try:
        current_data = yf.download(TICKER, period="1d", interval="1m", auto_adjust=True)
        if not current_data.empty:
            last_price = current_data['Close'].iloc[-1]
            print(f"Current market price for {TICKER}: {last_price:.2f}")
        else:
            print("❌ Could not fetch current price. Exiting.")
            return
    except Exception as e:
        print(f"❌ Error fetching initial data: {e}. Exiting.")
        return
        
    while True:
        try:
            # Step 1: Get the current price of the stock
            # In a real-time scenario, this would be a live feed
            try:
                current_data = yf.download(TICKER, period="1d", interval="1m", auto_adjust=True)
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                else:
                    print(f"⚠️ Could not fetch real-time price for {TICKER}. Retrying...")
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue
            except Exception as e:
                print(f"❌ Error fetching real-time price: {e}. Retrying...")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            
            # Step 2: Get signals from all agents
            news_signal, _ = news_agent.check_for_signals()
            quant_signal = quant_agent.check_for_signals()

            # Print current signals for user visibility
            print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 Quant Signal: {quant_signal} | 📰 News Signal: {news_signal}")
            print(f"Current Portfolio Value: {autonomous_agent.get_portfolio_value(current_price):.2f}")

            # Step 3: Autonomous Agent makes a decision
            autonomous_agent.run_if_enabled(news_signal, quant_signal, current_price)

            # --- Future improvements for the manual user control ---
            # if news_signal and "positive" in news_signal.lower():
            #     print("User Alert: Positive news detected! Do you want to BUY? (y/n)")
            #     user_input = input() # This would be a real-time chat interface
            #     if user_input.lower() == 'y':
            #         # Manual buy logic
            #         pass
            # ----------------------------------------------------

        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
