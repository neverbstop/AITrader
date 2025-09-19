import pandas as pd
from config import CAPITAL, BROKERAGE_PERCENT, TAX_PERCENT, THRESHOLD_PROFIT_PERCENT, AUTONOMOUS_BUDGET, AUTONOMOUS_ENABLED, TICKER
from risk_management import apply_fees

class AutonomousAgent:
    def __init__(self, ticker):
        self.ticker = ticker
        self.enabled = AUTONOMOUS_ENABLED
        self.budget = AUTONOMOUS_BUDGET
        self.cash = self.budget
        self.position = 0
        self.entry_price = 0.0
        self.trades = []
        self.is_in_position = False
        self.last_trade_action = None

    def enable(self):
        """
        Enables the autonomous trading functionality.
        """
        self.enabled = True
        print("✅ Autonomous Agent is now ENABLED. It will execute trades automatically.")

    def disable(self):
        """
        Disables the autonomous trading functionality.
        """
        self.enabled = False
        print("❌ Autonomous Agent is now DISABLED. It will continue to monitor but not trade.")

    def run_if_enabled(self, news_signal=None, quant_signal=None, current_price=None):
        """
        Main method to be called by the orchestrator.
        It executes trading logic only if the agent is enabled.
        """
        if not self.enabled or current_price is None:
            return

        print(f"🤖 Autonomous Agent is running... Current Price: {current_price:.2f}")

        # Check for sell conditions first to prioritize profit-taking and loss prevention
        # This includes both crossover signals and threshold sells
        self._handle_sells(news_signal, quant_signal, current_price)

        # Then, check for buy conditions
        self._handle_buys(news_signal, quant_signal, current_price)
    
    def _handle_buys(self, news_signal, quant_signal, current_price):
        """
        Handles the buying logic for the agent.
        """
        # The bot will only buy if it's not already in a position
        # AND both the news and quantitative agents give a BUY signal.
        if not self.is_in_position and news_signal and "positive" in news_signal.lower() and quant_signal == "QUANT_BUY":
            price_with_fees = current_price * (1 + BROKERAGE_PERCENT + TAX_PERCENT)
            if self.cash > price_with_fees:
                shares_to_buy = self.cash // price_with_fees
                if shares_to_buy > 0:
                    self.entry_price = current_price
                    buy_cost = shares_to_buy * current_price
                    self.position += shares_to_buy
                    self.cash -= buy_cost * (1 + BROKERAGE_PERCENT + TAX_PERCENT)
                    self.is_in_position = True
                    self.last_trade_action = "BUY"
                    self.trades.append({"date": pd.Timestamp.now(), "action": "BUY", "price": current_price,
                                        "shares": shares_to_buy, "pnl": None})
                    print(f"✅ BUY {shares_to_buy} @ {current_price:.2f} | Cash: {self.cash:.2f}")

    def _handle_sells(self, news_signal, quant_signal, current_price):
        """
        Handles the selling logic for the agent.
        """
        if self.is_in_position:
            # First, check for profit threshold sell
            profit_percent = ((current_price - self.entry_price) / self.entry_price) * 100
            if profit_percent >= THRESHOLD_PROFIT_PERCENT:
                self._execute_sell("SELL (threshold)", current_price)
            
            # Second, check for quantitative agent sell signal
            elif quant_signal == "QUANT_SELL":
                self._execute_sell("SELL (crossover)", current_price)

    def _execute_sell(self, action_type, current_price):
        """
        Executes a sell order and updates the agent's state.
        """
        if self.position > 0:
            sell_value = self.position * current_price
            cash_from_sell = apply_fees(sell_value)
            self.cash += cash_from_sell
            pnl = (current_price - self.entry_price) * self.position
            self.trades.append({"date": pd.Timestamp.now(), "action": action_type, "price": current_price,
                                "shares": self.position, "pnl": pnl})
            print(f"📉 {action_type} {self.position} @ {current_price:.2f} | Cash: {self.cash:.2f} | PnL: {pnl:.2f}")
            self.position = 0
            self.is_in_position = False
            self.last_trade_action = "SELL"
    
    def get_portfolio_value(self, current_price):
        """
        Calculates the current total portfolio value.
        """
        return self.cash + (self.position * current_price)
