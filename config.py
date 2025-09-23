# config.py

# ====================
# API KEYS & SECRETS
# ====================
# Best practice is to load these from environment variables or a .env file
# To set up, create a .env file in your project root with these variables:
# NEWS_API_KEY="your_news_api_key_here"
# TRADING_API_KEY="your_trading_api_key_here"
# TRADING_API_SECRET="your_trading_api_secret_here"
# Ensure the .env file is added to your .gitignore!
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TRADING_API_KEY = os.getenv("TRADING_API_KEY")
TRADING_API_SECRET = os.getenv("TRADING_API_SECRET")

# ====================
# BOT SETTINGS
# ====================
TICKER = "RELIANCE.NS"
CAPITAL = 10000.0  # Initial portfolio capital
CHECK_INTERVAL_SECONDS = 60  # How often the main loop runs

# ====================
# FILE PATHS
# ====================
STOCK_FILE = f"data/{TICKER}.csv"

# ====================
# AGENT CONFIGURATIONS
# ====================

# Quantitative Agent (Technical Analysis)
QUANT_SHORT_WINDOW = 20
QUANT_LONG_WINDOW = 50

# Autonomous Agent (Self-Managed)
AUTONOMOUS_ENABLED = False  # Set to True to enable the autonomous bot
AUTONOMOUS_BUDGET = 200.0   # Maximum expense limit for this agent
THRESHOLD_PROFIT_PERCENT = 2.0 # Sell when profit reaches this percentage

# ====================
# TRADING FEES
# ====================
BROKERAGE_PERCENT = 0.001 # 0.1%
TAX_PERCENT = 0.0005      # 0.05%