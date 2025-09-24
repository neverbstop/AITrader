# config.py - Enhanced for Indian Market with XAI Integration
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ====================
# API KEYS & SECRETS
# ====================
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TRADING_API_KEY = os.getenv("TRADING_API_KEY")
TRADING_API_SECRET = os.getenv("TRADING_API_SECRET")

# ====================
# BOT SETTINGS - INDIAN MARKET FOCUS
# ====================
TICKER = "AAPL" # Apple Inc. - Changed from RELIANCE.NS as requested
COMPANY_NAME = "Apple Inc."
MARKET = "INDIA" # For future expansion awareness
CURRENCY = "USD" # Apple trades in USD even when analyzed from India

CAPITAL = 10000.0 # Initial portfolio capital
CHECK_INTERVAL_SECONDS = 1800 # 30 minutes (Indian trading session is shorter)

# NEW: Processing mode for batch approach
PROCESSING_MODE = "batch" # batch processing as per our plan
ENABLE_XAI = True # Enable explainable AI features

# ====================
# INDIAN MARKET SPECIFIC SETTINGS
# ====================
INDIAN_MARKET_CONFIG = {
   'timezone': 'Asia/Kolkata',
   'trading_hours': {'open': '09:15', 'close': '15:30'},
   'currency_primary': 'INR', # For Indian investor perspective
   'currency_stock': 'USD',   # Apple stock currency
   'consider_currency_impact': True # USD-INR fluctuation impact
}

# ====================
# ENHANCED FILE PATHS
# ====================
# Ensure directories exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
EXPLANATION_DIR = Path("explanations")
EXPLANATION_DIR.mkdir(exist_ok=True)

# File paths with better naming
STOCK_FILE = DATA_DIR / f"{TICKER}_stock_data.csv"
NEWS_FILE = DATA_DIR / f"{TICKER}_news_data.csv"
SENTIMENT_FILE = DATA_DIR / f"{TICKER}_sentiment_data.csv"

# NEW: Logging and explanation files
PERFORMANCE_LOG = LOG_DIR / "performance.log"
DECISION_LOG = LOG_DIR / "trading_decisions.log"
EXPLANATION_LOG = LOG_DIR / "xai_explanations.log"

# ====================
# AGENT CONFIGURATIONS - ENHANCED
# ====================

# Quantitative Agent (Technical Analysis) - Enhanced
QUANT_CONFIG = {
   # Your existing parameters
   'short_window': 20, # QUANT_SHORT_WINDOW
   'long_window': 50,  # QUANT_LONG_WINDOW

   # NEW: Additional technical indicators
   'indicators': {
       'sma_periods': [5, 20, 50, 200],
       'ema_periods': [12, 26],
       'rsi_period': 14,
       'macd_fast': 12,
       'macd_slow': 26,
       'macd_signal': 9,
       'bollinger_period': 20,
       'atr_period': 14
   },

   # NEW: LSTM configuration for baseline testing
   'lstm_model': {
       'sequence_length': 60,
       'hidden_size': 128,
       'num_layers': 2,
       'dropout': 0.2,
       'batch_size': 32,
       'learning_rate': 0.001,
       'epochs': 50
   }
}

# Autonomous Agent - Enhanced
AUTONOMOUS_CONFIG = {
   'enabled': False, # AUTONOMOUS_ENABLED
   'budget': 200.0,  # AUTONOMOUS_BUDGET
   'profit_threshold_percent': 2.0, # THRESHOLD_PROFIT_PERCENT

   # NEW: Enhanced risk management
   'stop_loss_percent': 5.0,
   'max_position_size_percent': 10.0, # Max 10% of capital
   'confidence_threshold': 0.65,      # Minimum confidence for trading
   'max_trades_per_day': 3
}

# NEW: Baseline Model Configuration (Our Testing Phase)
BASELINE_CONFIG = {
   # FinBERT Sentiment Model
   'sentiment_model': {
       'name': 'ProsusAI/finbert',
       'max_length': 512,
       'batch_size': 8, # Memory optimized for Colab
       'confidence_threshold': 0.6
   },

   # Ensemble Configuration
   'ensemble': {
       'sentiment_weight': 0.5,     # Equal weight initially
       'technical_weight': 0.5,
       'confidence_threshold': 0.65,
       'signal_classes': ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
   }
}

# NEW: Apple-Specific Configuration
APPLE_CONFIG = {
   # Apple-specific keywords for Indian market context
   'keywords': {
       'high_impact': [
           'iPhone', 'earnings', 'revenue', 'guidance', 'quarterly results',
           'App Store', 'services revenue'
       ],
       'medium_impact': [
           'iPad', 'Mac', 'Apple Watch', 'Tim Cook', 'Apple event',
           'product launch', 'innovation'
       ],
       'low_impact': [
           'iOS', 'macOS', 'software update', 'developer conference'
       ]
   },

   # Seasonal patterns (Apple specific)
   'seasonal_events': {
       'earnings_months': [1, 4, 7, 10],    # Quarterly earnings
       'product_launch_months': [9, 10],    # Fall product launches
       'holiday_season_months': [11, 12, 1] # Holiday sales impact
   },

   # Indian market perspective factors
   'indian_market_factors': {
       'usd_inr_impact': True,              # Currency fluctuation impact
       'global_tech_sentiment': True,      # Global tech sector sentiment
       'us_market_correlation': 0.8        # High correlation with US Apple
   }
}

# ====================
# XAI (EXPLAINABLE AI) CONFIGURATION
# ====================
XAI_CONFIG = {
   'enabled': True,

   # Explanation types to generate
   'explanation_types': [
       'attention_weights',   # For transformer models
       'shap_values',        # Feature importance
       'feature_importance', # Traditional ML importance
       'natural_language'    # Human-readable explanations
   ],

   # Explanation settings
   'min_confidence_for_explanation': 0.7,
   'max_important_features': 10,
   'save_explanations': True,

   # Natural language explanation settings
   'natural_language': {
       'generate_summaries': True,
       'max_summary_length': 200,
       'include_confidence_levels': True,
       'include_risk_factors': True
   },

   # Visualization settings
   'visualization': {
       'save_plots': True,
       'plot_directory': 'explanations/plots',
       'attention_heatmaps': True,
       'feature_importance_charts': True
   }
}

# NEW: News Processing Configuration
NEWS_CONFIG = {
   # Source credibility (Indian context)
   'source_credibility': {
       'reuters.com': 1.0,
       'bloomberg.com': 1.0,
       'cnbc.com': 0.9,
       'economictimes.indiatimes.com': 0.9, # Indian financial news
       'business-standard.com': 0.85,       # Indian business news
       'moneycontrol.com': 0.8,             # Popular Indian finance site
       'yahoo.com': 0.8,
       'marketwatch.com': 0.8,
       'default': 0.5
   },

   # News filtering and processing
   'processing': {
       'min_article_length': 100,
       'max_articles_per_day': 50,
       'lookback_days': 7,
       'apple_relevance_threshold': 0.7
   },

   # Temporal weighting (recent news more important)
   'temporal_weighting': {
       'same_day': 1.0,
       '1_day_old': 0.8,
       '2_days_old': 0.6,
       '3_days_old': 0.4,
       'week_old': 0.2
   }
}

# ====================
# TRADING FEES - INDIAN CONTEXT
# ====================
BROKERAGE_PERCENT = 0.001 # 0.1% - Your existing
TAX_PERCENT = 0.0005      # 0.05% - Your existing

# NEW: Detailed fee structure for future enhancement
INDIAN_FEES_CONFIG = {
   'brokerage': BROKERAGE_PERCENT,
   'tax': TAX_PERCENT,
   'stt': 0.001,             # Securities Transaction Tax
   'gst_on_brokerage': 0.18, # 18% GST on brokerage
   'exchange_charges': 0.0000345, # NSE/BSE charges
   'total_estimated': 0.0025 # Approximately 0.25% total impact
}

# ====================
# PERFORMANCE MONITORING
# ====================
MONITORING_CONFIG = {
   'track_performance': True,
   'save_all_predictions': True,
   'calculate_metrics': [
       'accuracy', 'precision', 'recall', 'f1_score',
       'sharpe_ratio', 'max_drawdown', 'win_rate'
   ],
   'benchmark_comparison': True,

   # Logging configuration
   'logging': {
       'level': 'INFO',
       'save_to_file': True,
       'console_output': True,
       'detailed_explanations': True
   }
}

# ====================
# DATA VALIDATION SETTINGS
# ====================
VALIDATION_CONFIG = {
   'min_data_points': 252, # At least 1 year of data
   'max_missing_percentage': 5, # Max 5% missing data
   'price_sanity_checks': True,
   'volume_sanity_checks': True,
   'news_relevance_check': True,
   'min_news_articles_per_week': 5
}

# ====================
# TESTING CONFIGURATION
# ====================
TESTING_CONFIG = {
   'data_split': {
       'train': 0.7,
       'validation': 0.15,
       'test': 0.15
   },
   'min_test_period_days': 90,
   'cross_validation_folds': 5,
   'walk_forward_analysis': True,
   'backtesting_enabled': True
}

# ====================
# BACKWARD COMPATIBILITY
# ====================
# Keep your existing variable names for compatibility
QUANT_SHORT_WINDOW = QUANT_CONFIG['short_window']
QUANT_LONG_WINDOW = QUANT_CONFIG['long_window']
AUTONOMOUS_ENABLED = AUTONOMOUS_CONFIG['enabled']
AUTONOMOUS_BUDGET = AUTONOMOUS_CONFIG['budget']
THRESHOLD_PROFIT_PERCENT = AUTONOMOUS_CONFIG['profit_threshold_percent']