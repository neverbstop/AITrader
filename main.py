# main.py - Enhanced with XAI and robust error handling
import sys
import time
from datetime import datetime
from pathlib import Path
import polars as pl
import logging
import config
from core.autonomous_agent import EnhancedAutonomousAgent

agent = EnhancedAutonomousAgent(ticker=config.TICKER)

# Import your existing pipelines
from pipelines.data_pipeline import DataPipeline
from pipelines.news_pipeline import NewsPipeline
from pipelines.sentiment_pipeline import SentimentPipeline
from dashboard.dashboard import open_dashboard
from pipelines.data_pipeline import EnhancedDataPipeline
from pipelines.news_pipeline import EnhancedNewsPipeline
from pipelines.sentiment_pipeline import EnhancedSentimentPipeline

# Instantiate pipelines
data_pipeline = EnhancedDataPipeline(stockfile="data/data.csv", ticker=config.TICKER)
news_pipeline = EnhancedNewsPipeline(api_key=config.NEWSAPI_KEY)
sentiment_pipeline = EnhancedSentimentPipeline()

# Get current price (from data pipeline or external source)
current_price = data_pipeline.get_latest_price(config.TICKER)
# Get quant signal (from technical analysis)
quant_signal = data_pipeline.generate_quant_signal(config.TICKER)
# Get latest news summary
news_signal = news_pipeline.latest_signal(config.TICKER)
# Get sentiment score
sentiment_score = sentiment_pipeline.latest_score(config.TICKER)

# NEW: Import XAI components (we'll need to create these)
from xai.explainer import XAIExplainer
from xai.decision_tracker import DecisionTracker
from utils.performance_monitor import PerformanceMonitor
from utils.data_validator import DataValidator
from utils.logger_setup import setup_logging

import config
# After obtaining signal values from other pipelines/modules
# Assume: news_signal, quant_signal, sentiment_score, current_price have been calculated here

from core.autonomous_agent import EnhancedAutonomousAgent

agent = EnhancedAutonomousAgent(ticker=TICKER)
agent.enable()

# Example loop for simulation or live data:
while True:
    # update signal values as needed (e.g., fetch next batch of data)
    # call agent logic on every tick/data refresh
    agent.run_if_enabled(news_signal, quant_signal, sentiment_score, current_price)

    # Optional: break condition or sleep
    break
agent.run_if_enabled(news_signal, quant_signal, sentiment_score, current_price)

# Configure Polars
pl.Config.set_tbl_formatting("ASCII_FULL")

class TradingAIOrchestrator:
   """
   Main orchestrator for the AI Trading System with XAI integration
   """

   def __init__(self):
       self.performance_monitor = PerformanceMonitor()
       self.data_validator = DataValidator()
       self.xai_explainer = XAIExplainer()
       self.decision_tracker = DecisionTracker()
       self.logger = setup_logging()

   def validate_configuration(self):
       """Validate all required configuration parameters"""
       required_configs = ['TICKER', 'STOCK_FILE', 'NEWS_API_KEY']
       missing_configs = []

       for conf in required_configs:
           if not hasattr(config, conf) or not getattr(config, conf):
               missing_configs.append(conf)

       if missing_configs:
           raise ValueError(f"Missing required configurations: {missing_configs}")

       self.logger.info(":white_check_mark: Configuration validation passed")
       return True

   def run_data_pipeline(self):
       """Enhanced data pipeline with validation and monitoring"""
       self.logger.info(":arrows_counterclockwise: Starting Stock Data Pipeline...")

       try:
           with self.performance_monitor.track_time("data_pipeline"):
               data_pipeline = DataPipeline(
                   stock_file=config.STOCK_FILE,
                   ticker=config.TICKER
               )
               stock_data = data_pipeline.run()

           # Validate data quality
           validation_results = self.data_validator.validate_stock_data(stock_data)
           if not validation_results['is_valid']:
               self.logger.warning(f":warning: Data quality issues: {validation_results['issues']}")

           self.logger.info(f":white_check_mark: Stock data loaded: {len(stock_data)} rows")
           print(f"Stock Data Sample:\n{stock_data.head()}")

           return stock_data

       except Exception as e:
           self.logger.error(f":x: Data pipeline failed: {str(e)}")
           raise

   def run_news_pipeline(self):
       """Enhanced news pipeline with validation"""
       self.logger.info(":newspaper: Starting News Pipeline...")

       try:
           with self.performance_monitor.track_time("news_pipeline"):
               news_pipeline = NewsPipeline(api_key=config.NEWS_API_KEY)
               # Clean ticker for better search results
               clean_ticker = config.TICKER.split('.')[0]
               news_data = news_pipeline.run(company=clean_ticker, days=3)

           # Validate news data
           validation_results = self.data_validator.validate_news_data(news_data)
           if not validation_results['is_valid']:
               self.logger.warning(f":warning: News quality issues: {validation_results['issues']}")

           self.logger.info(f":white_check_mark: Collected {len(news_data)} news articles")

           return news_data

       except Exception as e:
           self.logger.error(f":x: News pipeline failed: {str(e)}")
           raise

   def run_sentiment_pipeline(self, news_data):
       """Enhanced sentiment pipeline with XAI explanations"""
       self.logger.info(":brain: Starting Sentiment Analysis Pipeline...")

       try:
           with self.performance_monitor.track_time("sentiment_pipeline"):
               sentiment_pipeline = SentimentPipeline(news_articles=news_data)
               sentiment_results = sentiment_pipeline.run()

           # NEW: Generate XAI explanations for sentiment analysis
           explanations = []
           for i, result in enumerate(sentiment_results[:5]): # Explain top 5 for performance
               try:
                   explanation = self.xai_explainer.explain_sentiment_prediction(
                       text=news_data[i].get('headline', '') + ' ' + news_data[i].get('content', ''),
                       sentiment_score=result.get('sentiment', 0),
                       model_confidence=result.get('confidence', 0.5)
                   )
                   explanations.append(explanation)
               except Exception as e:
                   self.logger.warning(f"Failed to explain sentiment for article {i}: {e}")

           self.logger.info(f":white_check_mark: Sentiment analysis completed: {len(sentiment_results)} results")
           self.logger.info(f":mag: Generated {len(explanations)} XAI explanations")

           # Sample output with explanations
           print("\nSentiment Analysis Results:")
           for i, result in enumerate(sentiment_results[:3]):
               print(f"Article {i+1}: {result}")
               if i < len(explanations):
                   print(f" Explanation: {explanations[i].get('summary', 'No explanation available')}")

           return sentiment_results, explanations

       except Exception as e:
           self.logger.error(f":x: Sentiment pipeline failed: {str(e)}")
           raise

   def generate_trading_signal(self, stock_data, sentiment_results, explanations):
       """NEW: Generate trading signal with comprehensive XAI explanation"""
       self.logger.info(":dart: Generating Trading Signal...")

       try:
           with self.performance_monitor.track_time("signal_generation"):
               # Simple baseline signal generation (you'll enhance this)
               avg_sentiment = sum(r.get('sentiment', 0) for r in sentiment_results) / len(sentiment_results) if sentiment_results else 0
               recent_price_change = (stock_data['Close'][-1] - stock_data['Close'][-5]) / stock_data['Close'][-5]

               # Basic ensemble logic
               if avg_sentiment > 0.1 and recent_price_change > 0.02:
                   signal = "BUY"
                   confidence = min(0.9, abs(avg_sentiment) + abs(recent_price_change))
               elif avg_sentiment < -0.1 and recent_price_change < -0.02:
                   signal = "SELL"
                   confidence = min(0.9, abs(avg_sentiment) + abs(recent_price_change))
               else:
                   signal = "HOLD"
                   confidence = 0.5

               # NEW: Generate comprehensive trading explanation
               trading_explanation = self.xai_explainer.explain_trading_decision(
                   signal=signal,
                   confidence=confidence,
                   sentiment_data={'average': avg_sentiment, 'explanations': explanations},
                   price_data={'recent_change': recent_price_change, 'current_price': stock_data['Close'][-1]},
                   technical_data={} # Add technical indicators later
               )

               # Track decision for learning
               self.decision_tracker.log_decision(
                   timestamp=datetime.now(),
                   signal=signal,
                   confidence=confidence,
                   explanation=trading_explanation,
                   input_data={
                       'sentiment': avg_sentiment,
                       'price_change': recent_price_change,
                       'news_count': len(sentiment_results)
                   }
               )

               self.logger.info(f":white_check_mark: Trading Signal: {signal} (Confidence: {confidence:.2%})")

               return {
                   'signal': signal,
                   'confidence': confidence,
                   'explanation': trading_explanation,
                   'supporting_data': {
                       'avg_sentiment': avg_sentiment,
                       'price_change': recent_price_change,
                       'news_articles': len(sentiment_results)
                   }
               }

       except Exception as e:
           self.logger.error(f":x: Signal generation failed: {str(e)}")
           raise

   def run_complete_pipeline(self):
       """Execute complete pipeline with comprehensive error handling"""
       start_time = time.time()

       try:
           # Validate configuration
           self.validate_configuration()

           # Run all pipelines
           stock_data = self.run_data_pipeline()
           news_data = self.run_news_pipeline()
           sentiment_results, explanations = self.run_sentiment_pipeline(news_data)

           # NEW: Generate trading signal with XAI
           trading_decision = self.generate_trading_signal(stock_data, sentiment_results, explanations)

           # Performance summary
           total_time = time.time() - start_time
           performance_summary = self.performance_monitor.get_summary()

           self.logger.info(f":checkered_flag: Pipeline completed successfully in {total_time:.2f}s")
           print(f"\n{'='*50}")
           print(f":dart: TRADING DECISION: {trading_decision['signal']}")
           print(f":bar_chart: CONFIDENCE: {trading_decision['confidence']:.2%}")
           print(f":mag: EXPLANATION: {trading_decision['explanation'].get('summary', 'Processing...')}")
           print(f":stopwatch: TOTAL EXECUTION TIME: {total_time:.2f}s")
           print(f"{'='*50}\n")

           return {
               'stock_data': stock_data,
               'news_data': news_data,
               'sentiment_results': sentiment_results,
               'explanations': explanations,
               'trading_decision': trading_decision,
               'performance': performance_summary
           }

       except Exception as e:
           self.logger.error(f":x: Pipeline failed: {str(e)}")
           print(f"\n:x: PIPELINE FAILED: {str(e)}")
           raise

   def open_enhanced_dashboard(self, results):
       """Open dashboard with XAI integration"""
       self.logger.info(":video_game: Opening Enhanced Dashboard...")

       try:
           # Pass all results including XAI explanations to dashboard
           open_dashboard(
               stock_data=results['stock_data'],
               news_data=results['news_data'],
               sentiment_results=results['sentiment_results'],
               explanations=results['explanations'],
               trading_decision=results['trading_decision'],
               performance_metrics=results['performance']
           )
       except Exception as e:
           self.logger.error(f":x: Dashboard failed to open: {str(e)}")
           print(f":x: Dashboard Error: {str(e)}")

def main():
   """Main entry point"""
   print(":rocket: Starting Advanced Trading AI Project with XAI...")
   print("=" * 60)

   try:
       orchestrator = TradingAIOrchestrator()
       results = orchestrator.run_complete_pipeline()
       orchestrator.open_enhanced_dashboard(results)

   except KeyboardInterrupt:
       print("\n:octagonal_sign: Process interrupted by user")
       sys.exit(0)
   except Exception as e:
       print(f"\n:boom: Fatal error: {str(e)}")
       sys.exit(1)

if __name__ == "__main__":
   main()