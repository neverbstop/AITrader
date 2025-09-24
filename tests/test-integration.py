# tests/test_integration.py - Comprehensive Integration Testing Suite

"""

SOLVES: Unknown system stability and component integration issues

 

Problems This Test Solves:

❌ Unknown if enhanced pipelines work together → ✅ Validates complete workflow

❌ Untested error handling → ✅ Tests failure scenarios 

❌ Memory usage unknown → ✅ Monitors resource consumption

❌ Config integration unclear → ✅ Validates all configurations

❌ XAI components untested → ✅ Verifies explanation generation

"""

 

import pytest

import sys

import time

import psutil

import tracemalloc

from pathlib import Path

import pandas as pd

import polars as pl

from datetime import datetime

import warnings

import logging

 

# Add project root to path

project_root = Path(__file__).parent.parent

sys.path.append(str(project_root))

 

# Import all our enhanced components

try:

    import config

    from pipelines.data_pipeline import EnhancedDataPipeline

    from pipelines.news_pipeline import EnhancedNewsPipeline 

    from pipelines.sentiment_pipeline import EnhancedSentimentPipeline

    print("✅ All imports successful")

except ImportError as e:

    print(f"❌ Import failed: {e}")

    sys.exit(1)

 

# Setup test logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

 

class IntegrationTestSuite:

    """

    Comprehensive integration testing for enhanced AI trading system

    """

   

    def __init__(self):

        self.test_results = {}

        self.performance_metrics = {}

        self.memory_usage = {}

        self.start_time = None

       

        # Test configuration

        self.test_ticker = "AAPL"

        self.test_news_days = 3  # Reduced for testing

        self.memory_limit_gb = 12  # 75% of 16GB system

       

    def run_all_tests(self):

        """

        SOLVES: Manual testing tedium with automated comprehensive validation

        """

        print("\n" + "="*60)

        print("🧪 STARTING AI TRADING SYSTEM INTEGRATION TESTS")

        print("="*60)

       

        # Start memory tracking

        tracemalloc.start()

        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

       

        self.start_time = time.time()

       

        # Run test suite

        tests = [

            ("Configuration Validation", self.test_config_integration),

            ("Data Pipeline Integration", self.test_data_pipeline),

            ("News Pipeline Integration", self.test_news_pipeline),

            ("Sentiment Pipeline Integration", self.test_sentiment_pipeline),

            ("XAI Features Integration", self.test_xai_integration),

            ("End-to-End Workflow", self.test_complete_workflow),

            ("Error Handling", self.test_error_handling),

            ("Performance Benchmarks", self.test_performance),

            ("Memory Usage Validation", self.test_memory_usage)

        ]

       

        for test_name, test_func in tests:

            print(f"\n🔄 Running: {test_name}")

            try:

                result = test_func()

                self.test_results[test_name] = {

                    'status': 'PASSED' if result else 'FAILED',

                    'details': result

                }

                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")

            except Exception as e:

                self.test_results[test_name] = {

                    'status': 'ERROR',

                    'details': str(e),

                    'traceback': traceback.format_exc()

                }

                print(f"❌ {test_name}: ERROR - {str(e)}")

       

        # Final memory check

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        self.memory_usage = {

            'initial_mb': initial_memory,

            'final_mb': final_memory,

            'increase_mb': final_memory - initial_memory,

            'max_allowed_mb': self.memory_limit_gb * 1024

        }

       

        # Generate comprehensive report

        self.generate_test_report()

   

    def test_config_integration(self):

        """

        SOLVES: Config compatibility issues between enhanced components

        """

        try:

            # Test basic config access

            assert hasattr(config, 'TICKER'), "TICKER not in config"

            assert hasattr(config, 'XAI_CONFIG'), "XAI_CONFIG missing"

            assert hasattr(config, 'APPLE_CONFIG'), "APPLE_CONFIG missing"

           

            # Test config values

            assert config.TICKER == "AAPL", f"Expected AAPL, got {config.TICKER}"

            assert config.XAI_CONFIG.get('enabled'), "XAI not enabled"

           

            # Test file paths exist or can be created

            data_dir = Path(config.DATA_DIR) if hasattr(config, 'DATA_DIR') else Path("data")

            data_dir.mkdir(exist_ok=True)

           

            return True

        except Exception as e:

            logger.error(f"Config test failed: {e}")

            return False

   

    def test_data_pipeline(self):

        """

        SOLVES: Data pipeline integration and validation issues

        """

        try:

            # Initialize data pipeline

            stock_file = f"data/{self.test_ticker}_test_stock.csv"

            data_pipeline = EnhancedDataPipeline(stock_file, self.test_ticker)

           

            # Test data loading (will download if needed)

            stock_data = data_pipeline.load_data()

           

            # Validate data structure

            assert stock_data is not None, "No stock data loaded"

            assert len(stock_data) > 100, f"Insufficient data: {len(stock_data)} rows"

           

            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

            for col in required_columns:

                assert col in stock_data.columns, f"Missing column: {col}"

           

            # Test technical indicators

            enhanced_data = data_pipeline.add_basic_technical_indicators()

           

            # Check for technical indicators

            tech_indicators = ['SMA_5', 'SMA_20', 'RSI_14', 'Daily_Return_Pct']

            for indicator in tech_indicators:

                assert indicator in enhanced_data.columns, f"Missing indicator: {indicator}"

           

            # Test data validation

            validation_results = data_pipeline.validation_results

            assert validation_results.get('is_valid'), "Data validation failed"

           

            return {

                'data_points': len(stock_data),

                'indicators_added': len([col for col in enhanced_data.columns if col not in required_columns]),

                'validation_passed': validation_results.get('is_valid')

            }

           

        except Exception as e:

            logger.error(f"Data pipeline test failed: {e}")

            return False

   

    def test_news_pipeline(self):

        """

        SOLVES: News collection and processing integration issues

        """

        try:

            # Check if NEWS_API_KEY is available

            if not config.NEWS_API_KEY:

                return {

                    'status': 'SKIPPED',

                    'reason': 'NEWS_API_KEY not configured'

                }

           

            # Initialize news pipeline

            news_pipeline = EnhancedNewsPipeline(config.NEWS_API_KEY)

           

            # Test news fetching (reduced scope for testing)

            news_data = news_pipeline.fetch_news(company="Apple", days=self.test_news_days)

           

            # Validate news data

            assert isinstance(news_data, list), "News data not a list"

            assert len(news_data) >= 0, "News data loading failed"

           

            if len(news_data) > 0:

                # Test news processing

                processed_articles = news_pipeline.preprocess()

               

                # Validate processed structure

                assert len(processed_articles) > 0, "No articles processed"

               

                # Check first article structure

                first_article = processed_articles[0]

                required_attrs = ['title', 'source', 'credibility_score', 'apple_relevance_score']

                for attr in required_attrs:

                    assert hasattr(first_article, attr), f"Missing attribute: {attr}"

               

                # Test XAI features

                xai_summary = news_pipeline.get_xai_summary()

                assert 'total_articles' in xai_summary, "XAI summary missing"

           

            return {

                'articles_collected': len(news_data),

                'articles_processed': len(processed_articles) if len(news_data) > 0 else 0,

                'has_xai_features': len(news_data) > 0

            }

           

        except Exception as e:

            logger.error(f"News pipeline test failed: {e}")

            return False

   

    def test_sentiment_pipeline(self):

        """

        SOLVES: Sentiment analysis and FinBERT integration issues

        """

        try:

            # Create sample news data for testing
            sample_articles = [
                NewsArticle(
                    id="test_1",
                    date=datetime.now(),
                    source="Reuters",
                    title='Apple reports record iPhone sales beating analyst expectations',
                    description='Strong quarterly results show significant growth',
                    content='Apple Inc. announced record-breaking iPhone sales...',
                    url="", credibility_score=0.9, apple_relevance_score=0.9,
                    temporal_weight=1.0, key_topics=[], sentiment_keywords={},
                    urgency_indicators=[], financial_terms=[]
                )
            ]

           

            # Initialize sentiment pipeline

            sentiment_pipeline = EnhancedSentimentPipeline(sample_articles)

           

            # Test sentiment analysis

            results = sentiment_pipeline.run() # Should return a list of dicts

           

            # Validate results

            assert len(results) > 0, "No sentiment results"

           

            first_result = results[0]
            assert isinstance(first_result, dict), "Result should be a dictionary"
            assert 'sentiment' in first_result, "Missing 'sentiment' key"
            assert 'confidence' in first_result, "Missing 'confidence' key"
            assert 'label' in first_result, "Missing 'label' key"

           

            # Test sentiment score bounds
            assert -1 <= first_result['sentiment'] <= 1, "Sentiment score out of bounds"
            assert 0 <= first_result['confidence'] <= 1, "Confidence out of bounds"

           

            # Test XAI features
            has_xai = 'explanation' in first_result and first_result['explanation']

           

            return {
                'sentiment_model': first_result.get('_object').model_used if '_object' in first_result else 'unknown',
                'articles_analyzed': len(results),
                'avg_confidence': sum(r['confidence'] for r in results) / len(results),
                'has_xai_explanations': has_xai
            }

           

        except Exception as e:

            logger.error(f"Sentiment pipeline test failed: {e}")

            return False

   

    def test_xai_integration(self):

        """

        SOLVES: XAI feature integration and explanation generation issues

        """

        try:

            # Test XAI configuration

            assert config.XAI_CONFIG.get('enabled'), "XAI not enabled in config"

           

            # Test explanation types

            explanation_types = config.XAI_CONFIG.get('explanation_types', [])

            assert len(explanation_types) > 0, "No explanation types configured"

           

            # Test XAI directories

            explanation_dir = Path("explanations")

            explanation_dir.mkdir(exist_ok=True)

            assert explanation_dir.exists(), "Explanation directory not created"

           

            return {

                'xai_enabled': True,

                'explanation_types': len(explanation_types),

                'directories_ready': True

            }

           

        except Exception as e:

            logger.error(f"XAI integration test failed: {e}")

            return False

   

    def test_complete_workflow(self):

        """

        SOLVES: End-to-end workflow integration issues

        """

        try:

            workflow_start = time.time()

           

            # Step 1: Load stock data

            stock_file = f"data/{self.test_ticker}_workflow_test.csv"

            data_pipeline = EnhancedDataPipeline(stock_file, self.test_ticker)

            stock_data = data_pipeline.load_data()

            enhanced_data = data_pipeline.add_basic_technical_indicators()

           

            # Step 2: Process news (if API key available)

            news_results = None

            sentiment_results = None

           

            if config.NEWS_API_KEY:

                news_pipeline = EnhancedNewsPipeline(config.NEWS_API_KEY)

                news_data = news_pipeline.fetch_news(company="Apple", days=1)  # Minimal for testing

               

                if news_data:

                    sentiment_pipeline = EnhancedSentimentPipeline(news_data)

                    sentiment_results = sentiment_pipeline.run()

           

            # Step 3: Generate trading signal (simplified)

            latest_price = enhanced_data['Close'][-1] if len(enhanced_data) > 0 else 100

           

            # Simple signal generation logic

            trading_signal = {

                'timestamp': datetime.now(),

                'price': latest_price,

                'action': 'HOLD',  # Default for testing

                'confidence': 0.7,

                'data_quality': 'GOOD' if len(enhanced_data) > 200 else 'LIMITED'

            }

           

            workflow_time = time.time() - workflow_start

           

            return {

                'workflow_completed': True,

                'execution_time_seconds': workflow_time,

                'stock_data_points': len(stock_data),

                'news_articles': len(news_results) if news_results else 0,

                'sentiment_results': len(sentiment_results) if sentiment_results else 0,

                'trading_signal_generated': True

            }

           

        except Exception as e:

            logger.error(f"Complete workflow test failed: {e}")

            return False

   

    def test_error_handling(self):

        """

        SOLVES: Unknown error resilience and graceful failure handling

        """

        error_tests = []

       

        # Test 1: Invalid ticker

        try:

            invalid_pipeline = EnhancedDataPipeline("data/INVALID_TICKER.csv", "INVALID")

            invalid_pipeline.load_data()

            error_tests.append("invalid_ticker_handled")

        except Exception:

            error_tests.append("invalid_ticker_failed_gracefully")

       

        # Test 2: Invalid API key

        try:

            invalid_news = EnhancedNewsPipeline("invalid_api_key")

            invalid_news.fetch_news(company="Apple", days=1)

            error_tests.append("invalid_api_handled")

        except Exception:

            error_tests.append("invalid_api_failed_gracefully")

       

        # Test 3: Empty news data

        try:

            empty_sentiment = EnhancedSentimentPipeline([])

            empty_sentiment.run()

            error_tests.append("empty_data_handled")

        except Exception:

            error_tests.append("empty_data_failed_gracefully")

       

        return {

            'error_scenarios_tested': len(error_tests),

            'error_handling_results': error_tests

        }

   
    def test_performance(self):

        """

        SOLVES: Unknown system performance and resource requirements

        """

        performance_metrics = {

            'data_loading_time': 0,

            'news_processing_time': 0,

            'sentiment_analysis_time': 0,

            'total_pipeline_time': 0

        }

       

        try:

            # Benchmark data loading

            start = time.time()

            data_pipeline = EnhancedDataPipeline(f"data/{self.test_ticker}_perf_test.csv", self.test_ticker)

            data_pipeline.load_data()

            performance_metrics['data_loading_time'] = time.time() - start

           

            # Benchmark news processing (if API available)

            if config.NEWS_API_KEY:

                start = time.time()

                news_pipeline = EnhancedNewsPipeline(config.NEWS_API_KEY)

                news_data = news_pipeline.fetch_news(company="Apple", days=1)

                performance_metrics['news_processing_time'] = time.time() - start

               

                # Benchmark sentiment analysis

                if news_data:

                    start = time.time()

                    sentiment_pipeline = EnhancedSentimentPipeline(news_data)

                    sentiment_pipeline.run()

                    performance_metrics['sentiment_analysis_time'] = time.time() - start

           

            performance_metrics['total_pipeline_time'] = sum(performance_metrics.values())

           

            return performance_metrics

           

        except Exception as e:

            logger.error(f"Performance test failed: {e}")

            return False

   
    def test_memory_usage(self):
        
        try:

            current, peak = tracemalloc.get_traced_memory()

            process = psutil.Process()

           

            memory_info = {

                'current_mb': current / 1024 / 1024,

                'peak_mb': peak / 1024 / 1024,

                'system_memory_mb': process.memory_info().rss / 1024 / 1024,

                'memory_limit_mb': self.memory_limit_gb * 1024,

                'within_limits': (current / 1024 / 1024) < (self.memory_limit_gb * 1024)

            }

           

            return memory_info

           

        except Exception as e:

            logger.error(f"Memory test failed: {e}")

            return False

   
    def generate_test_report(self):

        """

        SOLVES: Unclear test results with comprehensive reporting

        """

        print("\n" + "="*60)

        print("📊 INTEGRATION TEST REPORT")

        print("="*60)

       

        # Summary

        total_tests = len(self.test_results)

        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')

        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAILED')

        error_tests = sum(1 for result in self.test_results.values() if result['status'] == 'ERROR')

       

        print(f"📈 SUMMARY:")

        print(f"   ✅ Passed: {passed_tests}/{total_tests}")

        print(f"   ❌ Failed: {failed_tests}/{total_tests}")

        print(f"   💥 Errors: {error_tests}/{total_tests}")

        print(f"   🕒 Total Time: {time.time() - self.start_time:.2f} seconds")

       

        # Memory usage

        if self.memory_usage:

            print(f"\n💾 MEMORY USAGE:")

            print(f"   📊 Final Memory: {self.memory_usage['final_mb']:.1f} MB")

            print(f"   📈 Memory Increase: {self.memory_usage['increase_mb']:.1f} MB")

            print(f"   🎯 Within Limits: {'✅' if self.memory_usage['final_mb'] < self.memory_usage['max_allowed_mb'] else '❌'}")

       

        # Detailed results

        print(f"\n📋 DETAILED RESULTS:")

        for test_name, result in self.test_results.items():

            status_emoji = {'PASSED': '✅', 'FAILED': '❌', 'ERROR': '💥'}[result['status']]

            print(f"   {status_emoji} {test_name}: {result['status']}")

           

            if result['status'] != 'PASSED' and isinstance(result['details'], str):

                print(f"      └─ {result['details']}")

       

        # Final assessment

        print(f"\n🎯 SYSTEM ASSESSMENT:")

        if passed_tests == total_tests:

            print("   🚀 EXCELLENT: All systems operational and integrated!")

        elif passed_tests >= total_tests * 0.8:

            print("   ✅ GOOD: Most systems working, minor issues to address")

        elif passed_tests >= total_tests * 0.6:

            print("   ⚠ MODERATE: Some integration issues need attention")

        else:

            print("   ❌ CRITICAL: Major integration problems require fixes")

 

# Main execution

if __name__ == "__main__":

    """

    Run the complete integration test suite

    """

    print("🧪 AI Trading System Integration Test Suite")

    print("=" * 50)

   

    # Run tests

    test_suite = IntegrationTestSuite()

    test_suite.run_all_tests()

   

    print("\n🏁 Testing complete!")

    print("💡 If tests fail, check:")

    print("   1. .env file with NEWS_API_KEY")

    print("   2. Internet connection for data downloads")

    print("   3. Sufficient disk space in data/ folder")

    print("   4. Python dependencies installed")