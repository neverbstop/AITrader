#  - Professional Trading Dashboard with XAI

import streamlit as st

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import pandas as pd

import numpy as np

from datetime import datetime, timedelta

import json

import sys

import os

from pathlib import Path

 

# Add project root to path for imports

project_root = Path(__file__).parent.parent

sys.path.append(str(project_root))

 

# Import our enhanced components

import config

from pipelines.data_pipeline import EnhancedDataPipeline

from pipelines.news_pipeline import EnhancedNewsPipeline 

from pipelines.sentiment_pipeline import EnhancedSentimentPipeline

 

# Configure Streamlit page

st.set_page_config(

    page_title="🍎 Apple AI Trading Dashboard",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="expanded"

)

 

class AppleAITradingDashboard:

    """

    SOLVES: Static HTML limitation with interactive real-time dashboard

   

    Problems Solved:

    ❌ No real-time data → ✅ Live pipeline integration

    ❌ No XAI explanations → ✅ Interactive explanation widgets

    ❌ Static displays → ✅ Dynamic, updating visualizations

    ❌ No Apple focus → ✅ Company-specific analytics

    ❌ No agent coordination → ✅ Multi-agent decision center

    """

   

    def __init__(self):

        self.data_pipeline = None

        self.news_pipeline = None

        self.sentiment_pipeline = None

        self.current_data = {}

       

        # Initialize session state for real-time updates

        if 'last_update' not in st.session_state:

            st.session_state.last_update = datetime.now()

        if 'trading_signals' not in st.session_state:

            st.session_state.trading_signals = []

        if 'explanation_data' not in st.session_state:

            st.session_state.explanation_data = {}

 

    def load_data(self):

        """

        SOLVES: No data integration with enhanced pipeline loading

        """

        try:

            with st.spinner("🔄 Loading Apple data and AI models..."):

                # Load stock data

                self.data_pipeline = EnhancedDataPipeline(

                    stock_file=str(config.STOCK_FILE),

                    ticker=config.TICKER

                )

                stock_data = self.data_pipeline.run()

               

                # Load news data

                if config.NEWS_API_KEY:

                    self.news_pipeline = EnhancedNewsPipeline(api_key=config.NEWS_API_KEY)

                    news_data = self.news_pipeline.fetch_news(company="Apple", days=7)

                   

                    # Process sentiment

                    self.sentiment_pipeline = EnhancedSentimentPipeline(news_data)

                    sentiment_results = self.sentiment_pipeline.run()

                   

                    self.current_data = {

                        'stock_data': stock_data,

                        'news_data': news_data,

                        'sentiment_results': sentiment_results,

                        'xai_features': self.sentiment_pipeline.explanation_features

                    }

                else:

                    st.error("❌ NEWS_API_KEY not found. Please configure in .env file")

                   

        except Exception as e:

            st.error(f"❌ Data loading failed: {str(e)}")

            return False

       

        return True

 

    def render_header(self):

        """

        SOLVES: Generic dashboard with Apple-specific branding

        """

        st.markdown("""

        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #1f2937, #374151); border-radius: 10px; margin-bottom: 2rem;'>

            <h1 style='color: white; margin: 0;'>🍎 Apple AI Trading Command Center</h1>

            <p style='color: #9ca3af; margin: 0.5rem 0 0 0;'>Advanced XAI-Powered Trading System for AAPL</p>

        </div>

        """, unsafe_allow_html=True)

       

        # Real-time status indicators

        col1, col2, col3, col4 = st.columns(4)

       

        with col1:

            current_price = self.get_current_price()

            st.metric("📊 Current Price", f"${current_price:.2f}",

                     delta=f"{self.get_price_change():.2f}%")

       

        with col2:

            st.metric("🤖 AI Models", "Active", delta="FinBERT + XAI")

       

        with col3:

            news_count = len(self.current_data.get('news_data', []))

            st.metric("📰 News Articles", news_count, delta="Last 7 days")

       

        with col4:

            confidence = self.get_latest_confidence()

            st.metric("🎯 AI Confidence", f"{confidence:.1%}",

                     delta="High" if confidence > 0.7 else "Medium")

 

    def render_trading_signals(self):

        """

        SOLVES: No agent coordination with multi-agent decision display

        """

        st.markdown("## 🎯 AI Trading Signals & Explanations")

       

        # Get latest signals (simulated for now, will integrate with your core agents)

        latest_signal = self.generate_trading_signal()

       

        col1, col2 = st.columns([1, 2])

       

        with col1:

            # Signal display

            signal_color = {

                'BUY': 'green',

                'SELL': 'red',

                'HOLD': 'orange'

            }.get(latest_signal['action'], 'gray')

           

            st.markdown(f"""

            <div style='padding: 1rem; background: {signal_color}20; border-left: 4px solid {signal_color}; border-radius: 5px;'>

                <h3 style='color: {signal_color}; margin: 0;'>{latest_signal['action']} Signal</h3>

                <p><strong>Confidence:</strong> {latest_signal['confidence']:.1%}</p>

                <p><strong>Expected Return:</strong> {latest_signal['expected_return']:.1%}</p>

            </div>

            """, unsafe_allow_html=True)

       

        with col2:

            # XAI Explanation

            st.markdown("### 🔍 AI Decision Explanation")

            st.info(latest_signal['explanation'])

           

            # Show contributing factors

            with st.expander("📊 Detailed Analysis"):

                factors_df = pd.DataFrame(latest_signal['factors'])

                st.dataframe(factors_df, use_container_width=True)

 

    def render_sentiment_analysis(self):

        """

        SOLVES: No XAI explanations with interactive sentiment visualization

        """

        st.markdown("## 🧠 News Sentiment Analysis with XAI")

       

        if not self.current_data.get('sentiment_results'):

            st.warning("⚠ No sentiment data available")

            return

       

        # Sentiment overview

        col1, col2 = st.columns([1, 1])

       

        with col1:

            # Sentiment distribution

            sentiment_data = self.current_data['sentiment_results']

            sentiment_counts = self.calculate_sentiment_distribution(sentiment_data)

           

            fig_pie = px.pie(

                values=list(sentiment_counts.values()),

                names=list(sentiment_counts.keys()),

                title="📊 Sentiment Distribution",

                color_discrete_map={

                    'POSITIVE': '#22c55e',

                    'NEGATIVE': '#ef4444',

                    'NEUTRAL': '#6b7280'

                }

            )

            st.plotly_chart(fig_pie, use_container_width=True)

       

        with col2:

            # Sentiment timeline

            sentiment_timeline = self.create_sentiment_timeline(sentiment_data)

            fig_timeline = px.line(

                sentiment_timeline,

                x='date',

                y='sentiment_score',

                title="📈 Sentiment Trend (7 Days)",

                color_discrete_sequence=['#3b82f6']

            )

            st.plotly_chart(fig_timeline, use_container_width=True)

       

        # Interactive article analysis

        st.markdown("### 📰 Article-Level Analysis")

       

        selected_article = st.selectbox(

            "Select article for detailed XAI analysis:",

            options=range(len(sentiment_data[:10])),  # Top 10 articles

            format_func=lambda x: sentiment_data[x].title[:80] + "..." if len(sentiment_data[x].title) > 80 else sentiment_data[x].title

        )

       

        if selected_article is not None:

            self.render_article_xai_analysis(sentiment_data[selected_article])

 

    def render_article_xai_analysis(self, article):

        """

        SOLVES: Black box decisions with word-level explanation visualization

        """

        col1, col2 = st.columns([1, 1])

       

        with col1:

            st.markdown("#### 📝 Article Content")

            st.write(f"*Title:* {article.title}")

            st.write(f"*Source:* {article.source} (Credibility: {article.credibility_score:.2f})")

            st.write(f"*Date:* {article.date}")

            st.write(f"*Description:* {article.description}")

       

        with col2:

            st.markdown("#### 🔍 XAI Analysis")

            st.write(f"*Sentiment:* {article.sentiment_label}")

            st.write(f"*Score:* {article.sentiment_score:.3f}")

            st.write(f"*Confidence:* {article.confidence:.2%}")

            st.write(f"*Apple Relevance:* {article.apple_relevance_score:.2%}")

           

            # Show influential words

            if article.influential_words:

                st.markdown("🎯 Most Influential Words:")

                for word, weight in article.influential_words[:5]:

                    st.write(f"- *{word}*: {weight:.3f}")

 

    def render_stock_analysis(self):

        """

        SOLVES: Generic charts with Apple-specific technical analysis

        """

        st.markdown("## 📈 Apple Stock Technical Analysis")

       

        if 'stock_data' not in self.current_data:

            st.warning("⚠ No stock data available")

            return

       

        stock_data = self.current_data['stock_data'].to_pandas()  # Convert from Polars

       

        # Create comprehensive stock chart

        fig = make_subplots(

            rows=3, cols=1,

            shared_xaxes=True,

            subplot_titles=('Price & Moving Averages', 'Volume', 'Technical Indicators'),

            vertical_spacing=0.05,

            row_heights=[0.6, 0.2, 0.2]

        )

       

        # Price and moving averages

        fig.add_trace(

            go.Scatter(x=stock_data['Date'], y=stock_data['Close'],

                      name='Close Price', line=dict(color='#1f77b4')),

            row=1, col=1

        )

       

        if 'SMA_20' in stock_data.columns:

            fig.add_trace(

                go.Scatter(x=stock_data['Date'], y=stock_data['SMA_20'],

                          name='SMA 20', line=dict(color='#ff7f0e')),

                row=1, col=1

            )

       

        # Volume

        fig.add_trace(

            go.Bar(x=stock_data['Date'], y=stock_data['Volume'],

                   name='Volume', marker_color='#2ca02c'),

            row=2, col=1

        )

       

        # RSI if available

        if 'RSI_14' in stock_data.columns:

            fig.add_trace(

                go.Scatter(x=stock_data['Date'], y=stock_data['RSI_14'],

                          name='RSI (14)', line=dict(color='#d62728')),

                row=3, col=1

            )

            # Add RSI overbought/oversold lines

            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)

            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

       

        fig.update_layout(height=800, title="🍎 Apple Stock Comprehensive Analysis")

        st.plotly_chart(fig, use_container_width=True)

 

    def render_performance_metrics(self):

        """

        SOLVES: No performance tracking with comprehensive analytics

        """

        st.markdown("## 📊 AI Trading Performance")

       

        # Load performance data from your existing CSV files

        performance_data = self.load_performance_data()

       

        col1, col2, col3, col4 = st.columns(4)

       

        with col1:

            total_return = performance_data.get('total_return', 0)

            st.metric("💰 Total Return", f"{total_return:.2%}",

                     delta="vs Buy & Hold")

       

        with col2:

            sharpe_ratio = performance_data.get('sharpe_ratio', 0)

            st.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}",

                     delta="Risk-adjusted")

       

        with col3:

            max_drawdown = performance_data.get('max_drawdown', 0)

            st.metric("📉 Max Drawdown", f"{max_drawdown:.2%}",

                     delta="Peak to trough")

       

        with col4:

            win_rate = performance_data.get('win_rate', 0)

            st.metric("🎯 Win Rate", f"{win_rate:.1%}",

                     delta="Successful trades")

 

   

def render_control_panel(self):

        """

        SOLVES: No interactivity with real-time control capabilities

        """

        st.sidebar.markdown("## ⚙ AI Control Panel")

       

        # Model controls

        st.sidebar.markdown("### 🤖 Model Settings")

       

        confidence_threshold = st.sidebar.slider(

            "Minimum Confidence for Trading",

            min_value=0.5, max_value=0.95, value=0.65, step=0.05,

            help="Only execute trades above this confidence level"

        )

       

        enable_autonomous = st.sidebar.checkbox(

            "🤖 Enable Autonomous Trading",

            value=False,

            help="Allow AI to execute trades automatically"

        )

       

        max_position_size = st.sidebar.slider(

            "Maximum Position Size (%)",

            min_value=5, max_value=25, value=10, step=1,

            help="Maximum percentage of portfolio per trade"

        )

       

        # XAI controls

        st.sidebar.markdown("### 🔍 XAI Settings")

       

        show_attention = st.sidebar.checkbox(

            "Show Attention Weights",

            value=True,

            help="Display word-level attention in sentiment analysis"

        )

       

        explanation_detail = st.sidebar.select_slider(

            "Explanation Detail Level",

            options=["Basic", "Detailed", "Expert"],

            value="Detailed"

        )

       

        # Data refresh controls

        st.sidebar.markdown("### 🔄 Data Controls")

       

        auto_refresh = st.sidebar.checkbox(

            "Auto Refresh Data",

            value=True,

            help="Automatically update data every 5 minutes"

        )

       

        if st.sidebar.button("🔄 Refresh All Data"):

            st.cache_data.clear()

            st.rerun()

       

        # Performance controls

        st.sidebar.markdown("### 📊 Performance")

       

        show_benchmark = st.sidebar.checkbox(

            "Compare vs Buy & Hold",

            value=True,

            help="Show performance comparison with buy-and-hold strategy"

        )

       

        # Store settings in session state

        st.session_state.update({

            'confidence_threshold': confidence_threshold,

            'enable_autonomous': enable_autonomous,

            'max_position_size': max_position_size,

            'show_attention': show_attention,

            'explanation_detail': explanation_detail,

            'auto_refresh': auto_refresh,

            'show_benchmark': show_benchmark

        })

   

def render_news_analysis_detailed(self):

        """

        SOLVES: Basic news display with detailed XAI analysis

        """

        st.markdown("## 📰 Advanced News Analysis")

       

        if not self.current_data.get('news_data'):

            st.warning("⚠ No news data available")

            return

       

        news_data = self.current_data['news_data']

       

        # News filtering options

        col1, col2, col3 = st.columns([1, 1, 1])

       

        with col1:

            min_relevance = st.slider("Minimum Apple Relevance", 0.0, 1.0, 0.7, 0.1)

       

        with col2:

            min_credibility = st.slider("Minimum Source Credibility", 0.0, 1.0, 0.8, 0.1)

       

        with col3:

            max_articles = st.number_input("Max Articles to Show", 1, 50, 10)

       

        # Filter articles

        filtered_articles = [

            article for article in news_data

            if (hasattr(article, 'apple_relevance_score') and

                article.apple_relevance_score >= min_relevance and

                article.credibility_score >= min_credibility)

        ][:max_articles]

       

        if not filtered_articles:

            st.warning("No articles match the current filters")

            return

       

        # Display filtered articles with XAI

        for i, article in enumerate(filtered_articles):

            with st.expander(f"📰 {article.title[:80]}..." if len(article.title) > 80 else article.title):

               

                col1, col2 = st.columns([2, 1])

               

                with col1:

                    st.markdown(f"*Source:* {article.source}")

                    st.markdown(f"*Date:* {article.date}")

                    st.markdown(f"*Description:* {article.description}")

                   

                    if hasattr(article, 'content') and article.content:

                        st.markdown(f"*Content Preview:* {article.content[:200]}...")

               

                with col2:

                    # XAI metrics

                    st.metric("Sentiment", article.sentiment_label if hasattr(article, 'sentiment_label') else "N/A")

                    st.metric("Confidence", f"{article.confidence:.2%}" if hasattr(article, 'confidence') else "N/A")

                    st.metric("Relevance", f"{article.apple_relevance_score:.2%}")

                    st.metric("Credibility", f"{article.credibility_score:.2%}")

                   

                    # Temporal weight

                    if hasattr(article, 'temporal_weight'):

                        st.metric("Recency Weight", f"{article.temporal_weight:.2f}")

               

                # Show influential words if available

                if (hasattr(article, 'influential_words') and

                    article.influential_words and

                    st.session_state.get('show_attention', True)):

                   

                    st.markdown("🎯 Most Influential Words:")

                   

                    # Create word importance visualization

                    words_df = pd.DataFrame(

                        article.influential_words[:10],

                        columns=['Word', 'Importance']

                    )

                   

                    fig_words = px.bar(

                        words_df,

                        x='Importance',

                        y='Word',

                        orientation='h',

                        title="Word Importance Scores",

                        color='Importance',

                        color_continuous_scale='Viridis'

                    )

                    fig_words.update_layout(height=300)

                    st.plotly_chart(fig_words, use_container_width=True)

   

def render_model_performance_analysis(self):

        """

        SOLVES: No model performance tracking with detailed analytics

        """

        st.markdown("## 🧠 AI Model Performance Analysis")

       

        # Model accuracy tracking

        col1, col2 = st.columns([1, 1])

       

        with col1:

            st.markdown("### 📊 Prediction Accuracy")

           

            # Simulated accuracy data