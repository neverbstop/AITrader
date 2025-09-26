# core/autonomous_agent.py - Enhanced with XAI and Advanced Intelligence

"""

SOLVES: Basic trading logic limitations with sophisticated AI decision-making

 

Problems Solved:

❌ No decision explanations → ✅ Complete XAI for every trade

❌ Simple signal logic → ✅ Advanced ensemble decision making

❌ No confidence consideration → ✅ Confidence-based position sizing

❌ Basic risk management → ✅ Comprehensive risk controls

❌ No learning capability → ✅ Performance tracking and adaptation

❌ Generic trading → ✅ Apple-specific trading intelligence

"""

 
from xai.explainer import Explainer
from xai.decision_tracker import DecisionTracker

import pandas as pd

import numpy as np

from datetime import datetime, timedelta

from typing import Dict, List, Optional, Tuple, Any

import logging

import json

from dataclasses import dataclass, asdict

 

# Import configurations

from config import (

CAPITAL, BROKERAGE_PERCENT, TAX_PERCENT, THRESHOLD_PROFIT_PERCENT,

AUTONOMOUS_BUDGET, AUTONOMOUS_ENABLED, TICKER, AUTONOMOUS_CONFIG,

XAI_CONFIG, APPLE_CONFIG

)

from core.risk_management import apply_fees

 

# Setup logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(_name_)

 

@dataclass

class TradingDecision:

# """

# SOLVES: Lack of decision tracking with comprehensive decision structure

# """

    timestamp: datetime

    action: str  # BUY, SELL, HOLD

    price: float

    shares: int

    confidence: float

    reasoning: str

    contributing_factors: Dict[str, float]

    risk_assessment: Dict[str, Any]

    expected_return: float

    expected_risk: float

    market_conditions: Dict[str, Any]


    # XAI Explanation

    explanation: str

    key_signals: List[str]

    conflicting_signals: List[str]

    uncertainty_factors: List[str]

 

@dataclass

class PortfolioState:


    cash: float

    position: int

    entry_price: float

    current_value: float

    unrealized_pnl: float

    total_pnl: float

    win_rate: float

    sharpe_ratio: float

    max_drawdown: float

    consecutive_losses: int

    last_trade_date: datetime

 

class EnhancedAutonomousAgent:


    def __init__(self, ticker: str):

        # Basic configuration
        from xai.explainer import Explainer
        from xai.decision_tracker import DecisionTracker
        self.explainer = Explainer()
        self.decision_tracker = DecisionTracker()

        self.ticker = ticker

        self.enabled = AUTONOMOUS_ENABLED

        self.initial_budget = AUTONOMOUS_BUDGET

        self.budget = AUTONOMOUS_BUDGET

        self.cash = self.budget

        self.position = 0

        self.entry_price = 0.0

        self.is_in_position = False

        

        # Enhanced tracking

        self.trades = []

        self.decisions = []  # Track all decisions, not just executed trades

        self.performance_history = []

        self.last_trade_action = None

        self.last_decision_time = None

        

        # Risk management

        self.max_position_size = AUTONOMOUS_CONFIG.get('max_position_size_percent', 10) / 100

        self.stop_loss_percent = AUTONOMOUS_CONFIG.get('stop_loss_percent', 5) / 100

        self.confidence_threshold = AUTONOMOUS_CONFIG.get('confidence_threshold', 0.65)

        self.max_daily_trades = AUTONOMOUS_CONFIG.get('max_trades_per_day', 3)

        

        # Performance tracking

        self.daily_trades_count = 0

        self.last_trade_date = None

        self.consecutive_losses = 0

        self.win_rate = 0.0

        self.total_pnl = 0.0

        

        # Apple-specific intelligence

        self.apple_intelligence = AppleSpecificIntelligence()

        

        # XAI components

        self.enable_xai = XAI_CONFIG.get('enabled', True)

        self.decision_explainer = DecisionExplainer() if self.enable_xai else None

        

        logger.info(f"✅ Enhanced Autonomous Agent initialized for {ticker}")

        logger.info(f"🎯 Confidence threshold: {self.confidence_threshold:.2%}")

        logger.info(f"🛡 Stop loss: {self.stop_loss_percent:.2%}")

    

    def enable(self):

        """

        SOLVES: Simple enable/disable with enhanced activation

        """

        self.enabled = True

        activation_msg = f"""

        ✅ AUTONOMOUS AGENT ACTIVATED

        📊 Target: {self.ticker}

        💰 Budget: ${self.budget:,.2f}

        🎯 Confidence Threshold: {self.confidence_threshold:.1%}

        🛡 Risk Controls: Active

        🧠 XAI Explanations: {'Enabled' if self.enable_xai else 'Disabled'}

        """

        print(activation_msg)

        logger.info("Autonomous Agent enabled with enhanced features")


    def disable(self):

        """

        SOLVES: Simple disable with performance summary

        """

        self.enabled = False

        

        # Generate performance summary

        current_value = self.get_portfolio_value(self.get_last_known_price())

        total_return = (current_value - self.initial_budget) / self.initial_budget * 100

        

        summary_msg = f"""

        ❌ AUTONOMOUS AGENT DEACTIVATED

        📈 Total Return: {total_return:.2f}%

        💰 Current Value: ${current_value:,.2f}

        📊 Total Trades: {len(self.trades)}

        🎯 Win Rate: {self.win_rate:.1%}

        📉 Max Drawdown: {self.calculate_max_drawdown():.2%}

        """

        print(summary_msg)

        logger.info("Autonomous Agent disabled")


    def run_if_enabled(self, news_signal=None, quant_signal=None, current_price=None):

        """

        SOLVES: Basic signal processing with advanced decision-making engine

        """

        if not self.enabled or current_price is None:

            return

        

        # Reset daily trade count if new day

        self._reset_daily_counter()

        

        # Check if we've hit daily trade limit

        if self.daily_trades_count >= self.max_daily_trades:

            logger.info(f"📊 Daily trade limit reached ({self.max_daily_trades})")

            return

        

        # Gather all market intelligence

        market_intelligence = self._gather_market_intelligence(

            news_signal, quant_signal, current_price

        )

        

        # Generate comprehensive trading decision

        decision = self._make_enhanced_trading_decision(

            market_intelligence, current_price

        )

        

        # Log decision for learning

        self.decisions.append(decision)
        self.decision_tracker.log_decision(
            time=decision.timestamp,
            action=decision.action,
            quant_signal=market_intelligence['signals']['quant_signal'],
            news_signal=market_intelligence['signals']['news_signal'],
            sentiment_score=market_intelligence['signals']['news_sentiment_score'],
            explanation=decision.explanation
        )

        

        # Execute decision if confidence is sufficient

        if decision.confidence >= self.confidence_threshold:

            self._execute_decision(decision, current_price)

        else:
            logger.info(f"🤔 Decision confidence too low: {decision.confidence:.2%} < {self.confidence_threshold:.2%}")
            if self.enable_xai:
                print(decision.explanation)

    def _execute_decision(self, decision: TradingDecision, current_price: float):
        """
        Executes a trading decision (BUY or SELL) and logs the action.
        """
        action_log = f"EXECUTING TRADE: {decision.action} {decision.shares} shares of {self.ticker} at ${current_price:.2f}"
        logger.info(action_log)
        print(action_log)
        if self.enable_xai:
            print(decision.explanation)

        # Here you would add the actual brokerage execution logic
        # For now, we'll just simulate the portfolio change

        if decision.action == "BUY" and decision.shares > 0:
            cost = decision.shares * current_price
            self.cash -= cost
            self.position += decision.shares
            self.entry_price = ((self.entry_price * (self.position - decision.shares)) + cost) / self.position if self.position > 0 else current_price
            self.is_in_position = True

        elif decision.action == "SELL" and self.is_in_position:
            # For simplicity, we sell the entire position
            revenue = self.position * current_price
            self.cash += revenue
            pnl = (current_price - self.entry_price) * self.position
            self.total_pnl += pnl
            self.position = 0
            self.entry_price = 0.0
            self.is_in_position = False

        # Log the trade
        self.trades.append(asdict(decision))
        self.daily_trades_count += 1
        self.last_trade_action = decision.action


    def _gather_market_intelligence(self, news_signal, quant_signal, current_price) -> Dict[str, Any]:

        """

        SOLVES: Limited signal processing with comprehensive market analysis

        """

        intelligence = {

            'price_data': {

                'current_price': current_price,

                'entry_price': self.entry_price,

                'unrealized_pnl_pct': self._calculate_unrealized_pnl_percent(current_price) if self.is_in_position else 0

            },

            'signals': {

                'news_signal': news_signal,

                'quant_signal': quant_signal,

                'news_sentiment_score': self._extract_sentiment_score(news_signal),

                'technical_strength': self._assess_technical_strength(quant_signal)

            },

            'risk_factors': {

                'position_risk': self.position * current_price / self.budget if self.position > 0 else 0,

                'consecutive_losses': self.consecutive_losses,

                'recent_volatility': self._estimate_recent_volatility(current_price)

            },

            'market_regime': self._detect_market_regime(current_price),

            'apple_factors': self.apple_intelligence.analyze_apple_context(news_signal, current_price),

            'portfolio_state': self._get_current_portfolio_state(current_price)

        }

        

        return intelligence


    def _make_enhanced_trading_decision(self, intelligence: Dict, current_price: float) -> TradingDecision:

        """

        SOLVES: Simple buy/sell logic with sophisticated decision engine

        """

        # Initialize decision components

        action = "HOLD"

        confidence = 0.5

        reasoning_parts = []

        contributing_factors = {}

        key_signals = []

        conflicting_signals = []

        uncertainty_factors = []

        

        # Analyze sell conditions first (risk management priority)

        if self.is_in_position:

            sell_decision = self._analyze_sell_conditions(intelligence, current_price)

            if sell_decision['should_sell']:

                action = "SELL"

                confidence = sell_decision['confidence']

                reasoning_parts.extend(sell_decision['reasons'])

                contributing_factors.update(sell_decision['factors'])

                key_signals = sell_decision['signals']

                

        # Analyze buy conditions if not selling and not in position

        elif not self.is_in_position:

            buy_decision = self._analyze_buy_conditions(intelligence, current_price)

            if buy_decision['should_buy']:

                action = "BUY"

                confidence = buy_decision['confidence']

                reasoning_parts.extend(buy_decision['reasons'])

                contributing_factors.update(buy_decision['factors'])

                key_signals = buy_decision['signals']

            else:

                # Analyze why we're not buying

                uncertainty_factors = buy_decision.get('uncertainty_factors', [])

                conflicting_signals = buy_decision.get('conflicting_signals', [])

        

        # Calculate position sizing based on confidence

        shares = self._calculate_position_size(action, confidence, current_price)

        

        # Risk assessment

        risk_assessment = self._assess_trade_risk(action, shares, current_price, intelligence)

        

        # Expected return calculation

        expected_return = self._calculate_expected_return(action, intelligence, current_price)

        expected_risk = risk_assessment.get('value_at_risk', 0.05)

        
        # Generate explanation
        explanation = self.explainer.explain_decision(
            quant_signal=intelligence['signals']['quant_signal'],
            news_signal=intelligence['signals']['news_signal'],
            sentiment_score=intelligence['signals']['news_sentiment_score'],
            action=action,
            context={'reasons': reasoning_parts, 'confidence': confidence}
        )

        

        # Create comprehensive decision

        decision = TradingDecision(
            timestamp=datetime.now(),
            action=action,
            price=current_price,
            shares=shares,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "No strong signals detected",
            contributing_factors=contributing_factors,
            risk_assessment=risk_assessment,
            expected_return=expected_return,
            expected_risk=expected_risk,
            market_conditions=intelligence,
            explanation=explanation,
            key_signals=key_signals,
            conflicting_signals=conflicting_signals,
            uncertainty_factors=uncertainty_factors
        )


        

        return decision


    def _analyze_sell_conditions(self, intelligence: Dict, current_price: float) -> Dict:

        """

        SOLVES: Basic sell logic with comprehensive sell analysis

        """

        should_sell = False

        confidence = 0.0

        reasons = []

        factors = {}

        signals = []

        

        unrealized_pnl_pct = intelligence['price_data']['unrealized_pnl_pct']

        

        # 1. Profit taking condition

        if unrealized_pnl_pct >= THRESHOLD_PROFIT_PERCENT:

            should_sell = True

            confidence += 0.3

            reasons.append(f"Profit target reached ({unrealized_pnl_pct:.1f}% >= {THRESHOLD_PROFIT_PERCENT}%)")

            factors['profit_taking'] = 0.3

            signals.append("PROFIT_TARGET")

        

        # 2. Stop loss condition

        if unrealized_pnl_pct <= -self.stop_loss_percent * 100:

            should_sell = True

            confidence += 0.4

            reasons.append(f"Stop loss triggered ({unrealized_pnl_pct:.1f}% <= -{self.stop_loss_percent*100:.1f}%)")

            factors['stop_loss'] = 0.4

            signals.append("STOP_LOSS")

        

        # 3. Technical sell signal

        if intelligence['signals']['quant_signal'] == "QUANT_SELL":

            should_sell = True

            confidence += 0.2

            reasons.append("Technical indicators suggest selling")

            factors['technical_sell'] = 0.2

            signals.append("TECHNICAL_SELL")

        

        # 4. Negative news sentiment

        news_sentiment = intelligence['signals']['news_sentiment_score']

        if news_sentiment < -0.3:

            should_sell = True

            confidence += 0.15

            reasons.append(f"Strong negative sentiment ({news_sentiment:.2f})")

            factors['negative_sentiment'] = 0.15

            signals.append("NEGATIVE_NEWS")

        

        # 5. Apple-specific sell conditions

        apple_factors = intelligence['apple_factors']

        if apple_factors.get('negative_product_news', False):

            confidence += 0.1

            reasons.append("Apple product concerns detected")

            factors['apple_concerns'] = 0.1

            signals.append("APPLE_CONCERNS")

        

        # Cap confidence at 0.95

        confidence = min(confidence, 0.95)

        

        return {

            'should_sell': should_sell,

            'confidence': confidence,

            'reasons': reasons,

            'factors': factors,

            'signals': signals

        }



    def _analyze_buy_conditions(self, intelligence: Dict, current_price: float) -> Dict:

        """

        SOLVES: Simple buy logic with sophisticated buy analysis

        """

        should_buy = False

        confidence = 0.0

        reasons = []

        factors = {}

        signals = []

        uncertainty_factors = []

        conflicting_signals = []

        

        # Check if we have sufficient funds

        max_investment = self.cash * self.max_position_size

        if max_investment < current_price:

                uncertainty_factors.append("Insufficient funds for meaningful position")

                return {

                    'should_buy': False,

                    'confidence': 0.0,

                    'reasons': ["Insufficient funds"],

                    'factors': {},

                    'signals': [],

                    'uncertainty_factors': uncertainty_factors,

                    'conflicting_signals': conflicting_signals

            }

        

        # 1. Positive news sentiment

        news_sentiment = intelligence['signals']['news_sentiment_score']

        if news_sentiment > 0.2:

            confidence += 0.25

            reasons.append(f"Positive sentiment detected ({news_sentiment:.2f})")

            factors['positive_sentiment'] = 0.25

            signals.append("POSITIVE_NEWS")

        elif news_sentiment < -0.1:

                conflicting_signals.append(f"Negative sentiment ({news_sentiment:.2f})")

        

        # 2. Technical buy signal

        if intelligence['signals']['quant_signal'] == "QUANT_BUY":

            confidence += 0.3

            reasons.append("Technical indicators suggest buying")

            factors['technical_buy'] = 0.3

            signals.append("TECHNICAL_BUY")

        elif intelligence['signals']['quant_signal'] == "QUANT_SELL":

                conflicting_signals.append("Technical indicators suggest selling")

        

        # 3. Apple-specific positive factors

        apple_factors = intelligence['apple_factors']

        if apple_factors.get('positive_product_news', False):

            confidence += 0.2

            reasons.append("Positive Apple product news")

            factors['apple_positive'] = 0.2

            signals.append("APPLE_POSITIVE")

        

        if apple_factors.get('earnings_season', False):

            confidence += 0.1

            reasons.append("Approaching Apple earnings season")

            factors['earnings_season'] = 0.1 
            signals.append("EARNINGS_SEASON")

        

        # 4. Risk considerations

        if intelligence['risk_factors']['consecutive_losses'] >= 3:

            confidence *= 0.7  # Reduce confidence after consecutive losses
            uncertainty_factors.append("Recent consecutive losses")

        

        if intelligence['risk_factors']['recent_volatility'] > 0.05:  # >5% volatility

            confidence *= 0.8

            uncertainty_factors.append("High market volatility")

        

        # 5. Market regime consideration

        market_regime = intelligence['market_regime']

        if market_regime == 'BULL_MARKET':

            confidence += 0.1

            factors['bull_market'] = 0.1

        elif market_regime == 'BEAR_MARKET':

            confidence *= 0.6

            conflicting_signals.append("Bear market conditions")

        

        # Require minimum signals for buy decision

        if len(signals) >= 2 and confidence >= 0.4:

            should_buy = True

        

        # Cap confidence

        confidence = min(confidence, 0.95)

        

        return {

            'should_buy': should_buy,

            'confidence': confidence,

            'reasons': reasons,

            'factors': factors,

            'signals': signals,

            'uncertainty_factors': uncertainty_factors,

            'conflicting_signals': conflicting_signals

        }


    def _calculate_position_size(self, action: str, confidence: float, current_price: float) -> int:

        """

        SOLVES: Fixed position sizing with confidence-based dynamic sizing

        """

        if action != "BUY" or not self.cash:

            return 0

        

        # Base position size as percentage of available cash

        base_position_pct = self.max_position_size * confidence

        max_investment = self.cash * base_position_pct

        

        # Calculate shares, ensuring we don't exceed budget

        shares = int(max_investment / current_price)

        

        # Ensure we can afford the shares including fees

        total_cost = shares * current_price * (1 + BROKERAGE_PERCENT + TAX_PERCENT)

        if total_cost > self.cash:

            shares = int(self.cash / (current_price * (1 + BROKERAGE_PERCENT + TAX_PERCENT)))

        

        return max(0, shares)


    def _assess_trade_risk(self, action: str, shares: int, current_price: float, intelligence: Dict) -> Dict:

        """

        SOLVES: No risk assessment with comprehensive risk analysis

        """

        if action == "HOLD" or shares == 0:

            return {'risk_level': 'LOW', 'value_at_risk': 0.0}

        

        trade_value = shares * current_price

        portfolio_exposure = trade_value / self.budget

        

        # Calculate Value at Risk (simplified)

        volatility = intelligence['risk_factors']['recent_volatility']

        value_at_risk = trade_value * volatility * 2.33  # 99% confidence VaR

        

        risk_level = "LOW"

        if portfolio_exposure > 0.15:

            risk_level = "HIGH"

        elif portfolio_exposure > 0.08:

            risk_level = "MEDIUM"

        

        return {

            'risk_level': risk_level,

            'portfolio_exposure_pct': portfolio_exposure * 100,

            'value_at_risk': value_at_risk,

            'trade_value': trade_value,

            'estimated_volatility': volatility

        }


    def _calculate_expected_return(self, action: str, intelligence: Dict, current_price: float) -> float:

        """

        SOLVES: No return expectations with expected return calculation

        """

        if action == "HOLD":

            return 0.0

        

        # Base expected return from sentiment and technical signals

        news_sentiment = intelligence['signals']['news_sentiment_score']

        technical_strength = intelligence['signals']['technical_strength']

        

        expected_return = (news_sentiment * 0.5 + technical_strength * 0.5) * 0.1  # Scale to realistic range

        

        # Adjust for Apple-specific factors

        apple_factors = intelligence['apple_factors']

        if apple_factors.get('earnings_season'):

            expected_return *= 1.2

        

        # Adjust for market regime

        if intelligence['market_regime'] == 'BULL_MARKET':

            expected_return *= 1.1

        elif intelligence['market_regime'] == 'BEAR_MARKET':

            expected_return *= 0.7

        

        return expected_return
