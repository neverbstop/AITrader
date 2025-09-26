"""
Decision Tracker Module for XAI in Trading
Logs decisions, explanations, and supporting context for transparency and auditability.
"""

from typing import List, Dict, Any
from datetime import datetime

class DecisionTracker:
    """
    Logs every trading decision with signals and an explanation.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log_decision(
        self,
        time: datetime,
        action: str,
        quant_signal: str,
        news_signal: str,
        sentiment_score: float,
        explanation: str
    ):
        """
        Store a trading decision event with rationale and all input signals.
        """
        self.history.append({
            "timestamp": time.isoformat(),
            "action": action,
            "quant_signal": quant_signal,
            "news_signal": news_signal,
            "sentiment_score": sentiment_score,
            "explanation": explanation,
        })

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve a list of past decisions (most recent first).
        """
        return self.history[-limit:][::-1]
