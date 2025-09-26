"""
Explainer Module for XAI in Trading
Generates human-understandable explanations for trading decisions.
"""

from typing import Optional

class Explainer:
    """
    Generates natural language explanations for trading actions (BUY, SELL, HOLD)
    based on quantitative, news, and sentiment signals.
    """

    def explain_decision(
        self,
        quant_signal: Optional[str],
        news_signal: Optional[str],
        sentiment_score: Optional[float],
        action: str,
        context: Optional[dict] = None
    ) -> str:
        """
        Constructs a rich explanation for a trading action, emphasizing why
        the situation is significant for the user.
        """
        signals_summary = self.summarize_signals(quant_signal, news_signal, sentiment_score)
        explanation = ""
        if action == "BUY":
            explanation = (
                "The system recommends a BUY because current signals all align positively. "
                "Technical analysis reveals a strong bullish crossover. "
                f"{signals_summary} This alignment of factors indicates high probability for upward movement, "
                "and missing out now could mean losing early entry on a strong trend."
            )
        elif action == "SELL":
            explanation = (
                "A SELL is suggested as multiple indicators signal a potential downturn. "
                "Quantitative analysis shows weakening momentum or a bearish crossover. "
                f"{signals_summary} Acting now protects gains and reduces exposure to possible losses in a declining market."
            )
        elif action == "HOLD":
            explanation = (
                "No clear action (HOLD) is recommended since signals are mixed or weak. "
                f"{signals_summary} Waiting for a stronger signal prevents premature trade decisions."
            )
        else:
            explanation = "No valid trading action determined from the available signals."
        return explanation

    def summarize_signals(
        self,
        quant_signal: Optional[str],
        news_signal: Optional[str],
        sentiment_score: Optional[float]
    ) -> str:
        """
        Returns a concise summary of all input signals and their direction.
        """
        parts = []
        if quant_signal is not None:
            parts.append(f"Quantitative signals point to {quant_signal}")
        if news_signal is not None:
            parts.append(f"News sentiment is {news_signal.lower()}")
        if sentiment_score is not None:
            if sentiment_score > 0.5:
                senti_part = "and overall market mood is strongly positive"
            elif sentiment_score < -0.5:
                senti_part = "and overall market mood is strongly negative"
            else:
                senti_part = "and sentiment is neutral"
            parts.append(senti_part)
        if not parts:
            return "Insufficient signal data."
        return ", ".join(parts) + "."

