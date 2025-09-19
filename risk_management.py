import numpy as np
from config import BROKERAGE_PERCENT, TAX_PERCENT


def historical_var(returns_series, alpha=0.05):
    # returns series daily returns; VaR at alpha
    q = np.percentile(returns_series.dropna(), 100*alpha)
    return -q   # positive number = expected loss at alpha

def monte_carlo_price_simulation(current_price, returns, n_steps=5, n_sims=1000):
    # simple bootstrap Monte Carlo using historical returns
    sims = np.zeros((n_sims, n_steps))
    for i in range(n_sims):
        sampled = np.random.choice(returns, size=n_steps, replace=True)
        path = current_price * np.cumprod(1 + sampled)
        sims[i,:] = path
    return sims

def apply_fees(amount):
    """
    Deducts brokerage + tax from a trade amount.
    """
    fees = amount * (BROKERAGE_PERCENT + TAX_PERCENT)
    return amount - fees
