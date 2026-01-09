from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s

def equal_weight(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    return pd.Series([1.0 / n] * n, index=tickers, name="weight")

def max_diversification(cov: pd.DataFrame, bounds: tuple[float, float]=(0.0, 0.2)) -> pd.Series:
    """
    Maximize diversification ratio = (w' sigma) / sqrt(w' Cov w).
    
    Args:
        cov: Covariance matrix
        bounds: Min and max weight per asset
        
    Returns:
        Optimal weights as Series
    """
    tickers = list(cov.columns)
    sigma = np.sqrt(np.diag(cov.values))

    def obj(w):
        w_norm = normalize(w)
        num = w_norm @ sigma
        den = np.sqrt(w_norm @ cov.values @ w_norm)
        return -(num / den)  # maximize => minimize negative

    n = len(tickers)
    x0 = np.ones(n) / n
    bnds = [bounds] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
    w = normalize(res.x)
    return pd.Series(w, index=tickers, name="weight")


def min_variance(cov: pd.DataFrame, bounds: tuple[float, float]=(0.0, 0.35)) -> pd.Series:
    """
    Minimize portfolio variance.
    
    Args:
        cov: Covariance matrix
        bounds: Min and max weight per asset
        
    Returns:
        Optimal weights as Series
    """
    tickers = list(cov.columns)

    def obj(w):
        return w @ cov.values @ w

    n = len(tickers)
    x0 = np.ones(n) / n
    bnds = [bounds] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
    w = normalize(res.x)
    return pd.Series(w, index=tickers, name="weight")


def risk_parity(cov: pd.DataFrame, bounds: tuple[float, float]=(0.0, 0.35)) -> pd.Series:
    """
    Risk parity portfolio: equalize risk contribution from each asset.
    Uses inverse-volatility as starting point, then optimizes.
    
    Args:
        cov: Covariance matrix
        bounds: Min and max weight per asset
        
    Returns:
        Optimal weights as Series
    """
    tickers = list(cov.columns)
    sigma = np.sqrt(np.diag(cov.values))
    
    # Inverse volatility as initial weights
    inv_vol = 1.0 / sigma
    x0 = inv_vol / inv_vol.sum()

    def risk_contrib(w):
        """Calculate risk contribution for each asset."""
        port_var = w @ cov.values @ w
        port_vol = np.sqrt(port_var)
        # Marginal contribution to risk: dVol/dw = (Cov @ w) / Vol
        marginal = (cov.values @ w) / port_vol
        # Risk contribution: w * marginal
        return w * marginal

    def obj(w):
        """Minimize variance of risk contributions."""
        rc = risk_contrib(w)
        target = np.mean(rc)
        return np.sum((rc - target) ** 2)

    n = len(tickers)
    bnds = [bounds] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
    w = normalize(res.x)
    return pd.Series(w, index=tickers, name="weight")

def apply_satellite_cap(weights: pd.Series, universe: pd.DataFrame, satellite_cap: float = 0.5) -> pd.Series:
    """
    Enforce satellite cap constraint.
    
    If satellites exceed cap, scale them down and re-normalize core up.
    
    Args:
        weights: Portfolio weights
        universe: Universe DataFrame with 'ticker' and 'bucket' columns
        satellite_cap: Maximum weight for satellite bucket (default 0.5)
        
    Returns:
        Adjusted weights
    """
    u = universe.set_index("ticker")
    bucket = u.loc[weights.index, "bucket"]
    sat_mask = bucket.eq("satellite").values
    sat_sum = float(weights[sat_mask].sum())
    if sat_sum <= satellite_cap + 1e-9:
        return weights

    scale = satellite_cap / sat_sum
    w2 = weights.copy()
    w2[sat_mask] *= scale
    # distribute leftover to core proportionally
    leftover = 1.0 - float(w2.sum())
    core_mask = ~sat_mask
    core_sum = float(w2[core_mask].sum())
    if core_sum > 0:
        w2[core_mask] += w2[core_mask] / core_sum * leftover
    else:
        # all satellite? shouldn't happen if universe has core
        w2 /= float(w2.sum())
    return w2
