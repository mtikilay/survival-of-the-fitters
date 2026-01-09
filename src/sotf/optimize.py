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

def max_diversification(cov: pd.DataFrame, bounds: tuple[float, float]=(0.05, 0.35)) -> pd.Series:
    # Maximize diversification ratio = (w' sigma) / sqrt(w' Cov w)
    tickers = list(cov.columns)
    sigma = np.sqrt(np.diag(cov.values))

    def obj(w):
        w = normalize(w)
        num = w @ sigma
        den = np.sqrt(w @ cov.values @ w)
        return -(num / den)  # maximize => minimize negative

    n = len(tickers)
    x0 = np.ones(n) / n
    bnds = [bounds] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
    w = normalize(res.x)
    return pd.Series(w, index=tickers, name="weight")
