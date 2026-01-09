from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def annualized_return(rets: pd.DataFrame) -> pd.Series:
    # log returns: sum * (252 / n)
    mu = rets.mean() * TRADING_DAYS
    return mu

def annualized_vol(rets: pd.DataFrame) -> pd.Series:
    sig = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return sig

def corr(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.corr()

def cov_annual(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.cov() * TRADING_DAYS

def portfolio_stats(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> dict:
    w = weights.values.reshape(-1, 1)
    port_mu = float((weights * mu).sum())
    port_var = float(w.T @ cov.values @ w)
    port_vol = float(np.sqrt(port_var))
    sharpe = port_mu / port_vol if port_vol > 0 else np.nan
    return {"ann_return": port_mu, "ann_vol": port_vol, "sharpe_naive": sharpe}
