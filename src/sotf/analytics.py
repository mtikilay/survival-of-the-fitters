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
