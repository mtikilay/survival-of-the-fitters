from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable

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


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        Maximum drawdown as a negative fraction
    """
    cummax = equity_curve.expanding().max()
    drawdown = (equity_curve - cummax) / cummax
    return float(drawdown.min())


def backtest(
    rets: pd.DataFrame,
    weights_func: Callable,
    rebal_freq: str = "M",
    **weights_kwargs
) -> dict:
    """
    Simple backtest with periodic rebalancing.
    
    Args:
        rets: DataFrame of returns with tickers as columns
        weights_func: Function that takes a covariance matrix and returns weights
        rebal_freq: Rebalancing frequency ('M' for monthly, 'Q' for quarterly)
        **weights_kwargs: Additional arguments to pass to weights_func
        
    Returns:
        Dictionary with equity_curve, stats, turnover
    """
    # Group returns by rebalancing period
    rets_copy = rets.copy()
    rets_copy['period'] = rets_copy.index.to_period(rebal_freq)
    periods = rets_copy['period'].unique()
    
    equity = 100.0
    equity_curve = []
    dates = []
    turnovers = []
    prev_weights = None
    
    for period in periods:
        # Get returns for this period
        period_mask = rets_copy['period'] == period
        period_rets = rets_copy.loc[period_mask, rets.columns]
        
        if len(period_rets) == 0:
            continue
        
        # Calculate weights at start of period using trailing data
        lookback_data = rets_copy.loc[rets_copy.index < period_rets.index[0], rets.columns]
        
        if len(lookback_data) < 60:  # Need at least ~3 months of data
            continue
        
        # Calculate covariance from lookback data
        cov = lookback_data.cov() * TRADING_DAYS
        
        # Get new weights
        try:
            new_weights = weights_func(cov, **weights_kwargs)
        except Exception as e:
            print(f"Warning: weights_func failed for period {period}: {e}")
            continue
        
        # Calculate turnover
        if prev_weights is not None:
            # Align to same tickers
            common = prev_weights.index.intersection(new_weights.index)
            if len(common) > 0:
                turnover = (new_weights.loc[common] - prev_weights.loc[common]).abs().sum()
                turnovers.append(turnover)
        
        prev_weights = new_weights
        
        # Apply returns for this period
        for date, row in period_rets.iterrows():
            # Ensure weights align with available returns
            available = row.dropna().index.intersection(new_weights.index)
            if len(available) == 0:
                continue
            
            w = new_weights.loc[available]
            w = w / w.sum()  # Renormalize to available tickers
            r = row.loc[available]
            
            # For log returns: equity *= exp(sum(w * r))
            port_ret = (w * r).sum()
            equity *= np.exp(port_ret)
            
            equity_curve.append(equity)
            dates.append(date)
    
    if len(equity_curve) == 0:
        return {
            "equity_curve": pd.Series(),
            "cagr": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "avg_turnover": np.nan,
        }
    
    equity_series = pd.Series(equity_curve, index=dates)
    
    # Calculate statistics
    returns = np.log(equity_series / equity_series.shift(1)).dropna()
    years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
    
    if years > 0:
        cagr = (equity_series.iloc[-1] / 100.0) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan
    
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = cagr / vol if vol > 0 else np.nan
    max_dd = max_drawdown(equity_series)
    avg_turnover = np.mean(turnovers) if turnovers else np.nan
    
    return {
        "equity_curve": equity_series,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_turnover": avg_turnover,
    }
