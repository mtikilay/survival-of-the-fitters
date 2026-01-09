from __future__ import annotations
import pandas as pd
import yfinance as yf
import numpy as np
import numpy as np
import yfinance as yf
import warnings

def fetch_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # yfinance returns multi-index columns for multiple tickers; normalize to DataFrame[ticker]
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        # Typical: ("Adj Close", "LINC"), ...
        if ("Adj Close", tickers[0]) in data.columns:
            px = data["Adj Close"].copy()
        elif ("Close", tickers[0]) in data.columns:
            px = data["Close"].copy()
        else:
            raise ValueError("Unexpected yfinance columns; expected Adj Close/Close.")
    else:
        # Single ticker case
        px = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    px = px.dropna(how="all")
    px = px.sort_index()
    return px

def to_returns(px: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Convert price series to returns.
    
    Args:
        px: DataFrame of prices with tickers as columns
        method: 'log' for log returns or 'simple' for arithmetic returns
        
    Returns:
        DataFrame of returns
    """
    if method == "log":
        rets = np.log(px / px.shift(1))
    elif method == "simple":
        rets = px.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return rets.dropna(how="all")


def check_data_quality(px: pd.DataFrame, min_obs: int = 252, max_missing_pct: float = 0.3) -> pd.DataFrame:
    """
    Check data quality and warn about issues.
    
    Args:
        px: DataFrame of prices
        min_obs: Minimum number of observations required
        max_missing_pct: Maximum percentage of missing values allowed
        
    Returns:
        Filtered DataFrame with sufficient data quality
    """
    n_total = len(px)
    keep_cols = []
    
    for col in px.columns:
        n_valid = px[col].notna().sum()
        missing_pct = 1.0 - (n_valid / n_total)
        
        if n_valid < min_obs:
            warnings.warn(f"Ticker {col}: insufficient history ({n_valid} obs < {min_obs} required). Dropping.")
        elif missing_pct > max_missing_pct:
            warnings.warn(f"Ticker {col}: too many missing values ({missing_pct:.1%} > {max_missing_pct:.1%}). Dropping.")
        else:
            if missing_pct > 0.05:
                warnings.warn(f"Ticker {col}: {missing_pct:.1%} missing values (keeping).")
            keep_cols.append(col)
    
    if not keep_cols:
        raise ValueError("No tickers passed data quality checks!")
    
    return px[keep_cols]
