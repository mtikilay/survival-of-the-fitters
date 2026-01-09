from __future__ import annotations
import pandas as pd
import yfinance as yf
import numpy as np

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
    
    # Filter out columns with all NaN values (failed downloads)
    failed_tickers = px.columns[px.isna().all()].tolist()
    if failed_tickers:
        print(f"\nWarning: Excluding {len(failed_tickers)} ticker(s) with no valid data: {', '.join(failed_tickers)}")
        px = px.dropna(axis=1, how="all")
    
    if px.empty or len(px.columns) == 0:
        raise ValueError("No valid price data available for any ticker")
    
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
