from __future__ import annotations
import pandas as pd
import yfinance as yf

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
    if method == "log":
        rets = (px / px.shift(1)).applymap(lambda x: None if pd.isna(x) else x).astype(float)
        rets = (px / px.shift(1)).applymap(lambda x: x).astype(float)
        rets = (px / px.shift(1)).applymap(lambda x: x)
        # safer:
        rets = (px / px.shift(1))
        rets = rets.applymap(lambda x: float(x) if pd.notna(x) else float("nan"))
        import numpy as np
        rets = np.log(rets)
    elif method == "simple":
        rets = px.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return rets.dropna(how="all")
