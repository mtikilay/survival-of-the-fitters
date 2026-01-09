from __future__ import annotations
from pathlib import Path
import pandas as pd

from sotf.universe import load_universe
from sotf.fetch import fetch_adj_close, to_returns
from sotf.analytics import annualized_return, annualized_vol, cov_annual, portfolio_stats
from sotf.optimize import equal_weight, max_diversification, apply_satellite_cap
from sotf.report import summary_table, weights_table

def main():
    root = Path(__file__).resolve().parents[1]
    universe = load_universe(root / "data" / "universe.csv")
    tickers = universe["ticker"].tolist()

    # Pick a window you like; you can make this CLI args later.
    start = "2018-01-01"
    end = None  # yfinance uses "today" if None

    px = fetch_adj_close(tickers, start=start, end=end)
    rets = to_returns(px, method="log")

    mu = annualized_return(rets)
    vol = annualized_vol(rets)
    cov = cov_annual(rets)

    print("\n=== Universe summary ===")
    print(summary_table(mu, vol, universe).round(4).to_string())

    w_eq = equal_weight(list(mu.index))
    w_md = max_diversification(cov, bounds=(0.0, 0.35))

    # Enforce satellites cap even if you later add satellite names
    w_eq = apply_satellite_cap(w_eq, universe, satellite_cap=0.50)
    w_md = apply_satellite_cap(w_md, universe, satellite_cap=0.50)

    print("\n=== Equal-weight portfolio ===")
    print(weights_table(w_eq, universe).round(4).to_string())
    print(portfolio_stats(w_eq, mu.loc[w_eq.index], cov.loc[w_eq.index, w_eq.index]))

    print("\n=== Max-diversification-ish portfolio ===")
    print(weights_table(w_md, universe).round(4).to_string())
    print(portfolio_stats(w_md, mu.loc[w_md.index], cov.loc[w_md.index, w_md.index]))

if __name__ == "__main__":
    main()
