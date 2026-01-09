from __future__ import annotations
from pathlib import Path
import pandas as pd

from sotf.universe import load_universe
from sotf.fetch import fetch_adj_close, to_returns, check_data_quality
from sotf.analytics import (
    annualized_return, annualized_vol, cov_annual, corr, 
    portfolio_stats, backtest, max_drawdown
)
from sotf.optimize import (
    equal_weight, max_diversification, min_variance, risk_parity, 
    apply_satellite_cap
)
from sotf.report import (
    summary_table, weights_table, save_weights, save_summary,
    plot_correlation_heatmap, plot_equity_curves, generate_html_report
)

def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    universe = load_universe(root / "data" / "universe.csv")
    tickers = universe["ticker"].tolist()

    # Pick a window you like; you can make this CLI args later.
    start = "2018-01-01"
    end = None  # yfinance uses "today" if None

    print("Fetching price data...")
    px = fetch_adj_close(tickers, start=start, end=end)
    
    print("Checking data quality...")
    px = check_data_quality(px, min_obs=252, max_missing_pct=0.3)
    
    print("Converting to returns...")
    rets = to_returns(px, method="log")

    mu = annualized_return(rets)
    vol = annualized_vol(rets)
    cov = cov_annual(rets)
    correlation = corr(rets)

    print("\n=== Universe summary ===")
    summary = summary_table(mu, vol, universe)
    print(summary.round(4).to_string())
    summary.to_csv(output_dir / "universe_summary.csv")

    # Calculate weights for different strategies
    strategies = {}
    
    print("\n=== Calculating portfolio weights ===")
    
    # Equal weight
    w_eq = equal_weight(list(mu.index))
    w_eq = apply_satellite_cap(w_eq, universe, satellite_cap=0.50)
    strategies["Equal Weight"] = w_eq
    
    # Max diversification
    w_md = max_diversification(cov, bounds=(0.0, 0.35))
    w_md = apply_satellite_cap(w_md, universe, satellite_cap=0.50)
    strategies["Max Diversification"] = w_md
    
    # Minimum variance
    w_mv = min_variance(cov, bounds=(0.0, 0.35))
    w_mv = apply_satellite_cap(w_mv, universe, satellite_cap=0.50)
    strategies["Min Variance"] = w_mv
    
    # Risk parity
    w_rp = risk_parity(cov, bounds=(0.0, 0.35))
    w_rp = apply_satellite_cap(w_rp, universe, satellite_cap=0.50)
    strategies["Risk Parity"] = w_rp

    # Print and save weights
    weights_dict = {}
    stats_dict = {}
    
    for name, weights in strategies.items():
        print(f"\n=== {name} Portfolio ===")
        wt = weights_table(weights, universe)
        print(wt.round(4).to_string())
        weights_dict[name] = wt
        
        # Save weights
        save_weights(weights, output_dir / f"weights_{name.lower().replace(' ', '_')}.csv", universe)
        
        # Calculate stats
        stats = portfolio_stats(weights, mu.loc[weights.index], cov.loc[weights.index, weights.index])
        print(f"Stats: {stats}")
        stats_dict[name] = stats

    # Save summary statistics
    summary_df = pd.DataFrame(stats_dict).T
    summary_df.to_csv(output_dir / "portfolio_stats.csv")
    print("\n=== Portfolio Statistics ===")
    print(summary_df.round(4).to_string())

    # Generate correlation heatmap
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(correlation, output_dir / "correlation_heatmap.png")

    # Run backtests
    print("\n=== Running Backtests ===")
    backtest_results = {}
    equity_curves = {}
    
    from sotf.optimize import equal_weight, max_diversification, min_variance, risk_parity
    
    backtest_configs = {
        "Equal Weight": (equal_weight, {}),
        "Max Diversification": (max_diversification, {"bounds": (0.0, 0.35)}),
        "Min Variance": (min_variance, {"bounds": (0.0, 0.35)}),
        "Risk Parity": (risk_parity, {"bounds": (0.0, 0.35)}),
    }
    
    # Wrapper to handle equal_weight which takes tickers instead of cov
    def weights_wrapper(func, kwargs):
        def wrapped(cov):
            if func == equal_weight:
                return func(list(cov.columns))
            else:
                return func(cov, **kwargs)
        return wrapped
    
    for name, (func, kwargs) in backtest_configs.items():
        print(f"Backtesting {name}...")
        result = backtest(rets, weights_wrapper(func, kwargs), rebal_freq="M")
        backtest_results[name] = result
        equity_curves[name] = result["equity_curve"]
        
        print(f"  CAGR: {result['cagr']:.2%}")
        print(f"  Vol: {result['vol']:.2%}")
        print(f"  Sharpe: {result['sharpe']:.2f}")
        print(f"  Max DD: {result['max_dd']:.2%}")
        print(f"  Avg Turnover: {result['avg_turnover']:.2%}")

    # Save backtest results
    bt_stats = {name: {
        "cagr": res["cagr"],
        "vol": res["vol"],
        "sharpe": res["sharpe"],
        "max_dd": res["max_dd"],
        "avg_turnover": res["avg_turnover"]
    } for name, res in backtest_results.items()}
    
    bt_df = pd.DataFrame(bt_stats).T
    bt_df.to_csv(output_dir / "backtest_stats.csv")

    # Plot equity curves
    print("\nGenerating equity curve plots...")
    plot_equity_curves(equity_curves, output_dir / "equity_curves.png")

    # Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(
        output_dir,
        universe,
        summary,
        weights_dict,
        stats_dict,
        correlation
    )

    print(f"\n✓ All outputs saved to {output_dir}/")
    print("  - universe_summary.csv")
    print("  - weights_*.csv (one per strategy)")
    print("  - portfolio_stats.csv")
    print("  - backtest_stats.csv")
    print("  - correlation_heatmap.png")
    print("  - equity_curves.png")
    print("  - report.html")

if __name__ == "__main__":
    main()
