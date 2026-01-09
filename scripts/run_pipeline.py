#!/usr/bin/env python3
"""
Main pipeline script for running portfolio analysis and backtesting
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotf.universe import load_universe
from sotf.fetch import fetch_adj_close, to_returns
from sotf.analytics import annualized_return, annualized_vol, cov_annual, corr
from sotf.optimize import equal_weight, max_diversification
from sotf.backtest import run_backtest, calculate_performance_metrics
from sotf.report import (
    save_backtest_results, save_weights_over_time, plot_equity_curve,
    plot_weights_allocation, plot_correlation_heatmap, generate_summary_stats
)

def main():
    parser = argparse.ArgumentParser(description="Run vocational training portfolio analysis")
    parser.add_argument("--universe", type=str, default="data/universe.csv",
                       help="Path to universe CSV file")
    parser.add_argument("--start", type=str, default=None,
                       help="Start date (YYYY-MM-DD), default: 3 years ago")
    parser.add_argument("--end", type=str, default=None,
                       help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--strategy", type=str, default="equal_weight",
                       choices=["equal_weight", "max_div"],
                       help="Portfolio strategy to use")
    parser.add_argument("--rebalance", type=str, default="M",
                       choices=["M", "Q"],
                       help="Rebalancing frequency (M=monthly, Q=quarterly)")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory for results")
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                       help="Initial capital for backtest")
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if args.end is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = args.end
    
    if args.start is None:
        start_dt = datetime.now() - timedelta(days=3*365)
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        start_date = args.start
    
    print("=" * 60)
    print("SURVIVAL OF THE FITTERS - Portfolio Analysis")
    print("=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: {args.strategy}")
    print(f"Rebalance: {args.rebalance}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load universe
    print("Loading universe...")
    universe_path = Path(args.universe)
    universe = load_universe(universe_path)
    tickers = universe["ticker"].tolist()
    print(f"Loaded {len(tickers)} tickers: {', '.join(tickers)}")
    print()
    
    # Fetch price data
    print("Fetching price data...")
    try:
        prices = fetch_adj_close(tickers, start_date, end_date)
        print(f"Fetched {len(prices)} days of data")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        print()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return 1
    
    # Calculate returns
    print("Calculating returns...")
    returns = to_returns(prices, method="log")
    print(f"Calculated returns for {len(returns)} periods")
    print()
    
    # Calculate analytics
    print("Computing analytics...")
    mu = annualized_return(returns)
    vol = annualized_vol(returns)
    cov_mat = cov_annual(returns)
    corr_mat = corr(returns)
    
    print("\nAnnualized Returns:")
    for ticker, ret in mu.items():
        print(f"  {ticker:10s}: {ret:7.2%}")
    
    print("\nAnnualized Volatility:")
    for ticker, v in vol.items():
        print(f"  {ticker:10s}: {v:7.2%}")
    print()
    
    # Define strategy function
    if args.strategy == "equal_weight":
        def strategy_func(rets_df):
            return equal_weight(list(rets_df.columns))
    else:  # max_div
        def strategy_func(rets_df):
            cov = rets_df.cov() * 252
            return max_diversification(cov)
    
    # Run backtest
    print("Running backtest...")
    results, weights_over_time = run_backtest(
        prices,
        strategy_func,
        initial_capital=args.initial_capital,
        rebalance_freq=args.rebalance,
        lookback_days=252
    )
    
    if len(results) == 0:
        print("Error: Backtest produced no results")
        return 1
    
    print(f"Backtest complete: {len(results)} days simulated")
    print()
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(results)
    
    print("Performance Metrics:")
    print(f"  Total Return:        {metrics['total_return']:7.2%}")
    print(f"  Annualized Return:   {metrics['annualized_return']:7.2%}")
    print(f"  Annualized Vol:      {metrics['annualized_volatility']:7.2%}")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:7.2f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:7.2%}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:7.2f}")
    print()
    
    # Get final weights
    if len(weights_over_time) > 0:
        final_weights = weights_over_time.iloc[-1]
        print("Final Portfolio Weights:")
        for ticker in tickers:
            if ticker in final_weights:
                print(f"  {ticker:10s}: {final_weights[ticker]:7.2%}")
        print()
    
    # Generate reports
    print("Generating reports...")
    
    # Save backtest results
    save_backtest_results(results, output_dir / "backtest_results.csv")
    
    # Save weights over time
    if len(weights_over_time) > 0:
        save_weights_over_time(weights_over_time, output_dir / "weights_over_time.csv")
    
    # Plot equity curve
    plot_equity_curve(results, output_dir / "equity_curve.png")
    
    # Plot final allocation
    if len(weights_over_time) > 0:
        plot_weights_allocation(final_weights, output_dir / "allocation.png")
    
    # Plot correlation heatmap
    plot_correlation_heatmap(corr_mat, output_dir / "correlation_heatmap.png")
    
    # Generate summary stats
    if len(weights_over_time) > 0:
        generate_summary_stats(results, final_weights, mu, cov_mat, output_dir / "summary.txt")
    
    print()
    print("=" * 60)
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
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
