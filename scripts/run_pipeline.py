#!/usr/bin/env python3
"""
Main pipeline script for running portfolio analysis and backtesting
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
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
    run_started_at = datetime.now(timezone.utc)
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
    timestamp_dir = run_started_at.strftime("%Y%m%d-%H-%M-%S")
    output_root = Path(args.output)
    output_dir = output_root / timestamp_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        print(f"Fetched {len(prices)} days of data for {len(prices.columns)} ticker(s)")
        if len(prices.columns) < len(tickers):
            print(f"Successfully loaded tickers: {', '.join(prices.columns.tolist())}")
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
    
    if "warning" in metrics:
        print(f"WARNING: {metrics['warning']}")
        print()
    
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
