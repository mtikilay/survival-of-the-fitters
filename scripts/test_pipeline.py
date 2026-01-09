#!/usr/bin/env python3
"""
Test script with synthetic data to validate the pipeline works
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotf.universe import load_universe
from sotf.analytics import annualized_return, annualized_vol, cov_annual, corr
from sotf.optimize import equal_weight, max_diversification
from sotf.backtest import run_backtest, calculate_performance_metrics
from sotf.report import (
    save_backtest_results, save_weights_over_time, plot_equity_curve,
    plot_weights_allocation, plot_correlation_heatmap, generate_summary_stats
)

def generate_synthetic_prices(tickers, start_date, end_date, initial_price=100):
    """Generate synthetic price data for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    
    prices = {}
    for i, ticker in enumerate(tickers):
        # Different volatility and drift for each ticker
        vol = 0.15 + i * 0.03
        drift = 0.08 + i * 0.02
        
        # Generate random returns
        returns = np.random.normal(drift/252, vol/np.sqrt(252), len(dates))
        
        # Generate price series
        price_series = initial_price * np.exp(np.cumsum(returns))
        prices[ticker] = price_series
    
    return pd.DataFrame(prices, index=dates)

def main():
    print("=" * 60)
    print("SURVIVAL OF THE FITTERS - Test with Synthetic Data")
    print("=" * 60)
    print()
    
    # Load universe
    print("Loading universe...")
    universe = load_universe("data/universe.csv")
    tickers = universe["ticker"].tolist()
    print(f"Loaded {len(tickers)} tickers: {', '.join(tickers)}")
    print()
    
    # Generate synthetic data
    print("Generating synthetic price data...")
    start_date = "2021-01-01"
    end_date = "2023-12-31"
    prices = generate_synthetic_prices(tickers, start_date, end_date)
    print(f"Generated {len(prices)} days of data")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print()
    
    # Calculate returns
    print("Calculating returns...")
    returns = prices.pct_change().dropna()
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
    
    # Test both strategies
    for strategy_name in ["equal_weight", "max_div"]:
        print(f"\n{'=' * 60}")
        print(f"Testing {strategy_name.upper()} strategy")
        print('=' * 60)
        
        # Define strategy function
        if strategy_name == "equal_weight":
            def strategy_func(rets_df):
                return equal_weight(list(rets_df.columns))
        else:  # max_div
            def strategy_func(rets_df):
                cov = rets_df.cov() * 252
                return max_diversification(cov, bounds=(0.05, 0.30))
        
        # Run backtest
        print("Running backtest...")
        results, weights_over_time = run_backtest(
            prices,
            strategy_func,
            initial_capital=100000.0,
            rebalance_freq="Q",
            lookback_days=252
        )
        
        print(f"Backtest complete: {len(results)} days simulated")
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(results)
        
        print("\nPerformance Metrics:")
        print(f"  Total Return:        {metrics['total_return']:7.2%}")
        print(f"  Annualized Return:   {metrics['annualized_return']:7.2%}")
        print(f"  Annualized Vol:      {metrics['annualized_volatility']:7.2%}")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:7.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:7.2%}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:7.2f}")
        
        # Get final weights
        if len(weights_over_time) > 0:
            final_weights = weights_over_time.iloc[-1]
            print("\nFinal Portfolio Weights:")
            for ticker in tickers:
                if ticker in final_weights:
                    print(f"  {ticker:10s}: {final_weights[ticker]:7.2%}")
        
        # Create output directory
        output_dir = Path(f"output_{strategy_name}")
        output_dir.mkdir(exist_ok=True)
        
        # Generate reports
        print(f"\nGenerating reports to {output_dir}/...")
        save_backtest_results(results, output_dir / "backtest_results.csv")
        
        if len(weights_over_time) > 0:
            save_weights_over_time(weights_over_time, output_dir / "weights_over_time.csv")
            plot_weights_allocation(final_weights, output_dir / "allocation.png")
            generate_summary_stats(results, final_weights, mu, cov_mat, output_dir / "summary.txt")
        
        plot_equity_curve(results, output_dir / "equity_curve.png")
        plot_correlation_heatmap(corr_mat, output_dir / "correlation_heatmap.png")
    
    print("\n" + "=" * 60)
    print("Test complete! All outputs saved.")
    print("=" * 60)

if __name__ == "__main__":
    main()
