from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Callable

def run_backtest(
    prices: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float = 100000.0,
    rebalance_freq: str = "M",  # M=monthly, Q=quarterly
    lookback_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a backtest with periodic rebalancing
    
    Args:
        prices: DataFrame of adjusted close prices (dates x tickers)
        strategy_func: Function that takes returns DataFrame and returns weights Series
        initial_capital: Starting portfolio value
        rebalance_freq: Rebalancing frequency ('M' or 'Q')
        lookback_days: Number of days to look back for calculating statistics
        
    Returns:
        results: DataFrame with portfolio value over time
        weights_over_time: DataFrame with portfolio weights at each rebalance
    """
    # Sort prices by date
    prices = prices.sort_index()
    
    # Forward fill missing prices to avoid artificial portfolio value drops
    # This assumes that if a price is missing, we use the last known price
    prices = prices.ffill()
    
    # Check for any remaining NaN values (e.g., at the start of series)
    initial_nans = prices.isna().sum()
    if initial_nans.any():
        print(f"\nWarning: Some tickers have missing data at the start of the period:")
        for ticker, count in initial_nans[initial_nans > 0].items():
            print(f"  {ticker}: {count} missing values")
        # Back-fill any remaining NaN values at the start
        prices = prices.bfill()
    
    
    # Validate we have enough data
    if len(prices) < lookback_days + 20:
        print(f"\nWarning: Only {len(prices)} days of data available, but need at least {lookback_days + 20} days")
        print(f"         (lookback_days={lookback_days} + 20 minimum periods)")
        print(f"         Backtest will not rebalance, portfolio stays at initial capital.")
    
    # Get rebalance dates
    if rebalance_freq == "M":
        rebalance_dates = prices.resample("ME").last().index
    elif rebalance_freq == "Q":
        rebalance_dates = prices.resample("QE").last().index
    else:
        raise ValueError("rebalance_freq must be 'M' or 'Q'")
    
    # Filter rebalance dates that have sufficient lookback data
    valid_rebalance_dates = [d for i, d in enumerate(prices.index) if d in rebalance_dates and i >= lookback_days]
    
    if len(valid_rebalance_dates) == 0:
        print(f"\nWarning: No valid rebalance dates found.")
        print(f"         Data period: {prices.index[0]} to {prices.index[-1]} ({len(prices)} days)")
        print(f"         Rebalance frequency: {rebalance_freq}")
        print(f"         Lookback required: {lookback_days} days")
        print(f"         First possible rebalance would be after {prices.index[0] + pd.Timedelta(days=lookback_days)}")
        print(f"         Portfolio will remain at initial capital with no rebalancing.")
    
    # Initialize tracking variables
    results = []
    weights_history = []
    current_value = initial_capital
    current_shares = None
    
    for i, date in enumerate(prices.index):
        # Check if we need to rebalance
        if date in rebalance_dates and i >= lookback_days:
            # Get lookback window
            lookback_start = max(0, i - lookback_days)
            lookback_prices = prices.iloc[lookback_start:i+1]
            
            # Calculate returns for the lookback period
            lookback_returns = np.log(lookback_prices / lookback_prices.shift(1)).dropna()
            
            # Skip if not enough data
            if len(lookback_returns) < 20:
                continue
            
            # Get weights from strategy
            try:
                weights = strategy_func(lookback_returns)
                
                # Calculate number of shares to hold
                current_prices = prices.loc[date]
                total_value = current_value
                current_shares = {}
                
                for ticker in weights.index:
                    if ticker in current_prices.index and pd.notna(current_prices[ticker]):
                        allocation = total_value * weights[ticker]
                        shares = allocation / current_prices[ticker]
                        current_shares[ticker] = shares
                
                # Record weights
                weights_history.append({
                    "date": date,
                    **{ticker: weight for ticker, weight in weights.items()}
                })
            except Exception as e:
                # If strategy fails, keep current allocation
                print(f"Warning: Strategy failed on {date}: {e}")
                if current_shares is None:
                    continue
        
        # Calculate portfolio value
        if current_shares is not None:
            portfolio_value = 0.0
            current_prices = prices.loc[date]
            
            for ticker, shares in current_shares.items():
                if ticker in current_prices.index and pd.notna(current_prices[ticker]):
                    portfolio_value += shares * current_prices[ticker]
            
            current_value = portfolio_value
        else:
            portfolio_value = current_value
        
        # Record result
        results.append({
            "date": date,
            "portfolio_value": portfolio_value
        })
    
    # Convert to DataFrames
    results_df = pd.DataFrame(results).set_index("date")
    
    if weights_history:
        weights_df = pd.DataFrame(weights_history).set_index("date")
    else:
        weights_df = pd.DataFrame()
    
    return results_df, weights_df

def calculate_performance_metrics(results: pd.DataFrame) -> dict:
    """Calculate portfolio performance metrics"""
    if len(results) < 2:
        return {}
    
    # Check if portfolio actually changed (i.e., was rebalanced at least once)
    portfolio_values = results["portfolio_value"]
    if portfolio_values.std() == 0:
        # Portfolio never changed - stayed at initial capital
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": np.nan,
            "max_drawdown": 0.0,
            "calmar_ratio": np.nan,
            "start_date": results.index[0],
            "end_date": results.index[-1],
            "num_days": (results.index[-1] - results.index[0]).days,
            "warning": "Portfolio was never rebalanced - insufficient data or no valid rebalance dates"
        }
    
    total_return = (results["portfolio_value"].iloc[-1] / results["portfolio_value"].iloc[0]) - 1
    
    # Calculate annualized return
    n_days = (results.index[-1] - results.index[0]).days
    if n_days > 0:
        ann_return = (1 + total_return) ** (365.25 / n_days) - 1
    else:
        ann_return = 0.0
    
    # Calculate volatility
    returns = results["portfolio_value"].pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    
    # Max drawdown
    cumulative = results["portfolio_value"]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio (ann return / max drawdown)
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "start_date": results.index[0],
        "end_date": results.index[-1],
        "num_days": n_days,
    }
