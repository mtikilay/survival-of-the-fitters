#!/usr/bin/env python3
"""
Test script to demonstrate ASCII equity curve chart
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotf.report import print_ascii_equity_curve

def generate_test_equity_curve():
    """Generate a sample equity curve with a drawdown"""
    # Create dates for 2 years
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    
    # Generate portfolio values with a trend and a significant drawdown
    np.random.seed(42)
    
    # Start at 100000
    initial_value = 100000
    values = [initial_value]
    
    # Create returns with a trend and one major drawdown period
    for i in range(1, len(dates)):
        # General upward trend
        daily_return = np.random.normal(0.0005, 0.01)  # Small positive drift
        
        # Create a drawdown around day 400-500
        if 400 <= i <= 500:
            daily_return -= 0.002  # Negative bias during drawdown
        
        new_value = values[-1] * (1 + daily_return)
        values.append(new_value)
    
    # Create DataFrame
    results = pd.DataFrame({
        'portfolio_value': values
    }, index=dates)
    
    # Calculate drawdown metrics
    cumulative = results['portfolio_value']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    max_dd_date = drawdown.idxmin()
    max_dd_value = cumulative.loc[max_dd_date]
    peak_before_dd = running_max.loc[max_dd_date]
    running_max_before = running_max[:max_dd_date]
    peak_date = running_max_before[running_max_before == peak_before_dd].index[-1]
    
    metrics = {
        'max_drawdown': max_drawdown,
        'max_drawdown_date': max_dd_date,
        'max_drawdown_value': max_dd_value,
        'peak_before_drawdown': peak_before_dd,
        'peak_date': peak_date,
    }
    
    return results, metrics

def main():
    print("=" * 80)
    print("ASCII EQUITY CURVE CHART TEST")
    print("=" * 80)
    print()
    print("Generating sample equity curve with drawdown...")
    print()
    
    results, metrics = generate_test_equity_curve()
    
    print(f"Generated {len(results)} days of data")
    print(f"Start value: ${results['portfolio_value'].iloc[0]:,.2f}")
    print(f"End value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    print()
    
    # Display the ASCII chart
    print_ascii_equity_curve(results, metrics)
    
    print()
    print("Testing with --no-chart flag simulation:")
    print("(Chart would be skipped)")
    print()
    
    # Test with different chart sizes
    print()
    print("=" * 80)
    print("Smaller chart (height=15):")
    print("=" * 80)
    print_ascii_equity_curve(results, metrics, height=15)

if __name__ == "__main__":
    main()
