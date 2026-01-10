from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import asciichartpy as asciichart

def save_backtest_results(results: pd.DataFrame, output_path: str | Path) -> None:
    """Save backtest results to CSV"""
    results.to_csv(output_path, index=True)
    print(f"Backtest results saved to {output_path}")

def save_weights_over_time(weights_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save portfolio weights over time to CSV"""
    weights_df.to_csv(output_path, index=True)
    print(f"Weights over time saved to {output_path}")

def plot_equity_curve(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot portfolio equity curve"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results.index, results["portfolio_value"], label="Portfolio Value", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Portfolio Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Equity curve saved to {output_path}")

def plot_weights_allocation(weights: pd.Series, output_path: str | Path) -> None:
    """Plot current portfolio allocation as pie chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(weights.values, labels=weights.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Portfolio Allocation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Allocation chart saved to {output_path}")

def plot_correlation_heatmap(corr: pd.DataFrame, output_path: str | Path) -> None:
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Add correlation values
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title("Return Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

def generate_summary_stats(results: pd.DataFrame, weights: pd.Series, mu: pd.Series, 
                          cov: pd.DataFrame, output_path: str | Path) -> None:
    """Generate and save summary statistics"""
    total_return = (results["portfolio_value"].iloc[-1] / results["portfolio_value"].iloc[0]) - 1
    
    # Calculate annualized return
    n_days = (results.index[-1] - results.index[0]).days
    ann_return = (1 + total_return) ** (365.25 / n_days) - 1
    
    # Calculate volatility from returns
    portfolio_returns = results["portfolio_value"].pct_change().dropna()
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    
    # Max drawdown
    cumulative = results["portfolio_value"]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    summary = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{ann_return:.2%}",
        "Annualized Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Start Date": str(results.index[0].date()),
        "End Date": str(results.index[-1].date()),
        "Initial Value": f"${results['portfolio_value'].iloc[0]:.2f}",
        "Final Value": f"${results['portfolio_value'].iloc[-1]:.2f}",
    }
    
    # Add current weights
    for ticker, weight in weights.items():
        summary[f"Weight_{ticker}"] = f"{weight:.2%}"
    
    # Save to text file
    with open(output_path, 'w') as f:
        f.write("Portfolio Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key:30s}: {value}\n")
    
    print(f"Summary statistics saved to {output_path}")

def print_ascii_equity_curve(results: pd.DataFrame, metrics: dict, height: int = 20, width: int = 80) -> None:
    """
    Print an ASCII art equity curve chart to the console.
    
    Args:
        results: DataFrame with portfolio_value column and date index
        metrics: Dictionary of performance metrics including max_drawdown_date
        height: Height of the chart in lines (default: 20)
        width: Target width for chart display (default: 80)
    """
    if len(results) == 0:
        print("No data to chart")
        return
    
    # Prepare data
    values = results["portfolio_value"].values
    dates = results.index
    
    # Downsample if we have too many data points for the width
    if len(values) > width - 20:  # Leave room for axis labels
        # Sample evenly across the data
        indices = np.linspace(0, len(values) - 1, width - 20, dtype=int)
        values = values[indices]
        dates = dates[indices]
    
    # Find the max drawdown point
    max_dd_date = metrics.get('max_drawdown_date')
    dd_index = None
    if max_dd_date is not None:
        # Find closest date in our (potentially downsampled) data
        closest_idx = 0
        min_diff = abs((dates[0] - max_dd_date).total_seconds())
        for i, date in enumerate(dates):
            diff = abs((date - max_dd_date).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        dd_index = closest_idx
    
    # Create the chart
    config = {
        'height': height,
        'format': '{:10,.0f}'
    }
    
    chart = asciichart.plot(values.tolist(), config)
    
    # Split chart into lines for annotation
    chart_lines = chart.split('\n')
    
    # Print header
    print()
    print("=" * 80)
    print("EQUITY CURVE")
    print("=" * 80)
    print()
    
    # Print the chart
    for line in chart_lines:
        print(line)
    
    # Print X-axis (dates)
    print()
    
    # Create date labels - show start, middle, end, and drawdown point
    label_positions = []
    label_texts = []
    
    # Start date
    label_positions.append(0)
    label_texts.append(dates[0].strftime('%Y-%m-%d'))
    
    # Max drawdown date if available
    if dd_index is not None and dd_index not in label_positions:
        label_positions.append(dd_index)
        label_texts.append(f"{dates[dd_index].strftime('%Y-%m-%d')} (DD)")
    
    # Middle date
    mid_idx = len(dates) // 2
    if mid_idx not in label_positions:
        label_positions.append(mid_idx)
        label_texts.append(dates[mid_idx].strftime('%Y-%m-%d'))
    
    # End date
    if len(dates) - 1 not in label_positions:
        label_positions.append(len(dates) - 1)
        label_texts.append(dates[-1].strftime('%Y-%m-%d'))
    
    # Sort by position
    sorted_labels = sorted(zip(label_positions, label_texts))
    
    # Print date axis
    # Calculate spacing for labels
    axis_line = " " * 13  # Offset for y-axis labels
    for pos, text in sorted_labels:
        # Calculate position in the display
        # The chart has some offset from the y-axis labels
        display_pos = 13 + int((pos / len(dates)) * (len(chart_lines[0]) - 13))
        axis_line += " " * max(0, display_pos - len(axis_line))
        axis_line += "|"
    
    print(axis_line)
    
    # Print labels below markers
    label_line = " " * 13
    for pos, text in sorted_labels:
        display_pos = 13 + int((pos / len(dates)) * (len(chart_lines[0]) - 13))
        # Center the label under the marker
        label_start = display_pos - len(text) // 2
        label_line += " " * max(0, label_start - len(label_line))
        label_line += text
    
    print(label_line)
    print()
    
    # Print legend
    print("LEGEND:")
    print(f"  Start Value:  ${values[0]:,.2f} on {dates[0].strftime('%Y-%m-%d')}")
    print(f"  End Value:    ${values[-1]:,.2f} on {dates[-1].strftime('%Y-%m-%d')}")
    
    if dd_index is not None and 'max_drawdown' in metrics:
        dd_value = values[dd_index]
        peak_value = metrics.get('peak_before_drawdown', values[0])
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%} on {dates[dd_index].strftime('%Y-%m-%d')}")
        print(f"                (Peak: ${peak_value:,.2f} → Trough: ${dd_value:,.2f})")
    
    print()
    print("=" * 80)
