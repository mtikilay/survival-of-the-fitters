from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
