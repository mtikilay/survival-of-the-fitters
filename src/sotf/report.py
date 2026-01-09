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
import seaborn as sns
from pathlib import Path
from typing import Optional

def summary_table(mu, vol, universe: pd.DataFrame) -> pd.DataFrame:
    u = universe.set_index("ticker")
    out = pd.DataFrame({"ann_return": mu, "ann_vol": vol}).join(u, how="left")
    return out.sort_values(["bucket", "ann_vol"], ascending=[True, True])

def weights_table(weights: pd.Series, universe: pd.DataFrame) -> pd.DataFrame:
    u = universe.set_index("ticker")
    out = pd.DataFrame({"weight": weights}).join(u, how="left")
    out["weight"] = out["weight"].astype(float)
    return out.sort_values("weight", ascending=False)


def save_weights(weights: pd.Series, filepath: Path, universe: Optional[pd.DataFrame] = None):
    """Save portfolio weights to CSV."""
    if universe is not None:
        df = weights_table(weights, universe)
    else:
        df = pd.DataFrame({"weight": weights})
    df.to_csv(filepath)


def save_summary(stats_dict: dict, filepath: Path):
    """Save portfolio summary statistics to CSV."""
    df = pd.DataFrame([stats_dict])
    df.to_csv(filepath, index=False)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, filepath: Path):
    """Generate and save correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Asset Correlation Matrix")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_equity_curves(curves_dict: dict, filepath: Path):
    """Plot equity curves for multiple strategies."""
    plt.figure(figsize=(12, 6))
    
    for name, curve in curves_dict.items():
        if len(curve) > 0:
            plt.plot(curve.index, curve.values, label=name, linewidth=2)
    
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Backtest - Equity Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def generate_html_report(
    output_dir: Path,
    universe: pd.DataFrame,
    summary: pd.DataFrame,
    weights_dict: dict,
    stats_dict: dict,
    corr_matrix: Optional[pd.DataFrame] = None,
):
    """Generate simple HTML report."""
    html = ["<!DOCTYPE html>", "<html>", "<head>", 
            "<title>Vocational Training Portfolio Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "img { max-width: 100%; height: auto; margin: 20px 0; }",
            "h2 { color: #333; margin-top: 30px; }",
            "</style>",
            "</head>", "<body>",
            "<h1>Vocational Training Portfolio Analysis</h1>"]
    
    # Universe summary
    html.append("<h2>Universe Summary</h2>")
    html.append(summary.to_html(index=False))
    
    # Portfolio weights
    for strategy, weights in weights_dict.items():
        html.append(f"<h2>{strategy} - Portfolio Weights</h2>")
        html.append(weights.to_html())
    
    # Portfolio statistics
    html.append("<h2>Portfolio Statistics</h2>")
    stats_df = pd.DataFrame(stats_dict).T
    html.append(stats_df.to_html())
    
    # Correlation heatmap
    if corr_matrix is not None and (output_dir / "correlation_heatmap.png").exists():
        html.append("<h2>Correlation Heatmap</h2>")
        html.append('<img src="correlation_heatmap.png" alt="Correlation Heatmap">')
    
    # Equity curves
    if (output_dir / "equity_curves.png").exists():
        html.append("<h2>Backtest Results - Equity Curves</h2>")
        html.append('<img src="equity_curves.png" alt="Equity Curves">')
    
    html.extend(["</body>", "</html>"])
    
    with open(output_dir / "report.html", "w") as f:
        f.write("\n".join(html))
