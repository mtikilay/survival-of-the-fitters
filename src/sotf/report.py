from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
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
