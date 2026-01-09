from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

from sotf.universe import load_universe
from sotf.fetch import fetch_adj_close, to_returns, check_data_quality
from sotf.analytics import annualized_return, annualized_vol, cov_annual, corr


def screen_satellite_candidates(
    core_universe: pd.DataFrame,
    candidates_path: Path,
    start: str = "2018-01-01",
    end: str = None,
) -> pd.DataFrame:
    """
    Screen satellite candidates based on correlation and diversification.
    
    Args:
        core_universe: DataFrame with core holdings
        candidates_path: Path to CSV with satellite candidates
        start: Start date for historical data
        end: End date for historical data
        
    Returns:
        DataFrame with screening results
    """
    # Load candidate list
    if not candidates_path.exists():
        print(f"Candidate file not found: {candidates_path}")
        print("Please create data/satellite_candidates.csv with columns: ticker,name,notes")
        return pd.DataFrame()
    
    candidates = pd.read_csv(candidates_path)
    if "ticker" not in candidates.columns:
        raise ValueError("Candidates CSV must have 'ticker' column")
    
    candidate_tickers = candidates["ticker"].str.strip().tolist()
    core_tickers = core_universe["ticker"].tolist()
    
    print(f"Screening {len(candidate_tickers)} satellite candidates...")
    print(f"Against {len(core_tickers)} core holdings...")
    
    # Fetch all data
    all_tickers = core_tickers + candidate_tickers
    
    try:
        px = fetch_adj_close(all_tickers, start=start, end=end)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    
    # Check data quality
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px = check_data_quality(px, min_obs=252, max_missing_pct=0.5)
    
    # Only keep candidates that passed quality check
    available_candidates = [t for t in candidate_tickers if t in px.columns]
    available_core = [t for t in core_tickers if t in px.columns]
    
    if not available_candidates:
        print("No candidates passed data quality checks!")
        return pd.DataFrame()
    
    print(f"{len(available_candidates)} candidates have sufficient data quality")
    
    # Calculate returns and statistics
    rets = to_returns(px, method="log")
    mu = annualized_return(rets)
    vol = annualized_vol(rets)
    correlation = corr(rets)
    
    # Calculate metrics for each candidate
    results = []
    
    for ticker in available_candidates:
        if ticker not in correlation.columns:
            continue
        
        # Get correlation with core holdings
        core_corrs = correlation.loc[ticker, available_core].dropna()
        
        if len(core_corrs) == 0:
            continue
        
        avg_core_corr = core_corrs.mean()
        max_core_corr = core_corrs.max()
        min_core_corr = core_corrs.min()
        
        # Diversification score (lower correlation = better diversification)
        div_score = 1.0 - avg_core_corr
        
        # Risk-adjusted return
        sharpe_proxy = mu[ticker] / vol[ticker] if vol[ticker] > 0 else np.nan
        
        # Combined score: diversification benefit + risk-adjusted return
        # Weight diversification more heavily
        combined_score = 0.6 * div_score + 0.4 * (sharpe_proxy / 2.0 if not np.isnan(sharpe_proxy) else 0)
        
        results.append({
            "ticker": ticker,
            "ann_return": mu[ticker],
            "ann_vol": vol[ticker],
            "sharpe_proxy": sharpe_proxy,
            "avg_core_corr": avg_core_corr,
            "max_core_corr": max_core_corr,
            "min_core_corr": min_core_corr,
            "diversification_score": div_score,
            "combined_score": combined_score,
        })
    
    results_df = pd.DataFrame(results)
    
    # Merge with candidate info
    results_df = results_df.merge(candidates, on="ticker", how="left")
    
    # Sort by combined score
    results_df = results_df.sort_values("combined_score", ascending=False)
    
    return results_df


def main():
    root = Path(__file__).resolve().parents[1]
    
    # Load core universe
    universe = load_universe(root / "data" / "universe.csv")
    core_universe = universe[universe["bucket"] == "core"].copy()
    
    # Path to candidates
    candidates_path = root / "data" / "satellite_candidates.csv"
    
    # Screen candidates
    results = screen_satellite_candidates(
        core_universe,
        candidates_path,
        start="2018-01-01",
        end=None
    )
    
    if len(results) == 0:
        return
    
    # Save results
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    results.to_csv(output_dir / "satellite_screening.csv", index=False)
    
    # Print top candidates
    print("\n=== Top Satellite Candidates ===")
    print("(Ranked by diversification + risk-adjusted return)")
    print()
    
    display_cols = [
        "ticker", "ann_return", "ann_vol", "sharpe_proxy",
        "avg_core_corr", "diversification_score", "combined_score"
    ]
    
    print(results[display_cols].head(20).round(4).to_string(index=False))
    
    print(f"\n✓ Full results saved to {output_dir}/satellite_screening.csv")
    print("\nInterpretation:")
    print("- Higher combined_score = better satellite candidate")
    print("- Lower avg_core_corr = better diversification benefit")
    print("- Higher sharpe_proxy = better risk-adjusted returns")
    print("\nNext steps:")
    print("1. Review top candidates and their business descriptions")
    print("2. Verify they align with vocational training mandate")
    print("3. Add approved tickers to data/universe.csv with bucket='satellite'")


if __name__ == "__main__":
    main()
