from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_universe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ticker", "bucket"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universe missing columns: {missing}")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["bucket"] = df["bucket"].astype(str).str.strip().str.lower()
    return df
