from __future__ import annotations
import pandas as pd

def summary_table(mu, vol, universe: pd.DataFrame) -> pd.DataFrame:
    u = universe.set_index("ticker")
    out = pd.DataFrame({"ann_return": mu, "ann_vol": vol}).join(u, how="left")
    return out.sort_values(["bucket", "ann_vol"], ascending=[True, True])

def weights_table(weights: pd.Series, universe: pd.DataFrame) -> pd.DataFrame:
    u = universe.set_index("ticker")
    out = pd.DataFrame({"weight": weights}).join(u, how="left")
    out["weight"] = out["weight"].astype(float)
    return out.sort_values("weight", ascending=False)
