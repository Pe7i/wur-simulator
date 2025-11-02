from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd


# ----------------------------
# Pomocné funkce
# ----------------------------

def _slug(x: str) -> str:
    import re
    x = str(x).strip().lower()
    x = re.sub(r"[^0-9a-z]+", "-", x)
    x = re.sub(r"-+", "-", x).strip("-")
    return x or "unknown"


def _pick_ci(df: pd.DataFrame, *names: str) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


# ----------------------------
# Výpočet metrik z forecastu + historie
# ----------------------------

def _pair_with_history_and_aggregate(
    forecast_subset: pd.DataFrame,
    history: pd.DataFrame,
    k_internal: str = "horizon_k",
) -> pd.DataFrame:
    """
    Spojí forecast s historií podle (uni_id, target_year) a vrátí MAE/RMSE per K.
    Vrací DataFrame s kolonami: ["Horizont (K)", "MAE", "RMSE"] nebo prázdný DF.
    """
    if forecast_subset is None or forecast_subset.empty:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])
    if history is None or history.empty:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    fs = forecast_subset.copy()

    uid_col = _pick_ci(fs, "uni_id")
    oy_col  = _pick_ci(fs, "origin_year")
    k_col   = k_internal if k_internal in fs.columns else _pick_ci(fs, "horizon_k", "k", "step")
    yh_col  = _pick_ci(fs, "y_hat")

    if not all([uid_col, oy_col, k_col, yh_col]):
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    fs["__uid"] = fs[uid_col].astype(str)
    fs["__oy"]  = pd.to_numeric(fs[oy_col], errors="coerce").astype("Int64")
    fs["__k"]   = pd.to_numeric(fs[k_col],  errors="coerce").astype("Int64")
    fs["__yh"]  = pd.to_numeric(fs[yh_col], errors="coerce")

    fs["__ty"]  = (fs["__oy"] + fs["__k"]).astype("Int64")
    fs = fs.dropna(subset=["__uid", "__ty", "__yh", "__k"])

    h = history.copy()
    uid_h = _pick_ci(h, "uni_id")
    year_h = _pick_ci(h, "year")
    score_h = _pick_ci(h, "score")
    if not all([uid_h, year_h, score_h]):
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    h["__uid"] = h[uid_h].astype(str)
    h["__year"] = pd.to_numeric(h[year_h], errors="coerce").astype("Int64")
    h["__yt"] = pd.to_numeric(h[score_h], errors="coerce")

    paired = fs.merge(
        h[["__uid", "__year", "__yt"]],
        left_on=["__uid", "__ty"],
        right_on=["__uid", "__year"],
        how="inner",
    )
    if paired.empty:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    paired["__err"] = paired["__yh"] - paired["__yt"]
    paired = paired.dropna(subset=["__err", "__k"])

    if paired.empty:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    out = (
        paired.groupby("__k", as_index=False)
              .agg(
                  MAE=("__err", lambda s: s.abs().mean()),
                  RMSE=("__err", lambda s: (s.pow(2).mean()) ** 0.5),
              )
              .rename(columns={"__k": "Horizont (K)"})
              .sort_values("Horizont (K)")
    )
    return out


# ----------------------------
# Fallback: metriky z raw backtestu
# ----------------------------

def _from_backtest(
    backtest_path: str | Path,
    selected_uni_models: Dict[str, str],
    k_max: int,
) -> pd.DataFrame:
    """
    Robustní čtení backtest CSV/Parquet a výpočet pooled MAE/RMSE per K.
    - horizont: detekuje 'step' / 'horizon' / 'k' (case-insensitive)
    - uni_id: použije 'uni_id', jinak slug z 'university'
    - model: 'model' (case-insensitive), filtrovaný podle selected_uni_models (uid->model)
    - chyba: používá 'mae' (per-row absolute error); RMSE = sqrt(mean(ae^2))

    Vrací DataFrame ["Horizont (K)", "MAE", "RMSE"] nebo prázdný DF.
    """
    p = Path(backtest_path)
    if not p.exists():
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    if p.suffix.lower() == ".parquet":
        bt = pd.read_parquet(p)
    else:
        bt = pd.read_csv(p)

    if bt.empty:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    cols_l = {c.lower(): c for c in bt.columns}

    if "uni_id" in cols_l:
        bt["uni_id"] = bt[cols_l["uni_id"]].astype(str)
    elif "university" in cols_l:
        bt["uni_id"] = bt[cols_l["university"]].map(_slug)
    else:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    if "model" not in cols_l:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])
    bt["model"] = bt[cols_l["model"]].astype(str).str.strip().str.lower()

    k_src = None
    for cand in ("step", "horizon", "k"):
        if cand in cols_l:
            k_src = cols_l[cand]
            break
    if k_src is None:
        for cand in bt.columns:
            if cand.lower() in {"steps", "h", "k-step"}:
                k_src = cand
                break
    if k_src is None:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    bt["K"] = pd.to_numeric(bt[k_src], errors="coerce").astype("Int64")

    err_col = None
    for cand in ("mae", "ae"):
        if cand in cols_l:
            err_col = cols_l[cand]
            break
    if err_col is None:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])
    bt["ae"] = pd.to_numeric(bt[err_col], errors="coerce")

    if selected_uni_models:
        bt = bt[bt["uni_id"].astype(str).isin(selected_uni_models.keys())]
        if bt.empty:
            return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])
        bt = bt[bt.apply(lambda r: selected_uni_models.get(str(r["uni_id"])) == r["model"], axis=1)]
        if bt.empty:
            return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    bt = bt[bt["K"].notna() & (bt["K"] <= int(k_max))]
    if bt.empty:
        return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    out = (
        bt.groupby("K", as_index=False)
          .agg(
              MAE=("ae", "mean"),
              RMSE=("ae", lambda s: (s.pow(2).mean()) ** 0.5),
          )
          .rename(columns={"K": "Horizont (K)"})
          .sort_values("Horizont (K)")
    )
    return out


# ----------------------------
# Veřejná funkce
# ----------------------------

def compute_metrics_per_k(
    forecast_subset: Optional[pd.DataFrame],
    history: Optional[pd.DataFrame],
    selected_uni_models: Optional[Dict[str, str]] = None,
    k_max: int = 3,
    backtest_path: Optional[str | Path] = None,
    k_internal: str = "horizon_k",
) -> pd.DataFrame:
    """
    Vrátí MAE/RMSE per K:
      1) Primárně párováním forecastu s historií (když existuje ground-truth),
      2) Jinak fallback z raw backtestu (musí být cesta backtest_path).

    Parametry:
      - forecast_subset: forecast už ořezaný na konkrétní univerzitu/e a model/y
      - history: celá historie
      - selected_uni_models: map {uni_id -> zvolený/vítězný model} pro filtrování backtestu
      - k_max: horní mez horizontu
      - backtest_path: cesta k raw backtest CSV/Parquet
      - k_internal: název sloupce s K ve forecast_subset (pokud není standardní)
    """
    try:
        paired_agg = _pair_with_history_and_aggregate(forecast_subset, history, k_internal=k_internal)
        if not paired_agg.empty:
            return paired_agg
    except Exception:
        pass

    if backtest_path:
        try:
            return _from_backtest(backtest_path, selected_uni_models or {}, k_max)
        except Exception:
            return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])

    return pd.DataFrame(columns=["Horizont (K)", "MAE", "RMSE"])
