import os
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from src.wur.ui import sidebar_logo
from src.wur.metrics import compute_metrics_per_k
from src.wur.io import load_history, load_forecast, load_overall, load_universities

st.set_page_config(page_title="WUR – Validace", layout="wide")
st.title("Validace modelů – agregace přes originy a kroky")

# --- Data ---
hist = load_history()
fc   = load_forecast()
ovr  = load_overall()
unis = load_universities()

BACKTEST_PATH = os.environ.get("BACKTEST_RAW_PATH", "outputs/backtest/backtest_raw_results.csv")

# --- Sidebar ---
with st.sidebar:
    st.caption(
    "MAE/RMSE měří průměrnou odchylku predikovaného skóre od skutečnosti v bodech "
    "(stejná jednotka jako skóre). RMSE je citlivější na velké chyby. "
    "Níže uvádíme pro každý model a horizont (K) tři agregace: makro-průměr, medián a vážený průměr."
)
    sidebar_logo()

# --- Helper: načtení raw backtestu a základní normalizace ---

import re
def _slug(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^0-9a-z]+", "-", x)
    x = re.sub(r"-+", "-", x).strip("-")
    return x or "unknown"

def _load_backtest_table(path: str | os.PathLike) -> pd.DataFrame | None:
    p = os.fspath(path)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p) if str(p).lower().endswith(".csv") else pd.read_parquet(p)
    df["model"] = (
        df.get("model", "unknown")
          .astype(str).str.strip().str.lower()
          .str.replace(r"\s+", " ", regex=True)
    )
    if "step" in df.columns:
        df["K"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    elif "horizon" in df.columns:
        df["K"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    else:
        df["K"] = pd.NA
    df["ae"] = pd.to_numeric(df.get("mae"), errors="coerce")
    if "university" in df.columns:
        df["uni_id"] = df["university"].map(_slug)
    else:
        df["uni_id"] = pd.NA

    df = df.dropna(subset=["model", "K", "ae"])
    return df

# --- Helper: makro-průměr MAE/RMSE per model×K (forecast+history fallback) ---

def _pivot_metrics_with_fallback(forecast: pd.DataFrame,
                                 history: pd.DataFrame,
                                 backtest_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_mae = {}
    out_rmse = {}

    if forecast.empty:
        return (pd.DataFrame(), pd.DataFrame())

    models = sorted(forecast["model"].dropna().astype(str).unique().tolist())
    k_vals = sorted(pd.to_numeric(forecast["horizon_k"], errors="coerce").dropna().astype(int).unique().tolist())
    if not models or not k_vals:
        return (pd.DataFrame(), pd.DataFrame())

    for m in models:
        sub = forecast[forecast["model"] == m].copy()
        uids = sub["uni_id"].dropna().astype(str).unique().tolist()
        uid_to_model = {uid: m for uid in uids}

        agg = compute_metrics_per_k(
            forecast_subset=sub.assign(_k_num=pd.to_numeric(sub["horizon_k"], errors="coerce")),
            history=history,
            selected_uni_models=uid_to_model,
            k_max=max(k_vals),
            backtest_path=backtest_path,
            k_internal="_k_num",
        )
        if not agg.empty:
            out_mae[m]  = dict(zip(agg["Horizont (K)"].astype(int), agg["MAE"].astype(float)))
            out_rmse[m] = dict(zip(agg["Horizont (K)"].astype(int), agg["RMSE"].astype(float)))

    mae_rows, rmse_rows = [], []
    for m in sorted(out_mae.keys() | out_rmse.keys()):
        mae_row = {"model": m}
        rmse_row = {"model": m}
        for k in k_vals:
            mae_row[k]  = out_mae.get(m, {}).get(k, np.nan)
            rmse_row[k] = out_rmse.get(m, {}).get(k, np.nan)
        mae_rows.append(mae_row); rmse_rows.append(rmse_row)

    mae_pivot  = (pd.DataFrame(mae_rows).set_index("model").reindex(sorted(k_vals), axis=1).sort_index()
                  if mae_rows else pd.DataFrame())
    rmse_pivot = (pd.DataFrame(rmse_rows).set_index("model").reindex(sorted(k_vals), axis=1).sort_index()
                  if rmse_rows else pd.DataFrame())
    return mae_pivot, rmse_pivot

# --- Helper: Makro-průměr (přes všechny páry) zůstává stejný ---

def _macro_mean_mae(bt: pd.DataFrame) -> pd.DataFrame:
    return (bt.groupby(["model","K"], as_index=False)["ae"]
              .mean(numeric_only=True)
              .pivot(index="model", columns="K", values="ae")
              .sort_index())

def _macro_mean_rmse(bt: pd.DataFrame) -> pd.DataFrame:
    b2 = bt.assign(ae2=bt["ae"]**2)
    return (b2.groupby(["model","K"], as_index=False)["ae2"]
              .mean(numeric_only=True)
              .assign(rmse=lambda d: np.sqrt(d["ae2"]))
              .pivot(index="model", columns="K", values="rmse")
              .sort_index())

# --- Helper: Medián (přes univerzity): nejdřív per-uni metrika, pak median napříč uni_id ---

def _median_mae(bt: pd.DataFrame) -> pd.DataFrame:
    g = (bt.dropna(subset=["uni_id"])
           .groupby(["model","K","uni_id"], as_index=False)
           .agg(mae_u=("ae","mean")))
    if g.empty:
        return pd.DataFrame()
    med = (g.groupby(["model","K"], as_index=False)["mae_u"]
             .median(numeric_only=True)
             .pivot(index="model", columns="K", values="mae_u")
             .sort_index())
    return med

def _median_rmse(bt: pd.DataFrame) -> pd.DataFrame:
    g = (bt.dropna(subset=["uni_id"])
           .assign(ae2=lambda d: d["ae"]**2)
           .groupby(["model","K","uni_id"], as_index=False)
           .agg(mse_u=("ae2","mean")))
    if g.empty:
        return pd.DataFrame()
    g = g.assign(rmse_u=lambda d: np.sqrt(d["mse_u"]))
    med = (g.groupby(["model","K"], as_index=False)["rmse_u"]
             .median(numeric_only=True)
             .pivot(index="model", columns="K", values="rmse_u")
             .sort_index())
    return med

# --- Helper: Vážený průměr (přes univerzity): rovné váhy per univerzita ---
# (Liší se od makro-průměru přes páry, kde univerzity s více páry dominují.)
def _weighted_mean_mae(bt: pd.DataFrame) -> pd.DataFrame:
    g = (bt.dropna(subset=["uni_id"])
           .groupby(["model","K","uni_id"], as_index=False)
           .agg(mae_u=("ae","mean")))
    if g.empty:
        return pd.DataFrame()
    out = (g.groupby(["model","K"], as_index=False)["mae_u"]
             .mean(numeric_only=True)
             .pivot(index="model", columns="K", values="mae_u")
             .sort_index())
    return out

def _weighted_mean_rmse(bt: pd.DataFrame) -> pd.DataFrame:
    g = (bt.dropna(subset=["uni_id"])
           .assign(ae2=lambda d: d["ae"]**2)
           .groupby(["model","K","uni_id"], as_index=False)
           .agg(mse_u=("ae2","mean")))
    if g.empty:
        return pd.DataFrame()
    g = g.assign(rmse_u=lambda d: np.sqrt(d["mse_u"]))
    out = (g.groupby(["model","K"], as_index=False)["rmse_u"]
             .mean(numeric_only=True)
             .pivot(index="model", columns="K", values="rmse_u")
             .sort_index())
    return out

# --- Výpočet pivotů ---

bt = _load_backtest_table(BACKTEST_PATH)

if bt is not None and not bt.empty:
    mae_macro = _macro_mean_mae(bt)
    rmse_macro = _macro_mean_rmse(bt)
else:
    mae_macro, rmse_macro = _pivot_metrics_with_fallback(fc, hist, BACKTEST_PATH)

mae_median  = _median_mae(bt)   if (bt is not None and not bt.empty) else pd.DataFrame()
rmse_median = _median_rmse(bt)  if (bt is not None and not bt.empty) else pd.DataFrame()
mae_wavg    = _weighted_mean_mae(bt) if (bt is not None and not bt.empty) else pd.DataFrame()
rmse_wavg   = _weighted_mean_rmse(bt) if (bt is not None and not bt.empty) else pd.DataFrame()

colL, colR = st.columns(2)

# ---- MAE ----

with colL:
    st.subheader("MAE podle modelu a horizontu")
    if not mae_macro.empty:
        st.markdown("**Makro-průměr (přes páry)**")
        st.dataframe(mae_macro.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Stáhnout MAE – makro (CSV)", data=mae_macro.to_csv().encode("utf-8"),
                           file_name="mae_macro.csv", mime="text/csv")
    if not mae_median.empty:
        st.markdown("**Medián (přes univerzity)**")
        st.dataframe(mae_median.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Stáhnout MAE – medián (CSV)", data=mae_median.to_csv().encode("utf-8"),
                           file_name="mae_median.csv", mime="text/csv")
    if not mae_wavg.empty:
        st.markdown("**Průměr přes univerzity (rovné váhy)**")
        st.dataframe(mae_wavg.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Stáhnout MAE – vážený (CSV)", data=mae_wavg.to_csv().encode("utf-8"),
                           file_name="mae_weighted.csv", mime="text/csv")

# ---- RMSE ----

with colR:
    st.subheader("RMSE podle modelu a horizontu")
    if not rmse_macro.empty:
        st.markdown("**Makro-průměr (přes páry)**")
        st.dataframe(rmse_macro.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Stáhnout RMSE – makro (CSV)", data=rmse_macro.to_csv().encode("utf-8"),
                           file_name="rmse_macro.csv", mime="text/csv")
    if not rmse_median.empty:
        st.markdown("**Medián (přes univerzity)**")
        st.dataframe(rmse_median.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Stáhnout RMSE – medián (CSV)", data=rmse_median.to_csv().encode("utf-8"),
                           file_name="rmse_median.csv", mime="text/csv")
    if not rmse_wavg.empty:
        st.markdown("**Průměr přes univerzity (rovné váhy)**")
        st.dataframe(rmse_wavg.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Stáhnout RMSE – vážený (CSV)", data=rmse_wavg.to_csv().encode("utf-8"),
                           file_name="rmse_weighted.csv", mime="text/csv")

# ---- Diagnostika - počty párů N ----

if bt is not None and not bt.empty:
    st.markdown("---")
    st.caption("Počty párů (N) použité v jednotlivých buňkách (diagnostika):")
    n_pivot = (bt.groupby(["model","K"], as_index=False).size()
                 .pivot(index="model", columns="K", values="size")
                 .sort_index()
                 .fillna(0).astype(int))
    st.dataframe(n_pivot, use_container_width=True)
    st.download_button("Stáhnout N pivot (CSV)", data=n_pivot.to_csv().encode("utf-8"),
                       file_name="n_pivot.csv", mime="text/csv")
