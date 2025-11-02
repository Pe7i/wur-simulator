import json
from io import BytesIO
from pathlib import Path
import logging
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.wur.ui import sidebar_logo
from src.wur.metrics import compute_metrics_per_k
from src.wur.io import load_universities, load_history, load_forecast, load_overall
from src.wur import viz

# --- Logging & stránka ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("wur.app")
logger.info("Startuji Streamlit app…")

st.set_page_config(page_title="WUR – Predikce", layout="wide")
st.title("Predikce pro vybranou instituci")

# --- Data ---
unis = load_universities()
hist = load_history()
fc = load_forecast()
ovr = load_overall()

# --- Helper: vítězný model pro univerzitu (tichý výběr, bez UI) ---
def _pick_winner_model(uni_id: str, fc: pd.DataFrame, ovr: pd.DataFrame) -> str:
    try:
        win = ovr[(ovr["uni_id"] == str(uni_id)) & (ovr.get("winner_flag", False) == True)]
        if not win.empty:
            return str(win["model"].iloc[0])
    except Exception:
        pass
    subset = fc.loc[fc["uni_id"] == str(uni_id), "model"].dropna()
    if not subset.empty:
        return str(subset.value_counts().idxmax())
    all_models = fc["model"].dropna().unique().tolist()
    return (all_models[0] if all_models else "linear")

# --- Sidebar ---
with st.sidebar:
    st.header("Možnosti")

    uni_name = st.selectbox("Vybraná univerzita", options=unis["name"].sort_values().tolist())
    uni_id = unis.loc[unis["name"] == uni_name, "uni_id"].iloc[0]

    model_choice = _pick_winner_model(uni_id=str(uni_id), fc=fc, ovr=ovr)

    max_k_global = int(pd.to_numeric(fc["horizon_k"], errors="coerce").max()) if not fc.empty else 3
    k_slider = st.slider(
        "Max. horizont (roky)",
        min_value=1,
        max_value=max(3, int(max_k_global)),
        value=min(3, int(max_k_global)),
        step=1
    )

    # ---- Slider "Začátek období (rok)" ----
    h_uni_all = hist.loc[hist["uni_id"] == str(uni_id)].copy()
    h_uni_all["year_num"] = pd.to_numeric(h_uni_all["year"], errors="coerce").astype("Int64")
    hist_min_year = int(h_uni_all["year_num"].min()) if not h_uni_all.empty and h_uni_all["year_num"].notna().any() else 2016
    hist_max_year = int(h_uni_all["year_num"].max()) if not h_uni_all.empty and h_uni_all["year_num"].notna().any() else 2016

    fc_uni_model = fc.loc[
        (fc["uni_id"] == str(uni_id)) &
        (fc["model"] == model_choice) &
        (pd.to_numeric(fc["horizon_k"], errors="coerce").astype("Int64") <= int(k_slider))
    ].copy()

    if not fc_uni_model.empty:
        origin_num = pd.to_numeric(fc_uni_model["origin_year"], errors="coerce").astype("Int64")
        k_num = pd.to_numeric(fc_uni_model["horizon_k"], errors="coerce").astype("Int64")
        target_year = (origin_num + k_num).astype("Int64")
        fc_min_year = int(target_year.min()) if target_year.notna().any() else None
        fc_max_year = int(target_year.max()) if target_year.notna().any() else None
    else:
        fc_min_year = None
        fc_max_year = None

    min_year_all = min([y for y in [hist_min_year, fc_min_year] if y is not None], default=2016)
    max_year_all = max([y for y in [hist_max_year, fc_max_year] if y is not None], default=2016)

    default_start = max(min_year_all, min(2016, max_year_all))

    start_year = st.slider(
        "Začátek období (rok)",
        min_value=int(min_year_all),
        max_value=int(max_year_all),
        value=int(default_start),
        step=1
    )

    sidebar_logo()

# --- Filtrace dat pro vybranou instituci/model + start_year ---
h_uni = h_uni_all.loc[h_uni_all["year_num"] >= start_year].copy()

last_hist_year = int(h_uni_all["year_num"].max()) if not h_uni_all.empty else start_year
min_pred_year = max(start_year, last_hist_year + 1)

# --- Příprava forecastu pro graf/tabulku – jen budoucí target roky a s y_hat ---
fc_win = fc.loc[
    (fc["uni_id"] == str(uni_id)) &
    (pd.to_numeric(fc["horizon_k"], errors="coerce").astype("Int64") <= int(k_slider))
].copy()

def _slice_for_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
    tmp = df[df["model"] == model].copy()
    if tmp.empty:
        return tmp
    tmp["origin_num"] = pd.to_numeric(tmp["origin_year"], errors="coerce").astype("Int64")
    tmp["k_num"] = pd.to_numeric(tmp["horizon_k"], errors="coerce").astype("Int64")
    tmp["target_year"] = (tmp["origin_num"] + tmp["k_num"]).astype("Int64")
    tmp = tmp.dropna(subset=["y_hat"])
    tmp = tmp.loc[tmp["target_year"] >= int(min_pred_year)]
    return tmp

fc_uni = _slice_for_model(fc_win, model_choice)
final_model = model_choice

if fc_uni.empty:
    order = ["naive", "linear", "prophet"]
    avail = [m for m in order if m in fc_win["model"].unique().tolist()]
    for m in avail:
        cand = _slice_for_model(fc_win, m)
        if not cand.empty:
            fc_uni = cand
            final_model = m
            break
    if fc_uni.empty and not fc_win.empty:
        any_m = fc_win["model"].iloc[0]
        fc_uni = _slice_for_model(fc_win, any_m)
        final_model = any_m

# --- Layout ---
col_left, col_right = st.columns((2, 1))

with col_left:
    st.subheader("Predikce skóre")
    st.caption(f"{uni_name} – použitý model: {final_model}")
    if h_uni.empty and fc_uni.empty:
        st.warning("Pro vybranou instituci a nastavené období nejsou k dispozici data.")
    else:
        hist_for_plot = h_uni.rename(columns={"year": "year", "score": "score"})
        fc_for_plot = fc_uni[["uni_id", "origin_year", "horizon_k", "y_hat", "model", "mae_k", "rmse_k"]].copy() if not fc_uni.empty else fc_uni

        fig = viz.plot_score_forecast(hist_for_plot, fc_for_plot, str(uni_id))
        st.pyplot(fig)

        png_buf = BytesIO()
        fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button(
            label="Stáhnout graf (PNG)",
            data=png_buf.getvalue(),
            file_name=f"{uni_name}_forecast_{final_model}.png",
            mime="image/png",
        )

with col_right:
    st.subheader("Přesnost a výběr modelu")
    if fc_uni.empty:
        st.info("Pro tento model a období nejsou k dispozici validované předpovědi.")
    else:
        cols = ["origin_year", "horizon_k", "target_year", "y_hat"]
        show_cols = [c for c in cols if c in fc_uni.columns]
        st.dataframe(
            fc_uni.sort_values(["origin_year", "horizon_k"])[show_cols],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            label="Stáhnout tabulku (CSV)",
            data=fc_uni[show_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"{uni_name}_metrics_{final_model}.csv",
            mime="text/csv",
        )

    uid_to_model = {str(uni_id): final_model}

    agg_home = compute_metrics_per_k(
        forecast_subset=fc_uni.assign(_k_num=fc_uni["k_num"]),
        history=hist,
        selected_uni_models=uid_to_model,
        k_max=int(k_slider),
        backtest_path=os.environ.get("BACKTEST_RAW_PATH", "outputs/backtest/backtest_raw_results.csv"),
        k_internal="_k_num",
)

    if agg_home.empty:
        st.caption("")
    else:
        st.caption("Agregované metriky z validace:")
        st.dataframe(agg_home, hide_index=True, width="stretch")

# --- Predikce pořadí (historie + budoucnost) ---
st.markdown("---")
st.subheader("Predikce pořadí")

fc_all = fc.copy()
fc_all["origin_num"] = pd.to_numeric(fc_all["origin_year"], errors="coerce").astype("Int64")
fc_all["k_num"] = pd.to_numeric(fc_all["horizon_k"], errors="coerce").astype("Int64")
fc_all["target_year"] = (fc_all["origin_num"] + fc_all["k_num"]).astype("Int64")

hist_uni_rank = h_uni.loc[h_uni["year_num"] >= start_year]

fc_slice = fc_all.loc[
    (fc_all["k_num"] <= int(k_slider)) &
    (fc_all["target_year"] >= int(min_pred_year))
].copy()

if not fc_slice.empty:
    fc_slice = fc_slice.sort_values(["origin_num", "k_num", "y_hat"], ascending=[True, True, False])
    fc_slice["rank_pred"] = fc_slice.groupby(["origin_num", "k_num"]).cumcount() + 1
    fc_rank_uni = fc_slice.loc[
        (fc_slice["uni_id"] == str(uni_id)) & (fc_slice["model"] == final_model),
        ["target_year", "rank_pred"]
    ].sort_values("target_year")
else:
    fc_rank_uni = pd.DataFrame(columns=["target_year", "rank_pred"])

rank_fig, axr = plt.subplots(figsize=(8, 4))
has_any = False

if not hist_uni_rank.empty and "rank" in hist_uni_rank.columns:
    axr.plot(pd.to_numeric(hist_uni_rank["year"]), pd.to_numeric(hist_uni_rank["rank"]),
             marker="o", linewidth=1.2, label="Historie")
    has_any = True

if not fc_rank_uni.empty:
    axr.plot(fc_rank_uni["target_year"].astype(int), fc_rank_uni["rank_pred"].astype(int),
             linestyle="--", marker="o", label=f"Predikce ({final_model})")
    has_any = True

axr.set_xlabel("Rok")
axr.set_ylabel("Pořadí (1 = nejlepší)")
axr.set_title(f"{uni_name} – historie a predikce pořadí")
axr.grid(True, linewidth=0.3)
axr.invert_yaxis()
axr.legend(loc="best")
rank_fig.tight_layout()

if has_any:
    st.pyplot(rank_fig)
    rank_buf = BytesIO()
    rank_fig.savefig(rank_buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button(
        label="Stáhnout graf (PNG)",
        data=rank_buf.getvalue(),
        file_name=f"{uni_name}_rank_forecast_{final_model}.png",
        mime="image/png",
    )
else:
    st.caption("Pro zvolenou kombinaci zatím nemám data k vykreslení pořadí.")

# --- Manifest ---
st.markdown("---")
st.subheader("Verze dat / běhu")
m_path = Path("outputs/final/pipeline_manifest.json")
if m_path.exists():
    try:
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
        st.caption(f"Nalezen manifest: `{m_path}`")
        st.json(manifest)
        st.download_button(
            label="Stáhnout manifest JSON",
            data=m_path.read_bytes(),
            file_name=m_path.name,
            mime="application/json",
            type="secondary",
        )
    except Exception as e:
        st.error(f"Chyba při načítání manifestu `{m_path}`: {e}")
else:
    st.caption(f"Manifest `{m_path}` nenalezen.")
