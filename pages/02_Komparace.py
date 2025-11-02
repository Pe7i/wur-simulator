import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import matplotlib.pyplot as plt

from src.wur.ui import sidebar_logo
from src.wur.metrics import compute_metrics_per_k
from src.wur.io import load_universities, load_history, load_forecast, load_overall

st.set_page_config(page_title="WUR – Srovnání", layout="wide")
st.title("Srovnání vybraných institucí")
st.text("Graf zobrazuje historii a predikci pro vybrané univerzity a jejich vítězné modely a to jen pro společné období, pro které mají univerzity data.")

# --- Data ---
unis = load_universities()
hist = load_history()
fc   = load_forecast()
ovr  = load_overall()

# --- Helper: vítězný model pro univerzitu (fallback = první dostupný ve forecastu) ---
def pick_winner_model(uni_id: str) -> str | None:
    uid = str(uni_id)

    if "winner_flag" in ovr.columns:
        w = ovr[(ovr["uni_id"] == uid) & (ovr["winner_flag"] == True)]
        if not w.empty:
            return str(w["model"].iloc[0])

    f_uni = fc[fc["uni_id"] == uid].copy()
    if not f_uni.empty:
        has_mae  = "mae_k"  in f_uni.columns and f_uni["mae_k"].notna().any()
        has_rmse = "rmse_k" in f_uni.columns and f_uni["rmse_k"].notna().any()

        if has_mae or has_rmse:
            agg = (
                f_uni.groupby("model", as_index=False)[["mae_k", "rmse_k"]]
                .mean(numeric_only=True)
            )
            if "mae_k" not in agg.columns:  agg["mae_k"]  = np.inf
            if "rmse_k" not in agg.columns: agg["rmse_k"] = np.inf
            agg["mae_k"]  = agg["mae_k"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(np.inf)
            agg["rmse_k"] = agg["rmse_k"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(np.inf)

            agg = agg.sort_values(["mae_k", "rmse_k"], na_position="last")
            if not agg.empty and np.isfinite(agg["mae_k"].iloc[0]):
                return str(agg["model"].iloc[0])

        pref  = ["naive", "linear", "prophet"]
        avail = f_uni["model"].dropna().unique().tolist()
        for m in pref:
            if m in avail:
                return m
        if avail:
            return str(avail[0])

    return None

hist = hist.copy()
hist["_year_num"] = pd.to_numeric(hist["year"], errors="coerce").astype("Int64")
fc = fc.copy()
fc["_origin_num"] = pd.to_numeric(fc["origin_year"], errors="coerce").astype("Int64")
fc["_k_num"] = pd.to_numeric(fc["horizon_k"], errors="coerce").astype("Int64")

default_names_all = [
    "University of Oxford",
    "Princeton University",
    "Cornell University",
]
default_names = [n for n in default_names_all if n in set(unis["name"])]

# --- Sidebar ---
with st.sidebar:
    st.header("Možnosti")
    selected = st.multiselect(
        "Vyber instituce",
        options=unis["name"].sort_values().tolist(),
        default=default_names if default_names else unis["name"].sort_values().head(3).tolist(),
    )

    max_k = int(fc["_k_num"].max()) if not fc.empty else 3
    k_slider = st.slider(
        "Max. horizont (roky)",
        min_value=1,
        max_value=max(3, max_k),
        value=min(3, max_k),
        step=1
    )

    sel_unis = unis[unis["name"].isin(selected)].copy()
    sel_ids = sel_unis["uni_id"].astype(str).tolist()

    hist_ranges = (
        hist[hist["uni_id"].isin(sel_ids)]
        .groupby("uni_id")["_year_num"]
        .agg(["min", "max"])
        .rename(columns={"min": "min_year", "max": "max_year"})
    )

    ids_without_hist = set(sel_ids) - set(hist_ranges.index.astype(str).tolist())
    if ids_without_hist:
        missing_names = sel_unis[sel_unis["uni_id"].astype(str).isin(ids_without_hist)]["name"].tolist()
        st.warning("Následující instituce nemají v datasetu historická data a budou vynechány z grafu: " + ", ".join(missing_names))

    if not hist_ranges.empty:
        common_start = int(hist_ranges["min_year"].max())
        common_end = int(hist_ranges["max_year"].min())
    else:
        common_start, common_end = None, None

    if not fc.empty:
        fc_target_year = (fc["_origin_num"] + fc["_k_num"]).max()
        if fc_target_year is not pd.NA and fc_target_year is not None:
            if common_end is None:
                common_end = int(fc_target_year)
            else:
                common_end = int(max(common_end, int(fc_target_year)))

    if common_start is None or common_end is None or common_start > common_end:
        st.info("Vybrané instituce nemají společné historické období. Uprav výběr nebo přidej instituce s historií.")
        hist_min_global = int(hist["_year_num"].min()) if not hist.empty else 2010
        hist_max_global = int(hist["_year_num"].max()) if not hist.empty else 2025
        start_year = st.slider(
            "Začátek období (rok)",
            min_value=hist_min_global,
            max_value=hist_max_global,
            value=hist_min_global,
            step=1
        )
    else:
        start_year = st.slider(
            "Začátek období (rok)",
            min_value=common_start,
            max_value=common_end,
            value=2016,
            step=1
        )

    sidebar_logo()

if not selected:
    st.info("Vyber prosím alespoň jednu instituci v levém panelu.")
    st.stop()

# --- Graf: overlay historie + predikce (vítězný model per univerzita), oříznuté od start_year ---
fig, ax = plt.subplots(figsize=(10, 5))

name_to_uid = dict(zip(unis["name"], unis["uni_id"]))
used_rows = []

for name in selected:
    uid = name_to_uid[name]
    h = hist.loc[(hist["uni_id"] == str(uid)) & (hist["_year_num"] >= start_year)]
    if not h.empty:
        ax.plot(h["year"].astype(int), h["score"].astype(float), marker="o", linewidth=1.2, label=f"{name} – historie")

    wmodel = pick_winner_model(uid)
    if wmodel is None:
        continue

    f_win = fc.loc[
        (fc["uni_id"] == str(uid)) &
        (fc["model"] == wmodel) &
        (fc["_k_num"] <= k_slider)
    ].copy()
    f_win = f_win.dropna(subset=["y_hat"])

    if f_win.empty:
        cand = fc.loc[(fc["uni_id"] == str(uid)) & (fc["_k_num"] <= k_slider)].copy()
        cand = cand.dropna(subset=["y_hat"])
        if not cand.empty:
            pref = ["naive", "linear", "prophet"]
            avail = [m for m in pref if m in cand["model"].unique().tolist()]
            use_model = avail[0] if avail else cand["model"].iloc[0]
            f_plot = cand[cand["model"] == use_model].copy()
            model_label = use_model
        else:
            f_plot = pd.DataFrame(columns=fc.columns)
            model_label = wmodel
    else:
        f_plot = f_win
        model_label = wmodel

    if not f_plot.empty:
        f_plot["target_year"] = (f_plot["_origin_num"] + f_plot["_k_num"]).astype("Int64")
        f_plot = f_plot[f_plot["target_year"] >= start_year]
        if not f_plot.empty:
            ax.plot(
                f_plot["target_year"].astype(int),
                f_plot["y_hat"].astype(float),
                linestyle="--",
                marker="o",
                label=f"{name} – {model_label}"
            )
            used_rows.append(f_plot.assign(__name=name, __model=model_label))



ax.set_xlabel("Rok")
ax.set_ylabel("Skóre")
ax.set_title("Overlay trajektorií – historie a predikce")
ax.grid(True, linewidth=0.3)
ax.legend(ncol=1)
fig.tight_layout()

st.pyplot(fig)

buf = BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
st.download_button("Stáhnout graf (PNG)", data=buf.getvalue(), file_name="comparison_overlay.png", mime="image/png")



# --- Agregovaná tabulka metrik (MAE/RMSE) pro vybrané instituce ---
st.subheader("Agregované metriky (MAE/RMSE) pro vybrané instituce")

if not used_rows:
    st.caption("Pro vybrané instituce a nastavené období nejsou dostupné předpovědi (nebo chybí model s daty).")
else:
    used = pd.concat(used_rows, ignore_index=True)

    name_to_uid = dict(zip(unis["name"], unis["uni_id"]))

    winners = {
        name: pick_winner_model(name_to_uid[name])
        for name in selected
        if name in name_to_uid
    }

    uid_to_model = {
        str(name_to_uid[name]): model
        for name, model in winners.items()
        if model is not None
    }

    agg = compute_metrics_per_k(
        forecast_subset=used,
        history=hist,
        selected_uni_models=uid_to_model,
        k_max=int(k_slider),
        backtest_path=os.environ.get("BACKTEST_RAW_PATH", "outputs/backtest/backtest_raw_results.csv"),
        k_internal="_k_num",
    )

    if agg.empty:
        st.caption("Per-horizontní metriky nejsou pro vybranou kombinaci dostupné.")
    else:
        st.dataframe(agg, hide_index=True, width="stretch")
        st.download_button(
            "Stáhnout metriky (CSV)",
            data=agg.to_csv(index=False).encode("utf-8"),
            file_name="comparison_metrics.csv",
            mime="text/csv",
        )

st.caption("Pozn.: Per-horizontní metriky nejsou pro vybranou kombinaci dostupné (forecast obsahuje jen budoucí origin bez ground truth).")
