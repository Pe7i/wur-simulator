import matplotlib.pyplot as plt
import pandas as pd


def _future_years(df: pd.DataFrame) -> pd.Series:
    return df["origin_year"].astype(int) + df["horizon_k"].astype(int)


def plot_score_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, uni_id: str):
    """Historie a predikce skóre pro danou instituci.

    Parameters
    ----------
    history_df : DataFrame with columns [uni_id, year, score]
    forecast_df : DataFrame with columns [uni_id, origin_year, horizon_k, y_hat, model, ...]
    uni_id : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    h = history_df.loc[history_df["uni_id"] == str(uni_id)]
    f = forecast_df.loc[forecast_df["uni_id"] == str(uni_id)]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if not h.empty:
        ax.plot(h["year"].astype(int), h["score"].astype(float), marker="o", linewidth=1.5, label="Historie (score)")

    if not f.empty:
        years = _future_years(f)
        yhat = f["y_hat"].astype(float)
        ax.plot(years, yhat, marker="o", linestyle="--", label="Predikce (y_hat)")

    ax.set_xlabel("Rok")
    ax.set_ylabel("Skóre (overall)")
    ax.set_title("Historie a predikce skóre")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rank_change(forecast_df: pd.DataFrame, uni_id: str):
    """Vizualizace změny (predikovaného) pořadí napříč budoucími roky.

    Rank se počítá v rámci každého cílového roku t+k přes všechny instituce,
    podle hodnoty y_hat (vyšší score = lepší, tj. rank 1 je nejlepší).
    Vykresluje se pouze křivka pro danou instituci.
    """
    if forecast_df.empty:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.text(0.5, 0.5, "Forecast data nejsou k dispozici", ha="center", va="center")
        ax.axis("off")
        return fig

    df = forecast_df.copy()
    df["target_year"] = df["origin_year"].astype(int) + df["horizon_k"].astype(int)

    def _rank_within(group):
        group = group.sort_values("y_hat", ascending=False)
        group["rank_pred"] = range(1, len(group) + 1)
        return group

    ranked = df.groupby(["target_year", "model"], group_keys=False).apply(_rank_within)

    me = ranked.loc[ranked["uni_id"] == str(uni_id)]

    fig, ax = plt.subplots(figsize=(9, 4))
    if me.empty:
        ax.text(0.5, 0.5, "Pro instituci chybí předpověď pro výpočet pořadí", ha="center", va="center")
        ax.axis("off")
        return fig

    for model, sub in me.groupby("model"):
        ax.plot(sub["target_year"].astype(int), sub["rank_pred"].astype(int), marker="o", label=str(model))

    ax.set_xlabel("Rok")
    ax.set_ylabel("Predikované pořadí (nižší = lepší)")
    ax.set_title("Změna predikovaného pořadí v čase")
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3)
    ax.legend(title="Model")
    fig.tight_layout()
    return fig