import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).describe().T

def plot_university_trend(df: pd.DataFrame, university: str, metric: str = "overall_score") -> None:
    d = df[df["university"] == university].sort_values("year")
    if d.empty:
        print("Nenalezeny záznamy pro:", university); return
    plt.figure()
    plt.plot(d["year"], d[metric], marker="o")
    plt.title(f"{university} – trend: {metric}")
    plt.xlabel("Rok"); plt.ylabel(metric); plt.grid(True); plt.show()

def country_trends(df: pd.DataFrame, top_n: int = 5) -> None:
    counts = df.groupby("country")["university"].nunique().sort_values(ascending=False)
    top = counts.head(top_n).index.tolist()
    agg = (df[df["country"].isin(top)]
           .groupby(["country","year"])["overall_score"].mean().reset_index())
    plt.figure()
    for c in top:
        d = agg[agg["country"] == c].sort_values("year")
        plt.plot(d["year"], d["overall_score"], marker="o", label=c)
    plt.title("Průměrné celkové skóre – top země"); plt.xlabel("Rok"); plt.ylabel("overall_score")
    plt.grid(True); plt.legend(); plt.show()

def corr_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr(method="pearson")

def show_corr_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    cax = ax.imshow(corr.values, aspect='auto')
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)));  ax.set_yticklabels(corr.index)
    fig.colorbar(cax); ax.set_title("Korelační matice"); plt.tight_layout(); plt.show()

def corr_with_targets(df: pd.DataFrame, targets=("overall_score","rank")) -> dict[str, pd.Series]:
    feats = [c for c in df.columns if c not in ["year","university","country"]]
    feats = df[feats].select_dtypes(include=[np.number]).columns.tolist()
    out = {}
    for t in targets:
        if t in df.columns:
            out[t] = df[feats].corrwith(df[t]).sort_values(ascending=False)
    return out
