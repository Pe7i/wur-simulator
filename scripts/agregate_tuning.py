import pandas as pd, numpy as np
from pathlib import Path

RAW = Path("outputs/tuning/prophet/tuning_raw_results.csv")
OUT = Path("outputs/tuning/prophet")
assert RAW.exists(), f"Nevidím {RAW}"
df = pd.read_csv(RAW)

df = df.rename(columns={c: c.strip() for c in df.columns})
lc_map = {c.lower(): c for c in df.columns}
for need in ("mae","rmse","step","params_id"):
    if need not in lc_map:
        raise SystemExit(f"Chybí sloupec: {need} (k dispozici: {list(df.columns)})")

df["MAE"]  = pd.to_numeric(df[lc_map["mae"]],  errors="coerce")
df["RMSE"] = pd.to_numeric(df[lc_map["rmse"]], errors="coerce")
df["step"] = pd.to_numeric(df[lc_map["step"]], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["MAE","RMSE","step"])

steps = (df.groupby(["params_id","step"])
           .agg(MAE=("MAE","mean"), RMSE=("RMSE","mean"), N=("MAE","size"))
           .reset_index())
overall = (df.groupby(["params_id"])
             .agg(MAE=("MAE","mean"), RMSE=("RMSE","mean"), N=("MAE","size"))
             .reset_index().assign(step="overall"))
agg = pd.concat([steps, overall], ignore_index=True)

OUT.mkdir(parents=True, exist_ok=True)
agg.to_csv(OUT / "tuning_aggregated.csv", index=False)

best_by_step = (agg[agg["step"].isin([1,2,3])]
                .sort_values(["step","MAE"])
                .groupby("step").head(5))
best_by_step.to_csv(OUT / "best_params_by_step.csv", index=False)

print("Uloženo:")
print(" - outputs/tuning/prophet/tuning_aggregated.csv")
print(" - outputs/tuning/prophet/best_params_by_step.csv")

print("\nTOP podle kroku (MAE):")
print(best_by_step[["step","params_id","MAE","RMSE","N"]].to_string(index=False))
