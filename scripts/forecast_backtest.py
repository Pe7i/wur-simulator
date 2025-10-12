import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

print(">>> forecast_backtest.py: start import", flush=True)


try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

DATA_PATH = Path("data/clean/wur_dataset.parquet")
OUT_DIR   = Path("outputs/backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "overall_score"
HORIZON = 3
MIN_TRAIN_YEARS = 6
MIN_SERIES_YEARS = MIN_TRAIN_YEARS + HORIZON

def get_university_series(df: pd.DataFrame, university: str) -> pd.DataFrame:
    uni = df[df["university"] == university].sort_values("year")
    out = uni[["year", TARGET]].dropna().copy()
    out["year"] = out["year"].astype(int)
    return out

def to_year_start_datetime(year_series: pd.Series) -> pd.Series:
    return pd.to_datetime(year_series.astype(int).astype(str) + "-01-01")

def naive_forecast(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last_year = int(series["year"].iloc[-1])
    last_value = float(series[TARGET].iloc[-1])
    years_future = np.arange(last_year + 1, last_year + horizon + 1, dtype=int)
    return pd.DataFrame({"year": years_future, "forecast": last_value})

def linear_forecast(series: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, LinearRegression]:
    X = series["year"].values.reshape(-1, 1)
    y = series[TARGET].values
    model = LinearRegression().fit(X, y)

    last_year = int(series["year"].iloc[-1])
    years_future = np.arange(last_year + 1, last_year + horizon + 1, dtype=int)
    y_future = model.predict(years_future.reshape(-1, 1))
    return pd.DataFrame({"year": years_future, "forecast": y_future}), model

def prophet_forecast(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet není k dispozici.")
    df_p = pd.DataFrame({
        "ds": to_year_start_datetime(series["year"]),
        "y":  series[TARGET].astype(float)
    })
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5
    )
    model.fit(df_p)
    future = model.make_future_dataframe(periods=horizon, freq="YE", include_history=False)
    fc = model.predict(future)[["ds", "yhat"]]
    fc["year"] = fc["ds"].dt.year.astype(int)
    fc = fc.rename(columns={"yhat": "forecast"})[["year", "forecast"]]
    return fc

def _as_1d_float_array(x):
    import numpy as np, pandas as pd
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
    else:
        arr = np.asarray([x])
    return arr.astype(float).reshape(-1)

def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    y_true = _as_1d_float_array(y_true)
    y_pred = _as_1d_float_array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse)}


@dataclass
class SplitResult:
    university: str
    origin_year: int
    horizon: int
    model: str
    mae: float
    rmse: float
    n_points: int
    step: int


def rolling_backtest_one(series: pd.DataFrame,
                         horizon: int = HORIZON,
                         min_train_years: int = MIN_TRAIN_YEARS,
                         use_prophet: bool = True) -> List[SplitResult]:
    if series is None or series.empty:
        return []
    if series["year"].nunique() < (min_train_years + 1):
        return []
    
    years = series["year"].astype(int).values
    results: List[SplitResult] = []

    first_origin = years[0] + min_train_years - 1
    last_origin  = years[-1] - horizon
    if first_origin > last_origin:
        return results

    for origin in range(first_origin, last_origin + 1):
        train = series[series["year"] <= origin]
        test  = series[(series["year"] > origin) & (series["year"] <= origin + horizon)]
        if test.empty:
            continue

        fc_naive = naive_forecast(train, horizon)
        fc_lin, _ = linear_forecast(train, horizon)
        if use_prophet and PROPHET_AVAILABLE and len(train) >= 3:
            fc_prophet = prophet_forecast(train, horizon)
        else:
            fc_prophet = None

        for step in range(1, horizon + 1):
            year_k = origin + step
            if year_k not in set(test["year"]):
                continue
            y_true = test.loc[test["year"] == year_k, TARGET]
            for name, fc in [("naive", fc_naive), ("linear", fc_lin)]:
                y_pred = fc.loc[fc["year"] == year_k, "forecast"]
                if not y_pred.empty:
                    mets = evaluate_forecast(y_true, y_pred)
                    results.append(SplitResult(
                        university=(str(series["university"].iloc[0]) 
                                if ("university" in series.columns and not series.empty) else ""),
                        origin_year=origin, horizon=horizon, model=name,
                        mae=mets["MAE"], rmse=mets["RMSE"], n_points=int(y_true.shape[0]), step=step
                    ))
            if fc_prophet is not None:
                y_pred = fc_prophet.loc[fc_prophet["year"] == year_k, "forecast"]
                if not y_pred.empty:
                    mets = evaluate_forecast(y_true, y_pred)
                    results.append(SplitResult(
                    university=(str(series["university"].iloc[0]) 
                            if ("university" in series.columns and not series.empty) else ""),
                        origin_year=origin, horizon=horizon, model="prophet",
                        mae=mets["MAE"], rmse=mets["RMSE"], n_points=int(y_true.shape[0]), step=step
                    ))
    return results

def aggregate_results(df_res: pd.DataFrame) -> pd.DataFrame:
    per_step = (df_res
                .groupby(["model", "step"])
                .agg(MAE=("mae", "mean"), RMSE=("rmse", "mean"), N=("n_points", "sum"))
                .reset_index())
    overall = (df_res
               .groupby(["model"])
               .agg(MAE=("mae", "mean"), RMSE=("rmse", "mean"), N=("n_points", "sum"))
               .reset_index()
               .assign(step="overall"))
    return pd.concat([per_step, overall], ignore_index=True)

def plot_mae_by_step(agg: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(8, 5))
    for model in agg["model"].unique():
        part = agg[(agg["model"] == model) & (agg["step"].isin([1,2,3]))]
        if part.empty: 
            continue
        plt.plot(part["step"], part["MAE"], marker="o", label=model)
    plt.xlabel("Relativní horizont (rok)")
    plt.ylabel("MAE")
    plt.title("Backtest: MAE podle horizontu")
    plt.grid(True)
    plt.xticks([1,2,3])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)

def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["university", "year"]).reset_index(drop=True)

    lengths = (df.dropna(subset=[TARGET])
             .groupby("university")["year"]
             .nunique()
             .reset_index(name="n_years_nonnull"))
    valid_unis = set(lengths[lengths["n_years_nonnull"] >= MIN_SERIES_YEARS]["university"])
    df_valid = df[df["university"].isin(valid_unis)].copy()

    all_results: List[SplitResult] = []

    for uni in sorted(valid_unis):
        series = get_university_series(df_valid, uni)
        series = series.copy()
        series["university"] = uni

        if series.empty or series["year"].nunique() < (MIN_TRAIN_YEARS + 1):
            print(f">>> {uni}: přeskočeno – málo nenull hodnot (n={len(series)})", flush=True)
            continue
        try:
            res = rolling_backtest_one(series, horizon=HORIZON, min_train_years=MIN_TRAIN_YEARS, use_prophet=True)
            all_results.extend(res)
            if all_results:
                pd.DataFrame([r.__dict__ for r in all_results]).to_csv(OUT_DIR / "backtest_raw_results.csv", index=False)
            print(f">>> backtest hotový: {uni}, zatím {len(all_results)} řádků", flush=True)
        except Exception as e:
            print(f">>> {uni}: ERROR {e}", flush=True)
        continue

    if not all_results:
        print("Nebyla nalezena žádná validní kombinace (zřejmě příliš krátké řady).")
        return

    df_res = pd.DataFrame([r.__dict__ for r in all_results])
    df_res.to_csv(OUT_DIR / "backtest_raw_results.csv", index=False)

    agg = aggregate_results(df_res)
    agg.to_csv(OUT_DIR / "backtest_aggregated.csv", index=False)

    plot_mae_by_step(agg, OUT_DIR / "backtest_mae_by_step.png")

    n_unis = len(valid_unis)
    n_rows = df_res.shape[0]
    print(f"Počet univerzit v backtestu: {n_unis}")
    print(f"Počet vyhodnocených bodů (model × origin × step): {n_rows}")
    print("Agregované metriky (viz CSV):")
    print(agg.sort_values(["step", "MAE"]))

if __name__ == "__main__":
    print(">>> forecast_backtest.py: entering main()", flush=True)
    main()
    print(">>> forecast_backtest.py: done", flush=True)