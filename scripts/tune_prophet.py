import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

DATA_PATH = Path("data/clean/wur_dataset.parquet")
OUT_DIR   = Path("outputs/tuning/prophet")
TARGET = "overall_score"

HORIZON_DEFAULT = 3
MIN_TRAIN_YEARS_DEFAULT = 6

for name in ("cmdstanpy", "prophet", "prophet.models", "prophet.forecaster"):
    logging.getLogger(name).setLevel(logging.WARNING)

def log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)

def ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def _as_1d_float_array(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)): arr = x.to_numpy()
    elif isinstance(x, (list, tuple, np.ndarray)): arr = np.asarray(x)
    else: arr = np.asarray([x])
    return arr.astype(float).reshape(-1)

def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    y_true = _as_1d_float_array(y_true)
    y_pred = _as_1d_float_array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse)}

def to_year_start_datetime(year_series: pd.Series) -> pd.Series:
    return pd.to_datetime(year_series.astype(int).astype(str) + "-01-01")

def get_university_series(df: pd.DataFrame, university: str) -> pd.DataFrame:
    uni = df[df["university"] == university].sort_values("year")
    out = uni[["year", TARGET]].dropna().copy()
    out["year"] = out["year"].astype(int)
    return out

def prophet_forecast(series: pd.DataFrame, horizon: int, params: Dict[str, Any]) -> pd.DataFrame:
    from prophet import Prophet

    df_p = pd.DataFrame({
        "ds": to_year_start_datetime(series["year"]),
        "y":  pd.to_numeric(series[TARGET], errors="coerce")
    }).dropna().drop_duplicates(subset=["ds"]).sort_values("ds")

    if df_p.empty or df_p["y"].nunique() <= 1:
        return pd.DataFrame(columns=["year", "forecast"])

    max_cp = max(0, len(df_p) - 2)
    n_cp = min(int(params.get("n_changepoints", 10)), max_cp)

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth=params.get("growth", "linear"),
        seasonality_mode=params.get("seasonality_mode", "additive"),
        changepoint_prior_scale=float(params.get("changepoint_prior_scale", 0.5)),
        changepoint_range=float(params.get("changepoint_range", 0.9)),
        n_changepoints=int(n_cp),
        mcmc_samples=0,
        uncertainty_samples=0
    )

    try:
        m.fit(df_p, algorithm="Newton")
    except Exception:
        try:
            m = Prophet(
                yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                growth="linear", seasonality_mode="additive",
                changepoint_prior_scale=min(0.3, float(params.get("changepoint_prior_scale", 0.5))),
                changepoint_range=min(0.9, float(params.get("changepoint_range", 0.9))),
                n_changepoints=min(5, n_cp), mcmc_samples=0, uncertainty_samples=0
            )
            m.fit(df_p, algorithm="Newton")
        except Exception:
            return pd.DataFrame(columns=["year", "forecast"])

    future = m.make_future_dataframe(periods=horizon, freq="YE", include_history=False)
    fc = m.predict(future)[["ds", "yhat"]]
    fc["year"] = fc["ds"].dt.year.astype(int)
    return fc.rename(columns={"yhat": "forecast"})[["year", "forecast"]]

@dataclass
class SplitResult:
    university: str
    origin_year: int
    step: int
    mae: float
    rmse: float
    params_id: str

def rolling_backtest_one(series: pd.DataFrame,
                         horizon: int,
                         min_train_years: int,
                         params: Dict[str, Any],
                         origin_stride: int = 1,
                         last_k_origins: int = 0) -> List[SplitResult]:
    if series is None or series.empty or series["year"].nunique() < (min_train_years + 1):
        return []
    years = series["year"].astype(int).values
    first_origin = int(years[0]) + min_train_years - 1
    last_origin  = int(years[-1]) - horizon
    if first_origin > last_origin:
        return []

    origins = list(range(first_origin, last_origin + 1, max(1, int(origin_stride))))
    if last_k_origins and last_k_origins > 0:
        origins = origins[-int(last_k_origins):]

    results: List[SplitResult] = []
    for origin in origins:
        train = series[series["year"] <= origin]
        test  = series[(series["year"] > origin) & (series["year"] <= origin + horizon)]
        if test.empty:
            continue

        fc_prophet = prophet_forecast(train, horizon, params)
        if fc_prophet is None or fc_prophet.empty:
            continue

        for step in range(1, horizon + 1):
            year_k = origin + step
            if year_k not in set(test["year"]):
                continue
            y_true = test.loc[test["year"] == year_k, TARGET]
            y_pred = fc_prophet.loc[fc_prophet["year"] == year_k, "forecast"]
            if len(y_pred) == 0:
                continue
            mets = evaluate_forecast(y_true, y_pred)
            results.append(SplitResult(
                university=str(series["university"].iloc[0]) if "university" in series.columns else "",
                origin_year=int(origin),
                step=int(step),
                mae=mets["MAE"],
                rmse=mets["RMSE"],
                params_id=params_to_id(params)
            ))
    return results

def default_grid(small: bool) -> Dict[str, List[Any]]:
    if small:
        return {"changepoint_prior_scale": [0.03, 0.05, 0.1, 0.3],
                "n_changepoints": [5, 10],
                "changepoint_range": [0.9, 0.95],
                "seasonality_mode": ["additive"]}
    else:
        return {"changepoint_prior_scale": [0.03, 0.05, 0.1, 0.3, 0.5, 1.0],
                "n_changepoints": [5, 10, 20],
                "changepoint_range": [0.8, 0.9, 0.95],
                "seasonality_mode": ["additive"]}

def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = []
    for values in product(*[grid[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos

def params_to_id(params: Dict[str, Any]) -> str:
    kv = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(kv)

def aggregate_case_insensitive(df_res: pd.DataFrame) -> pd.DataFrame:
    df = df_res.copy()
    df.columns = [c.strip() for c in df.columns]
    lc = {c.lower(): c for c in df.columns}

    need = ["params_id", "step", "mae", "rmse"]
    for k in need:
        if k not in lc:
            raise KeyError(f"Chybí sloupec '{k}' v: {list(df.columns)}")

    df["MAE"]  = pd.to_numeric(df[lc["mae"]],  errors="coerce")
    df["RMSE"] = pd.to_numeric(df[lc["rmse"]], errors="coerce")
    df["step"] = pd.to_numeric(df[lc["step"]], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["MAE","RMSE","step"])

    per_step = (df.groupby([lc["params_id"], "step"])
                  .agg(MAE=("MAE","mean"), RMSE=("RMSE","mean"), N=("MAE","size"))
                  .reset_index().rename(columns={lc["params_id"]: "params_id"}))
    overall = (df.groupby(lc["params_id"])
                 .agg(MAE=("MAE","mean"), RMSE=("RMSE","mean"), N=("MAE","size"))
                 .reset_index().rename(columns={lc["params_id"]: "params_id"})
                 .assign(step="overall"))
    return pd.concat([per_step, overall], ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    parser.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS_DEFAULT)
    parser.add_argument("--limit-unis", type=int, default=0, help="0 = všechny, jinak prvních N škol")
    parser.add_argument("--origin-stride", type=int, default=1, help="krok mezi originy (1=každý rok, 2=každý druhý...)")
    parser.add_argument("--last-k-origins", type=int, default=0, help="0=bez omezení; jinak posledních K originů")
    parser.add_argument("--small-grid", action="store_true", help="menší výchozí grid (rychlejší)")
    parser.add_argument("--grid-cps", type=str, default="", help="vlastní cps, např. 0.03,0.05,0.1,0.3")
    parser.add_argument("--aggregate", action="store_true", help="po doběhu vyrobit i agregace (jinak jen RAW)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out = args.out; out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data).sort_values(["university","year"]).reset_index(drop=True)
    lengths = (df.dropna(subset=[TARGET])
                 .groupby("university")["year"].nunique()
                 .reset_index(name="n_years_nonnull"))
    valid_unis = list(lengths[lengths["n_years_nonnull"] >= (args.min_train_years + args.horizon)]["university"])
    if args.limit_unis and args.limit_unis < len(valid_unis):
        valid_unis = valid_unis[:args.limit_unis]
    log(f"Školy v ladění: {len(valid_unis)}", True)

    grid_dict = default_grid(small=args.small_grid)
    if args.grid_cps:
        grid_dict = {"changepoint_prior_scale": [float(x) for x in args.grid_cps.split(",")],
                     "n_changepoints": [10],
                     "changepoint_range": [0.9],
                     "seasonality_mode": ["additive"]}
    grid = expand_grid(grid_dict)
    log(f"Počet kombinací: {len(grid)}; grid: {grid_dict}", True)

    all_rows: List[Dict[str, Any]] = []
    for gi, params in enumerate(grid, 1):
        pid = params_to_id(params)
        log(f"[{gi}/{len(grid)}] params={pid}", True)

        for ui, uni in enumerate(valid_unis, 1):
            series = get_university_series(df, uni)
            series["university"] = uni
            try:
                res = rolling_backtest_one(series,
                                           horizon=args.horizon,
                                           min_train_years=args.min_train_years,
                                           params=params,
                                           origin_stride=args.origin_stride,
                                           last_k_origins=args.last_k_origins)
            except Exception:
                res = []

            for r in res:
                all_rows.append({
                    "university": r.university,
                    "origin_year": r.origin_year,
                    "step": r.step,
                    "MAE": r.mae,
                    "RMSE": r.rmse,
                    "params_id": r.params_id
                })

            if all_rows and (ui % 10 == 0):
                pd.DataFrame(all_rows).to_csv(out / "tuning_raw_results.csv", index=False)

        if all_rows:
            pd.DataFrame(all_rows).to_csv(out / "tuning_raw_results.csv", index=False)

    if all_rows:
        df_res = pd.DataFrame(all_rows)
        df_res.to_csv(out / "tuning_raw_results.csv", index=False)
    else:
        print("Nebyla vygenerována žádná kombinace výsledků.")
        return

if __name__ == "__main__":
    ensure_out()
    main()