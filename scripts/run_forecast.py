import argparse, json, logging, os, platform, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

DATA_PATH = Path("data/clean/wur_dataset.parquet")
OUT_DIR   = Path("outputs/final")
TARGET = "overall_score"
HORIZON_DEFAULT = 3
MIN_TRAIN_YEARS_DEFAULT = 6
RANDOM_SEED = 7

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

np.random.seed(RANDOM_SEED)
for name in ("cmdstanpy", "prophet", "prophet.models", "prophet.forecaster"):
    logging.getLogger(name).setLevel(logging.WARNING)

def ensure_out(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_year_start_datetime(year_series: pd.Series) -> pd.Series:
    return pd.to_datetime(year_series.astype(int).astype(str) + "-01-01")

def get_university_series(df: pd.DataFrame, university: str) -> pd.DataFrame:
    uni = df[df["university"] == university].sort_values("year")
    out = uni[["year", TARGET]].dropna().copy()
    out["year"] = out["year"].astype(int)
    return out

def evaluate(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse)}

class NaiveModel:
    name = "naive"
    def fit(self, series: pd.DataFrame):
        self.last_year  = int(series["year"].iloc[-1])
        self.last_value = float(series[TARGET].iloc[-1])
        return self
    def predict(self, horizon: int) -> pd.DataFrame:
        years = np.arange(self.last_year + 1, self.last_year + horizon + 1, dtype=int)
        return pd.DataFrame({"year": years, "forecast": self.last_value})

class LinearTrendModel:
    name = "linear"
    def fit(self, series: pd.DataFrame):
        X = series["year"].values.reshape(-1,1)
        y = series[TARGET].values
        self.reg = LinearRegression().fit(X, y)
        self.last_year = int(series["year"].iloc[-1])
        return self
    def predict(self, horizon: int) -> pd.DataFrame:
        years = np.arange(self.last_year + 1, self.last_year + horizon + 1, dtype=int)
        yhat  = self.reg.predict(years.reshape(-1,1))
        return pd.DataFrame({"year": years, "forecast": yhat})

class ProphetTrendModel:
    name = "prophet"
    def __init__(self, cps=0.05, cpr=0.8, n_cp=5, growth="linear", seasonality_mode="additive"):
        self.cps, self.cpr, self.n_cp = float(cps), float(cpr), int(n_cp)
        self.growth, self.seasonality_mode = growth, seasonality_mode
        if not PROPHET_AVAILABLE:
            raise RuntimeError("Prophet není dostupný v prostředí.")
    def fit(self, series: pd.DataFrame):
        df_p = pd.DataFrame({
            "ds": to_year_start_datetime(series["year"]),
            "y":  pd.to_numeric(series[TARGET], errors="coerce")
        }).dropna().drop_duplicates(subset=["ds"]).sort_values("ds")
        if df_p.empty or df_p["y"].nunique() <= 1:
            self._degenerate = True
            self.last_year = int(series["year"].iloc[-1])
            self.last_value = float(series[TARGET].iloc[-1])
            return self
        self._degenerate = False
        max_cp = max(0, len(df_p) - 2)
        n_cp = min(self.n_cp, max_cp)
        self.m = Prophet(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
            growth=self.growth, seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.cps, changepoint_range=self.cpr,
            n_changepoints=n_cp, mcmc_samples=0, uncertainty_samples=0
        )
        self.m.fit(df_p, algorithm="Newton")
        self.last_year = int(series["year"].iloc[-1])
        return self
    def predict(self, horizon: int) -> pd.DataFrame:
        if getattr(self, "_degenerate", False):
            years = np.arange(self.last_year + 1, self.last_year + horizon + 1, dtype=int)
            return pd.DataFrame({"year": years, "forecast": self.last_value})
        future = self.m.make_future_dataframe(periods=horizon, freq="YE", include_history=False)
        fc = self.m.predict(future)[["ds", "yhat"]]
        fc["year"] = fc["ds"].dt.year.astype(int)
        return fc.rename(columns={"yhat": "forecast"})[["year", "forecast"]]

def recent_linear_slope(series: pd.DataFrame, tail_years: int = 6) -> float:
    s = series.tail(tail_years)
    if s["year"].nunique() < 3:
        return 0.0
    X = s["year"].values.reshape(-1,1)
    y = s[TARGET].values
    reg = LinearRegression().fit(X, y)
    return float(reg.coef_[0])

@dataclass
class PipelineConfig:
    model: str
    horizon: int
    min_train_years: int
    prophet_params: Dict[str, float]
    auto_slope_thresh: float

class ForecastPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def _instantiate(self, series: pd.DataFrame, chosen: str):
        if chosen == "naive":
            return NaiveModel().fit(series)
        elif chosen == "linear":
            return LinearTrendModel().fit(series)
        elif chosen == "prophet":
            pp = self.cfg.prophet_params
            return ProphetTrendModel(
                cps=pp.get("changepoint_prior_scale", 0.05),
                cpr=pp.get("changepoint_range", 0.8),
                n_cp=pp.get("n_changepoints", 5),
                growth=pp.get("growth", "linear"),
                seasonality_mode=pp.get("seasonality_mode", "additive"),
            ).fit(series)
        else:
            raise ValueError(f"Neznámý model: {chosen}")

    def _auto_rule(self, series: pd.DataFrame) -> str:
        slope = recent_linear_slope(series, tail_years=min(6, len(series)))
        if abs(slope) >= self.cfg.auto_slope_thresh:
            return "linear"
        return "naive"

    def forecast_one(self, series: pd.DataFrame, prefer: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
        if series["year"].nunique() < (self.cfg.min_train_years + 1):
            return "skip", pd.DataFrame(columns=["year","forecast"])
        choice = (prefer or self.cfg.model)
        if choice == "auto":
            choice = self._auto_rule(series)
        try:
            model = self._instantiate(series, choice)
        except Exception:
            model = NaiveModel().fit(series)
            choice = "naive"
        fc = model.predict(self.cfg.horizon)
        return choice, fc

def save_plot(history: pd.DataFrame, fc: pd.DataFrame, uni: str, model_name: str, out_dir: Path):
    plt.figure(figsize=(9,5))
    plt.plot(history["year"], history[TARGET], marker="o", label="Historie")
    if not fc.empty:
        plt.plot(fc["year"], fc["forecast"], marker="x", label=f"Predikce ({model_name})")
    plt.grid(True); plt.xlabel("Rok"); plt.ylabel("Skóre")
    plt.title(f"{uni} – predikce na {len(fc)} rok/roky ({model_name})")
    plt.legend(); plt.tight_layout()
    out = out_dir / f"forecast_{model_name}_{uni.replace(' ','_').replace('/','-')}.png"
    plt.savefig(out, dpi=200); plt.close()

def env_manifest() -> Dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "prophet_available": PROPHET_AVAILABLE,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out",  type=Path, default=OUT_DIR)
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    parser.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS_DEFAULT)
    parser.add_argument("--model", type=str, default="auto",
                        help="naive | linear | prophet | auto")
    parser.add_argument("--prophet-cps", type=float, default=0.05)
    parser.add_argument("--prophet-cpr", type=float, default=0.8)
    parser.add_argument("--prophet-ncp", type=int,   default=5)
    parser.add_argument("--auto-slope-thresh", type=float, default=1.0,
                        help="hraniční sklon (body/rok) pro přepnutí na linear v režimu auto")
    parser.add_argument("--limit-unis", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    ensure_out(args.out)
    plots_dir = args.out / "figs"; ensure_out(plots_dir)

    df = pd.read_parquet(args.data).sort_values(["university","year"]).reset_index(drop=True)
    lengths = (df.dropna(subset=[TARGET]).groupby("university")["year"]
                 .nunique().reset_index(name="n_years_nonnull"))
    valid_unis = list(lengths[lengths["n_years_nonnull"] >= (args.min_train_years + args.horizon)]["university"])
    if args.limit_unis and args.limit_unis < len(valid_unis):
        valid_unis = valid_unis[:args.limit_unis]

    cfg = PipelineConfig(
        model=args.model,
        horizon=args.horizon,
        min_train_years=args.min_train_years,
        prophet_params={
            "changepoint_prior_scale": args.prophet_cps,
            "changepoint_range": args.prophet_cpr,
            "n_changepoints": args.prophet_ncp,
            "growth": "linear",
            "seasonality_mode": "additive",
        },
        auto_slope_thresh=args.auto_slope_thresh
    )
    pipe = ForecastPipeline(cfg)

    all_rows = []
    for i, uni in enumerate(sorted(valid_unis), 1):
        series = get_university_series(df, uni)
        series["university"] = uni
        chosen, fc = pipe.forecast_one(series)
        if chosen == "skip" or fc.empty:
            continue
        for _, row in fc.iterrows():
            all_rows.append({
                "university": uni,
                "model": chosen,
                "origin_last_year": int(series["year"].iloc[-1]),
                "year": int(row["year"]),
                "forecast": float(row["forecast"])
            })
        if args.plot:
            save_plot(series, fc, uni, chosen, plots_dir)

    pred_path = args.out / "final_forecasts.csv"
    pd.DataFrame(all_rows).to_csv(pred_path, index=False)

    manifest = {
        "config": {
            "model": cfg.model,
            "horizon": cfg.horizon,
            "min_train_years": cfg.min_train_years,
            "prophet_params": cfg.prophet_params,
            "auto_slope_thresh": cfg.auto_slope_thresh
        },
        "data": str(args.data),
        "outputs": str(args.out),
        "env": env_manifest(),
        "note": "Výběr modelu vychází z kap. 7.2–7.4. 'auto' preferuje naive; linear se volí při výrazném posledním trendu."
    }
    with open(args.out / "pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Hotovo.")
    print(" - predikce:", pred_path)
    if args.plot:
        print(" - grafy:", plots_dir)
    print(" - manifest:", args.out / "pipeline_manifest.json")

if __name__ == "__main__":
    main()
