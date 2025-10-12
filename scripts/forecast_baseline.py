import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

DATA_PATH = Path("data/clean/wur_dataset.parquet")

def get_university_series(df, university):
    uni_df = df[df["university"] == university].sort_values("year")
    return uni_df[["year", "overall_score"]].dropna().copy()

def to_year_start_datetime(year_series: pd.Series) -> pd.Series:
    return pd.to_datetime(year_series.astype(int).astype(str) + "-01-01")

def naive_forecast(series, horizon=3):
    last_value = float(series["overall_score"].iloc[-1])
    last_year  = int(series["year"].iloc[-1])
    years_future = np.arange(last_year + 1, last_year + horizon + 1, dtype=int)
    return pd.DataFrame({"year": years_future, "forecast": last_value})

def linear_forecast(series, horizon=3):
    X = series["year"].values.reshape(-1, 1)
    y = series["overall_score"].values
    model = LinearRegression().fit(X, y)
    last_year  = int(series["year"].iloc[-1])
    years_future = np.arange(last_year + 1, last_year + horizon + 1, dtype=int)
    y_future = model.predict(years_future.reshape(-1, 1))
    return pd.DataFrame({"year": years_future, "forecast": y_future})

def prophet_forecast(series, horizon=3):
    if not PROPHET_AVAILABLE:
        return pd.DataFrame({"year": [], "forecast": []})
    df_p = pd.DataFrame({"ds": to_year_start_datetime(series["year"]),
                         "y":  series["overall_score"].astype(float)})
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=horizon, freq="YE", include_history=False)
    fc = m.predict(future)[["ds", "yhat"]]
    fc["year"] = fc["ds"].dt.year.astype(int)
    return fc.rename(columns={"yhat": "forecast"})[["year", "forecast"]]

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH).sort_values(["university", "year"]).reset_index(drop=True)
    uni = "Harvard University"
    series = get_university_series(df, uni)

    naive   = naive_forecast(series)
    linear  = linear_forecast(series)
    prophet = prophet_forecast(series)

    plt.figure(figsize=(10, 6))
    plt.plot(series["year"], series["overall_score"], label="Historie", marker="o")
    if not linear.empty:
        plt.plot(linear["year"], linear["forecast"], label="Linear trend", marker="x")
    if not prophet.empty:
        plt.plot(prophet["year"], prophet["forecast"], label="Prophet", linestyle="--")
    if not naive.empty:
        plt.plot(naive["year"], naive["forecast"], label="Naive", linestyle=":")
    plt.legend()
    plt.title(f"Predikce skóre: {uni}")
    plt.xlabel("Rok")
    plt.ylabel("Skóre")
    plt.grid(True)

    out_path = Path("outputs/forecast_harvard.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=200)
    print(f"Graf uložen do: {out_path.resolve()}")
