from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

MetricDict = Dict[str, float]

@dataclass
class SplitSpec:
    train_years: List[int]
    valid_years: List[int]

def build_feature_matrix(df: pd.DataFrame,
                         target: str = "overall_score",
                         feature_cols: Optional[List[str]] = None
                         ) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_cols is None:
        feature_cols = ["teaching", "research", "citations", "industry_income",
                        "international_outlook", "year"]
    keep = ["university", "country", target] + feature_cols
    df_ = df[keep].copy()
    df_ = df_.dropna(subset=[target])
    X = df_[feature_cols].copy()
    y = df_[target].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    return X, y

def temporal_splits(df: pd.DataFrame,
                    n_splits: int = 3,
                    min_year: Optional[int] = None,
                    max_year: Optional[int] = None
                    ) -> List[SplitSpec]:
    years = sorted(df["year"].dropna().unique().tolist())
    if min_year is not None:
        years = [y for y in years if y >= min_year]
    if max_year is not None:
        years = [y for y in years if y <= max_year]
    if len(years) < n_splits + 1:
        n_splits = max(1, len(years) - 1)
    specs: List[SplitSpec] = []
    for i in range(1, n_splits + 1):
        valid_year = years[-i]
        train_years = [y for y in years if y < valid_year]
        specs.append(SplitSpec(train_years=train_years, valid_years=[valid_year]))
    return list(reversed(specs))

def make_models(random_state: int = 42) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    scaler = ColumnTransformer([("num", StandardScaler(), slice(0, None))], remainder="drop")
    models["LinearRegression"] = Pipeline([
        ("scaler", scaler),
        ("est", LinearRegression())
    ])
    models["RandomForest"] = Pipeline([
        ("est", RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state
        ))
    ])
    if XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("est", XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_state,
                tree_method="hist"
            ))
        ])
    models["MLP"] = Pipeline([
        ("scaler", scaler),
        ("est", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=400,
            random_state=random_state
        ))
    ])
    return models

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def fit_and_evaluate(df: pd.DataFrame,
                     target: str = "overall_score",
                     feature_cols: Optional[List[str]] = None,
                     n_splits: int = 3,
                     save_dir: str | Path = "models"
                     ) -> Dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y = build_feature_matrix(df, target=target, feature_cols=feature_cols)
    specs = temporal_splits(df, n_splits=n_splits)
    year_series = df.dropna(subset=[target])["year"].reset_index(drop=True)

    models = make_models()
    full_report: Dict = {"target": target, "splits": [], "models": list(models.keys())}

    best_on_last_split = (None, np.inf, None)  # (name, rmse, pipeline)

    for si, spec in enumerate(specs):
        split_info = {"split_id": si, "train_years": spec.train_years, "valid_years": spec.valid_years, "results": {}}
        train_idx = year_series.isin(spec.train_years).values
        valid_idx = year_series.isin(spec.valid_years).values
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[valid_idx], y[valid_idx]

        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            m = regression_metrics(y_va, pred)
            split_info["results"][name] = m
            if si == len(specs) - 1 and m["RMSE"] < best_on_last_split[1]:
                best_on_last_split = (name, m["RMSE"], pipe)

        full_report["splits"].append(split_info)

    report_path = save_dir / f"report_{target}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    best_name, _, best_pipe = best_on_last_split
    if best_pipe is not None:
        import joblib
        model_path = save_dir / f"best_{target}_{best_name}.joblib"
        joblib.dump(best_pipe, model_path)
        full_report["best_model_path"] = str(model_path)

    full_report["report_path"] = str(report_path)
    return full_report

# --- helper functions for chapter 7.2 ---

def split_indices_for_years(df: pd.DataFrame, target: str, train_years: list[int], valid_years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    year_series = df.dropna(subset=[target])["year"].reset_index(drop=True)
    train_idx = year_series.isin(train_years).values
    valid_idx = year_series.isin(valid_years).values
    return train_idx, valid_idx


def describe_splits(df: pd.DataFrame, target: str = "overall_score", n_splits: int = 3) -> None:
    specs = temporal_splits(df, n_splits=n_splits)
    X, y = build_feature_matrix(df, target=target)

    print(f"Target: {target}")
    for i, spec in enumerate(specs):
        tr_idx, va_idx = split_indices_for_years(df, target, spec.train_years, spec.valid_years)
        print(f"\nSplit {i}: train_years≤{max(spec.train_years)} | valid_year={spec.valid_years[0]}")
        print(f"  train rows: {tr_idx.sum():>6} | valid rows: {va_idx.sum():>6}")
        intersect = (tr_idx & va_idx).sum()
        print(f"  intersection (should be 0): {intersect}")
