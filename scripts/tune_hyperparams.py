from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

from src.wur.dataio import load_clean_dataset
from src.wur.models.model import build_feature_matrix, split_indices_for_years, temporal_splits

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


def main():
    ROOT = Path(__file__).resolve().parents[1]
    df = load_clean_dataset(ROOT / "data/clean/wur_dataset.parquet")

    spec = temporal_splits(df, n_splits=3)[-1]
    train_idx, valid_idx = split_indices_for_years(df, "overall_score", spec.train_years, spec.valid_years)

    X, y = build_feature_matrix(df, target="overall_score")
    X_train, y_train = X[train_idx], y[train_idx]

    # ------------------------
    # 1. Random Forest – GridSearch
    # ------------------------
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    rf_search = GridSearchCV(rf, rf_grid, scoring=rmse_scorer, cv=3, n_jobs=-1, verbose=2)
    rf_search.fit(X_train, y_train)

    print("=== RandomForest GridSearch ===")
    print("Best params:", rf_search.best_params_)
    print("Best RMSE:", -rf_search.best_score_)

    # ------------------------
    # 2. MLP – RandomizedSearch
    # ------------------------
    mlp = MLPRegressor(max_iter=400, random_state=42)
    mlp_dist = {
        "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [1e-2, 1e-3, 1e-4],
        "alpha": [1e-5, 1e-4, 1e-3],  
    }
    mlp_search = RandomizedSearchCV(
        mlp, mlp_dist, scoring=rmse_scorer, cv=3,
        n_iter=10, random_state=42, n_jobs=-1, verbose=2
    )
    mlp_search.fit(X_train, y_train)

    print("=== MLP RandomSearch ===")
    print("Best params:", mlp_search.best_params_)
    print("Best RMSE:", -mlp_search.best_score_)

    # ------------------------
    # 3. (Optional) BayesSearch – RF
    # ------------------------
    if SKOPT_AVAILABLE:
        rf_bayes = BayesSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": (100, 1000),
                "max_depth": (3, 30),
                "min_samples_split": (2, 20),
            },
            n_iter=20, scoring=rmse_scorer, cv=3, n_jobs=-1, random_state=42, verbose=2
        )
        rf_bayes.fit(X_train, y_train)
        print("=== RF BayesSearch ===")
        print("Best params:", rf_bayes.best_params_)
        print("Best RMSE:", -rf_bayes.best_score_)


if __name__ == "__main__":
    main()
