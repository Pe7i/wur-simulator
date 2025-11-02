from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import pandas as pd

def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Tabulka '{name}' postrádá povinné sloupce: {sorted(missing)}. "
            f"Dostupné sloupce: {sorted(df.columns)}"
        )

def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str], integer: bool = False) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns: 
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if integer:
            df[c] = df[c].astype("Int64")
    return df

def _coerce_string(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns: 
            continue
        df[c] = df[c].astype(str)
    return df

def _coerce_boolean(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns: 
            continue
        df[c] = (
            df[c]
            .map(lambda x: str(x).strip().lower())
            .map({"true": True, "false": False, "1": True, "0": False})
            .astype("boolean")
        )
    return df

def _assert_no_duplicates(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    if not keys:
        return
    dup = df.duplicated(subset=list(keys), keep=False)
    if dup.any():
        sample = df.loc[dup, list(keys)].head(10)
        raise ValueError(
            f"Tabulka '{name}' obsahuje duplicitní klíče podle {list(keys)}. "
            f"Prvních 10 duplicitních kombinací:\n{sample}"
        )

@dataclass(frozen=True)
class UniversitiesSchema:
    required: tuple[str, ...] = ("uni_id", "name", "country", "region")
    key: tuple[str, ...] = ("uni_id",)

def validate_universities(df: pd.DataFrame) -> pd.DataFrame:
    name = "universities"
    _require_columns(df, UniversitiesSchema.required, name)
    df = _coerce_string(df, ["uni_id", "name", "country", "region"])
    _assert_no_duplicates(df, UniversitiesSchema.key, name)
    return df

@dataclass(frozen=True)
class HistorySchema:
    required: tuple[str, ...] = ("uni_id", "year", "score", "rank", "rank_band")
    key: tuple[str, ...] = ("uni_id", "year")

def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    name = "historical_scores"
    _require_columns(df, HistorySchema.required, name)
    df = _coerce_string(df, ["uni_id", "rank_band"])
    df = _coerce_numeric(df, ["year"], integer=True)
    df = _coerce_numeric(df, ["score"], integer=False)
    df = _coerce_numeric(df, ["rank"], integer=True)
    _assert_no_duplicates(df, HistorySchema.key, name)
    return df.sort_values(["uni_id", "year"]).reset_index(drop=True)

@dataclass(frozen=True)
class ForecastSchema:
    required: tuple[str, ...] = ("uni_id", "origin_year", "horizon_k", "y_hat", "model")
    key: tuple[str, ...] = ("uni_id", "origin_year", "horizon_k", "model")

def validate_forecast(df: pd.DataFrame) -> pd.DataFrame:
    name = "forecast"
    _require_columns(df, ForecastSchema.required, name)
    df = _coerce_string(df, ["uni_id", "model"])
    df = _coerce_numeric(df, ["origin_year", "horizon_k"], integer=True)
    df = _coerce_numeric(df, ["y_hat"], integer=False)
    # volitelné metriky:
    if "mae_k" in df.columns:
        df = _coerce_numeric(df, ["mae_k"], integer=False)
    else:
        df["mae_k"] = pd.NA
    if "rmse_k" in df.columns:
        df = _coerce_numeric(df, ["rmse_k"], integer=False)
    else:
        df["rmse_k"] = pd.NA

    if "lower" in df.columns:
        df = _coerce_numeric(df, ["lower"], integer=False)
    if "upper" in df.columns:
        df = _coerce_numeric(df, ["upper"], integer=False)

    _assert_no_duplicates(df, ForecastSchema.key, name)
    return df.sort_values(["uni_id", "origin_year", "horizon_k", "model"]).reset_index(drop=True)

@dataclass(frozen=True)
class OverallSchema:
    required: tuple[str, ...] = ("uni_id", "model", "mae_overall", "rmse_overall", "winner_flag")
    key: tuple[str, ...] = ("uni_id", "model")

def validate_overall(df: pd.DataFrame) -> pd.DataFrame:
    name = "overall"
    _require_columns(df, OverallSchema.required, name)
    df = _coerce_string(df, ["uni_id", "model"])
    df = _coerce_numeric(df, ["mae_overall", "rmse_overall"], integer=False)
    df = _coerce_boolean(df, ["winner_flag"])
    _assert_no_duplicates(df, OverallSchema.key, name)
    return df

def validate_table(df: pd.DataFrame, table: str) -> pd.DataFrame:
    t = table.lower().strip()
    if t in {"universities", "unis", "u"}:
        return validate_universities(df)
    if t in {"historical_scores", "history", "h"}:
        return validate_history(df)
    if t in {"forecast", "f"}:
        return validate_forecast(df)
    if t in {"overall", "o"}:
        return validate_overall(df)
    raise ValueError(f"Neznámý typ tabulky pro validaci: {table}")
