"""
Servisní vrstva pro načítání dat/artefaktů do Streamlit aplikace.

- RAW data (universities, historical_scores) se čtou výhradně z RAW_DIR/wur_dataset.(parquet|csv).
- Artefakty validací/inference (forecast, overall) se čtou z DATA_DIR, s podporou:
    * přímé cesty přes env FORECAST_PATH,
    * rekurzivního hledání souboru *forecast*.* v rámci DATA_DIR,
    * fallbacku pro 'overall' dopočtením z 'forecast'.
- Cesty jsou čteny z env proměnných:
    RAW_DIR  (default 'data/clean'),
    DATA_DIR (default '.' = kořen projektu).
- Funkce používají st.cache_data (bezstavové, pouze I/O).
- Validace schémat je v src.wur.schemas.*.
- Logování: při načtení souboru se zaloguje název, shape a odhad paměti.
"""
from __future__ import annotations

import os
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, List

import pandas as pd

from src.wur.schemas import (
    validate_universities,
    validate_history,
    validate_forecast,
    validate_overall,
)

logger = logging.getLogger("wur.io")

# ---------------------------------------------------------------------------
# Konfigurace a pomocné funkce (DATA_DIR)
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = (".parquet", ".csv", ".feather")

OPTIONAL_COLS = {
    "forecast": {"mae_k", "rmse_k", "lower", "upper"},
    "overall": {"mae_overall", "rmse_overall", "winner_flag"},
}

def _impute_metrics_from_history(forecast_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dovypočítá mae_k/rmse_k z historie:
      - target_year = origin_year + horizon_k
      - join s history_df (uni_id, year -> score jako y_true)
      - výpočet AE/SE a agregace per (uni_id, origin_year, horizon_k, model)
    Vrací dataframe s columns: [uni_id, origin_year, horizon_k, model, mae_k, rmse_k].
    Pokud žádné páry nevzniknou, vrátí prázdný df.
    """
    if forecast_df.empty or history_df.empty:
        return pd.DataFrame(columns=["uni_id","origin_year","horizon_k","model","mae_k","rmse_k"])

    df = forecast_df.copy()
    df["target_year"] = (df["origin_year"].astype("Int64") + df["horizon_k"].astype("Int64")).astype("Int64")

    hist = history_df[["uni_id", "year", "score"]].rename(columns={"score": "y_true"}).copy()
    paired = df.merge(hist, left_on=["uni_id", "target_year"], right_on=["uni_id", "year"], how="inner")
    if paired.empty:
        return pd.DataFrame(columns=["uni_id","origin_year","horizon_k","model","mae_k","rmse_k"])

    paired["ae"] = (paired["y_hat"] - paired["y_true"]).abs()
    paired["se"] = (paired["y_hat"] - paired["y_true"]) ** 2

    agg = (
        paired.groupby(["uni_id","origin_year","horizon_k","model"], as_index=False)
              .agg(mae_k=("ae","mean"), rmse_k=("se", lambda s: (s.mean())**0.5))
    )
    return agg


def _coerce_final_forecast_like(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Rozpozná formát podobný 'final_forecast.csv' (forecast, model, origin_last_year, university, year)
    a převede jej na kontrakt: [uni_id, origin_year, horizon_k, y_hat, model].
    Vrátí None, pokud formát neodpovídá (necháme běžet standardní validaci).
    """
    cols = set(df.columns)
    needed = {"forecast", "model", "origin_last_year", "university", "year"}
    if not needed.issubset(cols):
        return None

    out = pd.DataFrame({
        "y_hat": df["forecast"],
        "model": df["model"],
        "origin_year": df["origin_last_year"].astype("Int64"),
        "horizon_k": (df["year"].astype("Int64") - df["origin_last_year"].astype("Int64")).astype("Int64"),
        "uni_id": _make_uid_from_name(df["university"]),
    })

    try:
        unis = load_universities()
        if "name" in unis.columns:
            m = unis[["uni_id", "name"]].copy()
            m["__slug"] = _make_uid_from_name(m["name"])
            out = out.merge(m[["uni_id", "__slug"]], left_on="uni_id", right_on="__slug", how="left", suffixes=("", "_mapped"))
            out["uni_id"] = out["uni_id_mapped"].fillna(out["uni_id"])
            out = out.drop(columns=["__slug", "uni_id_mapped"], errors="ignore")
    except Exception:
        pass

    out["mae_k"] = pd.NA
    out["rmse_k"] = pd.NA

    out = out.dropna(subset=["origin_year", "horizon_k", "y_hat"])
    out = out.loc[out["horizon_k"].astype("Int64") >= 0]
    return out



def _data_dir() -> Path:
    """
    Kořen s artefakty (forecast/overall).
    Default nastaven na '.' (kořen projektu), aby šlo rekurzivně dohledat např. outputs/finals/final_forecast.csv
    i bez nastavování DATA_DIR. Lze přepsat envem DATA_DIR.
    """
    return Path(os.environ.get("DATA_DIR", ".")).resolve()


def _find_first_existing(stem: str, exts: Iterable[str]) -> Optional[Path]:
    base = _data_dir()
    for ext in exts:
        cand = base / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _read_by_exact_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Nepodporovaná přípona: {path.suffix}")
    try:
        mem = int(df.memory_usage(deep=True).sum())
    except Exception:
        mem = -1
    logger.info("Načten soubor (exact path): %s | shape=%s | bytes=%s", path, df.shape, mem)
    return df


def _glob_first(patterns: List[str], base: Path) -> Optional[Path]:
    """Vrátí první nalezený soubor podle seznamu glob patternů, rekurzivně v base."""
    for pat in patterns:
        for p in base.rglob(pat):
            if p.is_file():
                return p
    return None


def _read_table(stem: str) -> pd.DataFrame:
    """
    Načte tabulku podle stem (bez přípony) z DATA_DIR. Preferuje Parquet, pak CSV, pak Feather.
    Vyhodí FileNotFoundError s přehlednou zprávou, pokud soubor chybí.
    """
    for ext in (".parquet", ".csv", ".feather"):
        path = _find_first_existing(stem, [ext])
        if path is None:
            continue
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:  # .feather
            df = pd.read_feather(path)
        try:
            mem = int(df.memory_usage(deep=True).sum())
        except Exception:
            mem = -1
        logger.info("Načten soubor z DATA_DIR: %s | shape=%s | bytes=%s", path.name, df.shape, mem)
        return df

    base = _data_dir()
    tried = ", ".join(f"{stem}{e}" for e in SUPPORTED_EXTS)
    raise FileNotFoundError(
        f"Nenalezen žádný soubor {tried} v adresáři {base}. "
        f"Zvaž nastavení DATA_DIR nebo použij rekurzivní hledání/override (FORECAST_PATH)."
    )


def _ensure_optional(df: pd.DataFrame, optional: set[str]) -> pd.DataFrame:
    for col in sorted(optional - set(df.columns)):
        df[col] = pd.NA
    return df


# ---------------------------------------------------------------------------
# RAW_DIR / wur_dataset helpery (universities & history)
# ---------------------------------------------------------------------------

def _raw_dir() -> Path:
    """Kořen s raw/clean daty (z env RAW_DIR, default 'data/clean')."""
    return Path(os.environ.get("RAW_DIR", "data/clean")).resolve()


def _find_first_existing_in_dir(stem: str, exts: Iterable[str], base: Path) -> Optional[Path]:
    for ext in exts:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _read_wur_dataset() -> pd.DataFrame:
    """Načte RAW_DIR/wur_dataset.(parquet|csv) – bez plošného přejmenování sloupců."""
    base = _raw_dir()
    path = _find_first_existing_in_dir("wur_dataset", (".parquet", ".csv"), base)
    if not path:
        raise FileNotFoundError(f"Nenalezen 'wur_dataset.(parquet|csv)' v {base}. Nastav RAW_DIR nebo ulož soubor.")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    logger.info("Načten wur_dataset z RAW_DIR: %s | shape=%s | cols=%s", path.name, df.shape, list(df.columns))
    return df


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Vybere první existující sloupec z kandidátů (case-insensitive). Vrací originální název sloupce."""
    cols_lower = {c.lower(): c for c in df.columns}
    for want in candidates:
        if want.lower() in cols_lower:
            return cols_lower[want.lower()]
    return None


def _make_uid_from_name(name_s: pd.Series) -> pd.Series:
    """Deterministický 'uni_id' ze jména (slug)."""
    def slug(x: str) -> str:
        x = str(x).strip().lower()
        x = re.sub(r"[^0-9a-z]+", "-", x)
        x = re.sub(r"-+", "-", x).strip("-")
        return x or "unknown"
    return name_s.map(slug)

def _load_backtest_metrics() -> Optional[pd.DataFrame]:
    """
    Načte per-horizontní metriky z backtestu a vrátí je v kontraktu:
    [uni_id, origin_year, horizon_k, model, mae_k, rmse_k].
    Hledá:
      - BACKTEST_RAW_PATH (exact),
      - rekurzivně v DATA_DIR: *backtest*raw*results*.parquet|csv|feather
    """
    b_override = os.environ.get("BACKTEST_RAW_PATH")
    path: Optional[Path] = None
    if b_override:
        path = Path(b_override).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Soubor z BACKTEST_RAW_PATH neexistuje: {path}")
    else:
        base = _data_dir()
        path = _glob_first(
            ["*backtest*raw*results*.parquet", "*backtest*raw*results*.csv", "*backtest*raw*results*.feather"],
            base,
        )

    if not path:
        return None

    df = _read_by_exact_path(path)
    cols = set(df.columns)
    need = {"university", "origin_year", "model", "mae", "rmse"}
    if not need.issubset(cols):
        return None

    has_step = "step" in cols
    if has_step:
        df = df.rename(columns={"step": "horizon_k"}).copy()
    elif "horizon" in cols:
        df = df.rename(columns={"horizon": "horizon_k"}).copy()
    else:
        return None

    df["uni_id"] = _make_uid_from_name(df["university"])
    g = (
        df.groupby(["uni_id", "origin_year", "horizon_k", "model"], as_index=False)
          .agg(mae_k=("mae", "mean"), rmse_k=("rmse", "mean"))
    )
    return g




# ---------------------------------------------------------------------------
# Veřejné načítací funkce
# ---------------------------------------------------------------------------

# UNIVERSITIES – vždy z RAW_DIR/wur_dataset.*

def load_universities(path: Optional[str] = None) -> pd.DataFrame:
    """
    Načte univerzity **výhradně** z RAW_DIR/wur_dataset.(parquet|csv) a namapuje je na schéma
    [uni_id, name, country, region]. Nepřejmenováváme globálně – pouze vybíráme existující sloupce.
    Argument `path` je ignorován (ponechán kvůli kompatibilitě rozhraní).
    """
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def _load(_: Optional[str]) -> pd.DataFrame:
        raw = _read_wur_dataset()

        name_col    = _pick_col(raw, ["name", "university", "institution", "university_name", "inst_name"])
        uid_col     = _pick_col(raw, ["uni_id", "university_id", "univ_id", "id", "inst_id", "institution_id"])
        country_col = _pick_col(raw, ["country", "nation", "location"])
        region_col  = _pick_col(raw, ["region", "world_region"])

        if uid_col is None and name_col is None:
            raise ValueError(
                "wur_dataset neobsahuje sloupec s identifikátorem ani s názvem instituce.\n"
                f"Dostupné sloupce: {list(raw.columns)}"
            )

        if uid_col is None:
            unis = raw[[name_col]].dropna().drop_duplicates().rename(columns={name_col: "name"})
            unis["uni_id"] = _make_uid_from_name(unis["name"])
        else:
            unis = raw[[uid_col]].dropna().drop_duplicates().rename(columns={uid_col: "uni_id"})
            if name_col is not None:
                nm = (
                    raw[[uid_col, name_col]]
                    .dropna()
                    .drop_duplicates()
                    .rename(columns={uid_col: "uni_id", name_col: "name"})
                )
                unis = unis.merge(nm, on="uni_id", how="left")
            else:
                unis["name"] = unis["uni_id"]

        if country_col is not None and country_col in raw.columns:
            key = "uni_id" if uid_col is not None else "name"
            ctab = raw[[uid_col or name_col, country_col]].dropna().drop_duplicates()
            unis = (
                unis
                .merge(ctab, left_on=key, right_on=(uid_col or name_col), how="left")
                .drop(columns=[(uid_col or name_col)])
                .rename(columns={country_col: "country"})
            )

        if region_col is not None and region_col in raw.columns:
            key = "uni_id" if uid_col is not None else "name"
            rtab = raw[[uid_col or name_col, region_col]].dropna().drop_duplicates()
            unis = (
                unis
                .merge(rtab, left_on=key, right_on=(uid_col or name_col), how="left")
                .drop(columns=[(uid_col or name_col)])
                .rename(columns={region_col: "region"})
            )

        if "country" not in unis.columns:
            unis["country"] = ""
        if "region" not in unis.columns:
            unis["region"] = ""

        df = unis[["uni_id", "name", "country", "region"]].drop_duplicates()
        df = validate_universities(df)
        return df

    return _load(None)


# HISTORY – vždy z RAW_DIR/wur_dataset.*

def load_history(path: Optional[str] = None) -> pd.DataFrame:
    """
    Načte historii **výhradně** z RAW_DIR/wur_dataset.(parquet|csv) a namapuje ji na schéma
    [uni_id, year, score, rank, rank_band]. Argument `path` je ignorován.
    """
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def _load(_: Optional[str]) -> pd.DataFrame:
        raw = _read_wur_dataset()

        uid_col   = _pick_col(raw, ["uni_id", "university_id", "univ_id", "id", "inst_id", "institution_id"])
        name_col  = _pick_col(raw, ["name", "university", "institution", "university_name", "inst_name"])
        year_col  = _pick_col(raw, ["year", "rank_year", "ranking_year"])
        score_col = _pick_col(raw, ["score", "overall_score", "overall", "score_overall"])
        rank_col  = _pick_col(raw, ["rank"])
        band_col  = _pick_col(raw, ["rank_band", "band", "rank_range"])

        if year_col is None or score_col is None:
            raise ValueError(
                "wur_dataset neobsahuje minimální sloupce pro historii (year a score/overall_score).\n"
                f"Dostupné sloupce: {list(raw.columns)}"
            )

        if uid_col is None:
            if name_col is None:
                raise ValueError("Nelze odvodit 'uni_id' – chybí jak identifikátor, tak název.")
            tmp = raw[[name_col]].copy()
            tmp["uni_id"] = _make_uid_from_name(tmp[name_col])
            raw = raw.merge(tmp[[name_col, "uni_id"]].drop_duplicates(), on=name_col, how="left")

        df = (
            raw[[uid_col or "uni_id", year_col, score_col]]
            .rename(columns={(uid_col or "uni_id"): "uni_id", year_col: "year", score_col: "score"})
            .copy()
        )
        df["rank"] = raw[rank_col] if rank_col is not None else pd.NA
        df["rank_band"] = raw[band_col] if band_col is not None else pd.NA

        df = validate_history(df)
        return df

    return _load(None)


# FORECAST – artefakt (z DATA_DIR, s override a rekurzivním hledáním)

def load_forecast(path: Optional[str] = None) -> pd.DataFrame:
    """
    Načte forecast a doplní metriky:
      1) načte predikce (FORECAST_PATH → DATA_DIR/forecast.* → *forecast* rekurzivně),
      2) pokusí se převést formát „final_forecast“ na kontrakt,
      3) přimíchá MAE/RMSE z backtestu podle klíče (uni_id, origin_year, horizon_k, model) – modely srovná v lowercase,
      4) přidá i backtestové řádky, pro které forecast řádek neexistuje (y_hat = NaN),
      5) co pořád chybí, dopočítá z historie párováním na y_true,
      6) projde validací a doplní volitelné sloupce.
    """
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def _load(p: Optional[str]) -> pd.DataFrame:
        # --- 1) forecast artefakt ---
        f_override = os.environ.get("FORECAST_PATH")
        if f_override:
            df = _read_by_exact_path(Path(f_override).resolve())
        elif p is not None:
            df = _read_by_exact_path(Path(p).resolve())
        else:
            try:
                df = _read_table("forecast")
            except FileNotFoundError:
                base = _data_dir()
                cand = _glob_first(["*forecast*.parquet", "*forecast*.csv", "*forecast*.feather"], base)
                if not cand:
                    raise FileNotFoundError(
                        "Nebyl nalezen žádný soubor forecast v DATA_DIR (vč. podadresářů). "
                        f"Hledej '*forecast*.parquet|csv|feather' v {base}, nebo nastav FORECAST_PATH."
                    )
                df = _read_by_exact_path(cand)

        # --- 2) převod final_forecast → kontrakt ---
        if not {"uni_id", "origin_year", "horizon_k", "y_hat", "model"}.issubset(df.columns):
            maybe = _coerce_final_forecast_like(df)
            if maybe is not None:
                df = maybe

        for col in ["origin_year", "horizon_k"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df["model"] = df["model"].astype(str).str.strip().str.lower()

        # --- 3) načtení backtest metriky a left-merge na forecast ---
        bt = _load_backtest_metrics()
        if bt is not None:
            bt = bt.copy()
            bt["origin_year"] = pd.to_numeric(bt["origin_year"], errors="coerce").astype("Int64")
            bt["horizon_k"]   = pd.to_numeric(bt["horizon_k"],   errors="coerce").astype("Int64")
            bt["model"]       = bt["model"].astype(str).str.strip().str.lower()

            before = df[["mae_k","rmse_k"]].notna().sum().sum() if {"mae_k","rmse_k"}.issubset(df.columns) else 0
            df = df.merge(
                bt[["uni_id","origin_year","horizon_k","model","mae_k","rmse_k"]],
                on=["uni_id","origin_year","horizon_k","model"],
                how="left",
                suffixes=("", "_bt")
            )
            if "mae_k_bt" in df.columns:
                df["mae_k"]  = df.get("mae_k")  if "mae_k"  in df.columns else pd.NA
                df["rmse_k"] = df.get("rmse_k") if "rmse_k" in df.columns else pd.NA
                df["mae_k"]  = df["mae_k"].fillna(df["mae_k_bt"])
                df["rmse_k"] = df["rmse_k"].fillna(df["rmse_k_bt"])
                df = df.drop(columns=["mae_k_bt","rmse_k_bt"])
            after = df[["mae_k","rmse_k"]].notna().sum().sum()
            logger.info("Backtest merge: doplněno metrik (nenulových buněk) %s → %s", before, after)

            # --- 4) přidání sirotčích backtest řádků (kde forecast neexistuje) ---
            keys = ["uni_id","origin_year","horizon_k","model"]
            bt_only = bt.merge(df[keys].drop_duplicates(), on=keys, how="left", indicator=True)
            bt_only = bt_only[bt_only["_merge"] == "left_only"].drop(columns=["_merge"])
            if not bt_only.empty:
                add = bt_only.assign(y_hat=pd.NA)
                df = pd.concat([df, add[["uni_id","origin_year","horizon_k","model","y_hat","mae_k","rmse_k"]]], ignore_index=True)

        # --- 5) dopočet z historie (párování target_year s y_true) ---
        if ("mae_k" not in df.columns) or ("rmse_k" not in df.columns) or df[["mae_k","rmse_k"]].isna().any().any():
            try:
                hist = load_history()  # cached
                subset = df.loc[:, ["uni_id","origin_year","horizon_k","model","y_hat"]].copy()
                imputed = _impute_metrics_from_history(subset, hist)
                if not imputed.empty:
                    df = df.merge(imputed, on=["uni_id","origin_year","horizon_k","model"], how="left", suffixes=("", "_imp"))
                    if "mae_k_imp" in df.columns:
                        df["mae_k"]  = df.get("mae_k")  if "mae_k"  in df.columns else pd.NA
                        df["rmse_k"] = df.get("rmse_k") if "rmse_k" in df.columns else pd.NA
                        df["mae_k"]  = df["mae_k"].fillna(df["mae_k_imp"])
                        df["rmse_k"] = df["rmse_k"].fillna(df["rmse_k_imp"])
                        df = df.drop(columns=["mae_k_imp","rmse_k_imp"])
            except Exception as e:
                logger.warning("Nepodařilo se dopočítat metriky z historie: %s", e)

        # --- 6) validace + doplnění volitelných sloupců ---
        df = validate_forecast(df)
        df = _ensure_optional(df, OPTIONAL_COLS["forecast"])  # (mae_k, rmse_k, lower, upper)
        df = df[["uni_id","origin_year","horizon_k","y_hat","model","mae_k","rmse_k"]].sort_values(
            ["uni_id","origin_year","horizon_k","model"]
        ).reset_index(drop=True)
        return df

    return _load(path)




# OVERALL – artefakt (z DATA_DIR, s fallbackem z 'forecast')

def load_overall(path: Optional[str] = None) -> pd.DataFrame:
    """
    Načte 'overall' z DATA_DIR; pokud chybí, odvodí ho z 'forecast':
      - mae_overall = průměr mae_k napříč origin_year × horizon_k (ignoruje NaN),
      - rmse_overall = průměr rmse_k napříč origin_year × horizon_k (ignoruje NaN),
      - winner_flag = True pro model s minimálním mae_overall v rámci uni_id (jen tam, kde je k dispozici).
    """
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def _load(p: Optional[str]) -> pd.DataFrame:
        try:
            if p is None:
                df = _read_table("overall")
            else:
                df = _read_by_exact_path(Path(p).resolve())
            df = validate_overall(df)
            df = _ensure_optional(df, OPTIONAL_COLS["overall"])
            return df

        except FileNotFoundError:
            fcast = load_forecast()
            if fcast.empty:
                raise FileNotFoundError(
                    "Soubor 'overall' chybí a 'forecast' je prázdný – není z čeho odvodit agregace."
                )

            agg = (
                fcast.groupby(["uni_id", "model"], as_index=False)[["mae_k", "rmse_k"]]
                .mean(numeric_only=True)
                .rename(columns={"mae_k": "mae_overall", "rmse_k": "rmse_overall"})
            )

            agg["winner_flag"] = False

            valid = agg.dropna(subset=["mae_overall"])
            if not valid.empty:
                idx = valid.groupby("uni_id")["mae_overall"].idxmin()
                agg.loc[idx.values, "winner_flag"] = True

            agg = validate_overall(agg)
            return _ensure_optional(agg, OPTIONAL_COLS["overall"])

    return _load(path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def list_available_files() -> pd.DataFrame:
    """Vrátí přehled dostupných souborů v DATA_DIR (užitečné pro debug v UI)."""
    base = _data_dir()
    files = [p for p in base.rglob("*") if p.is_file() and p.suffix in SUPPORTED_EXTS]
    df = pd.DataFrame({
        "file": [str(p.relative_to(base)) for p in files],
        "size_bytes": [p.stat().st_size for p in files],
        "modified": [pd.to_datetime(p.stat().st_mtime, unit="s") for p in files],
    }).sort_values("file").reset_index(drop=True)
    logger.info("DATA_DIR=%s | %d soubor(ů) nalezeno (rekurzivně)", base, len(df))
    return df
