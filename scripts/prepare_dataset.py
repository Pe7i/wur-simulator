import json
import re
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw")
OUT_PATH = Path("data/clean")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Politika převodu intervalů skóre ---
# "mid"  = střed pásma (výchozí, nestranný odhad)
# "low"  = dolní mez (konzervativní)
# "high" = horní mez (optimistická)
RANGE_POLICY = "mid"

DROP_ALL_EMPTY_ROWS = True
NUMERIC_COLS = [
    "rank", "overall_score", "teaching", "research",
    "citations", "industry_income", "international_outlook"
]

def parse_rank(rank):
    if rank is None or rank == "-" or str(rank).strip() == "":
        return None
    if isinstance(rank, (int, float)):
        try:
            return int(rank)
        except Exception:
            return None
    m = re.search(r"(\d+)", str(rank))
    return int(m.group(1)) if m else None

_DASHES = ["\u2013", "\u2014", "\u2212"]
def _normalize_num_string(s: str) -> str:
    s = s.strip()
    for d in _DASHES:
        s = s.replace(d, "-")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    return s

_range_re = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*$")

def _from_range(a: float, b: float) -> float:
    if RANGE_POLICY == "low":
        return min(a, b)
    if RANGE_POLICY == "high":
        return max(a, b)
    return (a + b) / 2.0

def parse_score(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s == "-":
        return None
    s = _normalize_num_string(s)

    try:
        return float(s)
    except Exception:
        pass

    m = _range_re.match(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return _from_range(a, b)

    return None

def load_json_records(file: Path):
    with open(file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        return raw["data"]
    if isinstance(raw, list):
        return raw
    return []

def main():
    files = sorted(list(RAW_PATH.glob("WUR_*.json")) + list(RAW_PATH.glob("wur_*.json")))
    print(f"🧭 Nalezeno JSON souborů: {len(files)}")
    if not files:
        print("⚠️ Nenalezeny žádné soubory. Ujisti se, že jsou v data/raw/ a jmenují se WUR_YYYY.json nebo wur_YYYY.json.")
        return

    rows = []
    total_records = 0
    converted_range_count = 0

    for file in files:
        m = re.search(r"(\d{4})", file.stem)
        if not m:
            print(f"⚠️ Přeskočen {file.name} – nebyl nalezen rok v názvu.")
            continue
        year = int(m.group(1))

        recs = load_json_records(file)
        if not isinstance(recs, list):
            print(f"⚠️ {file.name}: neočekávaná struktura, 'data' není list. Přeskočeno.")
            continue

        total_records += len(recs)

        for rec in recs:
            s_overall_raw = rec.get("scores_overall")
            s_overall_norm = _normalize_num_string(str(s_overall_raw)) if s_overall_raw is not None else ""
            if _range_re.match(s_overall_norm):
                converted_range_count += 1

            rows.append({
                "year": year,
                "university": rec.get("name"),
                "country": rec.get("location"),
                "rank": parse_rank(rec.get("rank")),
                "overall_score": parse_score(rec.get("scores_overall")),
                "teaching": parse_score(rec.get("scores_teaching")),
                "research": parse_score(rec.get("scores_research")),
                "citations": parse_score(rec.get("scores_citations")),
                "industry_income": parse_score(rec.get("scores_industry_income")),
                "international_outlook": parse_score(rec.get("scores_international_outlook")),
            })

    print(f"📦 Načteno záznamů (přes všechny roky): {total_records}")

    df = pd.DataFrame(rows)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before_dups = len(df)
    df = (df
          .dropna(subset=["university", "year"])
          .drop_duplicates(subset=["university", "year"])
          .sort_values(["university", "year"])
          .reset_index(drop=True))
    after_dups = len(df)

    print(f"✅ Záznamů po čištění: {len(df)} (odstraněno duplicit: {before_dups - after_dups})")
    print(f"🔎 Pásmových hodnot 'scores_overall' převedeno: {converted_range_count} ({RANGE_POLICY})")

    out_csv_full  = OUT_PATH / "wur_dataset_full.csv"
    out_parq_full = OUT_PATH / "wur_dataset_full.parquet"
    df.to_csv(out_csv_full, index=False)
    df.to_parquet(out_parq_full, index=False)

    if DROP_ALL_EMPTY_ROWS:
        nonempty_num = ~df[NUMERIC_COLS].isna().all(axis=1)
        df_work = df.loc[nonempty_num].copy()
        dropped = int((~nonempty_num).sum())
    else:
        df_work = df.copy()
        dropped = 0

    out_csv_work  = OUT_PATH / "wur_dataset.csv"
    out_parq_work = OUT_PATH / "wur_dataset.parquet"
    df_work.to_csv(out_csv_work, index=False)
    df_work.to_parquet(out_parq_work, index=False)

    print(f"💾 Uloženo (plná verze):            {out_csv_full}, {out_parq_full}")
    print(f"💾 Uloženo (pracovní, filtrovaná): {out_csv_work}, {out_parq_work}")
    if DROP_ALL_EMPTY_ROWS:
        print(f"🧹 Odstraněno řádků s prázdnými numeriky: {dropped}")

if __name__ == "__main__":
    main()
