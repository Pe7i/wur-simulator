import pandas as pd
import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw")
OUT_PATH = Path("data/clean")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# vezmeme oba tvary názvů: WUR_*.json i wur_*.json
files = sorted(list(RAW_PATH.glob("WUR_*.json")) + list(RAW_PATH.glob("wur_*.json")))
print(f"🧭 Nalezeno JSON souborů: {len(files)}")
if not files:
    print("⚠️ Nenalezeny žádné soubory. Zkontroluj prosím, že JSONy jsou v data/raw/ a jmenují se WUR_YYYY.json nebo wur_YYYY.json.")
    exit(1)

def parse_rank(rank):
    """Převeď rank na číslo (dolní hranice intervalu)."""
    if rank is None or rank == "-":
        return None
    if isinstance(rank, int):
        return rank
    m = re.match(r"(\d+)", str(rank))
    return int(m.group(1)) if m else None

def parse_score(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

rows = []
total_records = 0

for file in files:
    # rok odvozujeme z názvu souboru (WUR_2019.json → 2019)
    m = re.search(r"(\d{4})", file.stem)
    if not m:
        print(f"⚠️ Přeskakuji {file.name} – nenašel jsem rok v názvu.")
        continue
    year = int(m.group(1))

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("data", data)
    if not isinstance(records, list):
        print(f"⚠️ {file.name}: neočekávaná struktura, klíč 'data' není list. Přeskakuji.")
        continue

    total_records += len(records)

    for rec in records:
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
print(f"✅ Záznamů po čištění: {len(df)}")
if df.empty:
    print("⚠️ Výsledek je prázdný DataFrame. Zkontroluj prosím klíče uvnitř JSONu (scores_*) a pole 'data'.")
else:
    out_csv = OUT_PATH / "wur_dataset.csv"
    out_parq = OUT_PATH / "wur_dataset.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parq, index=False)
    print(f"💾 Uloženo do: {out_csv}")
    print(f"💾 Uloženo do: {out_parq}")
    # kontrolní výpis prvních řádků
    print(df.head().to_string(index=False))
