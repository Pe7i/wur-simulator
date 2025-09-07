from pathlib import Path
import json
import sys
import traceback

print("🚀 Spouštím train_models.py")
print("CWD:", Path.cwd())
print("Python:", sys.version)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("ROOT:", ROOT)

try:
    from src.wur.dataio import load_clean_dataset
    from src.wur.models.model import fit_and_evaluate
except Exception as e:
    print("❌ Problém s importem z src/:")
    traceback.print_exc()
    sys.exit(1)

def main():
    try:
        data_path = ROOT / "data/clean/wur_dataset.parquet"
        print("📦 Načítám dataset:", data_path)
        df = load_clean_dataset(str(data_path))
        print("✅ Načteno řádků:", len(df))
        print("Sloupce:", list(df.columns))

        print("🧪 Spouštím trénování + validaci (3 časové splity)…")
        report = fit_and_evaluate(
            df,
            target="overall_score",
            feature_cols=["teaching","research","citations","industry_income","international_outlook","year"],
            n_splits=3,
            save_dir=ROOT / "models"
        )

        print("✅ Hotovo")
        print("📝 Report:", report.get("report_path"))
        print("🏆 Nejlepší model:", report.get("best_model_path"))

    except Exception as e:
        print("❌ Během trénování došlo k chybě:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
