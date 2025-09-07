from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wur.dataio import load_clean_dataset
from src.wur.models.model import describe_splits

def main():
    df = load_clean_dataset(ROOT / "data/clean/wur_dataset.parquet")
    describe_splits(df, target="overall_score", n_splits=3)

if __name__ == "__main__":
    main()
