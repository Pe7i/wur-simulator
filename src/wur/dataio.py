from pathlib import Path
import pandas as pd

def load_clean_dataset(path: str | Path = "data/clean/wur_dataset.parquet") -> pd.DataFrame:
    return pd.read_parquet(Path(path))
