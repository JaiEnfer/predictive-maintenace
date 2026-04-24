from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"] 
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_measurement_{i}" for i in range(1, 22)] 
)

def _read_txt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = [c.strip() for c in COLUMN_NAMES]

    # Safety rename in case an old typo exists in code or cached runs
    df = df.rename(columns={"time_in_cyyles": "time_in_cycles"})

    return df

def load_training_data(fd:str = "FD001") -> pd.DataFrame:
    return _read_txt(DATA_DIR / f"train_{fd}.txt")

def load_test_data(fd:str = "FD001") -> pd.DataFrame:
    return _read_txt(DATA_DIR / f"test_{fd}.txt")

def load_rul(fd:str = "FD001") -> pd.Series:
    path = DATA_DIR / f"RUL_{fd}.txt"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_csv(path, header=None)[0]

