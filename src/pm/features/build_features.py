import pandas as pd


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each engine (unit_number), compute Remaining Useful Life (RUL) at each cycle:
    RUL = max_cycle_for_engine - current_cycle
    """
    max_cycles = df.groupby("unit_number")["time_in_cycles"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_number")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    df = df.drop(columns=["max_cycle"])
    return df


def add_binary_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """
    Create a label indicating whether the engine will fail within `threshold` cycles.
    1 = will fail soon, 0 = not soon.
    """
    if "RUL" not in df.columns:
        raise ValueError("RUL column not found. Run add_rul() first.")
    df["will_fail_soon"] = (df["RUL"] <= threshold).astype(int)
    return df
