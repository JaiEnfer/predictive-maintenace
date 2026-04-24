from __future__ import annotations
import numpy as np
import pandas as pd


def split_by_unit(
    df: pd.DataFrame,
    unit_col: str = "unit_number",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = df[unit_col].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(units)

    n_test = max(1, int(len(units) * test_size))
    test_units = set(units[:n_test])

    df_train = df[~df[unit_col].isin(test_units)].copy()
    df_val = df[df[unit_col].isin(test_units)].copy()
    return df_train, df_val
