import pandas as pd
from pm.constants import SENSOR_FEATURES


def add_rolling_features(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Adds rolling mean/std per engine for each sensor measurement.
    Keeps it simple and fast for a portfolio demo.
    """
    df = df.sort_values(["unit_number", "time_in_cycles"]).copy()

    for col in SENSOR_FEATURES:
        df[f"{col}_roll_mean_{window}"] = (
            df.groupby("unit_number")[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_roll_std_{window}"] = (
            df.groupby("unit_number")[col].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
        )

    return df
