from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from pm.constants import SENSOR_FEATURES
from pm.data.load_data import load_training_data
from pm.features.build_features import add_binary_label, add_rul
from pm.models.preprocess import RollingFeatureTransformer
from pm.models.split import split_by_unit

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Loading data...")
    df = load_training_data("FD001")

    print("Building labels...")
    df = add_rul(df)
    df = add_binary_label(df, threshold=30)

    print("Splitting by unit_number (no leakage)...")
    train_df, val_df = split_by_unit(df, test_size=0.2, random_state=42)

    X_train: pd.DataFrame = train_df[SENSOR_FEATURES].copy()
    y_train = train_df["will_fail_soon"].copy()

    X_val: pd.DataFrame = val_df[SENSOR_FEATURES].copy()
    y_val = val_df["will_fail_soon"].copy()

    print(
        f"Train engines: {train_df['unit_number'].nunique()} | "
        f"Val engines: {val_df['unit_number'].nunique()}"
    )
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    pipeline = Pipeline(
        steps=[
            ("roll", RollingFeatureTransformer(window=5)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training pipeline (preprocess + model)...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating...")
    preds = pipeline.predict(X_val)
    probs = pipeline.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, preds))
    print("ROC AUC:", roc_auc_score(y_val, probs))

    model_path = MODEL_DIR / "rf_predictive_maintenance.joblib"

    artifact = {
        "pipeline": pipeline,
        "sensor_features": SENSOR_FEATURES,
        "rolling_window": 5,
    }

    joblib.dump(artifact, model_path)
    print(f"\nSaved pipeline artifact to: {model_path.resolve()}")

if __name__ == "__main__":
    main()
