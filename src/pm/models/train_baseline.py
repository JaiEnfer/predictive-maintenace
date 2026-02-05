from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from pm.data.load_data import load_training_data
from pm.features.build_features import add_rul, add_binary_label
from pm.features.time_series_features import add_rolling_features
from pm.models.split import split_by_unit
from pm.constants import SENSOR_FEATURES


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def main():
    print("Loading data...")
    df = load_training_data("FD001")
    df = add_rul(df)
    df = add_binary_label(df, threshold=30)

    print("Adding rolling features...")
    df = add_rolling_features(df, window=5)

    # Feature columns: raw sensors + rolling stats
    feature_cols = []
    feature_cols += SENSOR_FEATURES
    feature_cols += [f"{c}_roll_mean_5" for c in SENSOR_FEATURES]
    feature_cols += [f"{c}_roll_std_5" for c in SENSOR_FEATURES]

    print("Splitting by unit_number (no leakage)...")
    train_df, val_df = split_by_unit(df, test_size=0.2, random_state=42)

    X_train = train_df[feature_cols]
    y_train = train_df["will_fail_soon"]

    X_val = val_df[feature_cols]
    y_val = val_df["will_fail_soon"]

    print(f"Train engines: {train_df['unit_number'].nunique()} | Val engines: {val_df['unit_number'].nunique()}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, preds))
    print("ROC AUC:", roc_auc_score(y_val, probs))

    model_path = MODEL_DIR / "rf_predictive_maintenance.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path.resolve()}")


if __name__ == "__main__":
    main()
