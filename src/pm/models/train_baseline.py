from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from pm.data.load_data import load_training_data
from pm.features.build_features import add_rul, add_binary_label


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def main():
    print("Loading data...")
    df = load_training_data("FD001")
    df = add_rul(df)
    df = add_binary_label(df, threshold=30)

    print("Preparing features...")
    feature_cols = [c for c in df.columns if "sensor_measurement" in c]

    X = df[feature_cols]
    y = df["will_fail_soon"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1
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
