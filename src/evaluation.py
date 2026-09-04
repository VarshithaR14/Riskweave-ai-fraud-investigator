import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.feature_engineering import create_features, FEATURE_COLUMNS


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "transactions.csv",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_model.pkl",
)


def main():
    print("=" * 60)
    print("RISKWEAVE MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print("Dataset not found:", DATA_PATH)
        return

    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        print("Train the model first using: python -m src.fraud_model")
        return

    df = pd.read_csv(DATA_PATH)

    X = create_features(df)[FEATURE_COLUMNS]
    y = df["is_fraud"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nDataset information")
    print("-" * 60)
    print(f"Total records       : {len(df)}")
    print(f"Fraud records       : {int(y.sum())}")
    print(f"Legitimate records : {int((y == 0).sum())}")
    print(f"Test records        : {len(y_test)}")

    print("\nAccuracy")
    print("-" * 60)
    print(f"{accuracy_score(y_test, predictions):.4f}")

    print("\nClassification report")
    print("-" * 60)
    print(classification_report(y_test, predictions))

    print("\nConfusion matrix")
    print("-" * 60)
    print(confusion_matrix(y_test, predictions))

    print("\nROC-AUC")
    print("-" * 60)
    print(f"{roc_auc_score(y_test, probabilities):.4f}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()