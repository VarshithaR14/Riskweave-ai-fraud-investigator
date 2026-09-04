import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.feature_engineering import (
    create_features,
    FEATURE_COLUMNS,
)


# ============================================================
# RISKWEAVE — FRAUD RISK MODEL
# ============================================================


MODEL_PATH = "models/fraud_model.pkl"
DATA_PATH = "data/raw/transactions.csv"


def train_model():

    print()
    print("========================================")
    print("       RISKWEAVE FRAUD MODEL")
    print("========================================")

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n[1/6] Loading transaction data...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded {len(df)} transactions."
    )

    # --------------------------------------------------------
    # 2. Create features
    # --------------------------------------------------------

    print("\n[2/6] Creating features...")

    df = create_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    print(
        f"Features used: {len(FEATURE_COLUMNS)}"
    )

    # --------------------------------------------------------
    # 3. Train/test split
    # --------------------------------------------------------

    print("\n[3/6] Creating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Held-out test samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # 4. Train model
    # --------------------------------------------------------

    print("\n[4/6] Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Model training completed.")

    # --------------------------------------------------------
    # 5. Evaluate
    # --------------------------------------------------------

    print("\n[5/6] Evaluating model...")

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print()
    print("========================================")
    print("              PERFORMANCE")
    print("========================================")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(
        f"\nROC-AUC: {auc:.4f}"
    )

    # --------------------------------------------------------
    # 6. Feature importance
    # --------------------------------------------------------

    print("\nFeature Importance:")

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("\n[6/6] Saving model...")

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print()
    print("========================================")
    print("         MODEL BUILD COMPLETE")
    print("========================================")


if __name__ == "__main__":

    train_model()