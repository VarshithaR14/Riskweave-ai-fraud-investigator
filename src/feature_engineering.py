import numpy as np
import pandas as pd


# ============================================================
# RISKWEAVE — FEATURE ENGINEERING
# ============================================================

FEATURE_COLUMNS = [
    "amount",
    "account_age_days",
    "failed_attempts",
    "refund_count",
    "amount_log",
    "high_amount",
    "new_account",
    "failed_attempt_risk",
    "refund_risk",
]


def create_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # Amount transformation
    # --------------------------------------------------------

    data["amount_log"] = np.log1p(
        data["amount"]
    )

    # --------------------------------------------------------
    # High-value transaction
    # --------------------------------------------------------

    data["high_amount"] = (
        data["amount"] > 10000
    ).astype(int)

    # --------------------------------------------------------
    # New account
    # --------------------------------------------------------

    data["new_account"] = (
        data["account_age_days"] < 30
    ).astype(int)

    # --------------------------------------------------------
    # Failed payment risk
    # --------------------------------------------------------

    data["failed_attempt_risk"] = (
        data["failed_attempts"] >= 3
    ).astype(int)

    # --------------------------------------------------------
    # Refund risk
    # --------------------------------------------------------

    data["refund_risk"] = (
        data["refund_count"] >= 2
    ).astype(int)

    return data