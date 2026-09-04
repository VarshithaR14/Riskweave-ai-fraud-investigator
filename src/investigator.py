import os

import joblib
import pandas as pd
import networkx as nx

from src.feature_engineering import (
    create_features,
    FEATURE_COLUMNS,
)

from src.graph_detector import (
    build_user_relationship_graph,
)


# ============================================================
# RISKWEAVE - AI INVESTIGATION ENGINE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load transaction dataset.
    """

    return pd.read_csv(
        DATA_PATH
    )


# ============================================================
# LOAD ML MODEL
# ============================================================

def load_model():
    """
    Load trained Random Forest fraud model.
    """

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# FIND RELATED USERS
# ============================================================

def find_related_users(
    df,
    target_user,
):
    """
    Find users directly connected to the target user
    through shared device or IP infrastructure.
    """

    graph = build_user_relationship_graph(
        df
    )

    # User is not present in graph
    if target_user not in graph:

        return []

    related_users = list(
        graph.neighbors(
            target_user
        )
    )

    return related_users


# ============================================================
# FIND FRAUD RING
# ============================================================

def find_user_ring(
    df,
    target_user,
):
    """
    Identify the fraud ring containing a user.

    Ring detection is based on connected components
    in the user relationship graph.

    A component is considered a meaningful ring when
    it contains at least 5 users.
    """

    graph = build_user_relationship_graph(
        df
    )

    # --------------------------------------------------------
    # User does not exist in graph
    # --------------------------------------------------------

    if target_user not in graph:

        return None


    # --------------------------------------------------------
    # Find all connected components
    # --------------------------------------------------------

    components = list(
        nx.connected_components(
            graph
        )
    )


    # --------------------------------------------------------
    # Find component containing target user
    # --------------------------------------------------------

    target_component = None

    for component in components:

        if target_user in component:

            target_component = component

            break


    if target_component is None:

        return None


    component_users = list(
        target_component
    )


    # --------------------------------------------------------
    # Ignore very small networks
    # --------------------------------------------------------

    if len(component_users) < 5:

        return None


    # --------------------------------------------------------
    # Find all meaningful components
    # --------------------------------------------------------

    large_components = [
        component
        for component in components
        if len(component) >= 5
    ]


    # --------------------------------------------------------
    # Sort components deterministically
    #
    # This makes ring IDs stable:
    #
    # RING_001
    # RING_002
    # RING_003
    # RING_004
    # --------------------------------------------------------

    large_components.sort(
        key=lambda component: min(
            int(
                user.split("_")[1]
            )
            for user in component
        )
    )


    # --------------------------------------------------------
    # Determine ring number
    # --------------------------------------------------------

    ring_number = (
        large_components.index(
            target_component
        )
        + 1
    )


    ring_id = (
        f"RING_{ring_number:03d}"
    )


    # --------------------------------------------------------
    # Transactions belonging to this ring
    # --------------------------------------------------------

    ring_df = df[
        df["user_id"].isin(
            component_users
        )
    ].copy()


    # --------------------------------------------------------
    # Fraud transactions
    # --------------------------------------------------------

    fraud_transactions = ring_df[
        ring_df["is_fraud"] == 1
    ]


    # --------------------------------------------------------
    # Ring statistics
    # --------------------------------------------------------

    total_transactions = len(
        ring_df
    )


    fraud_count = len(
        fraud_transactions
    )


    fraud_ratio = (

        fraud_count
        / total_transactions
        * 100

        if total_transactions > 0

        else 0
    )


    shared_devices = (
        ring_df["device_id"]
        .nunique()
    )


    shared_ips = (
        ring_df["ip_address"]
        .nunique()
    )


    shared_merchants = (
        fraud_transactions[
            "merchant_id"
        ]
        .nunique()
    )


    # --------------------------------------------------------
    # Return ring information
    # --------------------------------------------------------

    return {

        "ring_id":
            ring_id,

        "ring_user_count":
            len(component_users),

        "ring_shared_devices":
            shared_devices,

        "ring_shared_ips":
            shared_ips,

        "ring_shared_merchants":
            shared_merchants,

        "ring_transactions":
            total_transactions,

        "ring_fraud_transactions":
            fraud_count,

        "ring_fraud_ratio":
            round(
                fraud_ratio,
                2,
            ),
    }


# ============================================================
# COLLECT EVIDENCE
# ============================================================

def collect_evidence(
    transaction_id,
):
    """
    Collect all available evidence for a transaction.

    Evidence sources:

    1. Machine-learning risk score
    2. User behaviour
    3. Device sharing
    4. IP sharing
    5. User relationship graph
    6. Fraud ring
    7. Merchant history
    8. Ground truth for offline evaluation
    """

    # --------------------------------------------------------
    # Load resources
    # --------------------------------------------------------

    df = load_data()

    model = load_model()


    # --------------------------------------------------------
    # Find transaction
    # --------------------------------------------------------

    matches = df[
        df["transaction_id"]
        == transaction_id
    ]


    if matches.empty:

        raise ValueError(
            f"Transaction {transaction_id} "
            "was not found."
        )


    transaction = matches.iloc[0]


    # --------------------------------------------------------
    # ML risk score
    # --------------------------------------------------------

    feature_df = create_features(
        df
    )


    transaction_features = (
        feature_df[
            feature_df["transaction_id"]
            == transaction_id
        ][FEATURE_COLUMNS]
    )


    probability = model.predict_proba(
        transaction_features
    )[0][1]


    risk_score = round(
        probability * 100,
        2,
    )


    # --------------------------------------------------------
    # User information
    # --------------------------------------------------------

    user_id = transaction[
        "user_id"
    ]


    user_transactions = df[
        df["user_id"]
        == user_id
    ]


    # --------------------------------------------------------
    # User behaviour
    # --------------------------------------------------------

    user_transaction_count = len(
        user_transactions
    )


    user_fraud_count = int(
        user_transactions[
            "is_fraud"
        ].sum()
    )


    # --------------------------------------------------------
    # Shared device users
    # --------------------------------------------------------

    device_users = df[
        df["device_id"]
        == transaction["device_id"]
    ]["user_id"].nunique()


    # --------------------------------------------------------
    # Shared IP users
    # --------------------------------------------------------

    ip_users = df[
        df["ip_address"]
        == transaction["ip_address"]
    ]["user_id"].nunique()


    # --------------------------------------------------------
    # Related users
    # --------------------------------------------------------

    related_users = find_related_users(
        df,
        user_id,
    )


    # --------------------------------------------------------
    # Related fraud activity
    # --------------------------------------------------------

    if related_users:

        related_transactions = df[
            df["user_id"].isin(
                related_users
            )
        ]


        related_fraud_count = int(
            related_transactions[
                "is_fraud"
            ].sum()
        )

    else:

        related_fraud_count = 0


    # --------------------------------------------------------
    # Merchant activity
    # --------------------------------------------------------

    merchant_id = transaction[
        "merchant_id"
    ]


    merchant_transactions = df[
        df["merchant_id"]
        == merchant_id
    ]


    merchant_fraud_count = int(
        merchant_transactions[
            "is_fraud"
        ].sum()
    )


    # --------------------------------------------------------
    # Fraud ring
    # --------------------------------------------------------

    ring = find_user_ring(
        df,
        user_id,
    )


    # ========================================================
    # EVIDENCE OBJECT
    # ========================================================

    evidence = {

        # ----------------------------------------------------
        # Transaction
        # ----------------------------------------------------

        "transaction_id":
            transaction_id,

        "user_id":
            user_id,

        "merchant_id":
            merchant_id,

        "amount":
            float(
                transaction["amount"]
            ),

        "device_id":
            transaction["device_id"],

        "ip_address":
            transaction["ip_address"],


        # ----------------------------------------------------
        # ML risk
        # ----------------------------------------------------

        "risk_score":
            risk_score,


        # ----------------------------------------------------
        # Account
        # ----------------------------------------------------

        "account_age_days":
            int(
                transaction[
                    "account_age_days"
                ]
            ),

        "failed_attempts":
            int(
                transaction[
                    "failed_attempts"
                ]
            ),

        "refund_count":
            int(
                transaction[
                    "refund_count"
                ]
            ),


        # ----------------------------------------------------
        # User behaviour
        # ----------------------------------------------------

        "user_transaction_count":
            user_transaction_count,

        "user_fraud_count":
            user_fraud_count,


        # ----------------------------------------------------
        # Infrastructure
        # ----------------------------------------------------

        "shared_device_users":
            int(
                device_users
            ),

        "shared_ip_users":
            int(
                ip_users
            ),


        # ----------------------------------------------------
        # Relationship network
        # ----------------------------------------------------

        "related_users":
            related_users,

        "related_user_count":
            len(
                related_users
            ),

        "related_fraud_count":
            related_fraud_count,


        # ----------------------------------------------------
        # Merchant
        # ----------------------------------------------------

        "merchant_fraud_count":
            merchant_fraud_count,


        # ----------------------------------------------------
        # Fraud ring
        # ----------------------------------------------------

        "fraud_ring_detected":
            ring is not None,

        "fraud_ring_id":
            (
                ring["ring_id"]
                if ring
                else None
            ),

        "ring_user_count":
            (
                ring["ring_user_count"]
                if ring
                else 0
            ),

        "ring_shared_devices":
            (
                ring["ring_shared_devices"]
                if ring
                else 0
            ),

        "ring_shared_ips":
            (
                ring["ring_shared_ips"]
                if ring
                else 0
            ),

        "ring_shared_merchants":
            (
                ring["ring_shared_merchants"]
                if ring
                else 0
            ),

        "ring_transactions":
            (
                ring["ring_transactions"]
                if ring
                else 0
            ),

        "ring_fraud_transactions":
            (
                ring["ring_fraud_transactions"]
                if ring
                else 0
            ),

        "ring_fraud_ratio":
            (
                ring["ring_fraud_ratio"]
                if ring
                else 0
            ),


        # ----------------------------------------------------
        # Ground truth
        #
        # ONLY used for offline evaluation/demo validation.
        # ----------------------------------------------------

        "ground_truth":
            int(
                transaction[
                    "is_fraud"
                ]
            ),
    }


    return evidence


# ============================================================
# POLICY ENGINE
# ============================================================

def make_decision(
    evidence,
):
    """
    Convert evidence into a bounded policy action.

    Possible actions:

    APPROVE
    ADDITIONAL VERIFICATION
    REVIEW
    HOLD
    """

    risk = float(
        evidence["risk_score"]
    )


    related_users = evidence[
        "related_user_count"
    ]


    shared_devices = evidence[
        "shared_device_users"
    ]


    shared_ips = evidence[
        "shared_ip_users"
    ]


    related_fraud = evidence[
        "related_fraud_count"
    ]


    ring_detected = evidence[
        "fraud_ring_detected"
    ]


    ring_fraud_count = evidence[
        "ring_fraud_transactions"
    ]


    # ========================================================
    # VERY STRONG FRAUD RING SIGNAL
    # ========================================================

    if (

        ring_detected

        and risk >= 80

        and ring_fraud_count >= 10

    ):

        return {

            "decision":
                "HOLD",

            "severity":
                "HIGH",

            "reason":
                (
                    "High transaction risk combined "
                    "with confirmed coordinated fraud-ring "
                    "evidence."
                ),
        }


    # ========================================================
    # STRONG COORDINATED RISK
    # ========================================================

    if (

        risk >= 80

        and related_users >= 5

        and shared_devices >= 2

        and shared_ips >= 2

    ):

        return {

            "decision":
                "HOLD",

            "severity":
                "HIGH",

            "reason":
                (
                    "High transaction risk combined "
                    "with strong shared-infrastructure "
                    "evidence."
                ),
        }


    # ========================================================
    # NETWORK SUSPICIOUS
    # ========================================================

    if (

        risk >= 60

        and related_users >= 3

        and related_fraud >= 2

    ):

        return {

            "decision":
                "REVIEW",

            "severity":
                "MEDIUM",

            "reason":
                (
                    "Transaction risk is elevated and "
                    "the associated user network contains "
                    "suspicious activity."
                ),
        }


    # ========================================================
    # HIGH TRANSACTION RISK
    # ========================================================

    if risk >= 80:

        return {

            "decision":
                "REVIEW",

            "severity":
                "HIGH",

            "reason":
                "High transaction-level risk detected.",
        }


    # ========================================================
    # MODERATE RISK
    # ========================================================

    if risk >= 40:

        return {

            "decision":
                "ADDITIONAL VERIFICATION",

            "severity":
                "MEDIUM",

            "reason":
                "Moderate transaction risk detected.",
        }


    # ========================================================
    # LOW RISK
    # ========================================================

    return {

        "decision":
            "APPROVE",

        "severity":
            "LOW",

        "reason":
            "No significant risk signals detected.",
    }


# ============================================================
# HUMAN-READABLE AI INVESTIGATION
# ============================================================

def generate_investigation(
    evidence,
    decision,
):
    """
    Convert structured evidence into a human-readable
    investigation report.

    This is intentionally evidence-grounded.
    The explanation only uses signals actually calculated
    by the detection pipeline.
    """

    risk = float(
        evidence["risk_score"]
    )


    reasons = []


    # ========================================================
    # ML EVIDENCE
    # ========================================================

    if risk >= 80:

        reasons.append(
            f"Machine-learning risk score is "
            f"{risk:.2f}/100."
        )


    elif risk >= 40:

        reasons.append(
            f"Machine-learning risk score is "
            f"moderately elevated at "
            f"{risk:.2f}/100."
        )


    # ========================================================
    # FRAUD RING EVIDENCE
    # ========================================================

    if evidence[
        "fraud_ring_detected"
    ]:

        reasons.append(
            f"The transaction belongs to "
            f"{evidence['fraud_ring_id']}, "
            f"a coordinated network containing "
            f"{evidence['ring_user_count']} users."
        )


        reasons.append(
            f"The ring shares "
            f"{evidence['ring_shared_devices']} "
            f"devices and "
            f"{evidence['ring_shared_ips']} "
            f"IP addresses."
        )


        reasons.append(
            f"The ring contains "
            f"{evidence['ring_fraud_transactions']} "
            f"fraudulent transactions out of "
            f"{evidence['ring_transactions']} "
            f"total transactions."
        )


        reasons.append(
            f"The ring's observed fraud ratio is "
            f"{evidence['ring_fraud_ratio']:.2f}%."
        )


    # ========================================================
    # DEVICE EVIDENCE
    # ========================================================

    if evidence[
        "shared_device_users"
    ] >= 2:

        reasons.append(
            f"The transaction's device is associated "
            f"with {evidence['shared_device_users']} "
            f"users."
        )


    # ========================================================
    # IP EVIDENCE
    # ========================================================

    if evidence[
        "shared_ip_users"
    ] >= 2:

        reasons.append(
            f"The transaction's IP address is associated "
            f"with {evidence['shared_ip_users']} "
            f"users."
        )


    # ========================================================
    # NETWORK EVIDENCE
    # ========================================================

    if evidence[
        "related_user_count"
    ] >= 3:

        reasons.append(
            f"The account is connected to "
            f"{evidence['related_user_count']} "
            f"other users through shared infrastructure."
        )


    # ========================================================
    # HISTORICAL FRAUD EVIDENCE
    # ========================================================

    if evidence[
        "related_fraud_count"
    ] >= 2:

        reasons.append(
            f"The connected network contains "
            f"{evidence['related_fraud_count']} "
            f"previously labelled fraudulent "
            f"transactions."
        )


    # ========================================================
    # ACCOUNT AGE
    # ========================================================

    if evidence[
        "account_age_days"
    ] < 30:

        reasons.append(
            "The account is less than 30 days old."
        )


    # ========================================================
    # FAILED ATTEMPTS
    # ========================================================

    if evidence[
        "failed_attempts"
    ] >= 3:

        reasons.append(
            f"{evidence['failed_attempts']} "
            f"failed payment attempts were observed."
        )


    # ========================================================
    # REFUND BEHAVIOUR
    # ========================================================

    if evidence[
        "refund_count"
    ] >= 2:

        reasons.append(
            f"The account has "
            f"{evidence['refund_count']} "
            f"recorded refunds."
        )


    # ========================================================
    # NO SIGNIFICANT EVIDENCE
    # ========================================================

    if not reasons:

        reasons.append(
            "No significant risk signals were detected."
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    if evidence[
        "fraud_ring_detected"
    ]:

        summary = (
            f"{evidence['fraud_ring_id']} detected with "
            f"{evidence['ring_user_count']} connected users, "
            f"{evidence['ring_shared_devices']} shared devices, "
            f"{evidence['ring_shared_ips']} shared IPs, and "
            f"{evidence['ring_fraud_transactions']} "
            f"fraudulent transactions. "
            f"Transaction risk is "
            f"{risk:.2f}/100. "
            f"Recommended action: "
            f"{decision['decision']}."
        )

    else:

        summary = (
            decision["reason"]
        )


    return {

        "decision":
            decision["decision"],

        "severity":
            decision["severity"],

        "summary":
            summary,

        "evidence":
            reasons,
    }


# ============================================================
# COMPLETE INVESTIGATION
# ============================================================

def investigate(
    transaction_id,
):
    """
    Run the complete RISKWEAVE investigation pipeline.
    """

    evidence = collect_evidence(
        transaction_id
    )


    decision = make_decision(
        evidence
    )


    investigation = generate_investigation(
        evidence,
        decision
    )


    return {

        "evidence":
            evidence,

        "decision":
            decision,

        "investigation":
            investigation,
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "========================================"
    )

    print(
        "       RISKWEAVE AI INVESTIGATOR"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()


    # --------------------------------------------------------
    # Test first transaction
    # --------------------------------------------------------

    transaction_id = (
        df.iloc[0]["transaction_id"]
    )


    print()

    print(
        f"Investigating: "
        f"{transaction_id}"
    )


    # --------------------------------------------------------
    # Run investigation
    # --------------------------------------------------------

    result = investigate(
        transaction_id
    )


    evidence = result[
        "evidence"
    ]


    investigation = result[
        "investigation"
    ]


    # ========================================================
    # TRANSACTION
    # ========================================================

    print()

    print(
        "TRANSACTION"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"ID: "
        f"{evidence['transaction_id']}"
    )

    print(
        f"User: "
        f"{evidence['user_id']}"
    )

    print(
        f"Amount: "
        f"Rs.{evidence['amount']:.2f}"
    )


    # ========================================================
    # RISK
    # ========================================================

    print()

    print(
        "RISK"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Risk Score: "
        f"{evidence['risk_score']:.2f}/100"
    )


    # ========================================================
    # NETWORK
    # ========================================================

    print()

    print(
        "NETWORK"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Related users: "
        f"{evidence['related_user_count']}"
    )

    print(
        f"Shared device users: "
        f"{evidence['shared_device_users']}"
    )

    print(
        f"Shared IP users: "
        f"{evidence['shared_ip_users']}"
    )

    print(
        f"Related fraud transactions: "
        f"{evidence['related_fraud_count']}"
    )


    # ========================================================
    # FRAUD RING
    # ========================================================

    print()

    print(
        "FRAUD RING"
    )

    print(
        "----------------------------------------"
    )


    if evidence[
        "fraud_ring_detected"
    ]:

        print(
            f"Ring ID: "
            f"{evidence['fraud_ring_id']}"
        )

        print(
            f"Ring users: "
            f"{evidence['ring_user_count']}"
        )

        print(
            f"Shared devices: "
            f"{evidence['ring_shared_devices']}"
        )

        print(
            f"Shared IPs: "
            f"{evidence['ring_shared_ips']}"
        )

        print(
            f"Ring transactions: "
            f"{evidence['ring_transactions']}"
        )

        print(
            f"Ring fraud transactions: "
            f"{evidence['ring_fraud_transactions']}"
        )

        print(
            f"Ring fraud ratio: "
            f"{evidence['ring_fraud_ratio']:.2f}%"
        )

    else:

        print(
            "No significant fraud ring detected."
        )


    # ========================================================
    # DECISION
    # ========================================================

    print()

    print(
        "DECISION"
    )

    print(
        "----------------------------------------"
    )

    print(
        investigation["decision"]
    )


    # ========================================================
    # AI INVESTIGATION
    # ========================================================

    print()

    print(
        "AI INVESTIGATION"
    )

    print(
        "----------------------------------------"
    )


    for reason in investigation[
        "evidence"
    ]:

        print(
            f"- {reason}"
        )


    print()

    print(
        f"Summary: "
        f"{investigation['summary']}"
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "       INVESTIGATION COMPLETE"
    )

    print(
        "========================================"
    )