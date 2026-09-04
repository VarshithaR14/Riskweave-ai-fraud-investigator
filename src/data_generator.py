import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# RISKWEAVE - CONTROLLED FRAUD RING DATASET GENERATOR
# ============================================================

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "transactions.csv"
)

NUM_TRANSACTIONS = 6000
NUM_USERS = 1500
NUM_MERCHANTS = 150

FRAUD_RATE = 0.12


# ------------------------------------------------------------
# USER POOLS
# ------------------------------------------------------------

all_users = [
    f"USER_{i:04d}"
    for i in range(1, NUM_USERS + 1)
]

# Four controlled fraud rings
RINGS = {
    "RING_A": [f"USER_{i:04d}" for i in range(1, 31)],
    "RING_B": [f"USER_{i:04d}" for i in range(31, 61)],
    "RING_C": [f"USER_{i:04d}" for i in range(61, 91)],
    "RING_D": [f"USER_{i:04d}" for i in range(91, 121)],
}

ring_users = set()

for users in RINGS.values():
    ring_users.update(users)

# Normal users are completely separated from fraud-ring users
normal_users = [
    user for user in all_users
    if user not in ring_users
]


# ------------------------------------------------------------
# MERCHANTS
# ------------------------------------------------------------

merchants = [
    f"MERCHANT_{i:03d}"
    for i in range(1, NUM_MERCHANTS + 1)
]


# ------------------------------------------------------------
# RING INFRASTRUCTURE
# ------------------------------------------------------------

RING_INFRASTRUCTURE = {
    "RING_A": {
        "devices": ["DEVICE_RING_A_01", "DEVICE_RING_A_02"],
        "ips": ["IP_RING_A_01", "IP_RING_A_02"],
        "merchants": [
            "MERCHANT_001",
            "MERCHANT_002",
            "MERCHANT_003",
        ],
    },

    "RING_B": {
        "devices": ["DEVICE_RING_B_01", "DEVICE_RING_B_02"],
        "ips": ["IP_RING_B_01", "IP_RING_B_02"],
        "merchants": [
            "MERCHANT_004",
            "MERCHANT_005",
            "MERCHANT_006",
        ],
    },

    "RING_C": {
        "devices": ["DEVICE_RING_C_01", "DEVICE_RING_C_02"],
        "ips": ["IP_RING_C_01", "IP_RING_C_02"],
        "merchants": [
            "MERCHANT_007",
            "MERCHANT_008",
            "MERCHANT_009",
        ],
    },

    "RING_D": {
        "devices": ["DEVICE_RING_D_01", "DEVICE_RING_D_02"],
        "ips": ["IP_RING_D_01", "IP_RING_D_02"],
        "merchants": [
            "MERCHANT_010",
            "MERCHANT_011",
            "MERCHANT_012",
        ],
    },
}


# ------------------------------------------------------------
# STABLE NORMAL INFRASTRUCTURE
# ------------------------------------------------------------

def normal_device(user_id):
    number = int(user_id.split("_")[1])
    return f"DEVICE_{number:04d}_A"


def normal_ip(user_id):
    number = int(user_id.split("_")[1])
    return f"IP_{number:04d}_A"


# ------------------------------------------------------------
# TRANSACTION GENERATION
# ------------------------------------------------------------

start_time = datetime.now() - timedelta(days=90)

rows = []

fraud_transactions_target = int(NUM_TRANSACTIONS * FRAUD_RATE)

# Exactly controlled fraud count
fraud_indices = set(
    random.sample(
        range(NUM_TRANSACTIONS),
        fraud_transactions_target
    )
)


for i in range(NUM_TRANSACTIONS):

    transaction_id = f"TXN_{i + 1:06d}"

    timestamp = start_time + timedelta(
        minutes=random.randint(0, 90 * 24 * 60)
    )

    # ========================================================
    # FRAUD TRANSACTION
    # ========================================================

    if i in fraud_indices:

        ring_name = random.choice(list(RINGS.keys()))

        user_id = random.choice(RINGS[ring_name])

        infrastructure = RING_INFRASTRUCTURE[ring_name]

        device_id = random.choice(
            infrastructure["devices"]
        )

        ip_address = random.choice(
            infrastructure["ips"]
        )

        merchant_id = random.choice(
            infrastructure["merchants"]
        )

        # Fraudulent behavior
        amount = round(
            np.random.lognormal(
                mean=np.log(2500),
                sigma=1.0
            ),
            2
        )

        amount = min(amount, 25000)

        account_age_days = random.randint(1, 90)

        failed_attempts = random.randint(2, 8)

        refund_count = random.randint(1, 5)

        is_fraud = 1

    # ========================================================
    # NORMAL TRANSACTION
    # ========================================================

    else:

        # IMPORTANT:
        # Normal transactions ONLY use normal users.
        # Fraud-ring users never appear in normal transactions.
        user_id = random.choice(normal_users)

        merchant_id = random.choice(merchants)

        device_id = normal_device(user_id)

        ip_address = normal_ip(user_id)

        amount = round(
            np.random.lognormal(
                mean=np.log(600),
                sigma=0.75
            ),
            2
        )

        amount = min(amount, 10000)

        account_age_days = random.randint(
            30,
            1500
        )

        failed_attempts = random.choices(
            [0, 1, 2],
            weights=[0.70, 0.22, 0.08]
        )[0]

        refund_count = random.choices(
            [0, 1, 2],
            weights=[0.80, 0.15, 0.05]
        )[0]

        is_fraud = 0


    rows.append({
        "transaction_id": transaction_id,
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "timestamp": timestamp,
        "device_id": device_id,
        "ip_address": ip_address,
        "account_age_days": account_age_days,
        "failed_attempts": failed_attempts,
        "refund_count": refund_count,
        "is_fraud": is_fraud,
    })


# ------------------------------------------------------------
# DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(rows)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print()
print("=" * 60)
print("RISKWEAVE DATASET GENERATED")
print("=" * 60)

print(f"Transactions : {len(df)}")
print(f"Users        : {df['user_id'].nunique()}")
print(f"Merchants    : {df['merchant_id'].nunique()}")

print()
print("Fraud distribution:")
print(df["is_fraud"].value_counts())

print()
print("Fraud percentage:")
print(
    round(df["is_fraud"].mean() * 100, 2),
    "%"
)

print()
print("Fraud infrastructure:")
print(
    df[df["is_fraud"] == 1][
        ["user_id", "device_id", "ip_address"]
    ].head(10).to_string(index=False)
)

print()
print("Ring validation:")

for ring_name, users in RINGS.items():

    ring_df = df[
        (df["is_fraud"] == 1) &
        (df["user_id"].isin(users))
    ]

    print(
        f"{ring_name}: "
        f"{ring_df['user_id'].nunique()} users | "
        f"{len(ring_df)} fraud transactions"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")
print("=" * 60)