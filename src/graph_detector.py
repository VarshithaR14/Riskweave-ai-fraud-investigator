import pandas as pd
import networkx as nx


# ============================================================
# RISKWEAVE — FRAUD RING GRAPH ENGINE
# ============================================================


def build_user_relationship_graph(df):
    """
    Build a user-to-user relationship graph.

    Users are connected when they share:
    - Device
    - IP address

    Merchants are deliberately NOT used as graph edges because
    legitimate customers naturally share merchants.
    """

    graph = nx.Graph()

    # --------------------------------------------------------
    # Add all users
    # --------------------------------------------------------

    for user in df["user_id"].unique():

        graph.add_node(
            user,
            node_type="user"
        )

    # --------------------------------------------------------
    # Shared DEVICE relationships
    # --------------------------------------------------------

    device_groups = (
        df.groupby("device_id")["user_id"]
        .unique()
    )

    for device_id, user_list in device_groups.items():

        user_list = list(user_list)

        if len(user_list) < 2:
            continue

        for i in range(len(user_list)):

            for j in range(i + 1, len(user_list)):

                user_a = user_list[i]
                user_b = user_list[j]

                if graph.has_edge(
                    user_a,
                    user_b
                ):

                    graph[user_a][user_b][
                        "shared_devices"
                    ] += 1

                else:

                    graph.add_edge(
                        user_a,
                        user_b,
                        shared_devices=1,
                        shared_ips=0
                    )

    # --------------------------------------------------------
    # Shared IP relationships
    # --------------------------------------------------------

    ip_groups = (
        df.groupby("ip_address")["user_id"]
        .unique()
    )

    for ip_address, user_list in ip_groups.items():

        user_list = list(user_list)

        if len(user_list) < 2:
            continue

        for i in range(len(user_list)):

            for j in range(i + 1, len(user_list)):

                user_a = user_list[i]
                user_b = user_list[j]

                if graph.has_edge(
                    user_a,
                    user_b
                ):

                    graph[user_a][user_b][
                        "shared_ips"
                    ] += 1

                else:

                    graph.add_edge(
                        user_a,
                        user_b,
                        shared_devices=0,
                        shared_ips=1
                    )

    return graph


# ============================================================
# FIND SHARED INFRASTRUCTURE
# ============================================================


def get_shared_infrastructure(
    df,
    user_id
):

    user_data = df[
        df["user_id"] == user_id
    ]

    if user_data.empty:

        return {
            "shared_devices": [],
            "shared_ips": [],
            "shared_merchants": []
        }

    # --------------------------------------------------------
    # Devices
    # --------------------------------------------------------

    shared_devices = []

    for device in user_data["device_id"].unique():

        users = df[
            df["device_id"] == device
        ]["user_id"].unique()

        if len(users) > 1:

            shared_devices.append({
                "device_id": device,
                "users": list(users)
            })

    # --------------------------------------------------------
    # IPs
    # --------------------------------------------------------

    shared_ips = []

    for ip in user_data["ip_address"].unique():

        users = df[
            df["ip_address"] == ip
        ]["user_id"].unique()

        if len(users) > 1:

            shared_ips.append({
                "ip_address": ip,
                "users": list(users)
            })

    # --------------------------------------------------------
    # Merchants
    # --------------------------------------------------------

    shared_merchants = []

    for merchant in user_data["merchant_id"].unique():

        users = df[
            df["merchant_id"] == merchant
        ]["user_id"].unique()

        if len(users) > 1:

            shared_merchants.append({
                "merchant_id": merchant,
                "users": list(users)
            })

    return {
        "shared_devices": shared_devices,
        "shared_ips": shared_ips,
        "shared_merchants": shared_merchants
    }


# ============================================================
# FRAUD RING DETECTION
# ============================================================


def detect_fraud_rings(df):

    graph = build_user_relationship_graph(
        df
    )

    components = nx.connected_components(
        graph
    )

    rings = []

    ring_number = 1

    for component in components:

        users = list(component)

        # Ignore tiny groups
        if len(users) < 3:
            continue

        ring_df = df[
            df["user_id"].isin(users)
        ]

        # ----------------------------------------------------
        # Fraud activity
        # ----------------------------------------------------

        fraud_count = int(
            ring_df["is_fraud"].sum()
        )

        total_transactions = len(
            ring_df
        )

        fraud_ratio = (
            fraud_count / total_transactions
            if total_transactions > 0
            else 0
        )

        # ----------------------------------------------------
        # Shared devices
        # ----------------------------------------------------

        device_counts = (
            ring_df.groupby(
                "device_id"
            )["user_id"]
            .nunique()
        )

        shared_device_count = int(
            (
                device_counts >= 2
            ).sum()
        )

        # ----------------------------------------------------
        # Shared IPs
        # ----------------------------------------------------

        ip_counts = (
            ring_df.groupby(
                "ip_address"
            )["user_id"]
            .nunique()
        )

        shared_ip_count = int(
            (
                ip_counts >= 2
            ).sum()
        )

        # ----------------------------------------------------
        # Shared merchants — evidence only
        # ----------------------------------------------------

        merchant_counts = (
            ring_df.groupby(
                "merchant_id"
            )["user_id"]
            .nunique()
        )

        shared_merchant_count = int(
            (
                merchant_counts >= 2
            ).sum()
        )

        # ----------------------------------------------------
        # Ring score
        # ----------------------------------------------------

        score = 0

        # Multiple users
        if len(users) >= 3:
            score += 20

        # Larger coordinated network
        if len(users) >= 10:
            score += 20

        # Shared infrastructure
        if shared_device_count >= 1:
            score += 20

        if shared_ip_count >= 1:
            score += 20

        # Historical fraud
        if fraud_count >= 5:
            score += 20

        score = min(
            score,
            100
        )

        rings.append({

            "ring_id":
                f"RING_{ring_number:03d}",

            "users":
                users,

            "user_count":
                len(users),

            "shared_devices":
                shared_device_count,

            "shared_ips":
                shared_ip_count,

            "shared_merchants":
                shared_merchant_count,

            "transactions":
                total_transactions,

            "fraud_transactions":
                fraud_count,

            "fraud_ratio":
                round(
                    fraud_ratio * 100,
                    2
                ),

            "ring_score":
                score
        })

        ring_number += 1

    result = pd.DataFrame(
        rings
    )

    if result.empty:

        return result

    return result.sort_values(
        "ring_score",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# MAIN TEST
# ============================================================


if __name__ == "__main__":

    print()
    print("========================================")
    print("     RISKWEAVE FRAUD GRAPH ENGINE")
    print("========================================")

    df = pd.read_csv(
        "data/raw/transactions.csv"
    )

    print(
        f"\nLoaded {len(df)} transactions."
    )

    graph = build_user_relationship_graph(
        df
    )

    print(
        f"User nodes : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"User edges : "
        f"{graph.number_of_edges()}"
    )

    rings = detect_fraud_rings(
        df
    )

    print()

    if rings.empty:

        print(
            "No potential fraud rings detected."
        )

    else:

        print(
            f"Potential rings detected: "
            f"{len(rings)}"
        )

        print()

        print(
            rings[
                [
                    "ring_id",
                    "user_count",
                    "shared_devices",
                    "shared_ips",
                    "shared_merchants",
                    "transactions",
                    "fraud_transactions",
                    "fraud_ratio",
                    "ring_score"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("========================================")
    print("       GRAPH ANALYSIS COMPLETE")
    print("========================================")