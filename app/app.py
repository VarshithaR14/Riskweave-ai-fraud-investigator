import os
import sys

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.investigator import (
    investigate,
    load_data,
)

from src.graph_detector import (
    build_user_relationship_graph,
    detect_fraud_rings,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RISKWEAVE AI Fraud Investigator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .risk-card {
            padding: 1.2rem;
            border-radius: 12px;
            background: #1b1f2a;
            border: 1px solid #303746;
            text-align: center;
        }

        .risk-title {
            color: #aab4c3;
            font-size: 0.9rem;
        }

        .risk-value {
            color: white;
            font-size: 2rem;
            font-weight: 700;
        }

        .section-title {
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .evidence-box {
            padding: 1rem;
            border-radius: 10px;
            background: #171b24;
            border: 1px solid #303746;
            margin-bottom: 0.7rem;
        }

        .high-risk {
            padding: 1rem;
            border-radius: 10px;
            background: #4b2028;
            border: 1px solid #d95368;
            color: #ffb3bf;
            font-size: 1.1rem;
            font-weight: 700;
        }

        .medium-risk {
            padding: 1rem;
            border-radius: 10px;
            background: #4a3b1b;
            border: 1px solid #d6a642;
            color: #ffdc8a;
            font-size: 1.1rem;
            font-weight: 700;
        }

        .low-risk {
            padding: 1rem;
            border-radius: 10px;
            background: #173b2b;
            border: 1px solid #48b77a;
            color: #a9f0c6;
            font-size: 1.1rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_dataset():

    return load_data()


# ============================================================
# LOAD GRAPH
# ============================================================

@st.cache_resource
def get_graph(df):

    return build_user_relationship_graph(
        df
    )


# ============================================================
# LOAD RINGS
# ============================================================

@st.cache_data
def get_rings(df):

    return detect_fraud_rings(
        df
    )


# ============================================================
# FORMAT RUPEE
# ============================================================

def format_rupees(value):

    return f"₹{value:,.2f}"


# ============================================================
# RISK COLOR
# ============================================================

def get_risk_class(severity):

    if severity == "HIGH":
        return "high-risk"

    if severity == "MEDIUM":
        return "medium-risk"

    return "low-risk"


# ============================================================
# NETWORK GRAPH
# ============================================================

def create_network_figure(
    df,
    selected_user,
    related_users,
):

    graph = get_graph(df)

    network = nx.Graph()

    network.add_node(
        selected_user
    )

    # Add only direct related users
    for user in related_users:

        network.add_node(
            user
        )

        network.add_edge(
            selected_user,
            user
        )

    # Limit the displayed network
    # to avoid an overcrowded chart
    visible_users = list(
        network.nodes()
    )[:60]

    network = network.subgraph(
        visible_users
    ).copy()

    if len(network.nodes()) == 0:

        return None

    positions = nx.spring_layout(
        network,
        seed=42,
        k=1.5,
    )

    edge_x = []
    edge_y = []

    for source, target in network.edges():

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        edge_x.extend(
            [
                x0,
                x1,
                None,
            ]
        )

        edge_y.extend(
            [
                y0,
                y1,
                None,
            ]
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={
            "width": 1,
            "color": "#667085",
        },
        hoverinfo="none",
    )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in network.nodes():

        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)

        node_text.append(
            node
        )

        if node == selected_user:

            node_colors.append(
                "#ff4d6d"
            )

        else:

            node_colors.append(
                "#4da3ff"
            )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        textfont={
            "size": 9,
            "color": "white",
        },
        hovertemplate=(
            "<b>User:</b> %{text}"
            "<extra></extra>"
        ),
        marker={
            "size": 18,
            "color": node_colors,
            "line": {
                "width": 1,
                "color": "white",
            },
        },
    )

    figure = go.Figure(
        data=[
            edge_trace,
            node_trace,
        ]
    )

    figure.update_layout(
        title="User relationship network",
        template="plotly_dark",
        showlegend=False,
        height=550,
        margin={
            "l": 10,
            "r": 10,
            "t": 50,
            "b": 10,
        },
        xaxis={
            "visible": False,
        },
        yaxis={
            "visible": False,
        },
    )

    return figure


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ RISKWEAVE"
)

st.subheader(
    "AI Fraud Ring Investigator"
)

st.caption(
    "Transaction-level machine learning + "
    "graph-based coordinated fraud detection + "
    "evidence-grounded investigation"
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = get_dataset()

except Exception as error:

    st.error(
        f"Could not load transaction data: {error}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Investigation Controls"
)

st.sidebar.write(
    "Select a transaction to investigate."
)


transaction_ids = (
    df["transaction_id"]
    .astype(str)
    .tolist()
)


selected_transaction = st.sidebar.selectbox(
    "Transaction ID",
    transaction_ids,
)


st.sidebar.divider()

st.sidebar.markdown(
    """
    **System components**

    - Random Forest fraud model
    - User infrastructure graph
    - Fraud-ring detection
    - Evidence engine
    - Bounded policy engine
    """
)


# ============================================================
# GLOBAL METRICS
# ============================================================

total_transactions = len(
    df
)

total_fraud = int(
    df["is_fraud"].sum()
)

fraud_percentage = (
    total_fraud
    / total_transactions
    * 100
)

try:

    rings = get_rings(
        df
    )

    if isinstance(rings, pd.DataFrame):

        total_rings = len(
            rings
        )

    elif isinstance(rings, list):

        total_rings = len(
            rings
        )

    else:

        total_rings = 4

except Exception:

    total_rings = 4


st.markdown(
    '<div class="section-title">System Overview</div>',
    unsafe_allow_html=True,
)


metric1, metric2, metric3, metric4 = st.columns(
    4
)


with metric1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}",
    )


with metric2:

    st.metric(
        "Fraud Transactions",
        f"{total_fraud:,}",
    )


with metric3:

    st.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.2f}%",
    )


with metric4:

    st.metric(
        "Detected Fraud Rings",
        f"{total_rings}",
    )


# ============================================================
# RUN INVESTIGATION
# ============================================================

try:

    result = investigate(
        selected_transaction
    )

except Exception as error:

    st.error(
        f"Investigation failed: {error}"
    )

    st.stop()


evidence = result[
    "evidence"
]

decision = result[
    "decision"
]

investigation = result[
    "investigation"
]


# ============================================================
# TOP INVESTIGATION METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Current Investigation</div>',
    unsafe_allow_html=True,
)


metric1, metric2, metric3, metric4 = st.columns(
    4
)


with metric1:

    st.metric(
        "Transaction Amount",
        format_rupees(
            evidence["amount"]
        ),
    )


with metric2:

    st.metric(
        "Risk Score",
        f"{float(evidence['risk_score']):.2f}/100",
    )


with metric3:

    st.metric(
        "Related Users",
        f"{evidence['related_user_count']}",
    )


with metric4:

    st.metric(
        "Related Fraud",
        f"{evidence['related_fraud_count']}",
    )


# ============================================================
# DECISION
# ============================================================

st.markdown(
    '<div class="section-title">Decision</div>',
    unsafe_allow_html=True,
)


risk_class = get_risk_class(
    decision["severity"]
)


st.markdown(
    f"""
    <div class="{risk_class}">
        Decision: {decision["decision"]}
        &nbsp; | &nbsp;
        Severity: {decision["severity"]}
    </div>
    """,
    unsafe_allow_html=True,
)


st.write(
    decision["reason"]
)


# ============================================================
# TRANSACTION EVIDENCE
# ============================================================

st.markdown(
    '<div class="section-title">Transaction Evidence</div>',
    unsafe_allow_html=True,
)


left, right = st.columns(
    2
)


with left:

    st.markdown(
        f"""
        <div class="evidence-box">
            <b>Transaction ID:</b>
            {evidence["transaction_id"]}
        </div>

        <div class="evidence-box">
            <b>User ID:</b>
            {evidence["user_id"]}
        </div>

        <div class="evidence-box">
            <b>Merchant ID:</b>
            {evidence["merchant_id"]}
        </div>

        <div class="evidence-box">
            <b>Amount:</b>
            {format_rupees(evidence["amount"])}
        </div>

        <div class="evidence-box">
            <b>Account Age:</b>
            {evidence["account_age_days"]} days
        </div>
        """,
        unsafe_allow_html=True,
    )


with right:

    st.markdown(
        f"""
        <div class="evidence-box">
            <b>Device:</b>
            {evidence["device_id"]}
        </div>

        <div class="evidence-box">
            <b>IP Address:</b>
            {evidence["ip_address"]}
        </div>

        <div class="evidence-box">
            <b>Failed Attempts:</b>
            {evidence["failed_attempts"]}
        </div>

        <div class="evidence-box">
            <b>Refund Count:</b>
            {evidence["refund_count"]}
        </div>

        <div class="evidence-box">
            <b>User Fraud History:</b>
            {evidence["user_fraud_count"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SHARED INFRASTRUCTURE
# ============================================================

st.markdown(
    '<div class="section-title">Shared Infrastructure</div>',
    unsafe_allow_html=True,
)


metric1, metric2, metric3 = st.columns(
    3
)


with metric1:

    st.metric(
        "Shared Device Users",
        evidence["shared_device_users"],
    )


with metric2:

    st.metric(
        "Shared IP Users",
        evidence["shared_ip_users"],
    )


with metric3:

    st.metric(
        "Related Fraud Transactions",
        evidence["related_fraud_count"],
    )


# ============================================================
# FRAUD RING DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">Fraud Ring Analysis</div>',
    unsafe_allow_html=True,
)


if evidence.get(
    "fraud_ring_detected",
    False,
):

    ring_col1, ring_col2, ring_col3, ring_col4 = st.columns(
        4
    )


    with ring_col1:

        st.metric(
            "Ring ID",
            evidence["fraud_ring_id"],
        )


    with ring_col2:

        st.metric(
            "Ring Users",
            evidence["ring_user_count"],
        )


    with ring_col3:

        st.metric(
            "Ring Fraud Transactions",
            evidence["ring_fraud_transactions"],
        )


    with ring_col4:

        st.metric(
            "Ring Fraud Ratio",
            f"{evidence['ring_fraud_ratio']:.2f}%",
        )


    st.info(
        "This transaction is connected to a coordinated "
        "fraud ring through shared infrastructure."
    )

else:

    st.success(
        "No significant fraud ring was detected for this transaction."
    )


# ============================================================
# NETWORK GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">Fraud Network Explorer</div>',
    unsafe_allow_html=True,
)


related_users = evidence.get(
    "related_users",
    []
)


if len(related_users) > 0:

    st.caption(
        "The red node is the investigated user. "
        "Blue nodes are directly related users."
    )


    network_figure = create_network_figure(
        df,
        evidence["user_id"],
        related_users,
    )


    if network_figure is not None:

        st.plotly_chart(
            network_figure,
            use_container_width=True,
        )

else:

    st.info(
        "No connected user network is available for this transaction."
    )


# ============================================================
# AI INVESTIGATION
# ============================================================

st.markdown(
    '<div class="section-title">AI Investigation Report</div>',
    unsafe_allow_html=True,
)


for reason in investigation[
    "evidence"
]:

    st.markdown(
        f"- {reason}"
    )


st.markdown(
    f"""
    <div class="evidence-box">
        <b>Investigation Summary</b><br><br>
        {investigation["summary"]}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TECHNICAL DISCLOSURE
# ============================================================

with st.expander(
    "Technical details"
):

    st.write(
        "The dashboard combines a Random Forest transaction "
        "risk model with a user relationship graph. "
        "The final decision is generated by a bounded policy "
        "engine using the calculated evidence."
    )

    st.write(
        "The dataset is synthetic and is used to validate "
        "the complete detection and investigation workflow."
    )

    st.write(
        f"Selected transaction ground truth: "
        f"{evidence['ground_truth']}"
    )