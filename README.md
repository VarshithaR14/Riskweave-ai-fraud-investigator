# RISKWEAVE – AI Fraud Ring Investigator

RISKWEAVE is an AI-powered fraud investigation system created for the Razorpay AI Buildathon 2026 under the AI Risk Manager track.

## Problem

Traditional fraud detection systems often inspect transactions individually. Coordinated fraud can involve multiple accounts sharing devices, IP addresses, and merchant relationships.

## Solution

RISKWEAVE combines:

- Random Forest transaction-level fraud prediction
- User infrastructure relationship graphs
- Coordinated fraud-ring detection
- Evidence-based investigation reports
- Bounded policy decisions such as APPROVE and HOLD
- Streamlit dashboard

## Architecture

```text
Transaction Data
       |
       v
Feature Engineering
       |
       v
Random Forest Risk Model
       |
       v
Infrastructure Relationship Graph
       |
       v
Fraud-Ring Detection
       |
       v
Evidence-Based Investigation
       |
       v
APPROVE / HOLD
