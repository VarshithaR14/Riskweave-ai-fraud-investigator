
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
Technology Stack

Python

Pandas

NumPy

Scikit-learn

NetworkX

Streamlit

Plotly

Faker

Joblib

Run the Project

Create and activate the virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Generate the dataset:

python src\data_generator.py

Train the model:

python -m src.fraud_model

Detect fraud rings:

python src\graph_detector.py

Run evaluation:

python -m src.evaluation

Run tests:

python -m pip install pytest
python -m pytest -v

Start the dashboard:

python -m streamlit run app\app.py

Open the dashboard at:

http://localhost:8501
Example Result
Risk Score: 100/100
Decision: HOLD
Severity: HIGH
Fraud Ring: RING_001
Ring Users: 30
Evaluation

The model is evaluated using a held-out test set with:

Accuracy

Precision

Recall

F1-score

Confusion matrix

ROC-AUC

The prototype uses synthetic and controlled data. Therefore, the reported results demonstrate technical functionality and pipeline behavior, not guaranteed real-world production performance.

Future Improvements

Razorpay test-mode webhook integration

Real-time event processing

Analyst feedback

Case management

Model drift monitoring

Role-based access control

Production policy configuration

Responsible AI

RISKWEAVE is designed to support fraud analysts using explainable evidence and bounded decisions. It does not execute irreversible live payment actions.


### After pasting

1. Press **Ctrl + S**.
2. Open the **VS Code terminal**, not the README editor.
3. Run:

```powershell
git add README.md
git commit -m "Add project documentation"
git push
=======
git add README.md

git commit -m "Add project documentation"

git push


