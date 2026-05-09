load_pd_pricer.py
"""
loan_pd_pricer.py
Prototype module to predict Probability of Default (PD) and Expected Loss for loans.


How it works:
- Loads trained scaler and models saved in 'pd_models.joblib' (created by the training script).
- Exposes `predict_pd` and `expected_loss` functions.
- `predict_pd` returns probability of default using an ensemble (LogReg + RandomForest average).
- `expected_loss` computes PD * (1 - recovery_rate) * loan_amount.


Usage example:
>>> from loan_pd_pricer import predict_pd, expected_loss
>>> features = {"credit_lines_outstanding": 1, "loan_amt_outstanding": 5000, "total_debt_outstanding": 10000, "income": 40000, "years_employed": 3, "fico_score": 600}
>>> print(predict_pd(features))
>>> print(expected_loss(features, recovery_rate=0.10))


Note: ensure 'pd_models.joblib' is in the same directory as this module or provide full path.
"""


import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "pd_models.joblib"
data = joblib.load(MODEL_PATH)
scaler = data["scaler"]
logreg = data["logreg"]
rf = data["rf"]
FEATURES = data["features"]
DEFAULT_RECOVERY_RATE = 0.10


def _to_array(loan_features):
    if isinstance(loan_features, dict):
        arr = np.array([loan_features[f] for f in FEATURES], dtype=float).reshape(1, -1)
    else:
        arr = np.array(loan_features, dtype=float).reshape(1, -1)
    return arr


def predict_pd(loan_features):
    """Return probability of default (PD) for given loan_features."""
    arr = _to_array(loan_features)
    arr_scaled = scaler.transform(arr)
    p1 = logreg.predict_proba(arr_scaled)[:,1]
    p2 = rf.predict_proba(arr_scaled)[:,1]
    return float((p1 + p2) / 2.0)


def expected_loss(loan_features, recovery_rate=DEFAULT_RECOVERY_RATE):
    """Return expected loss for the loan (PD * (1 - recovery_rate) * loan_amount)."""
    pd_val = predict_pd(loan_features)
    # determine loan amount from input
    if isinstance(loan_features, dict):
        loan_amt = float(loan_features["loan_amt_outstanding"])
    else:
        # FEATURES order is assumed
        loan_amt = float(np.array(loan_features, dtype=float).reshape(1, -1)[0, FEATURES.index("loan_amt_outstanding")])
    return float(pd_val * (1 - recovery_rate) * loan_amt)