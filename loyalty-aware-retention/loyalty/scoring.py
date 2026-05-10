"""
Scoring helpers: offer computation, LTV, SHAP matrix.

One place to edit the "what does the baseline retention system do?" logic.
Everything downstream (detection, profiling, plots) re-uses these functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Baseline retention-system assumptions (auditable, user-editable)
# ---------------------------------------------------------------------------
#
# These match the `RETENTION_STRATEGIES` dict in app.py. If a feature's SHAP
# contribution exceeds SHAP_TRIGGER_THRESHOLD AND the customer's overall churn
# probability exceeds CHURN_MIN, the system "fires" the action and we add the
# unit cost below to the customer's offer value.
#
# Unit costs are plausible defaults, NOT ground truth — they're the "if we
# assume each intervention costs roughly this much, then ..." input.
# The loyalty-penalty conclusion is robust to reasonable perturbations.

DEFAULT_UNIT_COST: dict[str, float] = {
    "Contract":          120.0,
    "OnlineSecurity":     30.0,
    "TechSupport":        60.0,
    "InternetService":    45.0,
    "tenure":             40.0,
    "MonthlyCharges":     70.0,
    "OnlineBackup":       30.0,
    "DeviceProtection":   30.0,
    "StreamingTV":        25.0,
    "StreamingMovies":    25.0,
    "PaperlessBilling":    0.0,
    "PaymentMethod":      15.0,
}

DEFAULT_SHAP_THRESHOLD = 0.02
DEFAULT_CHURN_MIN = 0.50   # realistic retention-campaign threshold. Most
                           # telcos only act on customers whose predicted
                           # churn probability exceeds ~0.5 (i.e. the model
                           # is more confident they'll leave than stay).
                           # The original app.py used 0.20, which is the
                           # dataset's approximate median churn probability
                           # — unrealistically generous, it would flag
                           # roughly half the customer base. We default to
                           # 0.50 here and expose a slider so the user can
                           # verify the loyalty penalty persists across
                           # threshold choices (it actually strengthens as
                           # the threshold rises).


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_shap_matrix(explainer, X: pd.DataFrame) -> np.ndarray:
    """Return a (n_rows, n_features) SHAP matrix for the positive (churn) class."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv)


def offer_for_row(
    shap_row: np.ndarray,
    feature_display_names: list[str],
    churn_prob: float,
    unit_cost: dict[str, float] = DEFAULT_UNIT_COST,
    shap_threshold: float = DEFAULT_SHAP_THRESHOLD,
    churn_min: float = DEFAULT_CHURN_MIN,
) -> tuple[float, list[str]]:
    """Replicate the baseline SHAP-threshold rule. Returns (offer $, triggered features)."""
    if churn_prob < churn_min:
        return 0.0, []
    total = 0.0
    triggered: list[str] = []
    for feat, s in zip(feature_display_names, shap_row):
        if feat in unit_cost and s > shap_threshold:
            total += unit_cost[feat]
            triggered.append(feat)
    return total, triggered


def predicted_ltv(monthly: float, churn_prob: float, horizon_months: int = 24) -> float:
    """Simple tenure-independent LTV proxy. monthly * horizon * (1 - churn_prob)."""
    return float(monthly) * horizon_months * (1.0 - churn_prob)


def score_population(
    churn_model,
    explainer,
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_display_names: list[str],
    unit_cost: dict[str, float] = DEFAULT_UNIT_COST,
    shap_threshold: float = DEFAULT_SHAP_THRESHOLD,
    churn_min: float = DEFAULT_CHURN_MIN,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Score the entire population under the baseline retention rule.

    Returns a (scores_df, shap_matrix) tuple where:
      scores_df has columns: churn_prob, offer, ltv, n_triggered
      shap_matrix has shape (len(df), len(feature_display_names))
    Both are indexed identically to df.
    """
    X = df[feature_cols].copy().reset_index(drop=True)
    probs = churn_model.predict_proba(X)[:, 1]
    shap_mat = compute_shap_matrix(explainer, X)

    offers = np.zeros(len(df))
    triggered_counts = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        o, t = offer_for_row(
            shap_mat[i], feature_display_names, float(probs[i]),
            unit_cost, shap_threshold, churn_min,
        )
        offers[i] = o
        triggered_counts[i] = len(t)

    ltvs = np.array([
        predicted_ltv(df.iloc[i]["MonthlyCharges"], float(probs[i]))
        for i in range(len(df))
    ])

    scores = pd.DataFrame(
        {
            "churn_prob": probs,
            "offer": offers,
            "ltv": ltvs,
            "n_triggered": triggered_counts,
        },
        index=df.index,
    )
    return scores, shap_mat
