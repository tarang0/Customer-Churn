"""
Victim profiling and loyalty-validation.

Answers two questions:
  1. Are the "loyal customers" we identified *really* loyal?  (validation)
  2. What pattern distinguishes victims from non-victims?      (profiling)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detect import CTFResult


def validate_loyalty(df: pd.DataFrame, loyal_idx: np.ndarray) -> dict:
    """Prove that the subset we call 'loyals' actually behaves loyally."""
    if len(loyal_idx) == 0:
        return {}
    sub = df.iloc[loyal_idx]
    return {
        "n": int(len(loyal_idx)),
        "pct_of_dataset": float(len(loyal_idx) / len(df) * 100),
        "min_tenure": int(sub["tenure"].min()),
        "mean_tenure": float(sub["tenure"].mean()),
        "max_tenure": int(sub["tenure"].max()),
        "mean_total_charges": float(sub["TotalCharges"].mean()),
        "historical_churn_rate": float(sub["Churn"].mean()),
        "population_churn_rate": float(df["Churn"].mean()),
        "long_contract_pct": float(
            ((sub["Contract"] == "One year") | (sub["Contract"] == "Two year")).mean() * 100
            if sub["Contract"].dtype == object else 0.0
        ),
    }


def _summary(sub: pd.DataFrame, sc_sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {}
    contract_raw = sub["Contract"] if sub["Contract"].dtype == object else None
    internet_raw = sub["InternetService"] if sub["InternetService"].dtype == object else None
    partner_raw = sub["Partner"] if sub["Partner"].dtype == object else None
    dependents_raw = sub["Dependents"] if sub["Dependents"].dtype == object else None

    def _pct(series: pd.Series, value) -> float:
        if series is None:
            return float("nan")
        return float((series == value).mean())

    return {
        "n": int(len(sub)),
        "avg_tenure":       float(sub["tenure"].mean()),
        "avg_monthly":      float(sub["MonthlyCharges"].mean()),
        "avg_total":        float(sub["TotalCharges"].mean()),
        "avg_ltv":          float(sc_sub["ltv"].mean()),
        "avg_churn_p":      float(sc_sub["churn_prob"].mean()),
        "avg_offer":        float(sc_sub["offer"].mean()),
        "actual_churn_rate": float(sub["Churn"].mean()),
        "contract_m2m_pct":    _pct(contract_raw, "Month-to-month"),
        "contract_oneyr_pct":  _pct(contract_raw, "One year"),
        "contract_twoyr_pct":  _pct(contract_raw, "Two year"),
        "fiber_pct":           _pct(internet_raw, "Fiber optic"),
        "dsl_pct":             _pct(internet_raw, "DSL"),
        "partner_yes_pct":     _pct(partner_raw, "Yes"),
        "dependents_yes_pct":  _pct(dependents_raw, "Yes"),
        "senior_pct": (
            float(sub["SeniorCitizen"].mean()) if "SeniorCitizen" in sub.columns else float("nan")
        ),
    }


def profile_victims(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    ctf_result: CTFResult,
    threshold: float = 0.0,
) -> dict:
    """
    Compare loyalty-penalty victims (CTF delta > threshold) against
    non-victim loyal customers.
    """
    victim_mask = ctf_result.deltas > threshold
    victim_idx = ctf_result.loyal_idx[victim_mask]
    nonvictim_idx = ctf_result.loyal_idx[~victim_mask]

    return {
        "victims":     _summary(df.iloc[victim_idx], scores.iloc[victim_idx]),
        "non_victims": _summary(df.iloc[nonvictim_idx], scores.iloc[nonvictim_idx]),
        "victim_idx": victim_idx,
        "nonvictim_idx": nonvictim_idx,
    }
