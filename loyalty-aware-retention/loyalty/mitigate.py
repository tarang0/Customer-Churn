"""
Phase 3 — Loyalty-Aware Retention Allocator.

Fixes the loyalty penalty detected in Phase 2 by re-allocating the retention
budget with loyalty awareness baked in.

Two complementary mechanisms:

  1. **λ-penalised objective** — maximise (prevented churn value) + λ × (loyalty value).
     λ is a single knob controlling the trade-off between raw churn prevention
     and loyal-customer retention.

  2. **Two-pool budget split** — reserve a fraction α of the total budget as a
     dedicated "loyalty-reward pool" that only targets predicted victims.
     The remaining (1 − α) is the traditional "churn-prevention pool".

Both mechanisms use the victim-probability classifier from Method 6 as the
primary signal for identifying loyal customers who deserve investment.

The allocator itself is a deterministic **greedy knapsack** (by score-to-cost
ratio) — no pulp/ILP dependency required. For N ≈ 7,000 customers with small
individual costs relative to the total budget, greedy is within rounding error
of the optimal ILP and runs in O(N log N).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .detect import CTFResult
from .scoring import (DEFAULT_CHURN_MIN, DEFAULT_UNIT_COST, offer_for_row,
                      predicted_ltv)


# ---------------------------------------------------------------------------
# Victim-probability scorer (production-ready version of Method 6)
# ---------------------------------------------------------------------------

@dataclass
class VictimScorer:
    """A fitted classifier that gives P(loyal-penalty victim) for any customer."""
    clf: LogisticRegression
    mean: np.ndarray
    std: np.ndarray
    feature_cols: list[str]
    trained_on_tenure_min: int

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns a length-N array of P(victim) ∈ [0, 1]."""
        X = df[self.feature_cols].values.astype(float)
        X_scaled = (X - self.mean) / np.where(self.std > 0, self.std, 1.0)
        return self.clf.predict_proba(X_scaled)[:, 1]


def fit_victim_scorer(
    df: pd.DataFrame,
    ctf: CTFResult,
    feature_cols: list[str],
) -> VictimScorer:
    """
    Fit a logistic-regression classifier on the loyal subset with labels
    {1 = CTF delta > 0 (victim), 0 = otherwise}.

    Features are standardized (mean=0, std=1) so coefficients are comparable.
    We fit on the *entire* loyal subset (no holdout) because this is meant for
    scoring the rest of the population, not for reporting test-set metrics —
    Method 6 in detect.py already reports those for the scientific audit.
    """
    loyal = df.iloc[ctf.loyal_idx].reset_index(drop=True)
    X = loyal[feature_cols].values.astype(float)
    y = (ctf.deltas > 0).astype(int)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / np.where(std > 0, std, 1.0)

    clf = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42,
    )
    clf.fit(X_scaled, y)

    return VictimScorer(
        clf=clf, mean=mean, std=std,
        feature_cols=feature_cols,
        trained_on_tenure_min=int(loyal["tenure"].min()),
    )


# ---------------------------------------------------------------------------
# The allocation problem
# ---------------------------------------------------------------------------

@dataclass
class RetentionDecision:
    """What the mitigator decides for each customer."""
    selected: np.ndarray            # bool mask, shape (N,)
    offer_amount: np.ndarray        # $ per customer, 0 if not selected
    pool_assignment: np.ndarray     # "churn", "loyalty", or "none" per customer
    score: np.ndarray               # the score that drove selection
    total_spend: float
    n_selected: int
    budget: float
    lam: float
    alpha: float
    churn_pool_cap: float
    loyalty_pool_cap: float
    churn_pool_spend: float
    loyalty_pool_spend: float
    strategy: str                   # "baseline" | "lambda" | "two_pool"


@dataclass
class MitigationResult:
    """Everything needed to evaluate a mitigation strategy."""
    decision: RetentionDecision
    tog_before: float
    tog_after: float
    tog_improvement: float
    cluster_equity_before: dict[int, float]
    cluster_equity_after: dict[int, float]
    mean_offer_by_quintile_before: list[float]
    mean_offer_by_quintile_after: list[float]
    # NOTE: these are "risk-weighted LTV we TOUCHED", not "money saved".
    # We do NOT have treatment/control data to claim actual churn prevention.
    # These numbers measure the QUALITY of customers our budget reaches, not
    # the dollar impact of the offers themselves.
    ltv_at_risk_touched: float      # Σ (P_churn × LTV × I{selected})   — was prevented_churn_value
    loyalty_ltv_touched: float      # Σ (P_victim × LTV × I{selected})  — was loyalty_value_retained
    avg_ltv_of_selected: float      # mean LTV of selected customers
    total_population_ltv: float     # Σ LTV across all customers
    pct_victims_reached: float
    pct_loyals_reached: float


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _per_customer_cost(
    shap_matrix: np.ndarray,
    feat_names: list[str],
    churn_probs: np.ndarray,
    unit_cost: dict[str, float] = DEFAULT_UNIT_COST,
    shap_threshold: float = 0.02,
    churn_min: float = DEFAULT_CHURN_MIN,
    override_churn_min: float | None = None,
) -> np.ndarray:
    """
    Compute the cost of "acting on" each customer using the same SHAP-threshold
    rule as the baseline. Two wrinkles:

    - If `override_churn_min` is given, we use it instead of the global
      `churn_min`. This lets the loyalty pool fire on customers with lower
      churn probability than the baseline rule would allow.
    - A customer who would get a $0 offer under the rule still has a nominal
      action cost of $0 (we just won't select them).
    """
    effective_min = override_churn_min if override_churn_min is not None else churn_min
    N = len(churn_probs)
    costs = np.zeros(N)
    for i in range(N):
        c, _ = offer_for_row(
            shap_matrix[i], feat_names, float(churn_probs[i]),
            unit_cost, shap_threshold, effective_min,
        )
        costs[i] = c
    return costs


def _greedy_knapsack(
    score: np.ndarray, cost: np.ndarray, budget: float,
) -> np.ndarray:
    """
    Standard greedy-by-ratio knapsack.

    Returns a boolean selection mask. Deterministic given identical inputs.
    For customer i: ratio_i = score_i / cost_i (higher is better).
    We sort descending by ratio and keep picking until the budget runs out.

    Customers with cost == 0 or score <= 0 are never selected (their ratio is
    undefined or non-positive).
    """
    N = len(score)
    selected = np.zeros(N, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((cost > 0) & (score > 0), score / cost, -np.inf)
    order = np.argsort(-ratio)
    spent = 0.0
    for i in order:
        if ratio[i] <= 0:
            break
        if spent + cost[i] <= budget:
            selected[i] = True
            spent += cost[i]
    return selected


# ---------------------------------------------------------------------------
# The three strategies
# ---------------------------------------------------------------------------

def allocate_baseline(
    df: pd.DataFrame,
    churn_probs: np.ndarray,
    shap_matrix: np.ndarray,
    feat_names: list[str],
    budget: float,
    churn_min: float = DEFAULT_CHURN_MIN,
) -> RetentionDecision:
    """Strategy A — the current baseline."""
    N = len(df)
    cost = _per_customer_cost(shap_matrix, feat_names, churn_probs, churn_min=churn_min)
    ltv = np.array([
        predicted_ltv(df.iloc[i]["MonthlyCharges"], float(churn_probs[i]))
        for i in range(N)
    ])
    score = churn_probs * ltv
    selected = _greedy_knapsack(score, cost, budget)

    pool_assignment = np.array(["churn" if s else "none" for s in selected], dtype=object)
    offer = np.where(selected, cost, 0.0)

    return RetentionDecision(
        selected=selected, offer_amount=offer, pool_assignment=pool_assignment,
        score=score, total_spend=float(offer.sum()), n_selected=int(selected.sum()),
        budget=budget, lam=0.0, alpha=0.0,
        churn_pool_cap=budget, loyalty_pool_cap=0.0,
        churn_pool_spend=float(offer[pool_assignment == "churn"].sum()),
        loyalty_pool_spend=0.0, strategy="baseline",
    )


def allocate_lambda(
    df: pd.DataFrame,
    churn_probs: np.ndarray,
    shap_matrix: np.ndarray,
    feat_names: list[str],
    victim_probs: np.ndarray,
    budget: float,
    lam: float,
    churn_min: float = DEFAULT_CHURN_MIN,
    loyalty_churn_min: float = 0.0,
) -> RetentionDecision:
    """Strategy B — single-pool, λ-penalised objective."""
    N = len(df)

    ltv = np.array([
        predicted_ltv(df.iloc[i]["MonthlyCharges"], float(churn_probs[i]))
        for i in range(N)
    ])

    # Everybody gets a cost computed under a permissive churn_min so the
    # loyalty signal can fire even for low-risk loyals. The *baseline* firing
    # filter is still implicitly there through the churn_probs term in the
    # score — customers with near-zero churn probability and near-zero victim
    # probability will have score ~ 0 and won't be selected.
    cost = _per_customer_cost(
        shap_matrix, feat_names, churn_probs, churn_min=loyalty_churn_min,
    )

    churn_score = churn_probs * ltv           # "how much churn value do we save?"
    loyalty_score = victim_probs * ltv        # "how much loyalty value do we invest in?"
    score = churn_score + lam * loyalty_score

    selected = _greedy_knapsack(score, cost, budget)

    pool = np.array(["none"] * N, dtype=object)
    churn_mask = selected & (churn_score >= lam * loyalty_score)
    loyal_mask = selected & ~churn_mask
    pool[churn_mask] = "churn"
    pool[loyal_mask] = "loyalty"

    offer = np.where(selected, cost, 0.0)

    return RetentionDecision(
        selected=selected, offer_amount=offer, pool_assignment=pool,
        score=score, total_spend=float(offer.sum()), n_selected=int(selected.sum()),
        budget=budget, lam=lam, alpha=0.0,
        churn_pool_cap=budget, loyalty_pool_cap=0.0,
        churn_pool_spend=float(offer[pool == "churn"].sum()),
        loyalty_pool_spend=float(offer[pool == "loyalty"].sum()),
        strategy="lambda",
    )


def allocate_two_pool(
    df: pd.DataFrame,
    churn_probs: np.ndarray,
    shap_matrix: np.ndarray,
    feat_names: list[str],
    victim_probs: np.ndarray,
    budget: float,
    alpha: float,
    churn_min: float = DEFAULT_CHURN_MIN,
    loyalty_churn_min: float = 0.0,
) -> RetentionDecision:
    """Strategy C — explicit two-pool budget split."""
    N = len(df)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

    ltv = np.array([
        predicted_ltv(df.iloc[i]["MonthlyCharges"], float(churn_probs[i]))
        for i in range(N)
    ])

    # Pool caps
    loyalty_cap = alpha * budget
    churn_cap = budget - loyalty_cap

    # --- Churn pool: baseline cost + baseline score, baseline threshold ---
    churn_cost = _per_customer_cost(
        shap_matrix, feat_names, churn_probs, churn_min=churn_min,
    )
    churn_score = churn_probs * ltv
    churn_selected = _greedy_knapsack(churn_score, churn_cost, churn_cap)

    # --- Loyalty pool: permissive cost + victim-probability score ---
    # A customer already picked by the churn pool is not eligible for loyalty
    # pool spending (no double-dipping, budgets are separate).
    loyalty_cost = _per_customer_cost(
        shap_matrix, feat_names, churn_probs, churn_min=loyalty_churn_min,
    )
    loyalty_score = np.where(churn_selected, 0.0, victim_probs * ltv)
    loyalty_selected = _greedy_knapsack(loyalty_score, loyalty_cost, loyalty_cap)

    # Combine
    selected = churn_selected | loyalty_selected
    pool = np.array(["none"] * N, dtype=object)
    pool[churn_selected] = "churn"
    pool[loyalty_selected & ~churn_selected] = "loyalty"

    # A customer's offer cost is whichever pool selected them
    offer = np.zeros(N)
    offer[churn_selected] = churn_cost[churn_selected]
    loyalty_only = loyalty_selected & ~churn_selected
    offer[loyalty_only] = loyalty_cost[loyalty_only]

    total = float(offer.sum())
    score = np.where(pool == "churn", churn_score, loyalty_score)

    return RetentionDecision(
        selected=selected, offer_amount=offer, pool_assignment=pool,
        score=score, total_spend=total, n_selected=int(selected.sum()),
        budget=budget, lam=0.0, alpha=alpha,
        churn_pool_cap=churn_cap, loyalty_pool_cap=loyalty_cap,
        churn_pool_spend=float(offer[pool == "churn"].sum()),
        loyalty_pool_spend=float(offer[pool == "loyalty"].sum()),
        strategy="two_pool",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _tog_from_offers(tenures: np.ndarray, offers: np.ndarray) -> tuple[float, list[float]]:
    edges = np.unique(np.quantile(tenures, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(edges) < 6:
        edges = np.array([0, 14, 29, 43, 57, 72], dtype=float)
    buckets = np.clip(np.digitize(tenures, edges[1:-1], right=False),
                      0, len(edges) - 2)
    mean_offer = []
    for q in range(len(edges) - 1):
        m = buckets == q
        mean_offer.append(float(offers[m].mean()) if m.any() else 0.0)
    tog = float("inf") if mean_offer[-1] == 0 else mean_offer[0] / mean_offer[-1]
    return tog, mean_offer


def evaluate(
    df: pd.DataFrame,
    baseline_offers: np.ndarray,
    churn_probs: np.ndarray,
    victim_probs: np.ndarray,
    decision: RetentionDecision,
) -> MitigationResult:
    """Compute before/after TOG, cluster equity, and outcome metrics."""
    tenures = df["tenure"].values

    tog_before, mean_off_before = _tog_from_offers(tenures, baseline_offers)
    tog_after, mean_off_after = _tog_from_offers(tenures, decision.offer_amount)
    tog_improvement = (tog_before - tog_after) if np.isfinite(tog_before) else float("inf")

    ltv = np.array([
        predicted_ltv(df.iloc[i]["MonthlyCharges"], float(churn_probs[i]))
        for i in range(len(df))
    ])

    # Cluster equity (spend per $100 LTV, per cluster)
    cluster_before, cluster_after = {}, {}
    for c in sorted(df["cluster"].unique()):
        m = (df["cluster"] == c).values
        tot_ltv = float(ltv[m].sum())
        if tot_ltv <= 0:
            cluster_before[int(c)] = 0.0
            cluster_after[int(c)] = 0.0
            continue
        cluster_before[int(c)] = float(baseline_offers[m].sum() / tot_ltv * 100)
        cluster_after[int(c)] = float(decision.offer_amount[m].sum() / tot_ltv * 100)

    sel = decision.selected
    # Interpretable "quality of customers reached" metrics (NOT "prevention" claims)
    ltv_at_risk_touched = float((churn_probs * ltv * sel).sum())
    loyalty_ltv_touched = float((victim_probs * ltv * sel).sum())
    avg_ltv_of_selected = float(ltv[sel].mean()) if sel.any() else 0.0
    total_population_ltv = float(ltv.sum())

    victim_mask = victim_probs >= 0.5   # high-confidence victims
    pct_victims_reached = float(sel[victim_mask].mean() * 100) if victim_mask.any() else 0.0
    loyal_mask = tenures >= 48
    pct_loyals_reached = float(sel[loyal_mask].mean() * 100) if loyal_mask.any() else 0.0

    return MitigationResult(
        decision=decision,
        tog_before=tog_before,
        tog_after=tog_after,
        tog_improvement=tog_improvement,
        cluster_equity_before=cluster_before,
        cluster_equity_after=cluster_after,
        mean_offer_by_quintile_before=mean_off_before,
        mean_offer_by_quintile_after=mean_off_after,
        ltv_at_risk_touched=ltv_at_risk_touched,
        loyalty_ltv_touched=loyalty_ltv_touched,
        avg_ltv_of_selected=avg_ltv_of_selected,
        total_population_ltv=total_population_ltv,
        pct_victims_reached=pct_victims_reached,
        pct_loyals_reached=pct_loyals_reached,
    )


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    lam: float
    tog_after: float
    ltv_at_risk_touched: float      # was prevented_churn_value
    loyalty_ltv_touched: float      # was loyalty_value_retained
    pct_victims_reached: float
    pct_loyals_reached: float
    total_spend: float
    avg_ltv_of_selected: float


def pareto_frontier_lambda(
    df: pd.DataFrame,
    churn_probs: np.ndarray,
    shap_matrix: np.ndarray,
    feat_names: list[str],
    victim_probs: np.ndarray,
    baseline_offers: np.ndarray,
    budget: float,
    lambdas: list[float] | None = None,
    churn_min: float = DEFAULT_CHURN_MIN,
) -> list[ParetoPoint]:
    """Sweep λ and record the (TOG, LTV-at-risk-touched) trade-off at each point."""
    if lambdas is None:
        lambdas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    out = []
    for lam in lambdas:
        dec = allocate_lambda(
            df, churn_probs, shap_matrix, feat_names, victim_probs,
            budget=budget, lam=lam, churn_min=churn_min,
        )
        ev = evaluate(df, baseline_offers, churn_probs, victim_probs, dec)
        out.append(ParetoPoint(
            lam=lam,
            tog_after=ev.tog_after,
            ltv_at_risk_touched=ev.ltv_at_risk_touched,
            loyalty_ltv_touched=ev.loyalty_ltv_touched,
            pct_victims_reached=ev.pct_victims_reached,
            pct_loyals_reached=ev.pct_loyals_reached,
            total_spend=ev.decision.total_spend,
            avg_ltv_of_selected=ev.avg_ltv_of_selected,
        ))
    return out


# ---------------------------------------------------------------------------
# Extended Pareto sweep + sweet-spot identification
# ---------------------------------------------------------------------------

@dataclass
class DetailedParetoPoint:
    lam: float
    tog_after: float
    n_selected: int
    total_spend: float
    avg_ltv_of_selected: float
    pct_high_churn_reached: float    # % of P(churn) > 0.5 customers selected
    pct_victims_reached: float
    pct_loyals_reached: float
    ltv_at_risk_touched: float
    loyalty_ltv_touched: float
    # Composite sweet-spot score (0 to 1 — higher is better)
    fairness_score: float             # low TOG → high score
    quality_score: float              # high avg_LTV → high score
    victim_coverage_score: float      # high % victims → high score
    composite_score: float            # geometric mean of the above


def detailed_pareto_sweep(
    df: pd.DataFrame,
    churn_probs: np.ndarray,
    shap_matrix: np.ndarray,
    feat_names: list[str],
    victim_probs: np.ndarray,
    baseline_offers: np.ndarray,
    budget: float,
    lambdas: list[float] | None = None,
    churn_min: float = DEFAULT_CHURN_MIN,
    high_churn_cutoff: float = 0.5,
) -> tuple[list[DetailedParetoPoint], int]:
    """
    Fine-grained Pareto sweep over λ.

    Computes multiple coverage metrics per point, plus a composite
    "sweet spot" score (geometric mean of fairness, quality, victim coverage).

    Returns (points, sweet_spot_index).
    """
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    tenures = df["tenure"].values
    N = len(df)

    high_churn_mask = churn_probs >= high_churn_cutoff
    victim_mask = victim_probs >= 0.5
    loyal_mask = tenures >= 48

    ltv = np.array([
        predicted_ltv(df.iloc[i]["MonthlyCharges"], float(churn_probs[i]))
        for i in range(N)
    ])

    raw = []
    for lam in lambdas:
        dec = allocate_lambda(
            df, churn_probs, shap_matrix, feat_names, victim_probs,
            budget=budget, lam=lam, churn_min=churn_min,
        )
        ev = evaluate(df, baseline_offers, churn_probs, victim_probs, dec)

        sel = dec.selected
        pct_high_churn = float(sel[high_churn_mask].mean() * 100) if high_churn_mask.any() else 0.0

        raw.append({
            "lam": lam,
            "tog_after": ev.tog_after,
            "n_selected": dec.n_selected,
            "total_spend": dec.total_spend,
            "avg_ltv_of_selected": ev.avg_ltv_of_selected,
            "pct_high_churn_reached": pct_high_churn,
            "pct_victims_reached": ev.pct_victims_reached,
            "pct_loyals_reached": ev.pct_loyals_reached,
            "ltv_at_risk_touched": ev.ltv_at_risk_touched,
            "loyalty_ltv_touched": ev.loyalty_ltv_touched,
        })

    # Normalize scores to [0, 1] for composite
    avg_ltvs = [r["avg_ltv_of_selected"] for r in raw]
    togs = [r["tog_after"] if r["tog_after"] != float("inf") else max(
        (x["tog_after"] for x in raw if x["tog_after"] != float("inf")), default=1.0
    ) for r in raw]
    victims = [r["pct_victims_reached"] for r in raw]

    max_ltv = max(avg_ltvs) if avg_ltvs else 1.0
    min_tog = min(togs) if togs else 1.0
    max_victims = max(victims) if victims else 1.0

    points = []
    for r, tog_clean in zip(raw, togs):
        # Fairness: invert TOG, normalize so best TOG = 1.0
        fairness = min_tog / tog_clean if tog_clean > 0 else 0.0
        # Quality: normalize avg LTV to [0, 1]
        quality = r["avg_ltv_of_selected"] / max_ltv if max_ltv > 0 else 0.0
        # Victim coverage: normalize to [0, 1]
        victim_cov = r["pct_victims_reached"] / max_victims if max_victims > 0 else 0.0
        # Composite: geometric mean (penalizes any single dimension being low)
        composite = float(np.cbrt(max(fairness, 1e-9) * max(quality, 1e-9) * max(victim_cov, 1e-9)))

        points.append(DetailedParetoPoint(
            lam=r["lam"],
            tog_after=r["tog_after"],
            n_selected=r["n_selected"],
            total_spend=r["total_spend"],
            avg_ltv_of_selected=r["avg_ltv_of_selected"],
            pct_high_churn_reached=r["pct_high_churn_reached"],
            pct_victims_reached=r["pct_victims_reached"],
            pct_loyals_reached=r["pct_loyals_reached"],
            ltv_at_risk_touched=r["ltv_at_risk_touched"],
            loyalty_ltv_touched=r["loyalty_ltv_touched"],
            fairness_score=float(fairness),
            quality_score=float(quality),
            victim_coverage_score=float(victim_cov),
            composite_score=composite,
        ))

    sweet_idx = int(np.argmax([p.composite_score for p in points]))
    return points, sweet_idx
