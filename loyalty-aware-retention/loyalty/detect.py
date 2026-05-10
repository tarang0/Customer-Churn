"""
Six detection methods for the loyalty penalty.

Each returns a dataclass containing both the raw numbers (so viz/UI can
re-plot) and a short verdict string.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .scoring import (
    DEFAULT_CHURN_MIN,
    DEFAULT_SHAP_THRESHOLD,
    DEFAULT_UNIT_COST,
    compute_shap_matrix,
    offer_for_row,
)


# ===========================================================================
# Method 1 — Tenure-Offer Gap (TOG)
# ===========================================================================

@dataclass
class TOGResult:
    quintile_edges: np.ndarray
    quintile_labels: list[str]
    per_quintile_mean_offer: list[float]
    per_quintile_mean_ltv: list[float]
    per_quintile_mean_churn_prob: list[float]
    per_quintile_n: list[int]
    tog_ratio: float
    verdict: str


def tenure_offer_gap(df: pd.DataFrame, scores: pd.DataFrame) -> TOGResult:
    tenures = df["tenure"].values
    offers = scores["offer"].values
    ltvs = scores["ltv"].values
    probs = scores["churn_prob"].values

    edges = np.unique(np.quantile(tenures, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(edges) < 6:
        edges = np.array([0, 14, 29, 43, 57, 72], dtype=float)
    labels = [
        f"Q{i+1}: {int(edges[i])}-{int(edges[i+1])} mo"
        for i in range(len(edges) - 1)
    ]
    buckets = np.clip(
        np.digitize(tenures, edges[1:-1], right=False), 0, len(labels) - 1
    )

    mo, ml, mp, ns = [], [], [], []
    for q in range(len(labels)):
        m = buckets == q
        ns.append(int(m.sum()))
        mo.append(float(offers[m].mean()) if m.any() else 0.0)
        ml.append(float(ltvs[m].mean()) if m.any() else 0.0)
        mp.append(float(probs[m].mean()) if m.any() else 0.0)

    tog = float("inf") if mo[-1] == 0 else mo[0] / mo[-1]
    verdict = (
        "LOYALTY PENALTY DETECTED (Tenure-Offer Gap > 1.5)" if tog > 1.5
        else "Mild loyalty penalty (1.2 < Tenure-Offer Gap ≤ 1.5)" if tog > 1.2
        else "No meaningful loyalty penalty at group level"
    )
    return TOGResult(edges, labels, mo, ml, mp, ns, tog, verdict)


# ===========================================================================
# Method 2 — Counterfactual Tenure Flip (CTF)  — flips ONLY tenure
# ===========================================================================

@dataclass
class CTFResult:
    loyal_idx: np.ndarray        # global row indices
    deltas: np.ndarray            # twin_offer - real_offer, per loyal customer
    offer_real: np.ndarray
    offer_twin: np.ndarray
    churn_prob_real: np.ndarray
    churn_prob_twin: np.ndarray
    triggered_twin: list[list[str]]
    mean_delta: float
    median_delta: float
    pct_penalised: float
    t_stat: float
    p_value: float
    is_significant: bool
    loyal_threshold: int
    synth_tenure: int


def counterfactual_tenure_flip(
    df: pd.DataFrame,
    churn_model,
    explainer,
    feature_cols: list[str],
    feature_display_names: list[str],
    unit_cost: dict[str, float] = DEFAULT_UNIT_COST,
    loyal_threshold: int = 48,
    synth_tenure: int = 3,
    churn_min: float = DEFAULT_CHURN_MIN,
) -> CTFResult:
    X_all = df[feature_cols].copy().reset_index(drop=True)
    loyal_idx = np.where(df["tenure"].values >= loyal_threshold)[0]

    X_real = X_all.iloc[loyal_idx].copy()
    X_twin = X_real.copy()
    X_twin["tenure"] = synth_tenure  # ONLY tenure flipped, nothing else

    cp_real = churn_model.predict_proba(X_real)[:, 1]
    cp_twin = churn_model.predict_proba(X_twin)[:, 1]
    sv_real = compute_shap_matrix(explainer, X_real)
    sv_twin = compute_shap_matrix(explainer, X_twin)

    o_real = np.zeros(len(loyal_idx))
    o_twin = np.zeros(len(loyal_idx))
    trig_twin: list[list[str]] = []
    for i in range(len(loyal_idx)):
        ro, _ = offer_for_row(sv_real[i], feature_display_names,
                              float(cp_real[i]), unit_cost, churn_min=churn_min)
        to, tt = offer_for_row(sv_twin[i], feature_display_names,
                               float(cp_twin[i]), unit_cost, churn_min=churn_min)
        o_real[i] = ro
        o_twin[i] = to
        trig_twin.append(tt)

    deltas = o_twin - o_real
    if len(deltas) > 1 and np.std(deltas) > 0:
        t_stat, p_value = stats.ttest_1samp(deltas, 0.0)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = 0.0, 1.0

    return CTFResult(
        loyal_idx=loyal_idx,
        deltas=deltas,
        offer_real=o_real,
        offer_twin=o_twin,
        churn_prob_real=cp_real,
        churn_prob_twin=cp_twin,
        triggered_twin=trig_twin,
        mean_delta=float(deltas.mean()) if len(deltas) else 0.0,
        median_delta=float(np.median(deltas)) if len(deltas) else 0.0,
        pct_penalised=float((deltas > 0).mean() * 100.0) if len(deltas) else 0.0,
        t_stat=t_stat,
        p_value=p_value,
        is_significant=(p_value < 0.05 and float(deltas.mean()) > 0),
        loyal_threshold=loyal_threshold,
        synth_tenure=synth_tenure,
    )


# ===========================================================================
# Method 3 — Regression audit
# ===========================================================================

@dataclass
class RegressionResult:
    n: int
    r2: float
    coefs: dict[str, float]
    pvals: dict[str, float]
    summary: str
    # Two-stage model parts (stricter-threshold-aware):
    selection_coefs: dict[str, float]   # P(get any offer | features)
    selection_pvals: dict[str, float]
    selection_pseudo_r2: float
    amount_coefs: dict[str, float]      # E[offer | offer > 0, features]
    amount_pvals: dict[str, float]
    amount_r2: float
    amount_n: int


def _count_services(df: pd.DataFrame) -> np.ndarray:
    cols = ["PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    total = np.zeros(len(df), dtype=int)
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            total += (df[c] == "Yes").astype(int).values
        elif (c + "_enc") in df.columns:
            # Encoded fallback — best-effort, treat non-zero as "has the service"
            # (this is approximate; we only fall back if raw strings are missing)
            total += (df[c + "_enc"] > 0).astype(int).values
    return total


def regression_audit(df: pd.DataFrame, scores: pd.DataFrame) -> RegressionResult:
    """
    Two-stage audit, designed to survive a strict churn-probability filter:

      Stage 1 (selection):  P(get any offer | features)  via logit
      Stage 2 (amount):     E[offer | offer > 0, features]  via OLS

    Why two stages? When the retention system uses a strict P(churn) cutoff,
    loyal customers are EXCLUDED from receiving any offer at all — they
    never reach the "how big is the offer?" stage. A single OLS on the whole
    population then confounds two mechanisms (whether you're offered anything
    vs how much). Splitting them makes the tenure effect interpretable at
    every threshold.

    A loyalty penalty in stage 1 means: "loyalty makes it LESS LIKELY you
    receive any offer." That's the strongest form of the penalty.
    """
    import statsmodels.api as sm

    X = pd.DataFrame({
        "churn_prob":   scores["churn_prob"].values,
        "ltv":          scores["ltv"].values,
        "tenure":       df["tenure"].values,
        "monthly":      df["MonthlyCharges"].values,
        "num_services": _count_services(df),
    })
    X_sm = sm.add_constant(X, has_constant="add")
    y = scores["offer"].values

    # Full-population OLS (kept for backwards-compatibility, but interpret with care)
    model = sm.OLS(y, X_sm).fit()

    # Stage 1: logistic — does tenure, independent of risk and revenue,
    # reduce the probability of being offered anything?
    # We use sklearn's LogisticRegression with scaled inputs (statsmodels
    # Logit can diverge under high-variance feature scales and mechanical
    # outcomes). We report the coefficient on scaled tenure and bootstrap
    # a p-value for it.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    try:
        X_s1 = X[["ltv", "tenure", "monthly", "num_services"]].copy()
        scaler = StandardScaler()
        X_s1_scaled = scaler.fit_transform(X_s1)
        y_binary = (y > 0).astype(int)
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_s1_scaled, y_binary)

        # Bootstrap 200 resamples for tenure coefficient p-value
        from sklearn.utils import resample
        n_boot = 200
        boot_coefs = np.zeros((n_boot, X_s1_scaled.shape[1]))
        for b in range(n_boot):
            idx = resample(np.arange(len(y_binary)), random_state=b)
            if len(np.unique(y_binary[idx])) < 2:
                boot_coefs[b] = np.nan
                continue
            c = LogisticRegression(max_iter=500, random_state=42)
            c.fit(X_s1_scaled[idx], y_binary[idx])
            boot_coefs[b] = c.coef_[0]
        boot_coefs = boot_coefs[~np.isnan(boot_coefs).any(axis=1)]
        selection_coefs = {
            "ltv":          float(clf.coef_[0, 0]),
            "tenure":       float(clf.coef_[0, 1]),
            "monthly":      float(clf.coef_[0, 2]),
            "num_services": float(clf.coef_[0, 3]),
        }
        # Two-sided p-value from bootstrap: fraction of samples crossing zero
        def _p(col):
            if len(boot_coefs) == 0:
                return float("nan")
            # standard-error-like: proportion of bootstrap coefs on opposite side of mean
            signs = np.sign(boot_coefs[:, col])
            mean_sign = np.sign(np.mean(boot_coefs[:, col]))
            return float(2 * min((signs != mean_sign).mean(), 0.5))
        selection_pvals = {
            "ltv":          _p(0),
            "tenure":       _p(1),
            "monthly":      _p(2),
            "num_services": _p(3),
        }
        # Pseudo-R^2 approximation (McFadden-like via score)
        y_prob = clf.predict_proba(X_s1_scaled)[:, 1]
        ll_full = np.sum(y_binary * np.log(np.clip(y_prob, 1e-10, 1 - 1e-10))
                          + (1 - y_binary) * np.log(np.clip(1 - y_prob, 1e-10, 1 - 1e-10)))
        base = y_binary.mean()
        ll_null = np.sum(y_binary * np.log(np.clip(base, 1e-10, 1 - 1e-10))
                         + (1 - y_binary) * np.log(np.clip(1 - base, 1e-10, 1 - 1e-10)))
        selection_pseudo_r2 = float(1 - ll_full / ll_null) if ll_null != 0 else float("nan")
    except Exception:
        selection_coefs, selection_pvals, selection_pseudo_r2 = {}, {}, float("nan")

    # Stage 2: OLS on only the offered customers
    mask = y > 0
    if mask.sum() > 50:
        stage2 = sm.OLS(y[mask], X_sm[mask]).fit()
        amount_coefs = {k: float(v) for k, v in stage2.params.items()}
        amount_pvals = {k: float(v) for k, v in stage2.pvalues.items()}
        amount_r2 = float(stage2.rsquared)
        amount_n = int(stage2.nobs)
    else:
        amount_coefs, amount_pvals, amount_r2, amount_n = {}, {}, float("nan"), 0

    return RegressionResult(
        n=int(model.nobs),
        r2=float(model.rsquared),
        coefs={k: float(v) for k, v in model.params.items()},
        pvals={k: float(v) for k, v in model.pvalues.items()},
        summary=str(model.summary()),
        selection_coefs=selection_coefs,
        selection_pvals=selection_pvals,
        selection_pseudo_r2=selection_pseudo_r2,
        amount_coefs=amount_coefs,
        amount_pvals=amount_pvals,
        amount_r2=amount_r2,
        amount_n=amount_n,
    )


# ===========================================================================
# Method 4 — K-Means cluster-level equity
# ===========================================================================

@dataclass
class ClusterEquityResult:
    per_cluster: pd.DataFrame
    verdict: str


def cluster_equity(df: pd.DataFrame, scores: pd.DataFrame) -> ClusterEquityResult:
    out = []
    for c in sorted(df["cluster"].unique()):
        m = (df["cluster"] == c).values
        n = int(m.sum())
        total_ltv = float(scores.loc[m, "ltv"].sum())
        total_spend = float(scores.loc[m, "offer"].sum())
        spl = total_spend / total_ltv if total_ltv > 0 else 0.0
        out.append({
            "cluster": int(c),
            "n": n,
            "avg_tenure": float(df.loc[m, "tenure"].mean()),
            "avg_monthly": float(df.loc[m, "MonthlyCharges"].mean()),
            "actual_churn_rate": float(df.loc[m, "Churn"].mean()),
            "avg_offer": float(scores.loc[m, "offer"].mean()),
            "total_ltv": total_ltv,
            "total_spend": total_spend,
            "spend_per_ltv": spl,
        })
    per_cluster = pd.DataFrame(out)
    premium = per_cluster.loc[per_cluster["avg_tenure"].idxmax()]
    flight = per_cluster.loc[per_cluster["actual_churn_rate"].idxmax()]

    if premium["spend_per_ltv"] < flight["spend_per_ltv"] * 0.5:
        verdict = (f"Premium cluster (cluster {int(premium['cluster'])}) is "
                   f"SUBSTANTIALLY under-invested: ${premium['spend_per_ltv']*100:.2f} "
                   f"vs ${flight['spend_per_ltv']*100:.2f} per $100 LTV for Flight Risks.")
    elif premium["spend_per_ltv"] < flight["spend_per_ltv"]:
        verdict = (f"Premium cluster gets LESS per $100 LTV "
                   f"(${premium['spend_per_ltv']*100:.2f} vs "
                   f"${flight['spend_per_ltv']*100:.2f}).")
    else:
        verdict = "Spending is balanced at cluster level."
    return ClusterEquityResult(per_cluster=per_cluster, verdict=verdict)


# ===========================================================================
# Method 5 — Contract-controlled TOG (rules out contract as the confounder)
# ===========================================================================

@dataclass
class ContractControlledTOGResult:
    per_contract: dict
    verdict: str


def contract_controlled_tog(df: pd.DataFrame, scores: pd.DataFrame) -> ContractControlledTOGResult:
    """
    Someone could argue: loyal customers get smaller offers because they're on
    annual contracts (which already signal "won't churn"), not because of tenure.

    Fix: compute TOG *within* each contract type separately using tenure
    tertiles. If the penalty persists within M2M, 1-year, and 2-year, then
    tenure itself — not contract — is what's driving it.
    """
    if "Contract" not in df.columns or df["Contract"].dtype != object:
        return ContractControlledTOGResult(per_contract={}, verdict="Contract column unavailable.")

    out: dict[str, dict] = {}
    for contract in sorted(df["Contract"].unique()):
        mask = (df["Contract"] == contract).values
        if mask.sum() < 50:
            continue
        sub_tenures = df.loc[mask, "tenure"].values
        sub_offers = scores.loc[mask, "offer"].values

        try:
            edges = np.unique(np.quantile(sub_tenures, [0, 0.33, 0.66, 1.0]))
            if len(edges) < 4:
                continue
        except Exception:
            continue

        buckets = np.clip(
            np.digitize(sub_tenures, edges[1:-1], right=False), 0, len(edges) - 2
        )
        mo, ns = [], []
        for q in range(len(edges) - 1):
            bm = buckets == q
            ns.append(int(bm.sum()))
            mo.append(float(sub_offers[bm].mean()) if bm.any() else 0.0)

        if mo[-1] == 0:
            tog = float("inf")
        else:
            tog = mo[0] / mo[-1]

        out[str(contract)] = {
            "edges": edges.tolist(),
            "mean_offer": mo,
            "n": ns,
            "tog": tog,
        }

    any_penalty = any(
        (r["tog"] != float("inf") and r["tog"] > 1.5) for r in out.values()
    )
    verdict = (
        "Loyalty penalty PERSISTS within contract types — confounding ruled out."
        if any_penalty else
        "No within-contract penalty — penalty may be partly contract-mediated."
    )
    return ContractControlledTOGResult(per_contract=out, verdict=verdict)


# ===========================================================================
# Method 6 — Predicting loyalty-penalty victims (predictive feasibility)
# ===========================================================================

@dataclass
class VictimPredictorResult:
    feature_importance: pd.Series
    auc: float
    accuracy: float
    precision_at_top_n: float
    recall_at_top_n: float
    top_n: int
    classification_report: str
    n_victims: int
    n_total: int


def victim_predictor(
    df: pd.DataFrame,
    ctf_result: CTFResult,
    feature_cols: list[str],
    feature_display_names: list[str],
    random_state: int = 42,
) -> VictimPredictorResult:
    """
    Train a logistic-regression classifier on the loyal subset to predict which
    loyal customers are CTF victims (delta > 0). If the classifier works well
    (AUC > 0.8, precision@top-N > 0.8), we can confidently target retention
    budget at predicted victims.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                  roc_auc_score)
    from sklearn.model_selection import train_test_split

    loyal_df = df.iloc[ctf_result.loyal_idx].reset_index(drop=True)
    X = loyal_df[feature_cols].values
    y = (ctf_result.deltas > 0).astype(int)

    n_victims = int(y.sum())
    n_total = len(y)
    if n_victims < 20:
        return VictimPredictorResult(
            feature_importance=pd.Series(dtype=float),
            auc=0.0, accuracy=0.0, precision_at_top_n=0.0, recall_at_top_n=0.0,
            top_n=0, classification_report="Too few victims to train.",
            n_victims=n_victims, n_total=n_total,
        )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state,
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=random_state)
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    auc = float(roc_auc_score(y_te, y_prob))
    acc = float(accuracy_score(y_te, y_pred))

    # Precision / recall @ top-N where N = number of actual victims in test set
    top_n = max(int(y_te.sum()), 10)
    top_idx = np.argsort(y_prob)[::-1][:top_n]
    precision_at_top = float(y_te[top_idx].mean()) if top_n > 0 else 0.0
    recall_at_top = float(y_te[top_idx].sum() / max(y_te.sum(), 1))

    fi = pd.Series(clf.coef_[0], index=feature_display_names).abs().sort_values(
        ascending=False
    )
    report = classification_report(
        y_te, y_pred, target_names=["Not victim", "Victim"], zero_division=0,
    )
    return VictimPredictorResult(
        feature_importance=fi, auc=auc, accuracy=acc,
        precision_at_top_n=precision_at_top, recall_at_top_n=recall_at_top,
        top_n=top_n, classification_report=report,
        n_victims=n_victims, n_total=n_total,
    )


# ===========================================================================
# Cross-method agreement
# ===========================================================================

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def cross_method_agreement(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    ctf_result: CTFResult,
) -> pd.DataFrame:
    """
    Do the methods flag the *same* customers?

    We compare four ways of identifying "under-served loyalists":
      - CTF victims       (delta > 0, among loyals)
      - TOG top-quintile  (customers in the longest-tenure quintile with offer=0)
      - Cluster premium   (customers in the premium cluster getting no offer)
      - Predictive top-N  (would be added after classifier run; optional)

    Returns a Jaccard-similarity matrix across the four sets.
    """
    # CTF victims (global indices)
    ctf_victims = set(
        ctf_result.loyal_idx[ctf_result.deltas > 0].tolist()
    )

    # TOG top-tenure-quintile customers who got zero offer
    tenure_q5 = df["tenure"].quantile(0.8)
    tog_flagged = set(
        df.index[(df["tenure"] >= tenure_q5) & (scores["offer"] == 0)].tolist()
    )

    # Premium cluster customers with zero offer
    cluster_avg_tenure = df.groupby("cluster", observed=True)["tenure"].mean()
    premium_cluster = int(cluster_avg_tenure.idxmax())
    cluster_flagged = set(
        df.index[(df["cluster"] == premium_cluster) & (scores["offer"] == 0)].tolist()
    )

    sets = {
        "Counterfactual Tenure Flip victims": ctf_victims,
        "Tenure-Offer Gap top-quintile (offer=0)": tog_flagged,
        "Premium cluster (offer=0)": cluster_flagged,
    }
    names = list(sets)
    agreement = pd.DataFrame(
        np.zeros((len(names), len(names))), index=names, columns=names,
    )
    for a in names:
        for b in names:
            agreement.loc[a, b] = _jaccard(sets[a], sets[b])

    # Also include raw counts as a separate attribute
    agreement.attrs["counts"] = {k: len(v) for k, v in sets.items()}
    return agreement


# ===========================================================================
# Threshold sensitivity analysis
# ===========================================================================

@dataclass
class ThresholdSensitivityResult:
    thresholds: list[float]
    tog: list[float]
    pct_targeted: list[float]
    pct_loyal_offered: list[float]
    pct_new_offered: list[float]
    mean_offer: list[float]
    default_threshold: float


def threshold_sensitivity(
    df: pd.DataFrame,
    churn_model,
    explainer,
    feature_cols: list[str],
    feature_display_names: list[str],
    thresholds: list[float] | None = None,
    unit_cost: dict[str, float] = DEFAULT_UNIT_COST,
    shap_threshold: float = DEFAULT_SHAP_THRESHOLD,
) -> ThresholdSensitivityResult:
    """
    Show how TOG and targeting coverage change as we sweep the churn-probability
    threshold. If the loyalty penalty only existed at one specific threshold,
    our conclusion would be fragile. In practice the penalty strengthens as
    the threshold rises (because loyal customers have low churn probability
    and get filtered out at every threshold, while newcomers still pass).
    """
    from .scoring import compute_shap_matrix as _shap, offer_for_row as _offer

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    X = df[feature_cols].copy().reset_index(drop=True)
    probs = churn_model.predict_proba(X)[:, 1]
    sv = _shap(explainer, X)

    edges = np.unique(np.quantile(df["tenure"].values, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    buckets = np.clip(
        np.digitize(df["tenure"].values, edges[1:-1], right=False), 0, len(edges) - 2
    )
    loyal_mask = df["tenure"].values >= 48
    new_mask = df["tenure"].values <= 12

    togs, targeted, loyal_pct, new_pct, means = [], [], [], [], []
    for t in thresholds:
        offers = np.zeros(len(df))
        for i in range(len(df)):
            o, _ = _offer(sv[i], feature_display_names, float(probs[i]),
                          unit_cost, shap_threshold, churn_min=t)
            offers[i] = o
        q1 = offers[buckets == 0].mean()
        q5 = offers[buckets == len(edges) - 2].mean()
        togs.append(float(q1 / q5) if q5 > 0 else float("inf"))
        targeted.append(float((offers > 0).mean() * 100))
        loyal_pct.append(float((offers[loyal_mask] > 0).mean() * 100))
        new_pct.append(float((offers[new_mask] > 0).mean() * 100))
        means.append(float(offers.mean()))

    return ThresholdSensitivityResult(
        thresholds=list(thresholds),
        tog=togs,
        pct_targeted=targeted,
        pct_loyal_offered=loyal_pct,
        pct_new_offered=new_pct,
        mean_offer=means,
        default_threshold=DEFAULT_CHURN_MIN,
    )
