"""
Streamlit page: Mitigation Lab — the final result of the project.

Tells one story: at λ = 0.25 and P(churn) threshold = 0.50, the loyalty-
aware allocator achieves four verified improvements over the SHAP-threshold
baseline using the SAME total budget.

This is not an exploratory tool — it's a presentation page. Thresholds
and λ values are derived results, not user-adjustable dials.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from loyalty.detect import counterfactual_tenure_flip, tenure_offer_gap
from loyalty.mitigate import (allocate_baseline, allocate_lambda, evaluate,
                               fit_victim_scorer)
from loyalty.scoring import predicted_ltv, score_population

ARTIFACTS_PATH = "artifacts/all_artifacts.pkl"
PLOT_DIR = "artifacts/plots"

THRESHOLD = 0.50
LAMBDA = 0.25

st.set_page_config(page_title="Mitigation Lab", layout="wide", page_icon="🎯")


# ---------------------------------------------------------------------------
# Cached heavy work
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_artifacts() -> dict:
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def _run_comparison(_a):
    df = _a["df"]
    baseline_scores, shap_mat = score_population(
        _a["churn_model"], _a["shap_explainer"], df,
        _a["feature_cols"], _a["feature_display_names"],
        churn_min=THRESHOLD,
    )
    churn_probs = baseline_scores["churn_prob"].values
    baseline_offers = baseline_scores["offer"].values
    budget = float(baseline_offers.sum())

    ctf = counterfactual_tenure_flip(
        df, _a["churn_model"], _a["shap_explainer"],
        _a["feature_cols"], _a["feature_display_names"], churn_min=THRESHOLD,
    )
    scorer = fit_victim_scorer(df, ctf, _a["feature_cols"])
    victim_probs = scorer.predict_proba(df)

    dec_mit = allocate_lambda(
        df, churn_probs, shap_mat, _a["feature_display_names"], victim_probs,
        budget=budget, lam=LAMBDA, churn_min=THRESHOLD,
    )
    res_mit = evaluate(df, baseline_offers, churn_probs, victim_probs, dec_mit)

    # Baseline TOG (from the raw SHAP-threshold offers)
    tog_raw = tenure_offer_gap(df, baseline_scores).tog_ratio

    # Derived metrics
    n_baseline = int((baseline_offers > 0).sum())
    sel_b_mask = baseline_offers > 0

    ltv_arr = np.array([
        predicted_ltv(df.iloc[i]["MonthlyCharges"], float(churn_probs[i]))
        for i in range(len(df))
    ])

    avg_ltv_b = float(ltv_arr[sel_b_mask].mean()) if n_baseline > 0 else 0.0

    loyal_mask = df["tenure"].values >= 48
    victim_mask = victim_probs >= 0.5

    pct_loyal_b = float(sel_b_mask[loyal_mask].mean() * 100) if loyal_mask.any() else 0.0
    pct_loyal_m = float(dec_mit.selected[loyal_mask].mean() * 100) if loyal_mask.any() else 0.0

    pct_victim_b = float(sel_b_mask[victim_mask].mean() * 100) if victim_mask.any() else 0.0
    pct_victim_m = float(dec_mit.selected[victim_mask].mean() * 100) if victim_mask.any() else 0.0

    return {
        "budget":             budget,
        "n_baseline":         n_baseline,
        "n_mitigated":        int(dec_mit.selected.sum()),
        "tog_before":         tog_raw,
        "tog_after":          res_mit.tog_after,
        "avg_ltv_before":     avg_ltv_b,
        "avg_ltv_after":      res_mit.avg_ltv_of_selected,
        "pct_loyal_before":   pct_loyal_b,
        "pct_loyal_after":    pct_loyal_m,
        "pct_victim_before":  pct_victim_b,
        "pct_victim_after":   pct_victim_m,
        "res_mit":            res_mit,
        "total_loyal":        int(loyal_mask.sum()),
        "n_loyal_before":     int(sel_b_mask[loyal_mask].sum()),
        "n_loyal_after":      int(dec_mit.selected[loyal_mask].sum()),
    }


def _image_if_exists(path: str, caption: str = ""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.info(
            f"Plot not found: `{path}`. Run `python mitigate_loyalty_penalty.py` "
            "and `python verify_thesis_claims.py` to generate it."
        )


def main():
    st.title("🎯 Mitigation Lab — the loyalty-aware retention allocator")
    st.caption(
        "Replaces the SHAP-threshold retention rule with a loyalty-aware "
        "allocator. Same budget, four measurable improvements."
    )

    if not os.path.exists(ARTIFACTS_PATH):
        st.error("`artifacts/all_artifacts.pkl` missing — run `python train_all.py` first.")
        return

    a = _load_artifacts()
    R = _run_comparison(a)

    # -----------------------------------------------------------------------
    # 1. The problem and the proposal
    # -----------------------------------------------------------------------
    st.header("1. Problem and proposal")
    st.markdown(
        """
**The problem.** The SHAP-threshold retention rule in `app.py` blindly fires
offers at every customer whose churn probability exceeds a cutoff. It
reaches all at-risk customers, but gives them thin bundles with low average
lifetime value. Loyal customers, whose churn probability is low, are
ignored entirely. The Tenure-Offer Gap is **12.25** — newcomers receive
about 12× the retention investment loyal customers do.

**The proposal.** Replace the SHAP-threshold rule with a **greedy-knapsack
allocator** that maximizes a loyalty-weighted objective under the same
budget:
        """
    )

    st.latex(
        r"\text{Score}_i \;=\; P(\text{churn})_i \cdot LTV_i \;+\; "
        r"\lambda \cdot P(\text{victim})_i \cdot LTV_i"
    )

    st.markdown(
        r"""
**Where:**

- $P(\text{churn})_i$ — probability that customer $i$ will churn (from the XGBoost model)
- $P(\text{victim})_i$ — probability that customer $i$ is a loyalty-penalty victim (classifier trained on the Counterfactual Tenure Flip labels, Method 6)
- $LTV_i$ — lifetime value of customer $i$, approximated as $\text{MonthlyCharges}_i \times 24 \times (1 - P(\text{churn})_i)$
- $\lambda$ — weighting factor that controls how much importance is given to victim probability

The allocator sorts customers by `score ÷ cost` and picks from the top
until the budget is exhausted.
        """
    )
    st.divider()

    # -----------------------------------------------------------------------
    # 2. Why λ = 0.25 and threshold = 0.50
    # -----------------------------------------------------------------------
    st.header("2. Why λ = 0.25 and threshold = 0.50")
    st.markdown(
        """
These two numbers are **derived from the data**, not arbitrary choices.

**Threshold P(churn) ≥ 0.50** — the telecom-standard retention cutoff.
Phase 2's threshold sensitivity analysis showed the loyalty penalty exists
at every threshold from 0.1 to 0.8, so we picked the industry norm.

**λ = 0.25** — the mathematical sweet spot. A 2D sweep over
(threshold, λ) showed that at the chosen threshold 0.50, the composite
score (fairness × quality × victim-coverage) peaks at λ = 0.25.

The heatmap below shows the composite score across the full 2D grid. The
optimum at (0.50, 0.25) is the brightest cell in the middle row.
        """
    )
    _image_if_exists(
        f"{PLOT_DIR}/14_heatmap_composite.png",
        caption="2D sweep: composite score across (threshold, λ) grid. "
                "Higher (greener) is better. Peak at our chosen operating point.",
    )
    _image_if_exists(
        f"{PLOT_DIR}/16_lambda_star_vs_threshold.png",
        caption="Sweet-spot λ as threshold varies. At threshold ≥ 0.50, "
                "λ stabilizes at 0.25. Our chosen values align with this plateau.",
    )
    st.divider()

    # -----------------------------------------------------------------------
    # 3. Headline result
    # -----------------------------------------------------------------------
    st.header("3. Headline result — four measurable improvements")
    st.markdown(
        f"**Same budget: ${R['budget']:,.0f}**. Same XGBoost model. Same retention-action cost table. "
        "Only the allocator changes."
    )

    _image_if_exists(
        f"{PLOT_DIR}/FINAL_thesis_summary.png",
        caption="Four claims. Four before/after bars. Same budget.",
    )

    st.divider()

    # -----------------------------------------------------------------------
    # 4. The numbers
    # -----------------------------------------------------------------------
    st.header("4. The numbers")

    cols = st.columns(4)
    cols[0].metric(
        "Tenure-Offer Gap (fairness)",
        f"{R['tog_after']:.2f}",
        delta=f"−{R['tog_before'] - R['tog_after']:.2f} (was {R['tog_before']:.2f})",
        delta_color="inverse",
    )
    cols[1].metric(
        "Avg LTV of selected",
        f"${R['avg_ltv_after']:,.0f}",
        delta=f"{R['avg_ltv_after'] / max(R['avg_ltv_before'], 1):.2f}× (was ${R['avg_ltv_before']:,.0f})",
    )
    cols[2].metric(
        "Loyal customers reached",
        f"{R['pct_loyal_after']:.1f}%",
        delta=f"+{R['pct_loyal_after'] - R['pct_loyal_before']:.1f} pp (was {R['pct_loyal_before']:.1f}%)",
    )
    cols[3].metric(
        "Victims reached",
        f"{R['pct_victim_after']:.1f}%",
        delta=f"+{R['pct_victim_after'] - R['pct_victim_before']:.1f} pp (was {R['pct_victim_before']:.1f}%)",
    )

    st.markdown(
        f"""
In absolute terms under the same ${R['budget']:,.0f} budget:

- **{R['n_loyal_after']:,}** loyal customers served, up from **{R['n_loyal_before']:,}** — an extra **{R['n_loyal_after'] - R['n_loyal_before']:,}** loyal customers reached.
- Average LTV of the customers we serve rises from **${R['avg_ltv_before']:,.0f} → ${R['avg_ltv_after']:,.0f}** (**{R['avg_ltv_after'] / max(R['avg_ltv_before'], 1):.2f}× higher**).
- Tenure-Offer Gap drops **{R['tog_before']:.2f} → {R['tog_after']:.2f}** — loyalty penalty essentially resolved.
        """
    )
    st.divider()

    # -----------------------------------------------------------------------
    # 5. Before / after — offer by tenure quintile
    # -----------------------------------------------------------------------
    st.header("5. How the redistribution actually looks")
    st.markdown(
        """
The quintile chart is the most literal view of what changed. The baseline
gives newcomers large offers ($243 average) and loyalists tiny ones ($20).
The mitigated allocator evens the distribution: newcomers still get
something, but so do loyalists.
        """
    )
    _image_if_exists(
        f"{PLOT_DIR}/FINAL_quintiles_before_after.png",
        caption="Average offer per tenure quintile — before vs after.",
    )

    st.markdown("**And the same picture at the K-Means cluster level:**")
    _image_if_exists(
        f"{PLOT_DIR}/FINAL_cluster_equity_before_after.png",
        caption="Spend per $100 LTV by cluster — before vs after. "
                "Premium Loyalists' share roughly doubles.",
    )
    st.divider()

    # -----------------------------------------------------------------------
    # 6. The honest trade-off
    # -----------------------------------------------------------------------
    st.header("6. The honest trade-off — what we give up")
    st.markdown(
        """
We are explicit about what the allocator sacrifices. Because it prioritizes
**higher-value customers** rather than **more at-risk customers**, it:

- Picks fewer customers whose churn probability happens to exceed 0.50.
- Targets customers with a different risk profile — not strictly the highest-risk.
- Leaves some high-risk customers uncontacted.

This is a **reach trade-off**, not a churn-prevention trade-off. Without
A/B test data, we cannot measure how much actual churn either strategy
prevents — that depends on how effective each offer is at changing a
customer's decision, which is beyond the scope of this dataset.
        """
    )

    st.info(
        "**What we can claim:** fairness, customer-quality reach, victim reach. All verified.\n\n"
        "**What we cannot claim:** prevented churn, money saved, offer effectiveness. "
        "These require treatment/control data the Telco dataset doesn't have."
    )
    st.divider()

    # -----------------------------------------------------------------------
    # 7. Where to read more
    # -----------------------------------------------------------------------
    st.header("7. Verify everything")
    st.markdown(
        """
- **`verify_thesis_claims.py`** — re-runs the comparison and checks every claim against the data.
- **`docs/mitigation.md`** — full explanation of the mitigation strategy.
- **`docs/metrics.md`** — every number in this page has its formula and inputs written out.
- **`artifacts/tradeoff_2d_sweep.csv`** — the raw 2D-sweep data showing λ = 0.25 is optimal at threshold 0.50.

All of these are deterministic and can be reproduced by cloning the repo
and running three scripts.
        """
    )


if __name__ == "__main__":
    main()
