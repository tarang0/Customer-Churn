"""
Phase 3 — Mitigating the loyalty penalty.

Runs three allocation strategies on the same data and compares them:
    A. Baseline              — today's system (SHAP-threshold on churn risk only)
    B. λ-penalised           — single-pool, loyalty-aware objective
    C. Two-pool (α-split)    — explicit loyalty-reward pool

Prints before/after Tenure-Offer Gap, cluster equity, and coverage numbers for each,
sweeps λ to produce a Pareto frontier, and saves plots to artifacts/plots/.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from loyalty.detect import counterfactual_tenure_flip
from loyalty.mitigate import (allocate_baseline, allocate_lambda,
                              allocate_two_pool, detailed_pareto_sweep,
                              evaluate, fit_victim_scorer,
                              pareto_frontier_lambda)
from loyalty.scoring import (DEFAULT_CHURN_MIN, compute_shap_matrix,
                             offer_for_row, predicted_ltv, score_population)
from loyalty.viz import (plot_mitigation_cluster_equity,
                          plot_mitigation_quintiles, plot_pareto_frontier,
                          plot_tradeoff_curves)

ARTIFACTS_PATH = "artifacts/all_artifacts.pkl"


def _banner(title: str, char: str = "="):
    print("\n" + char * 78)
    print(f"  {title}")
    print(char * 78)


def _load():
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


def _fmt(x: float) -> str:
    return "∞" if x == float("inf") else f"{x:.2f}"


def _print_result(label: str, r):
    dec = r.decision
    print(f"\n  {label}  (strategy: {dec.strategy})")
    print(f"  {'-' * 60}")
    print(f"  Budget:              ${dec.budget:>12,.0f}")
    print(f"  Total spend:         ${dec.total_spend:>12,.0f}   "
          f"({dec.total_spend / dec.budget * 100:.1f}% of budget used)")
    print(f"  Customers selected:  {dec.n_selected:>12,}")
    print(f"    churn pool:        {int((dec.pool_assignment == 'churn').sum()):>12,}   "
          f"(${dec.churn_pool_spend:,.0f})")
    print(f"    loyalty pool:      {int((dec.pool_assignment == 'loyalty').sum()):>12,}   "
          f"(${dec.loyalty_pool_spend:,.0f})")
    print(f"  Tenure-Offer Gap:    {_fmt(r.tog_before):>12}  →  {_fmt(r.tog_after)}")
    if r.tog_improvement != float("inf"):
        print(f"  Tenure-Offer Gap improvement: {r.tog_improvement:>7.2f}")
    print(f"  Victims reached:     {r.pct_victims_reached:>11.1f}%")
    print(f"  Loyals reached:      {r.pct_loyals_reached:>11.1f}%")
    print(f"  Avg LTV of selected: ${r.avg_ltv_of_selected:>12,.0f}")
    print(f"  LTV-at-risk touched: ${r.ltv_at_risk_touched:>12,.0f}  "
          "(P(churn)×LTV summed across selected — NOT 'money saved')")
    print(f"  Loyalty-LTV touched: ${r.loyalty_ltv_touched:>12,.0f}  "
          "(P(victim)×LTV summed across selected)")

    print(f"  Cluster equity (spend per $100 LTV):")
    for c in sorted(r.cluster_equity_before):
        b = r.cluster_equity_before[c]
        a = r.cluster_equity_after[c]
        print(f"    cluster {c}:  ${b:>8.2f}  →  ${a:>8.2f}")

    print(f"  Offer by tenure quintile:")
    for i, (b, a) in enumerate(zip(r.mean_offer_by_quintile_before,
                                    r.mean_offer_by_quintile_after)):
        label = ["Q1 (new)", "Q2", "Q3", "Q4", "Q5 (loyal)"][i]
        print(f"    {label:<12} ${b:>7.2f}  →  ${a:>7.2f}")


def main():
    _banner("LOYALTY PENALTY MITIGATION — Phase 3", "#")

    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(
            f"{ARTIFACTS_PATH} missing — run `python train_all.py` first."
        )

    a = _load()
    df = a["df"]
    churn_model = a["churn_model"]
    explainer = a["shap_explainer"]
    feat_cols = a["feature_cols"]
    feat_names = a["feature_display_names"]

    print(f"\n  Dataset:              {len(df):,} customers")
    print(f"  Retention threshold:  P(churn) ≥ {DEFAULT_CHURN_MIN}")

    # --- Score the population once (same as Phase 2) ---
    print("\n  [scoring population with baseline SHAP-threshold rule ...]")
    baseline_scores, shap_mat = score_population(
        churn_model, explainer, df, feat_cols, feat_names,
    )
    churn_probs = baseline_scores["churn_prob"].values
    baseline_offers = baseline_scores["offer"].values
    print(f"  Baseline total spend: ${baseline_offers.sum():,.0f} "
          f"(= budget we'll reuse)")
    print(f"  Baseline customers targeted: {int((baseline_offers > 0).sum()):,}")
    budget = float(baseline_offers.sum())

    # --- Run Counterfactual Tenure Flip so we have victim labels for the scorer ---
    print("\n  [running Counterfactual Tenure Flip to label loyalty-penalty victims ...]")
    ctf = counterfactual_tenure_flip(
        df, churn_model, explainer, feat_cols, feat_names,
    )
    print(f"  Counterfactual Tenure Flip victims in training data: {int((ctf.deltas > 0).sum())}"
          f" / {len(ctf.loyal_idx)} loyal customers")

    # --- Fit the victim scorer + score everyone ---
    print("\n  [fitting victim-probability classifier (sklearn LogReg) ...]")
    scorer = fit_victim_scorer(df, ctf, feat_cols)
    victim_probs = scorer.predict_proba(df)
    print(f"  Mean P(victim) across population: {victim_probs.mean():.3f}")
    print(f"  Mean P(victim) among loyals (≥48mo): "
          f"{victim_probs[df['tenure'] >= 48].mean():.3f}")
    print(f"  Mean P(victim) among newcomers (≤12mo): "
          f"{victim_probs[df['tenure'] <= 12].mean():.3f}")

    # -----------------------------------------------------------------
    # Strategy A — Baseline
    # -----------------------------------------------------------------
    _banner("STRATEGY A — Baseline (today's system)")
    dec_a = allocate_baseline(
        df, churn_probs, shap_mat, feat_names, budget=budget,
    )
    res_a = evaluate(df, baseline_offers, churn_probs, victim_probs, dec_a)
    _print_result("Strategy A", res_a)

    # -----------------------------------------------------------------
    # Strategy B — λ = 2.0 (moderate loyalty weight)
    # -----------------------------------------------------------------
    _banner("STRATEGY B — λ-penalised (single pool, loyalty-weighted)")
    LAMBDA = 2.0
    dec_b = allocate_lambda(
        df, churn_probs, shap_mat, feat_names, victim_probs,
        budget=budget, lam=LAMBDA,
    )
    res_b = evaluate(df, baseline_offers, churn_probs, victim_probs, dec_b)
    _print_result(f"Strategy B (λ={LAMBDA})", res_b)

    # -----------------------------------------------------------------
    # Strategy C — α = 0.3 (30% to loyalty pool)
    # -----------------------------------------------------------------
    _banner("STRATEGY C — Two-pool (α=0.3 loyalty, 0.7 churn)")
    ALPHA = 0.3
    dec_c = allocate_two_pool(
        df, churn_probs, shap_mat, feat_names, victim_probs,
        budget=budget, alpha=ALPHA,
    )
    res_c = evaluate(df, baseline_offers, churn_probs, victim_probs, dec_c)
    _print_result(f"Strategy C (α={ALPHA})", res_c)

    # -----------------------------------------------------------------
    # Side-by-side comparison
    # -----------------------------------------------------------------
    _banner("COMPARISON — All three strategies side by side")
    comparison = pd.DataFrame([
        {
            "strategy": "A: Baseline",
            "Tenure-Offer Gap after": _fmt(res_a.tog_after),
            "victims reached": f"{res_a.pct_victims_reached:.1f}%",
            "loyals reached": f"{res_a.pct_loyals_reached:.1f}%",
            "avg LTV selected": f"${res_a.avg_ltv_of_selected:,.0f}",
            "LTV-at-risk touched": f"${res_a.ltv_at_risk_touched:,.0f}",
            "loyalty-LTV touched": f"${res_a.loyalty_ltv_touched:,.0f}",
            "spend": f"${res_a.decision.total_spend:,.0f}",
        },
        {
            "strategy": f"B: λ={LAMBDA}",
            "Tenure-Offer Gap after": _fmt(res_b.tog_after),
            "victims reached": f"{res_b.pct_victims_reached:.1f}%",
            "loyals reached": f"{res_b.pct_loyals_reached:.1f}%",
            "avg LTV selected": f"${res_b.avg_ltv_of_selected:,.0f}",
            "LTV-at-risk touched": f"${res_b.ltv_at_risk_touched:,.0f}",
            "loyalty-LTV touched": f"${res_b.loyalty_ltv_touched:,.0f}",
            "spend": f"${res_b.decision.total_spend:,.0f}",
        },
        {
            "strategy": f"C: α={ALPHA}",
            "Tenure-Offer Gap after": _fmt(res_c.tog_after),
            "victims reached": f"{res_c.pct_victims_reached:.1f}%",
            "loyals reached": f"{res_c.pct_loyals_reached:.1f}%",
            "avg LTV selected": f"${res_c.avg_ltv_of_selected:,.0f}",
            "LTV-at-risk touched": f"${res_c.ltv_at_risk_touched:,.0f}",
            "loyalty-LTV touched": f"${res_c.loyalty_ltv_touched:,.0f}",
            "spend": f"${res_c.decision.total_spend:,.0f}",
        },
    ])
    print("\n" + comparison.to_string(index=False))

    # -----------------------------------------------------------------
    # Detailed Pareto sweep + sweet spot
    # -----------------------------------------------------------------
    _banner("DETAILED SWEEP — finding the mathematical sweet spot")
    detailed, sweet_idx = detailed_pareto_sweep(
        df, churn_probs, shap_mat, feat_names, victim_probs,
        baseline_offers=baseline_offers, budget=budget,
    )
    sweet = detailed[sweet_idx]
    print(f"\n  {'λ':>6} {'T-O Gap':>7} {'N sel':>7} {'avgLTV':>8} "
          f"{'highChurn%':>11} {'victims%':>10} {'loyals%':>9} "
          f"{'fair':>6} {'qual':>6} {'vic':>6} {'comp':>7}")
    print("  " + "-" * 100)
    for i, p in enumerate(detailed):
        marker = "  <-- sweet spot" if i == sweet_idx else ""
        print(f"  {p.lam:>6.2f} {_fmt(p.tog_after):>7} {p.n_selected:>7} "
              f"${p.avg_ltv_of_selected:>7.0f} "
              f"{p.pct_high_churn_reached:>10.1f}% "
              f"{p.pct_victims_reached:>9.1f}% "
              f"{p.pct_loyals_reached:>8.1f}% "
              f"{p.fairness_score:>6.3f} {p.quality_score:>6.3f} "
              f"{p.victim_coverage_score:>6.3f} "
              f"{p.composite_score:>7.3f}{marker}")

    print(f"\n  >>> SWEET SPOT at λ = {sweet.lam}")
    print(f"      - Composite score: {sweet.composite_score:.3f}")
    print(f"      - Fairness: {sweet.fairness_score:.3f} (Tenure-Offer Gap = {_fmt(sweet.tog_after)})")
    print(f"      - Quality:  {sweet.quality_score:.3f} (avg LTV ${sweet.avg_ltv_of_selected:.0f})")
    print(f"      - Victim coverage: {sweet.victim_coverage_score:.3f} "
          f"({sweet.pct_victims_reached:.1f}% victims)")

    # -----------------------------------------------------------------
    # Simpler coarse Pareto sweep (existing)
    # -----------------------------------------------------------------
    _banner("PARETO FRONTIER — coarse sweep")
    points = pareto_frontier_lambda(
        df, churn_probs, shap_mat, feat_names, victim_probs,
        baseline_offers=baseline_offers, budget=budget,
    )
    print(f"  {'λ':>6} {'T-O Gap after':>14} {'victims%':>10} {'loyals%':>10} "
          f"{'avg LTV sel':>14} {'LTV-at-risk':>14} {'loyalty-LTV':>14}")
    print("  " + "-" * 82)
    for p in points:
        print(f"  {p.lam:>6.2f} {_fmt(p.tog_after):>10} "
              f"{p.pct_victims_reached:>9.1f}% {p.pct_loyals_reached:>9.1f}% "
              f"${p.avg_ltv_of_selected:>12,.0f} "
              f"${p.ltv_at_risk_touched:>12,.0f} "
              f"${p.loyalty_ltv_touched:>12,.0f}")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    _banner("Saving plots")
    plot_mitigation_quintiles(res_b, path="artifacts/plots/10_mitigation_quintiles_lambda.png")
    plot_mitigation_quintiles(res_c, path="artifacts/plots/10_mitigation_quintiles_twopool.png")
    plot_mitigation_cluster_equity(res_b, path="artifacts/plots/11_mitigation_cluster_lambda.png")
    plot_mitigation_cluster_equity(res_c, path="artifacts/plots/11_mitigation_cluster_twopool.png")
    plot_pareto_frontier(points)
    plot_tradeoff_curves(detailed, sweet_idx)
    print("  Saved: 10_mitigation_quintiles_lambda.png")
    print("  Saved: 10_mitigation_quintiles_twopool.png")
    print("  Saved: 11_mitigation_cluster_lambda.png")
    print("  Saved: 11_mitigation_cluster_twopool.png")
    print("  Saved: 12_pareto_frontier.png")
    print("  Saved: 13_tradeoff_curves.png")

    _banner("DONE — loyalty-penalty mitigation compared", "#")
    print("  Next: `streamlit run app.py` — the 'Mitigation Lab' page will")
    print("  render the side-by-side comparison interactively.\n")


if __name__ == "__main__":
    main()
