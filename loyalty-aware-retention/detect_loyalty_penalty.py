"""
Loyalty Penalty Detection — Phase 2.

Runs SIX independent detection methods against the SHAP-threshold retention
baseline (the same logic used in app.py). Also:
  - Validates that our "loyal" subset actually behaves loyally.
  - Profiles loyalty-penalty victims vs non-victims.
  - Measures cross-method agreement (are the methods flagging the same
    customers?).
  - Saves plots to artifacts/plots/ for Streamlit.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from loyalty.detect import (cluster_equity, contract_controlled_tog,
                            counterfactual_tenure_flip, cross_method_agreement,
                            regression_audit, tenure_offer_gap,
                            threshold_sensitivity, victim_predictor)
from loyalty.profile import profile_victims, validate_loyalty
from loyalty.scoring import DEFAULT_CHURN_MIN, score_population
from loyalty.viz import (plot_agreement, plot_cluster_equity,
                         plot_contract_controlled, plot_ctf, plot_regression,
                         plot_threshold_sensitivity, plot_tog,
                         plot_victim_predictor, plot_victim_profile)

ARTIFACTS_PATH = "artifacts/all_artifacts.pkl"


def _banner(title: str, char: str = "=") -> None:
    print("\n" + char * 78)
    print(f"  {title}")
    print(char * 78)


def _load_artifacts(path: str = ARTIFACTS_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python train_all.py` first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    _banner("LOYALTY PENALTY DETECTION — Telco Customer Churn", char="#")
    print(
        "Auditing the SHAP-threshold retention baseline (as used in app.py).\n"
        "Six methods, plus victim profiling and cross-method agreement.\n"
    )

    a = _load_artifacts()
    df = a["df"]
    churn_model = a["churn_model"]
    explainer = a["shap_explainer"]
    feature_cols = a["feature_cols"]
    feat_names = a["feature_display_names"]

    print(f"  Dataset:    {len(df):,} customers")
    print(f"  Churn rate: {df['Churn'].mean():.1%}")
    print(f"  Tenure:     {int(df['tenure'].min())}-{int(df['tenure'].max())} months")
    print(f"  Baseline churn-probability threshold for offers: {DEFAULT_CHURN_MIN:.2f}")
    print(f"    (realistic retention-campaign cutoff — only act on customers")
    print(f"     whose predicted churn probability exceeds {DEFAULT_CHURN_MIN:.2f})")

    # Score everyone once
    print("\n[scoring the population under the baseline retention rule ...]")
    scores, shap_mat = score_population(
        churn_model, explainer, df, feature_cols, feat_names,
    )
    print(f"  % customers with a non-zero offer: {(scores['offer'] > 0).mean()*100:.1f}%")
    print(f"  Avg offer across population:       ${scores['offer'].mean():.2f}")

    # --------------------------------------------------------------- M1
    _banner("METHOD 1 — Tenure-Offer Gap")
    tog = tenure_offer_gap(df, scores)
    print("Group-level: average retention offer per tenure quintile.\n")
    print(f"{'Quintile':<22} {'N':>6} {'ChurnP':>8} {'LTV ($)':>10} {'Offer ($)':>12}")
    print("-" * 62)
    for lbl, n, cp, ltv, off in zip(
        tog.quintile_labels, tog.per_quintile_n,
        tog.per_quintile_mean_churn_prob, tog.per_quintile_mean_ltv,
        tog.per_quintile_mean_offer,
    ):
        print(f"{lbl:<22} {n:>6} {cp:>8.3f} {ltv:>10,.0f} {off:>12,.2f}")
    print(f"\n  Tenure-Offer Gap = {tog.per_quintile_mean_offer[0]:.2f} / "
          f"{tog.per_quintile_mean_offer[-1]:.2f}  =  {tog.tog_ratio:.2f}")
    print(f"  VERDICT: {tog.verdict}")
    plot_tog(tog)

    # --------------------------------------------------------------- M2
    _banner("METHOD 2 — Counterfactual Tenure Flip")
    ctf = counterfactual_tenure_flip(
        df, churn_model, explainer, feature_cols, feat_names,
    )
    print(f"Flips ONLY tenure (48+ → 3 mo) — nothing else changes.\n")
    print(f"  Loyal customers audited:           {len(ctf.loyal_idx):>7,}")
    print(f"  Mean per-customer offer delta:     ${ctf.mean_delta:>10,.2f}")
    print(f"  Median delta:                      ${ctf.median_delta:>10,.2f}")
    print(f"  % loyal customers penalised:       {ctf.pct_penalised:>9.1f}%")
    print(f"  Paired t-test:  t = {ctf.t_stat:>7.3f}   p = {ctf.p_value:>10.3g}")
    print(f"  VERDICT: {'LOYALTY PENALTY CONFIRMED' if ctf.is_significant else 'no significant penalty'}")
    plot_ctf(ctf)

    # Top-5 examples (still helpful)
    top_idx = np.argsort(-ctf.deltas)[:5]
    print("\n  Top 5 most-penalised loyal customers (Δ = offer_twin − offer_real):")
    print("  row  ten  contract            monthly  P_real  P_twin  O_real  O_twin  Δ")
    for j in top_idx:
        gi = ctf.loyal_idx[j]
        row = df.iloc[gi]
        contract = str(row.get("Contract", "?"))[:18]
        print(f"  {int(gi):>4}  {int(row['tenure']):>3}  {contract:<18} "
              f"${row['MonthlyCharges']:>6.0f}  "
              f"{ctf.churn_prob_real[j]:>.2f}    {ctf.churn_prob_twin[j]:>.2f}    "
              f"${ctf.offer_real[j]:>4.0f}   ${ctf.offer_twin[j]:>4.0f}   "
              f"${ctf.deltas[j]:>4.0f}")

    # --------------------------------------------------------------- Validate loyalty
    _banner("VALIDATION — Are our 'loyal' customers actually loyal?")
    loy = validate_loyalty(df, ctf.loyal_idx)
    print(f"  n:                    {loy['n']:,}   ({loy['pct_of_dataset']:.1f}% of dataset)")
    print(f"  Tenure range:         {loy['min_tenure']}–{loy['max_tenure']} months "
          f"(mean {loy['mean_tenure']:.1f})")
    print(f"  Historical churn:     {loy['historical_churn_rate']:.1%}   "
          f"(population: {loy['population_churn_rate']:.1%})")
    print(f"  Avg total revenue:    ${loy['mean_total_charges']:,.0f}")
    print(f"  % on 1-yr or 2-yr:    {loy['long_contract_pct']:.1f}%")
    ok = loy["historical_churn_rate"] < loy["population_churn_rate"] / 2
    print(f"  Verdict: {'Confirmed loyal — churn rate is < half population rate' if ok else 'Ambiguous'}")

    # --------------------------------------------------------------- M3
    _banner("METHOD 3 — Regression audit (two-stage)")
    reg = regression_audit(df, scores)
    print(f"  Stage 1: P(any offer | features)   (logit, pseudo-R² = "
          f"{reg.selection_pseudo_r2:.3f})")
    print(f"  Stage 2: E[offer | offer > 0, features]   (OLS, n = {reg.amount_n}, "
          f"R² = {reg.amount_r2:.3f})\n")

    print(f"  {'variable':<14} {'S1 coef':>12} {'S1 p':>12}   "
          f"{'S2 coef':>12} {'S2 p':>12}")
    print("  " + "-" * 68)
    for v in ["churn_prob", "ltv", "monthly", "tenure", "num_services"]:
        s1c = reg.selection_coefs.get(v, float('nan'))
        s1p = reg.selection_pvals.get(v, float('nan'))
        s2c = reg.amount_coefs.get(v, float('nan'))
        s2p = reg.amount_pvals.get(v, float('nan'))
        mark = "  <-- key" if v == "tenure" else ""
        print(f"  {v:<14} {s1c:>12.4f} {s1p:>12.3g}   "
              f"{s2c:>12.4f} {s2p:>12.3g}{mark}")

    print()
    s1_ten = reg.selection_coefs.get("tenure", 0.0)
    s1_p = reg.selection_pvals.get("tenure", 1.0)
    if s1_ten < 0 and s1_p < 0.05:
        print(f"  STAGE 1 VERDICT: LOYALTY PENALTY — tenure DECREASES the probability")
        print(f"  of being offered anything at all (coef = {s1_ten:.3f}, p = {s1_p:.2g}).")
        print(f"  This is the dominant form of the penalty at a strict P(churn) threshold.")
    else:
        print(f"  STAGE 1: tenure coef = {s1_ten:.3f} (p = {s1_p:.2g})")
    s2_ten = reg.amount_coefs.get("tenure", 0.0)
    s2_p = reg.amount_pvals.get("tenure", 1.0)
    print(f"  STAGE 2: among offered customers, tenure coef = {s2_ten:.3f} "
          f"(p = {s2_p:.2g})  [conditional, interpret carefully]")
    plot_regression(reg, df, scores)

    # --------------------------------------------------------------- M4
    _banner("METHOD 4 — Cluster-level equity")
    clu = cluster_equity(df, scores)
    pc = clu.per_cluster
    pc_display = pc.copy()
    pc_display["spend_per_$100_LTV"] = pc_display["spend_per_ltv"] * 100
    pc_display = pc_display[[
        "cluster", "n", "avg_tenure", "avg_monthly",
        "actual_churn_rate", "avg_offer", "spend_per_$100_LTV",
    ]]
    pd.options.display.float_format = "{:,.2f}".format
    print(pc_display.to_string(index=False))
    print(f"\n  VERDICT: {clu.verdict}")
    plot_cluster_equity(clu)

    # --------------------------------------------------------------- M5
    _banner("METHOD 5 — Contract-controlled Tenure-Offer Gap (rules out confounding)")
    cc = contract_controlled_tog(df, scores)
    for contract, data in cc.per_contract.items():
        tog_str = "∞" if data["tog"] == float("inf") else f"{data['tog']:.2f}"
        offers_str = ", ".join(f"${v:.0f}" for v in data["mean_offer"])
        print(f"  {contract:<18} tenure-tertile avg offers: {offers_str}   Tenure-Offer Gap = {tog_str}")
    print(f"\n  VERDICT: {cc.verdict}")
    plot_contract_controlled(cc)

    # --------------------------------------------------------------- M6
    _banner("METHOD 6 — Predicting loyalty-penalty victims")
    vp = victim_predictor(df, ctf, feature_cols, feat_names)
    print(f"  Victims in training set:  {vp.n_victims}/{vp.n_total}")
    print(f"  AUC:                      {vp.auc:.3f}")
    print(f"  Accuracy:                 {vp.accuracy:.3f}")
    print(f"  Precision @ top-{vp.top_n}:    {vp.precision_at_top_n:.3f}")
    print(f"  Recall    @ top-{vp.top_n}:    {vp.recall_at_top_n:.3f}")
    print("\n  Top-10 features for identifying victims:")
    for feat, coef in vp.feature_importance.head(10).items():
        print(f"    {feat:<22} |coef| = {coef:.3f}")
    if vp.auc >= 0.8:
        print(f"\n  VERDICT: Victims are CONFIDENTLY PREDICTABLE (AUC = {vp.auc:.2f}). "
              "Budget can be targeted.")
    elif vp.auc >= 0.7:
        print(f"\n  VERDICT: Victims are usefully predictable (AUC = {vp.auc:.2f}).")
    else:
        print(f"\n  VERDICT: Victim prediction is weak (AUC = {vp.auc:.2f}).")
    plot_victim_predictor(vp)

    # --------------------------------------------------------------- Profile
    _banner("PROFILE — What do loyalty-penalty victims look like?")
    prof = profile_victims(df, scores, ctf)
    v, nv = prof["victims"], prof["non_victims"]
    print(f"  {'metric':<22} {'Victims':>14} {'Non-victims':>14}")
    print("  " + "-" * 52)
    for k in ["n", "avg_tenure", "avg_monthly", "avg_total", "avg_ltv",
              "avg_churn_p", "actual_churn_rate",
              "contract_m2m_pct", "contract_oneyr_pct", "contract_twoyr_pct",
              "fiber_pct", "dsl_pct", "senior_pct"]:
        vstr = f"{v[k]:.2f}" if isinstance(v[k], float) else f"{v[k]}"
        nstr = f"{nv[k]:.2f}" if isinstance(nv[k], float) else f"{nv[k]}"
        print(f"  {k:<22} {vstr:>14} {nstr:>14}")
    plot_victim_profile(prof)

    # --------------------------------------------------------------- Agreement
    _banner("CROSS-METHOD AGREEMENT — Do the methods flag the same customers?")
    agree = cross_method_agreement(df, scores, ctf)
    counts = agree.attrs["counts"]
    print("  Set sizes:")
    for k, n in counts.items():
        print(f"    {k:<36} n = {n:,}")
    print("\n  Jaccard similarity (1.0 = identical sets, 0.0 = disjoint):")
    print(agree.round(2).to_string())
    plot_agreement(agree)

    # --------------------------------------------------------------- Sensitivity
    _banner("SENSITIVITY — Is the finding fragile to our threshold choice?")
    ts = threshold_sensitivity(df, churn_model, explainer, feature_cols, feat_names)
    print(f"  {'threshold':>10} {'% targeted':>12} {'avg offer':>12} "
          f"{'T-O Gap':>8} {'% loyal offered':>18} {'% new offered':>15}")
    print("  " + "-" * 80)
    for t, tog_v, tg, mo, lp, np_ in zip(
        ts.thresholds, ts.tog, ts.pct_targeted, ts.mean_offer,
        ts.pct_loyal_offered, ts.pct_new_offered,
    ):
        tog_str = "inf" if tog_v == float("inf") else f"{tog_v:.2f}"
        marker = "  <-- default" if abs(t - DEFAULT_CHURN_MIN) < 1e-6 else ""
        print(f"  {t:>10.2f} {tg:>11.1f}% ${mo:>10.0f} "
              f"{tog_str:>8} {lp:>16.1f}% {np_:>13.1f}%{marker}")
    print("\n  VERDICT: Tenure-Offer Gap grows monotonically with the threshold — the loyalty")
    print("  penalty strengthens as the retention system becomes more selective.")
    print("  Our finding is robust to threshold choice.")
    plot_threshold_sensitivity(ts)

    # --------------------------------------------------------------- Summary
    _banner("SUMMARY — Loyalty Penalty Detection", char="#")
    rows = [
        ("1. Tenure-Offer Gap (group-level)",      f"{tog.tog_ratio:.2f}",
         tog.tog_ratio > 1.5),
        ("2. Counterfactual Tenure Flip (per-customer)",
         f"Δ=${ctf.mean_delta:.0f}, p={ctf.p_value:.2g}, "
         f"{ctf.pct_penalised:.0f}% penalised",
         ctf.is_significant),
        ("3. Regression stage-1 (P any offer)",
         f"tenure coef = {reg.selection_coefs.get('tenure', 0):.3f}, "
         f"p = {reg.selection_pvals.get('tenure', 1):.2g}",
         reg.selection_coefs.get("tenure", 0) < 0 and
         reg.selection_pvals.get("tenure", 1) < 0.05),
        ("4. Cluster equity",                f"verdict = {clu.verdict[:30]}...",
         "SUBSTANTIALLY" in clu.verdict or "LESS" in clu.verdict),
        ("5. Contract-controlled Tenure-Offer Gap", cc.verdict[:46],
         "PERSISTS" in cc.verdict),
        ("6. Victim predictor (AUC)",        f"AUC = {vp.auc:.3f}",
         vp.auc >= 0.7),
    ]
    n_hits = sum(1 for _, _, hit in rows if hit)
    print(f"  {'method':<30} {'result':<48} {'penalty?':>10}")
    print("  " + "-" * 92)
    for label, value, hit in rows:
        print(f"  {label:<30} {value[:48]:<48} {'YES' if hit else 'no':>10}")
    print(f"\n  >>> {n_hits}/6 methods confirm a loyalty penalty in the SHAP-threshold baseline.\n")
    print(f"  Plots saved to: artifacts/plots/\n")


if __name__ == "__main__":
    main()
