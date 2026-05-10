"""
One-off exploratory script (not part of the main pipeline).

Runs a 2D sweep: for each combination of (churn_threshold, λ), compute
the composite sweet-spot score and record key metrics. Output:

  1. A CSV with one row per (threshold, λ) combination.
  2. A heatmap showing the composite score across the grid.
  3. A second heatmap showing the Tenure-Offer Gap across the grid.
  4. A table of the optimal λ at each threshold.

This lets us answer: does the sweet-spot λ depend on the threshold?
"""

from __future__ import annotations

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, ".")

from loyalty.detect import counterfactual_tenure_flip
from loyalty.mitigate import (allocate_lambda, evaluate, fit_victim_scorer)
from loyalty.scoring import predicted_ltv, score_population

ARTIFACTS_PATH = "artifacts/all_artifacts.pkl"
OUT_DIR = "artifacts/plots"


def main():
    print("=" * 70)
    print("2D SWEEP — (churn_threshold, λ) → composite score")
    print("=" * 70)

    with open(ARTIFACTS_PATH, "rb") as f:
        a = pickle.load(f)

    df = a["df"]
    churn_model = a["churn_model"]
    explainer = a["shap_explainer"]
    feat_cols = a["feature_cols"]
    feat_names = a["feature_display_names"]

    # Grids
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    lambdas = [0.0, 0.10, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"  Thresholds: {thresholds}")
    print(f"  Lambdas:    {lambdas}")
    print(f"  Total cells: {len(thresholds) * len(lambdas)}")
    print()

    # For each threshold we need:
    #   - baseline scoring (budget = total baseline offers)
    #   - Counterfactual Tenure Flip to label victims
    #   - victim classifier
    # These are expensive, so compute once per threshold.
    results = []

    for t_idx, t in enumerate(thresholds):
        print(f"  [{t_idx+1}/{len(thresholds)}] threshold = {t}")

        # Baseline scoring at this threshold
        baseline_scores, shap_mat = score_population(
            churn_model, explainer, df, feat_cols, feat_names, churn_min=t,
        )
        churn_probs = baseline_scores["churn_prob"].values
        baseline_offers = baseline_scores["offer"].values
        budget = float(baseline_offers.sum())

        # Counterfactual Tenure Flip + victim classifier at this threshold
        ctf = counterfactual_tenure_flip(
            df, churn_model, explainer, feat_cols, feat_names, churn_min=t,
        )
        scorer = fit_victim_scorer(df, ctf, feat_cols)
        victim_probs = scorer.predict_proba(df)

        # Now sweep λ at this threshold
        for lam in lambdas:
            dec = allocate_lambda(
                df, churn_probs, shap_mat, feat_names, victim_probs,
                budget=budget, lam=lam, churn_min=t,
            )
            ev = evaluate(df, baseline_offers, churn_probs, victim_probs, dec)

            results.append({
                "threshold": t,
                "lam": lam,
                "budget": budget,
                "n_selected": dec.n_selected,
                "tog_after": ev.tog_after,
                "avg_ltv_selected": ev.avg_ltv_of_selected,
                "pct_victims_reached": ev.pct_victims_reached,
                "pct_loyals_reached": ev.pct_loyals_reached,
                "high_churn_reached": _pct_high_churn(sel=dec.selected,
                                                       churn_probs=churn_probs,
                                                       cutoff=t),
            })
        print(f"        baseline Tenure-Offer Gap = {_safe_tog_row(results, t, 0.0):.2f}, "
              f"budget = ${budget:,.0f}")

    df_r = pd.DataFrame(results)

    # Composite score per threshold (normalize within each threshold row)
    df_r = _add_composite(df_r)

    # Save CSV
    csv_path = "artifacts/tradeoff_2d_sweep.csv"
    df_r.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV: {csv_path}")

    # Summary table: optimal λ per threshold
    print("\n" + "=" * 70)
    print("OPTIMAL λ AT EACH THRESHOLD")
    print("=" * 70)
    print(f"\n  {'threshold':>10} {'optimal λ':>10} {'T-O Gap':>8} {'avgLTV':>8} "
          f"{'victims%':>10} {'composite':>10}")
    print("  " + "-" * 65)
    for t in thresholds:
        sub = df_r[df_r["threshold"] == t].reset_index(drop=True)
        best_idx = sub["composite"].idxmax()
        best = sub.loc[best_idx]
        print(f"  {t:>10.2f} {best['lam']:>10.2f} "
              f"{best['tog_after']:>8.2f} ${best['avg_ltv_selected']:>6.0f} "
              f"{best['pct_victims_reached']:>9.1f}% "
              f"{best['composite']:>10.3f}")

    # Heatmap: composite score
    _plot_heatmap(
        df_r, value_col="composite",
        title="Composite score: (threshold, λ) grid — higher is better",
        cmap="YlGn", path=f"{OUT_DIR}/14_heatmap_composite.png",
        thresholds=thresholds, lambdas=lambdas,
    )
    print(f"  Saved heatmap: {OUT_DIR}/14_heatmap_composite.png")

    # Heatmap: Tenure-Offer Gap
    _plot_heatmap(
        df_r, value_col="tog_after",
        title="Tenure-Offer Gap after mitigation — lower is better",
        cmap="RdYlGn_r", path=f"{OUT_DIR}/15_heatmap_tog.png",
        thresholds=thresholds, lambdas=lambdas,
    )
    print(f"  Saved heatmap: {OUT_DIR}/15_heatmap_tog.png")

    # λ* vs threshold line plot
    _plot_lambda_star(df_r, thresholds,
                       path=f"{OUT_DIR}/16_lambda_star_vs_threshold.png")
    print(f"  Saved line plot: {OUT_DIR}/16_lambda_star_vs_threshold.png")

    print()


def _pct_high_churn(sel: np.ndarray, churn_probs: np.ndarray,
                     cutoff: float) -> float:
    mask = churn_probs >= cutoff
    if not mask.any():
        return 0.0
    return float(sel[mask].mean() * 100)


def _safe_tog_row(results, t, lam) -> float:
    for r in results:
        if r["threshold"] == t and r["lam"] == lam:
            tog = r["tog_after"]
            return tog if tog != float("inf") else 9999.0
    return float("nan")


def _add_composite(df_r: pd.DataFrame) -> pd.DataFrame:
    """Compute composite score (geometric mean of normalized components)
    within each threshold, so comparisons are fair across thresholds."""
    out = []
    for t, sub in df_r.groupby("threshold"):
        sub = sub.copy()

        # Normalize within this threshold
        tog_clean = sub["tog_after"].replace(float("inf"), np.nan)
        tog_min = tog_clean.min()
        tog_max = tog_clean.max()

        def _fair(x):
            if np.isnan(x) or x <= 0:
                return 0.0
            return float(tog_min / x)

        avg_max = sub["avg_ltv_selected"].max()
        def _qual(x): return float(x / avg_max) if avg_max > 0 else 0.0

        vic_max = sub["pct_victims_reached"].max()
        def _vic(x): return float(x / vic_max) if vic_max > 0 else 0.0

        sub["fairness"] = sub["tog_after"].apply(_fair)
        sub["quality"] = sub["avg_ltv_selected"].apply(_qual)
        sub["victim_cov"] = sub["pct_victims_reached"].apply(_vic)
        sub["composite"] = np.cbrt(
            sub["fairness"].clip(1e-9) *
            sub["quality"].clip(1e-9) *
            sub["victim_cov"].clip(1e-9)
        )
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def _plot_heatmap(df_r, value_col: str, title: str, cmap: str, path: str,
                   thresholds, lambdas):
    pivot = df_r.pivot(index="threshold", columns="lam", values=value_col)
    pivot = pivot.reindex(index=thresholds, columns=lambdas)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.2f}" for l in lambdas])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("λ (loyalty weight)")
    ax.set_ylabel("P(churn) retention threshold")
    ax.set_title(title, fontweight="bold", fontsize=13)

    for i in range(len(thresholds)):
        for j in range(len(lambdas)):
            v = pivot.values[i, j]
            if np.isnan(v):
                text = "n/a"
            else:
                text = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
            col = "black"
            if cmap == "YlGn":
                col = "white" if v > pivot.values[~np.isnan(pivot.values)].max() * 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=col)

    plt.colorbar(im, ax=ax, label=value_col)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_lambda_star(df_r, thresholds, path: str):
    best_rows = []
    for t in thresholds:
        sub = df_r[df_r["threshold"] == t].reset_index(drop=True)
        idx = sub["composite"].idxmax()
        best_rows.append({
            "threshold": t,
            "lambda_star": float(sub.loc[idx, "lam"]),
            "composite":  float(sub.loc[idx, "composite"]),
            "tog_at_star":float(sub.loc[idx, "tog_after"]),
        })
    best = pd.DataFrame(best_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(best["threshold"], best["lambda_star"], marker="o", lw=2,
                  color="#AB47BC", markersize=11)
    for _, r in best.iterrows():
        axes[0].text(r["threshold"], r["lambda_star"] + 0.08,
                      f"λ*={r['lambda_star']}", ha="center", fontweight="bold")
    axes[0].set_xlabel("P(churn) retention threshold")
    axes[0].set_ylabel("Optimal λ (λ*)")
    axes[0].set_title("Does the sweet-spot λ depend on the threshold?",
                       fontweight="bold", fontsize=13)
    axes[0].grid(alpha=0.3)

    axes[1].plot(best["threshold"], best["tog_at_star"], marker="o", lw=2,
                  color="#EF5350", markersize=11)
    for _, r in best.iterrows():
        axes[1].text(r["threshold"], r["tog_at_star"] + 0.05,
                      f"Tenure-Offer Gap={r['tog_at_star']:.2f}", ha="center",
                      fontweight="bold")
    axes[1].set_xlabel("P(churn) retention threshold")
    axes[1].set_ylabel("Tenure-Offer Gap achieved at λ*")
    axes[1].set_title("Tenure-Offer Gap achieved at the sweet-spot λ", fontweight="bold",
                       fontsize=13)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
