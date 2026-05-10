"""
Find the optimal LOYALTY threshold (minimum tenure in months to call a
customer "loyal").

We evaluate TWO objectives and report both, then pick the one that makes
more sense on this data:

    Objective A: score(T) = log(N_T) × (churn_pop / churn_subset_T)
        The original "pure data-driven" objective. Tends to fail on
        bounded datasets (optimizer picks the maximum tenure value).

    Objective B: "KNEE POINT"
        Find T where the subset-vs-population churn-rate gap flattens
        out (diminishing returns). Specifically: the elbow point of
        the (1 - subset_churn / population_churn) curve.
        Requires N_T >= 15% of population (sample-size floor).

External anchor: Simon-Kucher 2025 Global Telecom Study found 95% of
customer lifetime value comes from customers with 3+ years (36+ months)
of tenure. Whatever we pick should be in that general neighbourhood.
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

ARTIFACTS_PATH = "artifacts/all_artifacts.pkl"
OUT_DIR = "artifacts/plots"
os.makedirs(OUT_DIR, exist_ok=True)


def _knee_point(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index of the knee / elbow point of the (x, y) curve.

    Uses the 'maximum distance to the line from first to last point' method
    — a standard and simple elbow detector.
    """
    p1 = np.array([x[0], y[0]], dtype=float)
    p2 = np.array([x[-1], y[-1]], dtype=float)
    line_vec = p2 - p1
    line_len = float(np.linalg.norm(line_vec))
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len

    distances = []
    for xi, yi in zip(x, y):
        point = np.array([xi, yi], dtype=float)
        point_vec = point - p1
        proj_len = float(np.dot(point_vec, line_unit))
        proj_vec = proj_len * line_unit
        perp = point_vec - proj_vec
        distances.append(float(np.linalg.norm(perp)))
    return int(np.argmax(distances))


def main():
    print("=" * 78)
    print("  OPTIMAL LOYALTY-THRESHOLD SEARCH")
    print("=" * 78)

    with open(ARTIFACTS_PATH, "rb") as f:
        a = pickle.load(f)
    df = a["df"]

    pop_churn = float(df["Churn"].mean())
    pop_n = len(df)
    print(f"\n  Population: {pop_n:,} customers, churn rate = {pop_churn:.2%}")
    print(f"  Tenure range: 0 to {int(df['tenure'].max())} months\n")

    # Sample-size floor: require at least 15% of the population in the subset
    MIN_FRACTION = 0.15
    MIN_N = int(MIN_FRACTION * pop_n)
    print(f"  Constraint: loyal subset must be >= {MIN_FRACTION*100:.0f}% "
          f"of the population ({MIN_N:,} customers minimum).\n")

    thresholds = list(range(12, 69 + 1, 3))  # Stop before max tenure (72)
    rows = []

    for T in thresholds:
        mask = df["tenure"] >= T
        n = int(mask.sum())
        if n < 50:
            continue
        subset_churn = float(df.loc[mask, "Churn"].mean())
        ratio = (pop_churn / subset_churn) if subset_churn > 0 else 999.0
        log_n = float(np.log(n))

        score_A = log_n * ratio
        # "How much loyalty lift?" — 1 means subset is no-churn, 0 means
        # subset is same as population.
        churn_gap = 1.0 - subset_churn / pop_churn

        rows.append({
            "threshold_months": T,
            "n_loyal": n,
            "pct_of_pop": n / pop_n * 100,
            "subset_churn_rate": subset_churn,
            "ratio_pop_over_subset": ratio,
            "churn_gap": churn_gap,
            "log_n": log_n,
            "score_A": score_A,
            "meets_min_n": n >= MIN_N,
        })

    sweep = pd.DataFrame(rows)

    # Objective A: max of (log N × ratio) under the min-N constraint
    feasible = sweep[sweep["meets_min_n"]].copy()
    if len(feasible) == 0:
        print("  No threshold meets the min-N constraint. Relaxing.")
        feasible = sweep.copy()
    best_A_idx = feasible["score_A"].idxmax()
    best_A = feasible.loc[best_A_idx]

    # Objective B: knee point of churn-gap vs threshold, within feasible set
    knee_idx_local = _knee_point(
        feasible["threshold_months"].values.astype(float),
        feasible["churn_gap"].values,
    )
    best_B = feasible.iloc[knee_idx_local]

    # Print table
    print(f"  {'T':>4} {'n':>6} {'% pop':>7} {'subset χ':>11} "
          f"{'ratio':>7} {'gap':>7} {'score_A':>9}")
    print("  " + "-" * 68)
    for _, r in sweep.iterrows():
        feas = "" if r["meets_min_n"] else " (N<min)"
        marker = ""
        if r["threshold_months"] == int(best_A["threshold_months"]):
            marker = "  <-- A"
        if r["threshold_months"] == int(best_B["threshold_months"]):
            marker += "  <-- B (knee)"
        print(f"  {int(r['threshold_months']):>4} "
              f"{int(r['n_loyal']):>6} "
              f"{r['pct_of_pop']:>6.1f}% "
              f"{r['subset_churn_rate']:>10.2%} "
              f"{r['ratio_pop_over_subset']:>7.2f} "
              f"{r['churn_gap']:>7.2%} "
              f"{r['score_A']:>9.3f}{feas}{marker}")

    # Decide — prefer knee point when the two disagree, because objective A
    # always pushes toward the upper feasible bound.
    choice = best_B
    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)

    print(f"\n  Objective A  (log N × ratio, feasible):  T = "
          f"{int(best_A['threshold_months'])} months")
    print(f"    n = {int(best_A['n_loyal']):,}, subset churn "
          f"{best_A['subset_churn_rate']:.2%}, gap {best_A['churn_gap']:.1%}")

    print(f"\n  Objective B  (knee of churn-gap curve):  T = "
          f"{int(best_B['threshold_months'])} months")
    print(f"    n = {int(best_B['n_loyal']):,}, subset churn "
          f"{best_B['subset_churn_rate']:.2%}, gap {best_B['churn_gap']:.1%}")

    print(f"\n  External anchor:")
    print(f"    Simon-Kucher 2025 Global Telecom Study: 95% of CLTV from "
          f"customers with 3+ years (36+ months).")

    print(f"\n  >>> Recommended: T* = {int(choice['threshold_months'])} months")
    print(f"      (knee point — balances loyalty signal strength with sample size)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    ax.plot(sweep["threshold_months"], sweep["subset_churn_rate"] * 100,
             marker="o", lw=2, color="#EF5350", markersize=7)
    ax.axhline(pop_churn * 100, color="black", ls="--", alpha=0.5,
                label=f"population churn = {pop_churn*100:.2f}%")
    ax.axvline(int(choice["threshold_months"]), color="green", ls="--", alpha=0.5,
                label=f"chosen T* = {int(choice['threshold_months'])} mo")
    ax.set_xlabel("Loyalty threshold (months)")
    ax.set_ylabel("Subset churn rate (%)")
    ax.set_title("(a) Subset churn rate falls as threshold grows",
                  fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(sweep["threshold_months"], sweep["n_loyal"],
             marker="s", lw=2, color="#42A5F5", markersize=7)
    ax.axhline(MIN_N, color="red", ls=":", alpha=0.6,
                label=f"min-N floor ({MIN_N:,})")
    ax.axvline(int(choice["threshold_months"]), color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Loyalty threshold (months)")
    ax.set_ylabel("Number of loyal customers")
    ax.set_title("(b) Subset size shrinks as threshold grows",
                  fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    # Knee plot — the headline figure
    ax = axes[2]
    colors = []
    for T in sweep["threshold_months"]:
        if T == int(choice["threshold_months"]):
            colors.append("#2E7D32")
        elif T == int(best_A["threshold_months"]):
            colors.append("#FFA726")
        else:
            colors.append("#BDBDBD")
    ax.plot(feasible["threshold_months"], feasible["churn_gap"] * 100,
             lw=2, color="#AB47BC", alpha=0.3)
    ax.scatter(sweep["threshold_months"], sweep["churn_gap"] * 100,
                c=colors, s=80, edgecolor="white", zorder=3)
    ax.axvline(int(choice["threshold_months"]), color="green", ls="--", alpha=0.5,
                label=f"knee point = {int(choice['threshold_months'])} mo")
    ax.axvline(36, color="orange", ls=":", alpha=0.6,
                label="Simon-Kucher anchor (36 mo)")
    ax.set_xlabel("Loyalty threshold (months)")
    ax.set_ylabel("Churn-rate gap: (1 − subset/population) × 100")
    ax.set_title(f"(c) Knee of churn-gap curve → T* = {int(choice['threshold_months'])} months",
                  fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Optimal loyalty-threshold search (data-driven)",
                  fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = f"{OUT_DIR}/17_loyalty_threshold_sweep.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    sweep.to_csv("artifacts/loyalty_threshold_sweep.csv", index=False)
    print(f"\n  Saved plot: {out_path}")
    print(f"  Saved CSV:  artifacts/loyalty_threshold_sweep.csv\n")


if __name__ == "__main__":
    main()
