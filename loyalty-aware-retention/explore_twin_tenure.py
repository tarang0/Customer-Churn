"""
Robustness check on the Counterfactual Tenure Flip's synthetic-twin tenure.

Our original method used synth_tenure = 3 months. Is that choice arbitrary,
or does it drive the finding? We sweep synth_tenure across 1, 3, 6, 9, 12,
18, 24 months and report how the CTF statistics move.

If the finding (mean Δ > 0, significant p-value, meaningful % victims)
holds across all of these values, then synth_tenure = 3 is a reasonable
methodological choice and the headline finding doesn't depend on it.
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

from loyalty.detect import counterfactual_tenure_flip

ARTIFACTS_PATH = "artifacts/all_artifacts.pkl"
OUT_DIR = "artifacts/plots"
os.makedirs(OUT_DIR, exist_ok=True)

LOYAL_THRESHOLD = 48
CHURN_MIN = 0.50


def main():
    print("=" * 78)
    print("  ROBUSTNESS CHECK — synthetic-twin tenure")
    print("=" * 78)

    with open(ARTIFACTS_PATH, "rb") as f:
        a = pickle.load(f)

    df = a["df"]
    churn_model = a["churn_model"]
    explainer = a["shap_explainer"]
    feat_cols = a["feature_cols"]
    feat_names = a["feature_display_names"]

    print(f"\n  Loyal threshold (fixed): {LOYAL_THRESHOLD} months")
    print(f"  Churn threshold (fixed): P(churn) >= {CHURN_MIN}")
    print(f"  Varying: synthetic-twin tenure\n")

    synth_values = [1, 3, 6, 9, 12, 18, 24]
    rows = []
    for s in synth_values:
        print(f"  [running synth_tenure = {s} mo ...]")
        ctf = counterfactual_tenure_flip(
            df, churn_model, explainer, feat_cols, feat_names,
            loyal_threshold=LOYAL_THRESHOLD,
            synth_tenure=s,
            churn_min=CHURN_MIN,
        )
        rows.append({
            "synth_tenure": s,
            "n_loyal": len(ctf.loyal_idx),
            "n_victims": int((ctf.deltas > 0).sum()),
            "pct_victims": float((ctf.deltas > 0).mean() * 100),
            "mean_delta": float(ctf.mean_delta),
            "median_delta": float(ctf.median_delta),
            "t_stat": float(ctf.t_stat),
            "p_value": float(ctf.p_value),
            "significant": ctf.is_significant,
        })

    res = pd.DataFrame(rows)

    # Print table
    print()
    print(f"  {'synth':>6} {'victims':>8} {'% victims':>10} "
          f"{'mean Δ':>10} {'median Δ':>10} {'t':>7} {'p':>10} {'sig?':>5}")
    print("  " + "-" * 72)
    for _, r in res.iterrows():
        sig = "yes" if r["significant"] else "no"
        print(f"  {int(r['synth_tenure']):>5}mo "
              f"{int(r['n_victims']):>8} "
              f"{r['pct_victims']:>9.1f}% "
              f"${r['mean_delta']:>8.2f} "
              f"${r['median_delta']:>8.2f} "
              f"{r['t_stat']:>7.2f} "
              f"{r['p_value']:>10.2g} "
              f"{sig:>5}")

    # Verdict
    all_sig = bool(res["significant"].all())
    mean_stability = float(res["mean_delta"].std() / res["mean_delta"].mean())
    pct_stability = float(res["pct_victims"].std())

    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"\n  All synth tenures produce significant findings? {all_sig}")
    print(f"  Coefficient of variation in mean Δ: {mean_stability*100:.1f}%")
    print(f"  SD of % victims across runs:         {pct_stability:.2f}%")

    default_row = res[res["synth_tenure"] == 3].iloc[0]
    extremes = res[res["synth_tenure"].isin([1, 24])]
    print()
    print(f"  Default choice (synth = 3): mean Δ = ${default_row['mean_delta']:.2f}, "
          f"% victims = {default_row['pct_victims']:.1f}%")
    print(f"  At synth = 1 (most extreme newcomer):  "
          f"mean Δ = ${extremes.iloc[0]['mean_delta']:.2f}, "
          f"% victims = {extremes.iloc[0]['pct_victims']:.1f}%")
    print(f"  At synth = 24 (least extreme):         "
          f"mean Δ = ${extremes.iloc[1]['mean_delta']:.2f}, "
          f"% victims = {extremes.iloc[1]['pct_victims']:.1f}%")

    if all_sig and mean_stability < 0.50:
        print(
            "\n  >>> ROBUST. The loyalty-penalty finding is significant at every "
            "\n      synthetic-tenure value tested, and the magnitudes are stable. "
            "\n      The specific choice of 3 months is methodologically arbitrary "
            "\n      within the newcomer range, not a driver of the result."
        )
    else:
        print("\n  >>> SENSITIVE. The finding depends on the synth-tenure choice. "
              "\n      Document carefully.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(res["synth_tenure"], res["mean_delta"],
                 marker="o", lw=2, color="#EF5350", markersize=9)
    axes[0].axvline(3, color="green", ls="--", alpha=0.5,
                    label="chosen value (3 mo)")
    for _, r in res.iterrows():
        axes[0].text(r["synth_tenure"], r["mean_delta"] + 2,
                      f"${r['mean_delta']:.1f}", ha="center", fontsize=9,
                      fontweight="bold")
    axes[0].set_xlabel("Synthetic twin tenure (months)")
    axes[0].set_ylabel("Mean Δ ($)")
    axes[0].set_title("(a) Mean per-customer penalty vs twin tenure",
                       fontweight="bold")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(res["synth_tenure"], res["pct_victims"],
                 marker="s", lw=2, color="#AB47BC", markersize=9)
    axes[1].axvline(3, color="green", ls="--", alpha=0.5)
    for _, r in res.iterrows():
        axes[1].text(r["synth_tenure"], r["pct_victims"] + 0.5,
                      f"{r['pct_victims']:.1f}%", ha="center", fontsize=9,
                      fontweight="bold")
    axes[1].set_xlabel("Synthetic twin tenure (months)")
    axes[1].set_ylabel("% of loyal customers flagged as victims")
    axes[1].set_title("(b) % victims vs twin tenure", fontweight="bold")
    axes[1].grid(alpha=0.3)

    axes[2].plot(res["synth_tenure"], -np.log10(np.maximum(res["p_value"], 1e-300)),
                 marker="^", lw=2, color="#42A5F5", markersize=9)
    axes[2].axhline(-np.log10(0.05), color="red", ls=":", alpha=0.6,
                     label="significance threshold (p = 0.05)")
    axes[2].axvline(3, color="green", ls="--", alpha=0.5)
    axes[2].set_xlabel("Synthetic twin tenure (months)")
    axes[2].set_ylabel("-log10(p-value)")
    axes[2].set_title("(c) Statistical significance vs twin tenure",
                       fontweight="bold")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(
        f"Robustness check: does CTF depend on the synthetic-tenure choice?  "
        f"(loyal ≥ {LOYAL_THRESHOLD} mo, P(churn) ≥ {CHURN_MIN})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = f"{OUT_DIR}/18_synth_tenure_robustness.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    res.to_csv("artifacts/synth_tenure_robustness.csv", index=False)
    print(f"\n  Saved plot: {out_path}")
    print(f"  Saved CSV:  artifacts/synth_tenure_robustness.csv\n")


if __name__ == "__main__":
    main()
