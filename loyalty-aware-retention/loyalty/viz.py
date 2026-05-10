"""
Matplotlib plots for the six detection methods + profiling.
All plots are also surfaced in the Streamlit "Loyalty Audit" page.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .detect import (CTFResult, ClusterEquityResult, ContractControlledTOGResult,
                     RegressionResult, TOGResult, VictimPredictorResult)

PLOT_DIR = "artifacts/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Method 1
# ---------------------------------------------------------------------------

def plot_tog(r: TOGResult, path: str = f"{PLOT_DIR}/01_tog.png") -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(r.quintile_labels)))
    axes[0].bar(r.quintile_labels, r.per_quintile_mean_offer, color=colors, edgecolor="white")
    axes[0].set_title("Avg Retention Offer by Tenure Quintile", fontweight="bold")
    axes[0].set_ylabel("Avg Offer ($)")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(r.per_quintile_mean_offer):
        axes[0].text(i, v + max(r.per_quintile_mean_offer) * 0.02, f"${v:.0f}",
                     ha="center", fontweight="bold")

    axes[1].bar(r.quintile_labels, r.per_quintile_mean_ltv,
                color="#42A5F5", edgecolor="white")
    axes[1].set_title("Avg LTV by Tenure Quintile", fontweight="bold")
    axes[1].set_ylabel("Avg LTV ($)")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(r.per_quintile_mean_ltv):
        axes[1].text(i, v + max(r.per_quintile_mean_ltv) * 0.02, f"${v:.0f}",
                     ha="center", fontweight="bold")

    axes[2].bar(r.quintile_labels, r.per_quintile_mean_churn_prob,
                color="#EF5350", edgecolor="white")
    axes[2].set_title("Avg Churn Probability by Tenure Quintile", fontweight="bold")
    axes[2].set_ylabel("P(churn)")
    axes[2].tick_params(axis="x", rotation=20)
    for i, v in enumerate(r.per_quintile_mean_churn_prob):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    fig.suptitle(
        f"METHOD 1 — Tenure-Offer Gap (TOG = {r.tog_ratio:.2f})",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Method 2
# ---------------------------------------------------------------------------

def plot_ctf(r: CTFResult, path: str = f"{PLOT_DIR}/02_ctf.png") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Distribution of per-customer deltas
    axes[0].hist(r.deltas, bins=40, color="#AB47BC", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].axvline(r.mean_delta, color="red", lw=2, ls="--",
                    label=f"mean = ${r.mean_delta:.0f}")
    axes[0].set_title("Per-customer offer delta (twin − real)", fontweight="bold")
    axes[0].set_xlabel("twin_offer − real_offer  ($)")
    axes[0].set_ylabel("# loyal customers")
    axes[0].legend()

    # Real vs twin offer scatter
    jitter = np.random.default_rng(42).normal(0, 3, len(r.offer_real))
    axes[1].scatter(r.offer_real + jitter, r.offer_twin + jitter,
                    s=12, alpha=0.4, c="#1E88E5")
    lim = max(r.offer_real.max(), r.offer_twin.max()) + 50
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="equal")
    axes[1].set_xlim(-20, lim)
    axes[1].set_ylim(-20, lim)
    axes[1].set_xlabel("Real offer  (loyal tenure)")
    axes[1].set_ylabel("Twin offer  (tenure = 3mo)")
    axes[1].set_title("Above the dashed line = loyalty-penalised", fontweight="bold")
    axes[1].legend()

    fig.suptitle(
        f"METHOD 2 — Counterfactual Tenure Flip  "
        f"(mean Δ = ${r.mean_delta:.0f}, {r.pct_penalised:.0f}% penalised, "
        f"p = {r.p_value:.2g})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Method 3
# ---------------------------------------------------------------------------

def plot_regression(
    r: RegressionResult, df: pd.DataFrame, scores: pd.DataFrame,
    path: str = f"{PLOT_DIR}/03_regression.png",
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    items = ["churn_prob", "ltv", "tenure", "monthly", "num_services"]
    s1 = [r.selection_coefs.get(k, 0.0) for k in items]
    s2 = [r.amount_coefs.get(k, 0.0) for k in items]

    # Normalize each stage's coefficients to [-1, 1] for comparability
    def _norm(vals):
        m = max(abs(v) for v in vals) if vals else 1.0
        return [v / m if m > 0 else 0.0 for v in vals]

    s1n = _norm(s1)
    s2n = _norm(s2)

    y = np.arange(len(items))
    w = 0.4
    colors_s1 = ["#EF5350" if k == "tenure" else "#42A5F5" for k in items]
    colors_s2 = ["#FFA726" if k == "tenure" else "#AB47BC" for k in items]

    axes[0].barh(y - w/2, s1n, w, color=colors_s1, edgecolor="white",
                 label=f"Stage 1: P(any offer) (pseudo-R²={r.selection_pseudo_r2:.2f})")
    axes[0].barh(y + w/2, s2n, w, color=colors_s2, edgecolor="white",
                 label=f"Stage 2: E[offer | offered] (R²={r.amount_r2:.2f}, n={r.amount_n})")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(items)
    axes[0].set_title("Two-stage regression — normalized coefficients",
                      fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    for i, (v1, v2) in enumerate(zip(s1, s2)):
        axes[0].text(s1n[i] + (0.03 if s1n[i] >= 0 else -0.03), i - w/2,
                     f"{v1:+.3f}", fontsize=8, va="center",
                     ha="left" if s1n[i] >= 0 else "right")
        axes[0].text(s2n[i] + (0.03 if s2n[i] >= 0 else -0.03), i + w/2,
                     f"{v2:+.2f}", fontsize=8, va="center",
                     ha="left" if s2n[i] >= 0 else "right")

    # Right: P(any offer) vs tenure scatter
    offered = (scores["offer"] > 0).astype(int)
    tenure_bins = pd.cut(df["tenure"], bins=np.arange(0, 80, 6))
    rate = offered.groupby(tenure_bins, observed=True).mean()
    centers = [(b.left + b.right) / 2 for b in rate.index]
    axes[1].plot(centers, rate.values, marker="o", lw=2, color="#EF5350")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].set_ylabel("P(receives any offer)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Empirical: probability of any offer vs tenure",
                      fontweight="bold")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "METHOD 3 — Two-stage regression audit",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Method 4
# ---------------------------------------------------------------------------

def plot_cluster_equity(
    r: ClusterEquityResult, path: str = f"{PLOT_DIR}/04_cluster_equity.png",
) -> str:
    pc = r.per_cluster.sort_values("avg_tenure")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = [f"C{int(c)}" for c in pc["cluster"]]

    axes[0].bar(labels, pc["avg_offer"], color="#FFA726", edgecolor="white")
    axes[0].set_title("Avg offer per cluster", fontweight="bold")
    axes[0].set_ylabel("$")
    for i, v in enumerate(pc["avg_offer"]):
        axes[0].text(i, v + max(pc["avg_offer"]) * 0.02, f"${v:.0f}",
                     ha="center", fontweight="bold")

    axes[1].bar(labels, pc["total_ltv"] / 1000, color="#42A5F5", edgecolor="white")
    axes[1].set_title("Total LTV (thousands of $)", fontweight="bold")
    axes[1].set_ylabel("$k")
    for i, v in enumerate(pc["total_ltv"]):
        axes[1].text(i, v / 1000 + max(pc["total_ltv"] / 1000) * 0.02,
                     f"${v/1000:.0f}k", ha="center", fontweight="bold")

    axes[2].bar(labels, pc["spend_per_ltv"] * 100, color="#66BB6A", edgecolor="white")
    axes[2].set_title("Retention spend per $100 LTV", fontweight="bold")
    axes[2].set_ylabel("$ / $100 LTV")
    for i, v in enumerate(pc["spend_per_ltv"] * 100):
        axes[2].text(i, v + max(pc["spend_per_ltv"] * 100) * 0.02, f"${v:.2f}",
                     ha="center", fontweight="bold")

    fig.suptitle("METHOD 4 — K-Means cluster-level equity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Method 5
# ---------------------------------------------------------------------------

def plot_contract_controlled(
    r: ContractControlledTOGResult, path: str = f"{PLOT_DIR}/05_contract_controlled.png",
) -> str:
    if not r.per_contract:
        return ""

    fig, axes = plt.subplots(1, len(r.per_contract), figsize=(6 * len(r.per_contract), 5),
                             squeeze=False)
    for ax, (contract, data) in zip(axes[0], r.per_contract.items()):
        edges = data["edges"]
        labels = [f"{int(edges[i])}-{int(edges[i+1])}mo" for i in range(len(edges) - 1)]
        ax.bar(labels, data["mean_offer"], color="#7E57C2", edgecolor="white")
        tog_v = data["tog"]
        tog_str = "∞" if tog_v == float("inf") else f"{tog_v:.2f}"
        ax.set_title(f"{contract}  (TOG = {tog_str})", fontweight="bold")
        ax.set_ylabel("Avg offer ($)")
        for i, v in enumerate(data["mean_offer"]):
            ax.text(i, v + max(data["mean_offer"]) * 0.02, f"${v:.0f}",
                    ha="center", fontweight="bold")

    fig.suptitle("METHOD 5 — Contract-controlled TOG  (rules out confounding)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Method 6
# ---------------------------------------------------------------------------

def plot_victim_predictor(
    r: VictimPredictorResult, path: str = f"{PLOT_DIR}/06_victim_predictor.png",
) -> str:
    top = r.feature_importance.head(10)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].barh(top.index, top.values, color="#26A69A", edgecolor="white")
    axes[0].set_title("Top-10 features for predicting CTF victims", fontweight="bold")
    axes[0].set_xlabel("|logistic-regression coefficient|")

    metrics_labels = ["AUC", "Accuracy", f"Precision@{r.top_n}", f"Recall@{r.top_n}"]
    metrics_values = [r.auc, r.accuracy, r.precision_at_top_n, r.recall_at_top_n]
    colors = ["#2E7D32" if v > 0.7 else ("#FB8C00" if v > 0.5 else "#C62828") for v in metrics_values]
    axes[1].bar(metrics_labels, metrics_values, color=colors, edgecolor="white")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(metrics_values):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    axes[1].set_title(f"Victim classifier quality "
                      f"({r.n_victims}/{r.n_total} victims in training)",
                      fontweight="bold")

    fig.suptitle("METHOD 6 — Predicting loyalty-penalty victims",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Profile comparison
# ---------------------------------------------------------------------------

def plot_victim_profile(
    profile: dict, path: str = f"{PLOT_DIR}/07_victim_profile.png",
) -> str:
    v, nv = profile["victims"], profile["non_victims"]
    metrics = [
        ("avg_tenure",        "Avg tenure (mo)"),
        ("avg_monthly",       "Avg monthly ($)"),
        ("avg_total",         "Avg total ($)"),
        ("avg_ltv",           "Avg LTV ($)"),
        ("avg_churn_p",       "Avg churn P"),
        ("actual_churn_rate", "Actual churn rate"),
        ("contract_m2m_pct",  "% month-to-month"),
        ("fiber_pct",         "% fiber optic"),
    ]
    names = [m[1] for m in metrics]
    vvals = [v[m[0]] for m in metrics]
    nvvals = [nv[m[0]] for m in metrics]

    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(16, 6))
    b1 = ax.bar(x - w/2, vvals, w, label=f"Victims (n={v['n']})", color="#EF5350", edgecolor="white")
    b2 = ax.bar(x + w/2, nvvals, w, label=f"Non-victims (n={nv['n']})", color="#66BB6A", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Victim vs non-victim profile (both groups are 'loyal': tenure ≥ 48mo)",
                 fontweight="bold", fontsize=13)
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            fmt = f"{h:.2f}" if h < 2 else f"{h:.0f}"
            ax.text(b.get_x() + b.get_width()/2, h, fmt,
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Cross-method agreement heatmap
# ---------------------------------------------------------------------------

def plot_agreement(
    matrix: pd.DataFrame, path: str = f"{PLOT_DIR}/08_agreement.png",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix.values, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticklabels(matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.values[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix.values[i,j] > 0.5 else "black",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_title("Do the methods flag the same customers?\n"
                 "(Jaccard similarity between flagged sets)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threshold_sensitivity(r, path: str = f"{PLOT_DIR}/09_threshold_sensitivity.png") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: TOG vs threshold
    ax = axes[0]
    tog_plot = [t if t != float("inf") else max(x for x in r.tog if x != float("inf")) * 1.2 for t in r.tog]
    ax.plot(r.thresholds, tog_plot, marker="o", lw=2, color="#EF5350",
            markersize=9, label="TOG ratio")
    ax.axhline(1.0, color="black", lw=1, ls=":", alpha=0.5, label="TOG = 1 (equal)")
    ax.axhline(1.5, color="orange", lw=1, ls="--", alpha=0.6, label="penalty threshold (1.5)")
    ax.axvline(r.default_threshold, color="#1E88E5", lw=2, ls="--", alpha=0.6,
               label=f"our default ({r.default_threshold:.2f})")
    for x, y in zip(r.thresholds, tog_plot):
        ax.text(x, y + 0.4, f"{y:.1f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_xlabel("Churn-probability threshold for firing retention action")
    ax.set_ylabel("Tenure-Offer Gap (TOG)")
    ax.set_title("Loyalty penalty persists AND strengthens as threshold rises",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    # Right: coverage by group
    ax = axes[1]
    ax.plot(r.thresholds, r.pct_new_offered, marker="o", lw=2, color="#EF5350",
            label="% newcomers (tenure ≤ 12mo) offered")
    ax.plot(r.thresholds, r.pct_loyal_offered, marker="s", lw=2, color="#2E7D32",
            label="% loyalists (tenure ≥ 48mo) offered")
    ax.axvline(r.default_threshold, color="#1E88E5", lw=2, ls="--", alpha=0.6)
    ax.set_xlabel("Churn-probability threshold")
    ax.set_ylabel("% of group receiving any offer")
    ax.set_title("Newcomers stay well-covered; loyalists are cut off", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Threshold sensitivity — is our finding fragile to the 0.5 choice?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
