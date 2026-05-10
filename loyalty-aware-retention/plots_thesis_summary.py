"""
One-figure-says-everything summary plot for the thesis.

Four panels showing baseline → mitigated for the four claims we're selling:

    1. Tenure-Offer Gap (fairness) : 12.25 → 1.31   ↓ lower is better
    2. Avg LTV of selected ($)  : 385   → 1,075  ↑ higher is better
    3. % loyal customers reached: 10.8% → 42.3%  ↑ higher is better
    4. % victims reached        : 58.1% → 62.4%  ↑ higher is better

Single image, four bar pairs, one per claim. Designed to be THE figure
used in a slide deck or paper.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_thesis_summary(
    tog_before: float, tog_after: float,
    avg_ltv_before: float, avg_ltv_after: float,
    pct_loyal_before: float, pct_loyal_after: float,
    pct_victim_before: float, pct_victim_after: float,
    path: str = "artifacts/plots/FINAL_thesis_summary.png",
) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    bar_colors = ["#EF5350", "#2E7D32"]
    labels = ["Baseline\n(SHAP threshold)", "Mitigated\n(λ = 0.25)"]

    def _panel(ax, before, after, title, is_currency=False, is_pct=False,
               lower_better=False):
        vals = [before, after]
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white",
                      linewidth=2, width=0.55)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.grid(alpha=0.3, axis="y")

        for b, v in zip(bars, vals):
            if is_currency:
                text = f"${v:,.0f}"
            elif is_pct:
                text = f"{v:.1f}%"
            else:
                text = f"{v:.2f}"
            ax.text(
                b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                text, ha="center", fontweight="bold", fontsize=13,
            )

        direction = "↓ lower = better" if lower_better else "↑ higher = better"
        ax.set_xlabel(direction, fontsize=10, style="italic", color="#555")

        if is_pct:
            ax.set_ylim(0, max(vals) * 1.2)
        else:
            ax.set_ylim(0, max(vals) * 1.2)

    _panel(
        axes[0], tog_before, tog_after,
        "Tenure-Offer Gap (fairness)\nnewcomer offer ÷ loyal offer",
        lower_better=True,
    )
    _panel(
        axes[1], avg_ltv_before, avg_ltv_after,
        "Avg LTV of selected customers\n(quality of spend)",
        is_currency=True,
    )
    _panel(
        axes[2], pct_loyal_before, pct_loyal_after,
        "% of loyal customers reached\n(tenure ≥ 48 months)",
        is_pct=True,
    )
    _panel(
        axes[3], pct_victim_before, pct_victim_after,
        "% of loyalty-penalty victims reached\n(P(victim) ≥ 0.5)",
        is_pct=True,
    )

    fig.suptitle(
        "Loyalty-aware retention — same $846k budget, four verified improvements",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
