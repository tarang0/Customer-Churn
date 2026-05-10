# Loyalty Penalty Detection — Method write-ups

Detailed per-method documentation. Each file covers:

- **What it is**: one-paragraph summary
- **Why we need it**: what question it answers, what objection it rules out
- **How it works**: algorithm, inputs, outputs
- **What we found**: numbers, tables, interpretation
- **Diagram**: the chart from the Streamlit audit page, with a guide to reading it
- **Takeaway**: what this method adds to the overall case

## Methods in order

| # | File | Role | Scope |
|---|---|---|---|
| 1 | [`01_tenure_offer_gap.md`](01_tenure_offer_gap.md) | Group by tenure | Aggregate |
| 2 | [`02_cluster_equity.md`](02_cluster_equity.md) | Group by behavior (K-Means) | Aggregate, business framing |
| 3 | [`03_counterfactual_tenure_flip.md`](03_counterfactual_tenure_flip.md) | Per-customer twin test | Individual-level causal |
| 4 | [`04_regression_audit.md`](04_regression_audit.md) | Control for risk + value | Statistical |
| 5 | [`05_contract_controlled_tog.md`](05_contract_controlled_tog.md) | Control for contract type | Statistical |
| 6 | [`06_victim_predictor.md`](06_victim_predictor.md) | Identify victims predictively | Deployment-ready |

## Methods 1 vs 2 — why both

Methods 1 and 2 are both "group customers and compare averages" — the key difference is what the groups are.

- **Method 1** groups customers by tenure alone (one variable).
- **Method 2** groups customers by behavior using the K-Means clusters from Layer 2 (seven variables).

Method 1's split lumps "old and cheap" customers (Budget Basics) with "old and valuable" customers (Premium Loyalists). Method 2's split separates them. Method 2 can answer the business-framed question "do our most valuable customers get under-served per dollar of value?" — Method 1 cannot.

Both methods flag the penalty; having both means the finding survives under two distinct group-construction logics.

## Methods 4 & 5 — why control for multiple confounders

The skeptic's natural defenses are:

1. *"Loyal customers get smaller offers because they're low-risk."* → ruled out by **Method 4** (regression controls for churn probability and LTV).
2. *"Loyal customers get smaller offers because they're on long-term contracts."* → ruled out by **Method 5** (TOG persists within month-to-month customers).

After Methods 4 and 5, the remaining explanation is: **tenure itself depresses retention offers**. That is the loyalty penalty.

## Method 6 — the bridge to Phase 3

The first five methods **prove** the loyalty penalty exists. Method 6 **identifies individual victims** so Phase 3 can redirect retention budget toward them. Without Method 6, we have a statistical finding but no operational fix. With Method 6 (AUC 0.97, precision @ top-N = 74%), the finding becomes actionable.
