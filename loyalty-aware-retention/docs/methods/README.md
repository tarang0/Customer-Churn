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
2. *"Loyal customers get smaller offers because they're on long-term contracts."* → ruled out by **Method 5** (the Tenure-Offer Gap persists within month-to-month customers).

After Methods 4 and 5, the remaining explanation is: **tenure itself depresses retention offers**. That is the loyalty penalty.

## Method 6 — the bridge to Phase 3

The first five methods **prove** the loyalty penalty exists. Method 6 **identifies individual victims** so Phase 3 can redirect retention budget toward them. Without Method 6, we have a statistical finding but no operational fix. With Method 6 (AUC 0.97, precision @ top-N = 74%), the finding becomes actionable.

## Parameter choices (all derived from analysis, not arbitrary)

The audit and mitigation pipelines use three fixed constants. Each was
chosen deliberately and has a reproducible justification. None are exposed
as user-adjustable dials — they are **findings**, not knobs.

### 1. Retention firing threshold: `P(churn) ≥ 0.50`

The retention system fires an offer only when the XGBoost churn probability
for a customer exceeds 0.50.

**Why 0.50:**
- It's the industry-typical telecom retention cutoff (the point where the
  model is more confident a customer will churn than stay).
- At lower thresholds (0.10–0.30) the retention pool balloons to 45–60% of
  the customer base — operationally infeasible for real campaigns.
- At higher thresholds (0.70+) the pool is very small (< 25%) — loyalty
  penalty is still present but the sample gets thin.
- Phase 2's threshold sensitivity analysis (`09_threshold_sensitivity.png`)
  confirmed the Tenure-Offer Gap is present at every threshold from 0.10 to
  0.80 and grows monotonically as the threshold rises. The finding is not
  an artifact of this specific cutoff.

**File:** `loyalty/scoring.py` → `DEFAULT_CHURN_MIN = 0.50`

### 2. Loyalty threshold: `tenure ≥ 48 months` (4 years)

A customer is considered "loyal" if they have been with the company at
least 48 months. This is the threshold used to build the loyal subset for
the Counterfactual Tenure Flip, Victim Predictor, and victim profiling.

**Why 48 months:**
- The **Simon-Kucher 2025 Global Telecommunications Study** found that
  *"95% of customer lifetime value comes from customers with three or more
  years of tenure. These customers account for 75% of the subscriber
  base."* ([source](https://www.simon-kucher.com/en/insights/loyalty-pays-monetization-insights-global-telecommunications-study-2025)).
  Our 48-month threshold sits cleanly in the "loyal" region of that finding.
- We validated this choice by sweeping tenure thresholds from 12 to 69
  months (`explore_loyalty_threshold.py`, output in
  `artifacts/loyalty_threshold_sweep.csv` and
  `17_loyalty_threshold_sweep.png`). Our data-driven optimizer (`score = log N × (population_churn / subset_churn)`) with a 15%-of-population
  sample-size floor peaked at T = 63 months. The elbow-point of the
  loyalty-lift curve sat at T = 18 months. **48 months is between the two
  and aligns with the Simon-Kucher anchor.**
- Verification that 48-month loyalists behave loyally: 2,303 customers
  (32.7% of population), 9.6% historical churn rate (vs 26.5% population),
  $4,641 average total revenue, 84.5% on 1-year or 2-year contracts. This
  check is in Section 2 ("Are our loyal customers actually loyal?") of the
  Loyalty Audit page and is part of `verify_thesis_claims.py`.

**File:** `loyalty/detect.py` → `counterfactual_tenure_flip(loyal_threshold=48)`

### 3. Synthetic-twin tenure: `3 months`

In the Counterfactual Tenure Flip (Method 3), each real loyal customer is
compared against a synthetic twin with tenure set to 3 months, every other
feature kept identical.

**Why 3 months:**
- 3 months represents a "newcomer" who has been through one billing cycle —
  the opposite of a loyal customer, and the framing most useful for asking
  *"would the retention system treat this person differently if it saw
  them as new?"*
- The specific value of 3 is methodologically arbitrary within the
  newcomer range. We verified this in `explore_twin_tenure.py` by
  sweeping synthetic tenures across 1, 3, 6, 9, 12, 18, and 24 months.
  Result: the Counterfactual Tenure Flip finding is statistically
  significant (p < 10⁻⁸) at every tested value. Mean per-customer Δ
  ranges from $9.90 (at synth = 18) to $29.58 (at synth = 1), with our
  chosen 3 producing $24.10 — near the middle of the range.
- See `artifacts/plots/18_synth_tenure_robustness.png` and
  `artifacts/synth_tenure_robustness.csv` for the full table.

**File:** `loyalty/detect.py` → `counterfactual_tenure_flip(synth_tenure=3)`

### Summary table

| Parameter | Value | How determined | Supporting artifact |
|---|---|---|---|
| Retention firing threshold | `P(churn) ≥ 0.50` | Industry convention; robustness verified | `09_threshold_sensitivity.png` |
| Loyalty threshold | `tenure ≥ 48 months` | Within Simon-Kucher 2025 anchor (36+); optimizer supported range | `17_loyalty_threshold_sweep.png` |
| Synthetic-twin tenure | `3 months` | Arbitrary within newcomer range; finding robust across 1–24 mo | `18_synth_tenure_robustness.png` |

All three are reproducible by running:

```bash
python explore_lambda_threshold.py      # confirms threshold = 0.50, λ = 0.25
python explore_loyalty_threshold.py     # supports loyalty ≥ 48
python explore_twin_tenure.py           # supports synth = 3
```
