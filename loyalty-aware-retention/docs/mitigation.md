# Phase 3 — Mitigating the Loyalty Penalty

This document explains the mitigation strategy, the math behind it, and
exactly what claims are supported by the data (and which are not).

## The final thesis pitch

> Standard retention systems based on SHAP thresholds blindly fire offers
> at everyone above a churn-probability cutoff. They reach all at-risk
> customers but invest pennies in each. Loyal customers are ignored,
> creating a measurable loyalty penalty (Tenure-Offer Gap = 12.25 on Telco data).
>
> We propose a loyalty-aware allocator: greedy knapsack, threshold
> P(churn) ≥ 0.50, objective `P(churn) × LTV + λ × P(victim) × LTV`, with
> λ = 0.25 identified as the mathematical sweet spot through a 2D sweep
> over (threshold, λ).
>
> Same $846k budget. Average LTV of served customers rises 2.8×,
> loyal-customer reach rises 3.9×, victim coverage rises 4.3 percentage
> points, and the Tenure-Offer Gap drops from 12.25 to 1.31.
>
> We do not claim to prevent more churn — that requires A/B testing we
> don't have. We claim demonstrably better customer-quality reach and
> substantially improved fairness under the same spend. Both claims are
> verifiable from the data, and we've published the verification script.

## The formula, in plain language

For each customer *i*, compute one combined score:

```
score_i = P(churn)_i × LTV_i  +  λ × P(victim)_i × LTV_i
```

The two terms:

- **P(churn)_i × LTV_i** — the standard "prevented churn value" term. Higher
  if the customer is likely to churn AND generates a lot of revenue.
- **λ × P(victim)_i × LTV_i** — the loyalty-awareness term. Higher if the
  customer is likely to be a loyalty-penalty victim (from Method 6's
  classifier) AND generates a lot of revenue.

The knob λ controls the balance. At λ = 0 we only care about churn. At
λ = ∞ we only care about loyalty. At λ = 0.25 (our chosen value), we
balance both.

## The allocator: greedy knapsack

Given each customer's `score_i` and `cost_i` (from the SHAP-threshold
cost table), solve:

```
maximize   Σ  selected_i × score_i
subject to Σ  selected_i × cost_i  ≤  budget
           selected_i ∈ {0, 1}
```

The algorithm:

1. Compute `ratio_i = score_i / cost_i` for every customer.
2. Sort customers by descending ratio.
3. Walk down the sorted list, selecting each customer if the remaining
   budget can afford them.

This is deterministic, takes O(N log N) for 7,043 customers, and is within
rounding error of the optimal ILP for this problem. No dependencies on
`pulp` or any other solver.

## Why λ = 0.25 and threshold = 0.50

Not arbitrary. Derived from data.

**Threshold P(churn) ≥ 0.50.** The industry-typical telecom retention
cutoff. Phase 2's threshold sensitivity analysis showed the loyalty penalty
exists at every threshold from 0.1 to 0.8. At thresholds below 0.5 the
retention pool balloons to 50%+ of customers, which is operationally
infeasible. At 0.7+ the pool is very small. 0.5 is the practical middle
ground and the industry default.

**λ = 0.25.** A 2D sweep over 50 combinations of (threshold, λ) scored each
point by a composite metric (geometric mean of fairness × quality ×
victim-coverage). At threshold 0.50, the sweet spot is λ = 0.25 with a
composite score of 0.967 out of 1.00.

Interesting fact from the sweep: for all thresholds ≥ 0.50, the optimal λ
stabilizes at 0.25. For permissive thresholds (0.30–0.40), the optimal λ
shifts slightly higher (0.50–0.75). This means the chosen operating point
is **robust** within the realistic threshold range a telecom would actually
use.

## Final result — four verified improvements

At threshold = 0.50 and λ = 0.25, with the same total budget of $846,325:

| Metric | Baseline (SHAP-threshold) | Mitigated (λ = 0.25) | Change |
|---|---|---|---|
| Customers selected | 2,496 | 3,557 | +1,061 |
| **Tenure-Offer Gap** | **12.25** | **1.31** | **−10.94** |
| **Avg LTV of selected** | **$385** | **$1,075** | **×2.79** |
| **Loyal customers reached** | **10.8%** | **42.3%** | **+31.5 pp** |
| **Victims reached** | **58.1%** | **62.4%** | **+4.3 pp** |

These four rows are the four claims in the thesis pitch. Each is derived
by a deterministic formula applied to the population. Each is reproducible
with `python verify_thesis_claims.py`.

## What we CAN claim

1. **Fairness.** Tenure-Offer Gap drops 12.25 → 1.31. This is measurable, verified, and
   independent of any offer-effectiveness assumption. The distribution of
   retention offers across tenure groups is objectively more balanced.

2. **Customer-base quality.** Average LTV of selected customers triples
   under the same spend. This is a direct consequence of the allocator
   preferring high-LTV customers over low-LTV ones.

3. **Loyal-customer coverage.** 3.9× more loyal customers receive a
   retention offer. Computed directly from the selection output.

4. **Victim coverage.** 4.3 percentage points more identified victims are
   reached. Measured against Method 6's classifier output.

5. **Compliance posture.** The allocator's rules are transparent,
   auditable, and tunable (via λ) without opaque heuristics. A regulator
   can inspect the scoring function, the classifier coefficients, and the
   selection algorithm.

## What we CANNOT claim

- **"We prevent more churn."** Unknown. Requires an uplift model trained
  on treatment/control data.
- **"We save the company more money."** Unknown. Depends on offer
  effectiveness, which we can't measure.
- **"Our offers are more effective."** Unknown for the same reason.

We have been explicit about this in every report, table, and script.

## The honest trade-off

The allocator re-ranks customers. It prefers high-value customers over
high-risk-alone customers. Because of this, it **reaches customers with a
different risk profile** than the baseline — not strictly the same
high-risk set.

Specifically, the allocator sometimes picks a loyal customer with moderate
churn probability (say 0.3) and high LTV over a short-tenure customer with
high churn probability (0.7) and low LTV. Whether this is the correct
decision depends on offer effectiveness, which is outside our data.

We frame this honestly: **the trade-off is about customer-base quality and
fairness, not about churn prevention.** Whether the trade is a net business
win requires A/B testing to answer.

## Why not an exact ILP solver?

Using `pulp` or `ortools` for the knapsack would give the provably optimal
allocation. We chose greedy-by-ratio instead because:

- **Zero extra dependencies.** No install required beyond scikit-learn
  and numpy.
- **Deterministic.** Sorting is stable; results reproduce exactly.
- **Fast.** O(N log N). A 7,043-row problem solves in milliseconds.
- **Near-optimal.** For this problem (small individual costs, large
  budget), greedy is within fractions of a percent of ILP.

If a production deployment wanted exact ILP, the allocator structure in
`loyalty/mitigate.py` is directly swappable — the greedy heuristic is a
single function call.

## Reproducing the result

```bash
cd loyalty-aware-retention

# Regenerate artifacts (optional — already saved)
python train_all.py

# Phase 2 — detection
python detect_loyalty_penalty.py

# 2D sweep — verify λ = 0.25 is optimal at threshold 0.50
python explore_lambda_threshold.py

# Phase 3 — mitigation
python mitigate_loyalty_penalty.py

# Final check — verify every thesis claim
python verify_thesis_claims.py

# Interactive dashboard
streamlit run app.py
```

The Streamlit app has a dedicated "Mitigation Lab" page showing the
final result with all the plots embedded.
