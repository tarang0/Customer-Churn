# Metrics — Verifiable Math

Every number in this project comes from a deterministic formula applied to
the Telco dataset. This document writes out each formula, shows the inputs,
and shows the output so every claim can be independently verified.

## Dataset-level context

- **N** = 7,043 customers (IBM Telco Customer Churn, Kaggle)
- **Population churn rate** = 26.54%
- **Retention firing threshold** = P(churn) ≥ 0.50 (the industry-typical cutoff, held fixed for all claims)
- **λ** = 0.25 (sweet spot from 2D sweep at the chosen threshold)
- **Loyal-customer threshold** = tenure ≥ 48 months (default)
- **LTV horizon** = 24 months
- **Retention budget** = $846,325 (the total of all baseline SHAP-threshold offers — held fixed across strategies for a fair comparison)

## Foundational formulas

### LTV proxy

```
LTV_i = MonthlyCharges_i × 24 × (1 − P(churn)_i)
```

Tenure-independent by construction (no tenure term) so the loyalty-penalty
audit can't be blamed on tenure leaking into the LTV score.

Example: a customer paying $90/month with P(churn) = 0.10:

```
LTV = 90 × 24 × (1 − 0.10) = 90 × 24 × 0.90 = $1,944
```

### Baseline retention offer (the SHAP-threshold rule from `app.py`)

```
offer_i = 0,                                                 if P(churn)_i < 0.50
        = Σ unit_cost[f]  for features f where               otherwise
          SHAP_i[f] > 0.02  and  f ∈ retention_features
```

If the customer passes the threshold, every feature with a meaningful SHAP
contribution adds its unit cost to the offer. The retention features and
costs are in `loyalty/scoring.py` (editable; values are plausible defaults).

## Phase 2 — Detection metrics

### Method 1 — Tenure-Offer Gap

```
Tenure-Offer Gap = mean(offer)_Q1 / mean(offer)_Q5
```

where Q1 = shortest-tenure quintile, Q5 = longest-tenure quintile.

Computed from data:

| Quintile | n | avg P(churn) | avg LTV | avg offer |
|---|---|---|---|---|
| Q1 (0–6 mo) | 1,371 | 0.664 | $335 | **$243.39** |
| Q2 | 1,436 | 0.465 | $605 | $171.80 |
| Q3 | 1,415 | 0.318 | $922 | $106.14 |
| Q4 | 1,338 | 0.244 | $1,147 | $64.48 |
| Q5 (60–72 mo) | 1,483 | 0.100 | $1,590 | **$19.87** |

```
Tenure-Offer Gap = 243.39 / 19.87 = 12.25
```

Verdict threshold: Tenure-Offer Gap > 1.5 → loyalty penalty detected.

### Method 3 — Counterfactual Tenure Flip

For each loyal customer *i* (tenure ≥ 48 mo):

```
Δ_i = offer(twin_i) − offer(real_i)
```

where `twin_i` is identical to `real_i` except tenure = 3.

Summary:

- Loyal customers: **n = 2,303**
- Victims (Δ > 0): **339** (14.7% of loyals)
- Mean Δ: **$24.10**
- Median Δ: $0.00
- Paired t-test: t = 11.20, **p = 2.2e-28**

### Method 4 — Regression audit (two-stage)

**Stage 1** (logistic): `offered = (offer > 0)` vs `[ltv, tenure, monthly, num_services]`.

- tenure coef = **−0.238**, p ≈ 0 (bootstrap 200×)
- Pseudo-R² = 0.953

**Stage 2** (OLS on offered customers only): `offer` vs all five features.

- tenure coef = **−0.523**, p = 2.2e-14
- R² = 0.616, n = 2,496

Interpretation: even after controlling for churn probability and LTV,
tenure independently depresses both the probability of being offered
anything AND the offer amount.

### Method 6 — Victim classifier

- Algorithm: `LogisticRegression(class_weight='balanced')`
- Training subset: 2,303 loyal customers (tenure ≥ 48)
- Target: 1 if the Counterfactual Tenure Flip delta (Δ_i) > 0, else 0
- 75/25 stratified train/test, seed 42

Test metrics:

- AUC = **0.968**
- Accuracy = 0.882
- Precision @ top-85 = 0.741
- Recall @ top-85 = 0.741

## Phase 3 — Mitigation metrics (the headline)

### Mitigated allocator selection rule

```
score_i = P(churn)_i × LTV_i + λ × P(victim)_i × LTV_i
cost_i  = baseline offer for customer i (from SHAP rule)

rank customers by descending (score_i / cost_i)
pick customers in order until cumulative_cost ≤ budget
```

λ = 0.25, threshold = 0.50, budget = $846,325.

### Tenure-Offer Gap (fairness)

Same Tenure-Offer Gap formula, but using the post-mitigation offers.

Computed at (threshold=0.50, λ=0.25):

- Q1 mean offer: $123 (down from $243)
- Q5 mean offer: $94 (up from $20)
- TOG_after = 123 / 94 = **1.31**
- Improvement: 12.25 − 1.31 = **−10.94 points**

### Average LTV of selected customers (customer-base quality)

```
avg_LTV_selected = mean(LTV_i | selected_i = 1)
```

Computed:

- Baseline: **$385**
- Mitigated (λ=0.25, T=0.50): **$1,075**
- Ratio: 1075 / 385 = **×2.79**

### Loyal customer coverage

```
pct_loyal_reached = |{i : selected_i = 1 AND tenure_i ≥ 48}| / |{i : tenure_i ≥ 48}| × 100
```

Computed:

- Baseline: **10.8%** (249 / 2,303 loyals served)
- Mitigated: **42.3%** (975 / 2,303 loyals served)
- Change: **+31.5 percentage points** (3.9× relative)

### Victim coverage

```
pct_victim_reached = |{i : selected_i = 1 AND P(victim)_i ≥ 0.5}| / |{i : P(victim)_i ≥ 0.5}| × 100
```

Computed:

- Baseline: **58.1%**
- Mitigated: **62.4%**
- Change: **+4.3 percentage points**

## Why λ = 0.25 — the 2D sweep

A grid search over (threshold × λ) with 5 × 10 = 50 combinations.

| Threshold | Optimal λ | Tenure-Offer Gap at optimum | avg LTV | composite |
|---|---|---|---|---|
| 0.30 | 0.75 | 1.48 | $1,052 | 0.999 |
| 0.40 | 0.50 | 1.52 | $1,032 | 0.983 |
| **0.50** | **0.25** | **1.31** | **$1,075** | **0.967** ← chosen |
| 0.60 | 0.25 | 1.06 | $1,137 | 0.969 |
| 0.70 | 0.25 | 0.85 | $1,192 | 0.956 |

The composite score is computed per-threshold (normalized within row) as:

```
fairness     = min_TOG / TOG_i
quality      = avg_LTV_i / max_avg_LTV
victim_cov   = pct_victims_i / max_pct_victims
composite    = cbrt(fairness × quality × victim_cov)
```

At thresholds ≥ 0.50, the optimal λ stabilizes at 0.25. Our chosen
operating point lives on this stable plateau.

## Reproducing every number

```bash
cd loyalty-aware-retention
python detect_loyalty_penalty.py       # Phase 2 detection
python explore_lambda_threshold.py     # 2D sweep
python mitigate_loyalty_penalty.py     # Phase 3 mitigation
python verify_thesis_claims.py         # final verification
```

All functions use `random_state=42`. Results are bit-for-bit reproducible.
