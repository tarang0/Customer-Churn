# Telco Customer Churn — Prediction + Retention System

## Dataset

**Telco Customer Churn** (Kaggle) — 7,043 real telecom customers with 19 features and a binary churn label (Yes/No).

- **Churn rate:** 26.5% (1,869 churned out of 7,043)
- **Features:** Demographics, account info, services subscribed, contract details, billing

---

## Layer 1 — Churn Prediction (XGBoost)

### Results

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.8298** |
| **Accuracy** | 75.8% |
| **F1-Score (churned)** | 0.6076 |
| **Precision (churned)** | 53% |
| **Recall (churned)** | 71% — catches 7 out of 10 churners |

### Top Features Driving Churn (by importance)

| Rank | Feature | Importance | What It Means |
|------|---------|-----------|---------------|
| 1 | **Contract** | 0.3309 | Month-to-month customers churn far more than annual/2-year contracts. This alone explains 33% of churn. |
| 2 | **OnlineSecurity** | 0.1062 | Customers WITHOUT online security churn more — they feel less invested in the service. |
| 3 | **InternetService** | 0.0925 | Fiber optic customers churn more than DSL — likely because fiber is more expensive with similar issues. |
| 4 | **TechSupport** | 0.0463 | No tech support = higher churn. Customers who don't feel supported leave. |
| 5 | **StreamingMovies** | 0.0420 | Streaming service usage indicates engagement. More services = more sticky. |
| 6 | **tenure** | 0.0383 | New customers (< 12 months) churn at much higher rates. Loyalty builds over time. |
| 7 | **MultipleLines** | 0.0306 | Having multiple phone lines = slightly more engaged. |
| 8 | **StreamingTV** | 0.0303 | Similar to streaming movies — more services = lower churn. |
| 9 | **MonthlyCharges** | 0.0288 | Higher monthly charges = higher churn. Price sensitivity matters. |
| 10 | **OnlineBackup** | 0.0287 | Customers with online backup are more invested in the ecosystem. |

### Key Insight

> **Contract type is the #1 churn driver by a huge margin (33%).** Month-to-month customers churn at 3-4x the rate of annual contract customers. The second biggest factor is whether they have online security — customers with fewer services feel less locked in and leave more easily.

---

## Layer 2 — Customer Value Segmentation (K-Means Clustering)

### How We Found the Segments

Used 7 engineered features for clustering:

| Feature | Description | Range |
|---------|-------------|-------|
| tenure | Months as customer | 0 – 72 |
| MonthlyCharges | Current monthly bill | $18 – $119 |
| TotalCharges | Total revenue generated | $19 – $8,685 |
| num_services | Count of services subscribed | 0 – 7 |
| contract_length | 0=month-to-month, 1=1yr, 2=2yr | 0 – 2 |
| has_internet | Has internet service or not | 0 – 1 |
| avg_monthly_revenue | TotalCharges / tenure | $14 – $121 |

Tested K=2 through K=8. **Silhouette score picked K=3** as optimal (score = 0.4461).

### The 3 Customer Segments

#### Cluster 2 — "Premium Loyalists" 🟢 (2,184 customers, 31%)

| Metric | Value |
|--------|-------|
| Avg Tenure | **58 months** (nearly 5 years) |
| Avg Monthly | **$90** |
| Avg Total Revenue | **$5,198** |
| Avg Services | **5.0** (heavily invested in ecosystem) |
| Contract | Mostly **1-year or 2-year** |
| **Churn Rate** | **14.2%** 🟢 LOW |

> **Who they are:** Long-term, high-spending customers who subscribe to many services and sign long contracts. They are the company's bread and butter.
>
> **Strategy:** Protect at all costs. Any churn prevention budget spent here has the highest ROI because losing one of these customers means losing ~$5,000 in lifetime revenue.

---

#### Cluster 1 — "High-Risk Flight Risks" 🔴 (3,317 customers, 47%)

| Metric | Value |
|--------|-------|
| Avg Tenure | **16 months** |
| Avg Monthly | **$69** |
| Avg Total Revenue | **$1,109** |
| Avg Services | **2.5** |
| Contract | Almost entirely **month-to-month** (avg 0.1) |
| **Churn Rate** | **43.6%** 🔴 HIGH |

> **Who they are:** The biggest segment AND the most dangerous. Moderate spenders on month-to-month contracts with few services. Nearly half of them churn. They have no commitment — they can leave any time.
>
> **Strategy:** This is the primary target for retention campaigns. Offer incentives to switch to annual contracts. Bundle additional services at a discount to increase stickiness. SHAP (Layer 3) will tell us exactly which factors are pushing each individual to churn.

---

#### Cluster 0 — "Budget Basics" 🟢 (1,542 customers, 22%)

| Metric | Value |
|--------|-------|
| Avg Tenure | **31 months** |
| Avg Monthly | **$21** |
| Avg Total Revenue | **$675** |
| Avg Services | **1.0** (phone only, no internet) |
| Contract | Mostly **1-year** |
| **Churn Rate** | **7.4%** 🟢 LOW |

> **Who they are:** Basic phone-only customers. Low spend, low services, but also very low churn. They're simple, stable, and don't generate much revenue.
>
> **Strategy:** Low priority for retention (they barely churn). Potential upsell target — if you can get them to add internet and streaming services, they could move into higher-value segments. But don't spend retention budget here.

---

### Churn Rate Comparison Across Segments

```
Cluster 1 (Flight Risks):    43.6%  ████████████████████████████████████████████  🔴
Cluster 2 (Premium):         14.2%  ██████████████  🟢
Cluster 0 (Budget):           7.4%  ███████  🟢
```

### Where to Focus Retention Budget

```
Priority 1: Cluster 1 "Flight Risks"
  - 47% of all customers
  - 43.6% churn rate
  - $1,109 avg revenue — worth saving
  - Month-to-month contracts — offer annual contract incentives

Priority 2: Cluster 2 "Premium Loyalists" (only the ones flagged by Layer 1)
  - 14.2% churn rate overall, but the ones who DO churn = $5,198 lost
  - Focus on the few high-risk individuals, not the whole segment

Priority 3: Cluster 0 "Budget Basics"
  - 7.4% churn rate — barely churn
  - $675 avg revenue — low loss even if they leave
  - Upsell opportunity, not retention priority
```

---

## Quick Start

```bash
cd telco-churn

# Layer 1: Train churn prediction model
python train_layer1.py

# Layer 2: Run customer clustering
python train_layer2.py
```

## Generated Charts

- `cluster_selection.png` — Elbow method + silhouette score for choosing K
- `cluster_profiles.png` — 6 bar charts comparing clusters across all metrics

## Next Layers (Coming)

- **Layer 3:** SHAP explainability — why each individual customer is churning
- **Layer 4:** Uplift modeling — which retention action works for which customer
- **Layer 5:** ROI optimization — budget allocation across segments
- **Layer 6:** Streamlit dashboard — unified interface for all layers
