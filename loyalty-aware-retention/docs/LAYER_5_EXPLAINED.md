# Layer 5 Explained — The Loyalty-Aware Retention Allocator

A plain-English walkthrough of what Layer 5 does, how it fixes what Layer 4 diagnosed, and why it does not claim anything it cannot measure. Intended for teammates and for the viva presentation.

---

## 1. Where we are in the pipeline

| Layer | What it does | Output |
|---|---|---|
| 1 | XGBoost predicts P(churn) per customer | Churn probability |
| 2 | K-Means assigns each customer to a behavioural cluster | Cluster label |
| 3 | SHAP + rule-based offers (original app.py) | Personalised retention offer |
| 4 | Six-method audit of Layer 3 | Proves the loyalty penalty + produces `P(victim)` |
| **5** | **Re-allocate the retention budget fairly, without losing churn coverage** | **A loyalty-aware offer per customer under the same total spend** |

Layer 4 said: *"the Layer 3 system is under-serving loyal customers by a factor of twelve."* Layer 5 is the fix: same budget, but spent more fairly.

## 2. Why a budget-allocator, not a model rewrite?

Two design choices to keep the fix clean:

1. **Don't retrain the XGBoost model.** The churn model is doing its job — predicting who is at risk. The problem is not in the model, it's in the *policy* that acts on the model.
2. **Don't change the SHAP-threshold retention rule either.** That rule still works for short-tenure customers. We only change *who gets selected* under a fixed budget.

So Layer 5 is a thin decision layer bolted on top of Layers 1-3. Input: every customer's churn probability (Layer 1), cluster (Layer 2), baseline retention offer (Layer 3), and victim probability (Layer 4). Output: a YES/NO selection flag per customer and the offer amount to send them.

## 3. The core trade-off, in one sentence

> The baseline spends the budget on **everyone who is likely to churn**. The loyalty-aware allocator spends the budget on **customers who are likely to churn OR who are being penalised for loyalty**, weighted by how valuable they are to the business.

That is the whole idea. Everything below is how we turn that sentence into a formula.

## 4. The scoring formula (the "what should we care about per customer" question)

For each customer *i*:

```
score_i  =  P(churn)_i × LTV_i   +   λ × P(victim)_i × LTV_i
             └──────────────┘       └──────────────────────┘
           churn-at-risk value       loyalty-penalty term
```

Two terms, both in units of dollars, both per customer.

### Term 1 — churn-at-risk value

`P(churn) × LTV` is the **expected revenue that leaves** if this customer churns. High P(churn) + high LTV = big loss if we lose them. This is what the original baseline already cares about (implicitly).

### Term 2 — loyalty-penalty term

`P(victim) × LTV` is the **expected loyalty-penalty damage** to a customer. `P(victim)` comes from Layer 4's Method 6 classifier and says *"how likely is it that this specific customer is being under-served by the Layer 3 policy?"*. Multiplying by LTV weights the term by how much of that customer's value is at stake.

### Why λ (the knob)

λ controls how much weight we put on fairness vs raw churn risk.

- λ = 0 → pure churn-risk mode. Identical to the baseline. Loyalty is ignored.
- λ = small (0.1-0.3) → churn risk dominates, but loyalty gets a small tiebreaker.
- λ = large (1+) → fairness dominates. You may start skipping high-churn customers to prioritise loyal ones.
- λ = ∞ → pure fairness mode. Ignores churn risk entirely.

**We use λ = 0.25** (justified in Section 7 below).

### Simple analogy

Imagine you have $100 to spend on a group dinner and two things matter: (a) feeding the hungriest people first, and (b) making sure long-time friends aren't ignored. λ is the dial: at λ = 0 you only feed the hungriest; at λ = ∞ you only feed old friends regardless of hunger; at λ = 0.25 you mostly feed hungriest but nudge a bit toward old friends when the hunger difference is small.

## 5. The allocator (the "who do we actually pick" question)

Once every customer has a `score_i`, we solve a classic **knapsack problem**:

> Given a list of customers, each with a score and a cost (the baseline retention offer), pick a subset that maximises the total score without exceeding the total budget.

We solve it with a **greedy algorithm**:

1. For every customer, compute `ratio_i = score_i / cost_i` — the bang-for-buck.
2. Sort everyone by ratio, descending.
3. Walk down the sorted list. For each customer, if we still have budget, pick them and subtract their cost.
4. Stop when the budget runs out.

### Why greedy and not a solver?

Two reasons:

- **Zero extra dependencies.** We don't pull in `pulp` or `ortools`. The allocator runs on numpy alone. That matters for keeping the production footprint small.
- **Near-optimal in practice.** For a 7,043-row problem where individual offers are small relative to the total budget, greedy-by-ratio is within fractions of a percent of the exact ILP solution. The structure is clean enough to swap in ILP later if needed.

### Simple analogy

You're at a buffet with a budget. Every dish has a price (cost) and a happiness rating (score). Greedy-by-ratio = sort dishes by "happiness per dollar," grab the best deal first, keep going until your wallet runs out. It won't always be provably optimal, but it works really well and is trivial to explain.

## 6. Three strategies tested (so you understand why we picked what we picked)

We ran three allocation strategies on the exact same data, same budget, same threshold, and compared their outputs.

### Strategy A — Baseline

The current Layer 3 policy. Every customer above P(churn) ≥ 0.50 gets their SHAP-rule offer. This is what we're trying to fix.

### Strategy B — Single-pool λ allocator

One budget pool. Each customer gets a combined score `P(churn)·LTV + λ·P(victim)·LTV`. Greedy-pick by ratio until the budget is empty. This is the formula above, one universal pot.

### Strategy C — Two-pool α-split allocator

Split the budget into two pots:

- **Churn pool (fraction 1 − α):** allocated purely on `P(churn) × LTV`. Classic retention.
- **Loyalty pool (fraction α):** allocated purely on `P(victim) × LTV`. Pure loyalty reward.

Then merge the two selection lists. This is a more literal interpretation of "set aside some money for loyalty." We tested α = 0.3 (30% to loyalty).

### Why we picked Strategy B (single-pool)

Both B and C give similar fairness improvements. Strategy B was chosen for deployment because:

- **One knob.** Only λ to explain, not a pool-split parameter.
- **Simpler to audit.** A regulator inspecting the allocator only needs to understand one score function.
- **Smoother behaviour.** No discrete jump between "churn customer" and "loyalty customer" — a single customer who is both high-risk AND a loyalty victim gets double-weighted naturally.

## 7. How we picked λ = 0.25 and threshold = 0.50

Neither number was guessed. Both come out of data-driven sweeps.

### Threshold = 0.50

Inherited from the original `app.py` baseline. P(churn) ≥ 0.5 is the industry-standard "act / don't act" cutoff. We did not tune it — it's a fixed operating assumption. Layer 4's threshold-sensitivity plot (figure 9) already proved the loyalty penalty exists at every threshold from 0.10 to 0.80, so 0.5 is not a cherry-pick.

### λ = 0.25

Found by a **2D grid sweep** over (threshold, λ), 5 × 10 = 50 combinations. At each combination we computed:

- **Fairness score** — how small is the post-mitigation Tenure-Offer Gap?
- **Quality score** — how high is the average LTV of selected customers?
- **Victim-coverage score** — how many loyalty-penalty victims are reached?

Each is normalised within its threshold row, then combined with a cube-root (geometric mean). At threshold 0.50, the composite peaks at **λ = 0.25 with a score of 0.967**. At stricter thresholds (0.60, 0.70), the optimal λ stays at 0.25 — so the choice is **stable**, not a one-off fit.

### Plain-English reading

λ = 0.25 says: *"for every dollar of churn-risk value, count 25 cents of loyalty-penalty value."* That is enough to pull the allocator's behaviour clearly away from the baseline, but not so much that it starts skipping genuinely at-risk customers.

## 8. What Layer 5 actually achieved (the numbers)

All under the same total spend of $846,325 (the exact total of the baseline Layer 3 offers — budget is not hardcoded, it is whatever the current system already spends).

| Metric | Baseline (Layer 3 as-is) | Mitigated (λ = 0.25) | Change |
|---|---|---|---|
| Customers selected | 2,496 | 3,557 | +1,061 |
| **Tenure-Offer Gap** | **12.25** | **1.31** | **÷9.4** |
| **Avg LTV of selected** | **$385** | **$1,075** | **×2.79** |
| **Loyal customer coverage** | **10.8%** | **42.3%** | **+31.5 pp** |
| **Victim coverage** | **58.1%** | **62.4%** | **+4.3 pp** |

Four wins:

1. **Fairness.** The 12.25× tenure gap collapses to a near-equal 1.31×. Both quintiles now get comparable retention money per customer.
2. **Customer-base quality.** Average LTV of the customers we're investing in nearly triples, because the allocator prefers high-value loyal customers that the baseline was ignoring.
3. **Loyal coverage.** Nearly four times more loyal customers are reached.
4. **Victim coverage.** The loyalty-penalty victims identified by Method 6 are now being served.

## 9. The honest limits — what we DO NOT claim

This is important for the presentation. If asked whether we saved the company more money, the answer is:

> **We don't know, and we don't claim we do.** Measuring "prevented churn" requires a randomised A/B test with treatment and control groups. We don't have that. What we claim is that under the same spend we reach a higher-value, loyalty-aware customer base, and the retention policy becomes measurably fairer.

Things we explicitly do NOT claim:

- "More churn is prevented." Unknown.
- "The company saves money." Unknown.
- "Our offers are more effective." Unknown.

Things we DO claim and can prove line by line:

- Tenure-Offer Gap drops from 12.25 to 1.31.
- Average LTV of selected customers rises 2.79×.
- Loyal-customer coverage rises from 10.8% to 42.3%.
- Victim coverage rises by 4.3 percentage points.
- Total spend is held constant.

## 10. The honest trade-off (the one bullet that is NOT a win)

The mitigated allocator sometimes **skips a genuinely at-risk short-tenure customer** in favour of a moderately-at-risk but high-value loyal customer. Specifically, high-churn-coverage falls slightly under mitigation.

This is not a bug, it's the explicit trade-off:

- Baseline optimises for *"hit every high-churn customer, regardless of value or tenure."*
- Mitigated optimises for *"hit the highest-value high-churn customers AND the high-value loyalty-penalty victims."*

Whether the trade is a net business win depends on offer effectiveness, which we don't measure. The thesis is: **we can make the allocation fairer and higher-value per dollar under the same spend, and we can do it transparently and auditably.** That is the contribution, not "we save more money."

## 11. The pipeline plumbing

For teammates wondering how the code hangs together:

```
Layer 4 Method 6 (detect.py)
   └─► trains victim classifier on Counterfactual Tenure Flip labels
   └─► returns AUC-0.968 LogisticRegression

Layer 5 (mitigate.py)
   ├─► fit_victim_scorer()           → gets P(victim) for every customer
   ├─► allocate_lambda(..., λ=0.25)  → runs the greedy knapsack
   ├─► evaluate(...)                 → computes before/after Tenure-Offer Gap,
   │                                     cluster equity, coverage metrics
   └─► detailed_pareto_sweep(...)    → sweeps λ to find the sweet spot
```

All of this is deterministic, seed-42, reproducible with:

```bash
python mitigate_loyalty_penalty.py    # full run
python verify_thesis_claims.py        # asserts every claim above is true
```

## 12. Reading the charts

For the presentation, two figures carry the Layer 5 story:

1. **`FINAL_thesis_summary.png`** — the four-panel before-vs-after: Tenure-Offer Gap, avg LTV, loyal coverage, victim coverage. All four arrows point the right way. One chart, whole thesis.
2. **`13_tradeoff_curves.png`** — four panels sweeping λ from 0 to 10 showing: (a) high-churn coverage falls, (b) avg LTV rises, (c) Tenure-Offer Gap falls, (d) composite score peaks at λ = 0.25. This is the chart that justifies *why* we picked 0.25 and not 0 or 5.

## 13. Frequently-anticipated questions

**Q: Why not just lower the P(churn) threshold so more loyal customers are covered?**
Tried. Lowering the threshold (0.3, 0.4) requires a much larger λ (0.5-0.75) to achieve the same fairness. That means pushing the allocator much further from the baseline, which is harder to defend. Threshold 0.5 with λ = 0.25 is the minimum-intervention operating point.

**Q: What if the business wants to tune λ differently?**
They can. The Mitigation Lab page in Streamlit lets an analyst move λ and immediately see the effect on all four metrics. λ = 0.25 is the mathematical sweet spot, but the policy choice can lean in either direction based on business appetite.

**Q: Is the budget "reused" from the baseline or set externally?**
Reused. We deliberately do not hardcode a budget. The allocator takes the total spend that the current Layer 3 system is already making, then reallocates the same amount more fairly. This means the thesis holds under whatever budget the company happens to be running, not just one special number.

**Q: Isn't this just handing money to loyal customers who wouldn't churn anyway?**
Some, yes. That is the honest trade-off from Section 10. But remember Layer 4 Method 6: the `P(victim)` score predicts *which* loyal customers the system is actively under-serving, not all loyal customers. We target those, which means we're not blasting random loyalists.

**Q: How does Strategy C (two-pool α-split) differ from B?**
Strategy C explicitly sets aside a fraction α of the budget for a pure loyalty pool. It gives similar results, with the advantage that the "loyalty budget" is a hard, promised number regardless of how the churn pool behaves. It's a more regulator-friendly framing. Strategy B was chosen for the main deployment because it has one parameter instead of two.

**Q: What is the runtime cost?**
At 7,043 customers, the full allocator runs in about 200 milliseconds. The 2D sweep (50 combinations) runs in under a minute. Production cost is negligible.

**Q: Can the allocator handle new customer arrivals in production?**
Yes. Every new customer gets `P(churn)`, `P(victim)`, and `LTV` computed from Layers 1, 4, and the LTV formula. The allocator just re-ranks the updated population. Nothing about Layer 5 requires retraining from scratch when a customer joins.

## 14. One-line pitch

> Layer 5 takes the loyalty-penalty evidence from Layer 4 and builds a deployable policy: same budget, a single tunable λ = 0.25, a transparent greedy knapsack. Result — the Tenure-Offer Gap drops from 12.25 to 1.31, the customers being served have 2.79× more lifetime value on average, and four times more loyal customers are reached. No change to the underlying churn model, no A/B claim we can't back up, no hidden parameters.
