# Layer 4 Explained — The Loyalty Audit

A plain-English walkthrough of what Layer 4 does, why we added it after Layers 1-3, and what each of the six audit methods proves. Intended for teammates and for the viva presentation.

---

## 1. Where we are in the pipeline

| Layer | What it does | Output |
|---|---|---|
| 1 | XGBoost predicts how likely each customer is to churn | P(churn) per customer |
| 2 | K-Means groups customers into 3 behavioural segments | Cluster 0 / 1 / 2 label per customer |
| 3 | SHAP explains each prediction and fires a retention offer | Personalised offer per customer |
| **4** | **Audit the Layer 3 offers for a loyalty penalty** | **Evidence that long-tenured customers are under-served** |
| 5 | Fix the penalty while keeping churn coverage | A re-allocated, loyalty-aware retention budget |

Layers 1-3 answer the question *"which customer is about to leave, and what offer should we make them?"* Layer 4 asks a different question: *"is the Layer 3 policy fair to customers who have stayed with us for years?"*

## 2. Why Layer 4 exists

### The real-world problem

In 2018 the UK Competition and Markets Authority (CMA) formally accused telecom, energy, insurance, and banking providers of running a **loyalty penalty**: customers who stay with a company the longest end up paying the most or receiving the worst deals, because the company quietly rolls their tariff up over time while reserving discounts for new sign-ups. Simon-Kucher's 2025 Global Telecommunications Study confirmed the same pattern: *95% of customer lifetime value comes from customers with three or more years of tenure*, yet most operators spend the bulk of their retention budget on recent joiners.

### Does our Layer 3 have the same problem?

It might. The Layer 3 retention rule fires an offer only when P(churn) is high. But short-tenure customers churn more often than long-tenure ones (look at any churn-rate-vs-tenure chart). So if the rule is *"offer only to high-P(churn) customers,"* it will naturally hit new customers and ignore loyal ones. That is the loyalty penalty in mathematical form.

Layer 4 audits whether this is happening, and if it is, how bad it is. It does not change anything yet. The fix is Layer 5.

## 3. Simple everyday analogy

Think of a gym.

- Layer 1-3 is the gym's "win-back" program. If your attendance drops, the staff offers you a free month or a personal training session.
- The loyalty penalty is: the staff only ever offers freebies to people who are about to cancel. The guy who has come three times a week for five years gets nothing.
- Layer 4 is an independent reviewer walking in and asking six different ways: *"Does your freebie-allocation actually discriminate against loyal members?"* Each of the six questions catches a different excuse the gym might give.

## 4. The six audit methods

Every method asks the same yes/no question — "is the Layer 3 policy penalising loyalty?" — but each one comes at it differently, and each blocks a different objection the business could raise. You need all six because if you only had one, the business would wriggle out of it.

### Method 1: Tenure-Offer Gap

**The simple version.** Line up every customer by how long they have stayed. Split them into five equal groups (quintiles): the newest 20%, the next 20%, ... the most loyal 20%. For each group, compute the average retention offer they received. Take the ratio (newest group) / (most loyal group). If the ratio is big, newcomers are being spoiled.

**What we found.** Newest quintile gets $243 on average. Most loyal quintile gets $20. Ratio = **12.25×**. Newcomers get twelve times more retention money than loyalists.

**Why it is beneficial.** It is the simplest, most visual proof of the problem. You can show this chart to a non-technical audience and they get it in five seconds.

**The objection it doesn't answer.** "Loyal customers just need less money because they churn less." — That is handled by Method 3.

### Method 2: Cluster Equity

**The simple version.** Instead of grouping by tenure, group by *behaviour* using the three K-Means clusters from Layer 2 (Budget Basics, Flight Risks, Premium Loyalists). For each cluster, compute the retention spend as "dollars per $100 of lifetime value." A fair policy would spend roughly the same *per dollar of value* on every cluster.

**What we found.** Flight Risks (the high-churn cluster, mostly month-to-month newcomers) get a dramatically bigger slice per $100 LTV than Premium Loyalists. Same penalty, visible through behaviour instead of tenure.

**Why it is beneficial.** This is the version a business audience will care about. "Are we under-serving our most valuable customer segment per dollar they generate?" is a question a board member asks. Method 2 translates the statistical finding into business language.

**Together with Method 1.** Two completely different ways of grouping customers (tenure vs behaviour) both flag the same bias. That rules out the "it's just how you cut the data" objection.

### Method 3: Counterfactual Tenure Flip

**The simple version.** Take each loyal customer (tenure ≥ 48 months). Make an imaginary copy of them — a "twin" — with **everything identical except tenure set to 3 months**. Run both the real customer and the twin through the Layer 3 pipeline. If the twin gets a bigger offer than the real customer, the system is penalising loyalty (and nothing else, because we held every other feature constant).

**Why "counterfactual"?** It is a tiny what-if experiment. What if this exact person was a new customer instead of a 5-year loyalist? What would our system offer them?

**What we found.** Out of 2,303 loyal customers, **339 of them get a bigger offer when the system thinks they are new**. The average uplift for these 339 victims is **$206 per person**, with a median of $265. The effect is overwhelmingly statistically significant.

**Why it is beneficial.** This is the killer argument. Method 1 and 2 compare groups; someone could always argue "those groups are different people with different needs." Method 3 compares *the same person to themselves* with only tenure changed. There is nothing else to blame.

**Simple analogy.** You walk into a shop. You are a regular customer. The cashier quotes you $100. Your twin (same clothes, same basket, same card) walks in five minutes later — the only difference is, the cashier thinks they're a new customer. Cashier quotes them $75. That $25 difference is the loyalty penalty, proven directly.

### Method 4: Regression Audit (two-stage)

**The simple version.** A statistician's version of "control for confounders." We fit two regression models:

- **Stage 1 (logit):** does tenure make a customer less likely to receive *any* offer, even after accounting for their churn risk, LTV, monthly spend, and number of services? If tenure comes out with a negative coefficient and a tiny p-value, the answer is yes.
- **Stage 2 (OLS):** among customers who *did* receive an offer, does tenure make the offer smaller, again controlling for the same features?

**What we found.** Stage 1 tenure coefficient = **−0.238** (massively significant). Stage 2 tenure coefficient = **−0.523** (p = 2.2e-14). Both stages say the same thing: tenure *independently* suppresses retention offers, even after controlling for churn probability and LTV.

**Why it is beneficial.** It kills the most natural business defence: *"Loyal customers get less because they're low-risk and we optimise risk."* The regression literally adds risk (and value) as control variables and tenure still comes out negative. Risk is not the explanation.

**Simple analogy.** If your boss says "men earn more here because men do harder jobs on average," you rerun the analysis *controlling for job type*. If men still earn more within the same job, the original explanation was wrong. That is what a two-stage regression does here.

### Method 5: Contract-Controlled Tenure-Offer Gap

**The simple version.** Split customers by contract type (Month-to-Month, One-Year, Two-Year). Within *each* contract bucket, compute the Tenure-Offer Gap separately. If the gap is ≥ 1.5× inside every bucket, the penalty can't be blamed on long contracts.

**What we found.** The Tenure-Offer Gap persists — and in some contract buckets is actually *worse* than the overall number — when we hold contract type fixed.

**Why it is beneficial.** It kills the second natural business defence: *"Of course loyal customers get less — they're locked into two-year contracts, so they won't churn anyway."* Method 5 says: no, even among month-to-month customers only, the longest-tenured ones get far less.

**Together with Method 4.** Method 4 rules out risk as the explanation. Method 5 rules out contract type. After both, the remaining explanation is tenure itself. Which is the loyalty penalty.

### Method 6: Victim Predictor

**The simple version.** Methods 1-5 prove the penalty exists. Method 6 builds a small **logistic regression classifier** that, given a loyal customer's features, predicts *"is this individual one of the 339 victims?"* — without having to rerun the Counterfactual Tenure Flip.

**Why this matters.** The Counterfactual Tenure Flip is expensive: for every customer you have to build a twin, re-run XGBoost, re-run SHAP, compute offer differences. That is fine for a one-time audit but impossible in live production. Method 6 trains once and then scores any new customer in milliseconds, giving back `P(victim)` — the probability that this loyal customer is being penalised.

**What we found.** AUC-ROC = **0.968**. Which means: if you show the classifier two loyal customers, one a real victim and one not, it ranks them correctly 97% of the time. The top five predictive features tell us *which* loyal customers are most likely to be victims: monthly charges, internet service type, contract, and so on.

**Why it is beneficial.** It is the bridge from Layer 4 to Layer 5. Layer 5 will use `P(victim)` as an input to the re-allocator. Without Method 6, we would have a statistical finding and no deployable fix. With Method 6, the finding becomes actionable — the `P(victim)` score can be read off for any customer, any day.

**Simple analogy.** Method 3 is a full medical workup (blood test, MRI, lab results) — accurate, but takes hours. Method 6 is a 30-second screening questionnaire that predicts the full workup's result with 97% accuracy. In production you run the questionnaire; in an audit you run the full workup.

## 5. The two things Layer 4 also does beyond the six methods

### (a) Loyal-subset validation

Before running any audit we check: are the customers we're calling "loyal" actually loyal? The loyal subset (tenure ≥ 48 months) must churn at less than half the rate of the general population. We verified this in code: loyal subset churns at **9.6%** vs population **26.5%**. Good. If the check ever failed, the loyalty threshold would be rewritten and the audit rerun.

### (b) Threshold sensitivity sweep

Someone will ask *"did you choose the P(churn) ≥ 0.50 cutoff because it made the penalty look bad?"* The sweep answers this: we re-ran Method 1 at every cutoff from 0.10 to 0.80 and the Tenure-Offer Gap is above 1.5 at every single point, and grows monotonically as the cutoff tightens. The 0.5 number is an inherited industry default, not a cherry-pick, and the penalty is there at every reasonable threshold.

## 6. What the audit gives us as a package

Six methods. Six different angles. Six different objections blocked. All six agree: **the Layer 3 system is penalising loyalty.**

| # | Method | Role |
|---|---|---|
| 1 | Tenure-Offer Gap | Visual, simple, tenure-grouped |
| 2 | Cluster Equity | Behaviour-grouped, business framing |
| 3 | Counterfactual Tenure Flip | Same-person comparison, the direct proof |
| 4 | Regression Audit | Rules out risk as explanation |
| 5 | Contract-Controlled TOG | Rules out contract type as explanation |
| 6 | Victim Predictor | Bridge to Layer 5 — identifies individual victims |

Method 6's classifier output is the handoff to Layer 5.

## 7. Reading the charts

For the presentation, four figures carry the story:

1. **`01_tog.png`** — the 12.25× gap. Headline visual.
2. **`02_ctf.png`** — the twin-vs-real scatter. The "same person, different tenure" chart.
3. **`06_victim_predictor.png`** — the AUC = 0.968 and top features. Shows the bridge is ready.
4. **`09_threshold_sensitivity.png`** — gap persists at every cutoff. Blocks cherry-picking objection.

## 8. Frequently-anticipated questions

**Q: If the penalty is so obvious, why hasn't it been fixed already?**
Because nobody audited for it. Most retention-ML papers stop at "we predicted churn well." The penalty is a second-order property of the *policy built on top* of the model. Without an explicit audit layer it stays invisible.

**Q: Is this specific to telecom?**
No. Any industry where the retention budget is tied to churn risk alone (insurance, energy, subscription SaaS, banking) will produce the same bias. The CMA super-complaint covered all four.

**Q: Could we just tell Layer 3 to care about tenure?**
That is Layer 5. Layer 4 is diagnosis, Layer 5 is treatment.

**Q: How is the Counterfactual Tenure Flip different from a SHAP score?**
SHAP says *"tenure contributes −0.08 to this customer's churn probability."* The Counterfactual Tenure Flip says *"if this customer's tenure was 3 months instead of 60, the retention system would offer them $200 more."* SHAP explains the model; the Counterfactual Tenure Flip tests the downstream policy.

**Q: Is p = 2.2 × 10⁻²⁸ real?**
Yes. It means the probability that the observed offer difference between real loyal customers and their synthetic new twins arose by chance is effectively zero. The penalty is statistically unambiguous.

**Q: Why is Method 4 called "two-stage"?**
Because a lot of customers receive zero offer (they're below the churn threshold). A plain regression on "offer amount" would mix the decision to offer with the decision of how much. Stage 1 models "did they get anything?", Stage 2 models "if they got something, how much?" This is a standard Heckman selection setup.

## 9. One-line pitch

> Layer 4 is the independent audit: six mathematically distinct methods, all agreeing that the Layer 3 retention policy under-serves long-tenure customers. Method 6 makes the finding operational by producing a deployable `P(victim)` score per customer, which is what Layer 5 uses to rebalance the budget.
