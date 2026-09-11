# Household Market Basket Analysis using Apriori — Design Spec

**Date:** 2026-09-12  
**Status:** Approved for planning (pending user review of this file)  
**Stack:** Python, Pandas, NumPy, mlxtend, Streamlit, Matplotlib, Plotly

## 1. Problem & objective

Build an interactive Streamlit application that performs **market basket analysis** on household shopping transactions using the **Apriori** association-rule algorithm. Users load sample or custom CSV data, tune support/confidence/lift thresholds, inspect frequent itemsets and rules, get product recommendations, view charts and a rule network, and download results.

**Presentation title:** Household Market Basket Analysis using Apriori (not “product combinations” alone).

**Viva core flow:**  
Transactions → Preprocessing → One-Hot Encoding → Apriori → Frequent Itemsets → Association Rules → Support / Confidence / Lift → Recommendations

## 2. Architecture (Approach 1 — approved)

Single `app.py` with sidebar + tabs; logic in modular `src/` packages.

```
household-apriori/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── household_transactions.csv
├── src/
│   ├── preprocessing.py
│   ├── apriori_analysis.py
│   ├── recommendations.py
│   └── visualization.py
└── assets/
```

**Apriori strategy (Hybrid — approved):**  
- **Runtime:** `mlxtend.frequent_patterns.apriori` + `association_rules`  
- **Education:** dedicated “How Apriori Works” tab with pipeline, pseudocode, and viva concepts (no second from-scratch miner)

## 3. Data flow

1. Load built-in CSV or user upload (`TransactionID`, `Product`)  
2. Validate columns; drop nulls/duplicates; group products by transaction  
3. Build one-hot encoded transaction–item matrix (boolean)  
4. Run Apriori with user `min_support`  
5. Generate rules with `min_confidence`; filter by `min_lift`  
6. Compute KPIs, insights, recommendations, charts, network  
7. Offer CSV downloads for itemsets, rules, recommendations  

## 4. Dataset

- **Format:** long transactional — columns `TransactionID`, `Product`  
- **Size:** 800–1000 realistic household baskets  
- **Products (examples):** Rice, Wheat, Milk, Bread, Eggs, Sugar, Tea, Coffee, Biscuits, Cooking Oil, Salt, Spices, Detergent, Dish Soap, Shampoo, Toothpaste, Soap, Toilet Cleaner, Floor Cleaner, Vegetables, Fruits, Snacks  
- **Generation:** scripted/synthetic with correlated pairs (e.g. Milk–Bread, Detergent–Dish Soap, Rice–Cooking Oil–Spices) so rules are demonstrable  
- **Upload:** same schema required; clear errors if columns missing  

## 5. UI (approved)

**Visual:** Premium-but-simple academic dashboard; teal/slate palette; KPI cards; tables; Plotly; no unnecessary animation.

**Sidebar:** Title/tagline; data source (sample | upload); sliders for Min Support, Min Confidence, Min Lift; results recalculate when inputs change.

**Tabs:**

| Tab | Purpose |
|-----|---------|
| Dashboard | KPIs + auto business insights |
| Dataset Explorer | Preview, stats, missing values, top products |
| Apriori Analysis | Frequent itemsets table + download |
| Association Rules | Rules table, sort by lift/confidence/support + download |
| Recommendations | Select product → ranked related products + download |
| Visualizations | Five charts (top products, itemsets, confidence, lift, scatter) |
| Rule Network | Interactive product association graph |
| How Apriori Works | Pipeline, pseudocode, why Apriori, viva checklist |

**KPI cards:** total transactions, total products, frequent itemsets count, rules count, top combination, average support, average confidence, maximum lift.

**Metric copy:** Short beginner explanations of support, confidence, and lift in-app.

## 6. Module responsibilities

### `preprocessing.py`
- `load_sample_data()`, `load_uploaded_csv()`  
- Validate required columns; clean nulls/duplicates  
- `transactions_to_onehot(df) → DataFrame`  
- Raise / return typed errors for empty data, bad schema  

### `apriori_analysis.py`
- `run_apriori(onehot, min_support)` → frequent itemsets  
- `generate_rules(itemsets, min_confidence, min_lift)` → rules  
- Format itemsets/rules for display (frozenset → readable strings)  
- `compute_kpis()`, `generate_insights()`  

### `recommendations.py`
- Given selected product + rules, return ranked consequents with confidence/lift and short “why” text  

### `visualization.py`
- Plotly: top-10 products, top itemsets, rules by confidence, rules by lift, support–confidence scatter (hover: antecedent, consequent, support, confidence, lift)  
- Rule network: nodes = products, edges = rules; stronger lift/confidence visually emphasized (Plotly)  

### `app.py`
- Layout, sidebar, tabs, error UI, download buttons, session wiring  

## 7. Error handling

| Condition | Behavior |
|-----------|----------|
| Empty dataset | Message; block analysis |
| Invalid CSV / parse fail | Message; keep prior or sample |
| Missing TransactionID or Product | Message listing required columns |
| Duplicates / missing values | Drop with note in Dataset Explorer |
| No frequent itemsets | Suggest lowering min support |
| No association rules | Suggest lowering confidence/lift |
| Very high thresholds | Same empty-state guidance |

No uncaught crashes in normal user paths.

## 8. Downloads

- Frequent itemsets → CSV  
- Association rules → CSV  
- Product recommendations → CSV  

## 9. Business insights & academic support

Auto-generate plain-language insights from top lift/confidence rules. Explain uses: placement, bundles, cross-sell, personalized suggestions, promos, inventory.

**How Apriori Works** includes: transaction DB, cleaning, encoding, candidate generation, prune, iterate, rules, confidence/lift, ranking; pseudocode steps 1–11 from the project brief; why Apriori suits household market baskets (interpretable, classic MBA, viva-friendly).

## 10. README deliverables

Title, problem, objective, features, stack, dataset, Apriori explanation, install, run, screenshots placeholder, sample results, business apps, future work, viva Q&A (in README or companion section).

## 11. Out of scope

- FP-Growth or other miners  
- User accounts / auth  
- Database persistence  
- Deployed cloud hosting (local `streamlit run` only)  
- From-scratch Apriori runtime (education tab only)  

## 12. Success criteria

User can: load data → explore → set thresholds → run Apriori → view itemsets/rules → get recommendations → view charts/network → download CSVs — without crashes, with metrics explained for a college viva.
