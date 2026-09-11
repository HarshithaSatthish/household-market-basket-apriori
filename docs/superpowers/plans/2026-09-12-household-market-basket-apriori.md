# Household Market Basket Analysis using Apriori — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully interactive Streamlit app that runs Apriori market basket analysis on household transactions, with KPIs, rules, recommendations, Plotly charts, rule network, downloads, and an educational Apriori walkthrough.

**Architecture:** Single `app.py` (sidebar + tabs) over modular `src/` helpers. Runtime mining via mlxtend; education tab explains the algorithm. Sample CSV (~900 transactions) plus user upload.

**Tech Stack:** Python 3.10+, pandas, numpy, mlxtend, streamlit, plotly, matplotlib, scikit-learn (optional)

## Global Constraints

- Title: **Household Market Basket Analysis using Apriori**
- Core algorithm: **Apriori only** (mlxtend for runtime; no FP-Growth)
- Dataset schema: `TransactionID`, `Product` → one-hot boolean matrix
- UI: Streamlit sidebar + tabs listed in design spec
- Errors: friendly messages, no crashes on empty rules / bad CSV
- Project root folder name: `household-apriori/` under workspace

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `household-apriori/requirements.txt` | Dependencies |
| `household-apriori/data/household_transactions.csv` | Sample dataset |
| `household-apriori/src/__init__.py` | Package marker |
| `household-apriori/src/preprocessing.py` | Load, validate, clean, one-hot |
| `household-apriori/src/apriori_analysis.py` | Apriori, rules, KPIs, insights |
| `household-apriori/src/recommendations.py` | Product recommendations |
| `household-apriori/src/visualization.py` | Plotly charts + network |
| `household-apriori/app.py` | Streamlit UI |
| `household-apriori/README.md` | Full project README + viva Q&A |
| `household-apriori/scripts/generate_dataset.py` | One-shot dataset generator |
| `household-apriori/tests/test_preprocessing.py` | Preprocess tests |
| `household-apriori/tests/test_apriori_analysis.py` | Apriori/rules tests |
| `household-apriori/tests/test_recommendations.py` | Recommendation tests |

---

### Task 1: Scaffold + sample dataset

**Files:**
- Create: `household-apriori/requirements.txt`
- Create: `household-apriori/src/__init__.py`
- Create: `household-apriori/scripts/generate_dataset.py`
- Create: `household-apriori/data/household_transactions.csv`
- Create: `household-apriori/assets/.gitkeep`

**Interfaces:**
- Produces: CSV with columns `TransactionID`, `Product`; ≥800 transactions; correlated pairs (Milk–Bread, Detergent–Dish Soap, Rice–Cooking Oil–Spices)

- [ ] **Step 1: Create requirements.txt**

```text
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
mlxtend>=0.23.0
plotly>=5.18.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
networkx>=3.1
```

- [ ] **Step 2: Write `scripts/generate_dataset.py`** that builds ~900 baskets with biased co-occurrence, writes `data/household_transactions.csv`, prints transaction/product counts.

- [ ] **Step 3: Run generator**

```bash
cd household-apriori
python scripts/generate_dataset.py
```

Expected: CSV created; ≥800 unique TransactionIDs.

- [ ] **Step 4: Commit**

```bash
git add household-apriori/requirements.txt household-apriori/scripts household-apriori/data household-apriori/src/__init__.py household-apriori/assets
git commit -m "chore: scaffold project and generate household transaction dataset"
```

---

### Task 2: Preprocessing module

**Files:**
- Create: `household-apriori/src/preprocessing.py`
- Create: `household-apriori/tests/test_preprocessing.py`

**Interfaces:**
- Produces:
  - `class DataValidationError(Exception)`
  - `load_transactions(path_or_buffer) -> pd.DataFrame`
  - `clean_transactions(df) -> tuple[pd.DataFrame, dict]`  # cleaned df + stats (dropped nulls/dupes)
  - `validate_schema(df) -> None`  # raises DataValidationError
  - `to_onehot(df) -> pd.DataFrame`  # boolean matrix, index=TransactionID
  - `dataset_summary(df) -> dict`  # n_transactions, n_products, top products, missing counts

- [ ] **Step 1: Write failing tests** for schema validation, cleaning, one-hot shape/values.

- [ ] **Step 2: Implement preprocessing.py**

- [ ] **Step 3: Run `pytest household-apriori/tests/test_preprocessing.py -v`** — all PASS

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add transaction preprocessing and one-hot encoding"
```

---

### Task 3: Apriori analysis + recommendations

**Files:**
- Create: `household-apriori/src/apriori_analysis.py`
- Create: `household-apriori/src/recommendations.py`
- Create: `household-apriori/tests/test_apriori_analysis.py`
- Create: `household-apriori/tests/test_recommendations.py`

**Interfaces:**
- Consumes: one-hot `pd.DataFrame` from Task 2
- Produces:
  - `run_apriori(onehot, min_support: float) -> pd.DataFrame`  # itemsets with support, length, rank
  - `generate_rules(frequent_itemsets, min_confidence, min_lift) -> pd.DataFrame`
  - `format_itemset(s) -> str` / `frozenset_to_str`
  - `compute_kpis(onehot, itemsets, rules) -> dict`
  - `generate_insights(rules, top_n=5) -> list[str]`
  - `recommend_products(selected: str, rules: pd.DataFrame, top_n=10) -> pd.DataFrame`

Rules columns for UI: `antecedents`, `consequents`, `support`, `confidence`, `lift` (string-formatted antecedents/consequents).

- [ ] **Step 1: Write tests** using a tiny hand-built one-hot matrix with known Milk–Bread co-occurrence.

- [ ] **Step 2: Implement apriori_analysis.py and recommendations.py** using mlxtend.

- [ ] **Step 3: pytest both test files** — PASS

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add Apriori mining, KPIs, insights, and recommendations"
```

---

### Task 4: Visualization module

**Files:**
- Create: `household-apriori/src/visualization.py`

**Interfaces:**
- Produces Plotly figures:
  - `chart_top_products(df_transactions, n=10)`
  - `chart_top_itemsets(itemsets, n=10)`
  - `chart_rules_by_confidence(rules, n=15)`
  - `chart_rules_by_lift(rules, n=15)`
  - `chart_support_confidence_scatter(rules)`
  - `chart_rule_network(rules, max_edges=40)`

Empty rules/itemsets → empty figure with annotation, not exception.

- [ ] **Step 1: Implement visualization.py**

- [ ] **Step 2: Smoke-import and call each chart with empty DataFrame** — no raise

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add Plotly charts and association rule network"
```

---

### Task 5: Streamlit application UI

**Files:**
- Create: `household-apriori/app.py`

**Interfaces:**
- Consumes all Task 2–4 APIs
- Sidebar: sample vs upload; min_support, min_confidence, min_lift sliders
- Tabs: Dashboard, Dataset Explorer, Apriori Analysis, Association Rules, Recommendations, Visualizations, Rule Network, How Apriori Works
- Downloads: itemsets, rules, recommendations CSVs
- Handle all error cases from design §7

- [ ] **Step 1: Implement full `app.py`** with professional teal/slate styling via `st.markdown` CSS

- [ ] **Step 2: Run `streamlit run app.py`** from `household-apriori/` — app loads sample data and shows KPIs

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add Streamlit dashboard for market basket Apriori analysis"
```

---

### Task 6: README + viva support

**Files:**
- Create: `household-apriori/README.md`

Content must include: title, problem, objective, features, stack, dataset, Apriori explanation, install, run, screenshots section, sample results, business apps, future enhancements, viva Q&A (≥8 Q&As).

- [ ] **Step 1: Write README.md**

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: add README with install guide and viva Q&A"
```

---

### Task 7: End-to-end verification

- [ ] **Step 1:** `pip install -r requirements.txt`
- [ ] **Step 2:** `pytest household-apriori/tests -v`
- [ ] **Step 3:** Run analysis on sample data programmatically (script or python -c) confirming ≥1 frequent itemset and ≥1 rule at defaults (support 0.05, confidence 0.3, lift 1.0)
- [ ] **Step 4:** Final commit if any fixes

---

## Spec coverage checklist

| Spec area | Task |
|-----------|------|
| Apriori via mlxtend | 3 |
| Hybrid education tab | 5 |
| Dataset 500–1000 + upload | 1, 5 |
| One-hot encoding | 2 |
| KPIs / insights | 3, 5 |
| Recommendations | 3, 5 |
| 5 charts + network | 4, 5 |
| Downloads | 5 |
| Error handling | 2, 5 |
| README + viva | 6 |
| Title Market Basket | 5, 6 |
