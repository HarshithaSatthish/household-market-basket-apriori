# Household Market Basket Analysis using Apriori

Interactive data-mining web application that discovers frequently purchased **household product combinations** using the **Apriori association rule mining algorithm**.

## 1. Problem statement

Retailers and household shoppers generate large volumes of transaction data, but raw receipts do not reveal which products are bought together. Without association analysis it is hard to design bundles, shelf layouts, or cross-sell offers. This project applies **market basket analysis** to household shopping data to find actionable product relationships.

## 2. Objective

Build a beginner-friendly Streamlit application that:

1. Loads household transaction data (sample or user CSV)
2. Preprocesses and one-hot encodes baskets
3. Runs the **Apriori** algorithm
4. Displays frequent itemsets and association rules with **support**, **confidence**, and **lift**
5. Recommends related products
6. Visualizes results (charts + rule network)
7. Explains the algorithm for academic viva presentations

## 3. Features

- Sample dataset (~900 realistic household transactions)
- CSV upload (`TransactionID`, `Product`)
- Adjustable minimum support, confidence, and lift
- KPI dashboard and auto-generated business insights
- Frequent itemsets and association rules tables
- Product recommendation engine
- Plotly charts and interactive association-rule network
- Downloadable CSV results
- “How Apriori Works” page with pipeline + pseudocode

## 4. Technology stack

| Layer | Tools |
|-------|--------|
| UI | Streamlit |
| Data | Pandas, NumPy |
| Mining | mlxtend (Apriori + association rules) |
| Charts | Plotly, Matplotlib, NetworkX |
| Optional | Scikit-learn |

**Hybrid design:** mlxtend runs Apriori reliably; the education tab explains every algorithmic step for viva.

## 5. Dataset description

File: `data/household_transactions.csv`

| Column | Meaning |
|--------|---------|
| TransactionID | Basket / receipt id |
| Product | Item purchased in that basket |

Products include Rice, Wheat, Milk, Bread, Eggs, Sugar, Tea, Coffee, Biscuits, Cooking Oil, Salt, Spices, Detergent, Dish Soap, Shampoo, Toothpaste, Soap, Toilet Cleaner, Floor Cleaner, Vegetables, Fruits, Snacks.

The generator (`scripts/generate_dataset.py`) injects correlated bundles (Milk–Bread, Detergent–Dish Soap, Rice–Cooking Oil–Spices, etc.) so rules are demonstrable.

Regenerate:

```bash
python scripts/generate_dataset.py
```

## 6. Apriori algorithm explanation

**Market basket flow used in this project:**

```
Transactions → Preprocessing → One-Hot Encoding → Apriori
→ Frequent Itemsets → Association Rules
→ Support / Confidence / Lift → Recommendations
```

- **Support:** how often an itemset appears in all transactions  
- **Confidence:** how often the consequent appears when the antecedent is present  
- **Lift:** how much stronger the association is than random co-occurrence (lift > 1 is positive)

## 7. Project structure

```
household-apriori/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── household_transactions.csv
├── scripts/
│   └── generate_dataset.py
├── src/
│   ├── preprocessing.py
│   ├── apriori_analysis.py
│   ├── recommendations.py
│   └── visualization.py
├── tests/
└── assets/
```

## 8. Installation

```bash
cd household-apriori
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If the sample CSV is missing:

```bash
python scripts/generate_dataset.py
```

## 9. How to run

```bash
cd household-apriori
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

### Suggested demo thresholds

- Minimum Support: `0.08`
- Minimum Confidence: `0.40`
- Minimum Lift: `1.0`

Frequent itemsets are capped at length ≤ 3 (Apriori `max_len`) so the dashboard stays interactive for demos.

## 10. Screenshots

Add screenshots after your first demo run:

1. Dashboard KPI cards  
2. Association rules table  
3. Support vs confidence scatter  
4. Rule network  
5. Recommendations for Milk  

Place images under `assets/` and link them here.

## 11. Sample results

Actual values depend on thresholds and the generated sample. Typical patterns you should see:

| Rule (example pattern) | What it means |
|------------------------|---------------|
| Milk → Bread | Breakfast staples bought together |
| Detergent → Dish Soap | Cleaning products co-purchased |
| Rice → Cooking Oil / Spices | Cooking essentials basket |

Use the **Association Rules** tab for exact support, confidence, and lift from your run.

## 12. Business applications

- Product placement and planograms  
- Bundle / combo offers  
- Cross-selling at POS  
- Personalized shopping suggestions  
- Promotional campaign design  
- Inventory and replenishment planning  

## 13. Future enhancements

- Compare Apriori with FP-Growth on larger catalogs  
- Multi-store / time-series baskets  
- Price and margin-aware rule ranking  
- Deploy to Streamlit Community Cloud  
- User login and saved analyses  

## 14. Tests

```bash
cd household-apriori
pytest tests -v
```

## 15. Viva questions and answers

**Q1. What is market basket analysis?**  
A: Finding products that are frequently purchased together from transaction (basket) data using association rules.

**Q2. Why use Apriori?**  
A: It is the classic association-rule algorithm; it uses the anti-monotone property to prune infrequent candidates and produces interpretable rules with support, confidence, and lift.

**Q3. What is support?**  
A: The fraction of transactions that contain an itemset.

**Q4. What is confidence?**  
A: The probability that the consequent is bought given the antecedent is bought: `support(A∪B) / support(A)`.

**Q5. What is lift?**  
A: `confidence(A→B) / support(B)`. Lift > 1 means A and B appear together more than by chance.

**Q6. What is one-hot encoding here?**  
A: Each transaction becomes a row; each product a column; 1/True if the product is in that basket.

**Q7. What is a frequent itemset?**  
A: An itemset whose support is at least the user-chosen minimum support.

**Q8. How are association rules generated?**  
A: From frequent itemsets, split into antecedent → consequent and keep rules meeting min confidence (and min lift).

**Q9. How does the recommendation feature work?**  
A: The user selects a product; the system finds rules where that product is in the antecedent and ranks consequents by lift and confidence.

**Q10. Why is Apriori suitable for household products?**  
A: Household catalogs are moderate in size, baskets are interpretable, and retail decisions need explainable “bought together” rules—exactly what Apriori provides.

**Q11. What happens if minimum support is too high?**  
A: Few or no frequent itemsets remain; the UI asks the user to lower the threshold.

**Q12. Did you implement Apriori from scratch?**  
A: Runtime uses mlxtend’s Apriori for correctness and speed; the app includes a full educational walkthrough and pseudocode for viva explanation (hybrid approach).

---

**Core viva flow to remember:**  
Transactions → Preprocessing → One-Hot Encoding → Apriori → Frequent Itemsets → Association Rules → Support/Confidence/Lift → Recommendations
