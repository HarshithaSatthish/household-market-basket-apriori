"""Generate a realistic household shopping transaction dataset."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

PRODUCTS = [
    "Rice",
    "Wheat",
    "Milk",
    "Bread",
    "Eggs",
    "Sugar",
    "Tea",
    "Coffee",
    "Biscuits",
    "Cooking Oil",
    "Salt",
    "Spices",
    "Detergent",
    "Dish Soap",
    "Shampoo",
    "Toothpaste",
    "Soap",
    "Toilet Cleaner",
    "Floor Cleaner",
    "Vegetables",
    "Fruits",
    "Snacks",
]

# Correlated bundles — probabilities tuned so Apriori finds clear rules
# without exploding the number of frequent itemsets.
BUNDLES = [
    (["Milk", "Bread"], 0.42),
    (["Milk", "Eggs"], 0.18),
    (["Bread", "Eggs"], 0.12),
    (["Rice", "Cooking Oil"], 0.35),
    (["Rice", "Cooking Oil", "Spices"], 0.18),
    (["Detergent", "Dish Soap"], 0.38),
    (["Shampoo", "Toothpaste"], 0.22),
    (["Tea", "Sugar"], 0.28),
    (["Coffee", "Sugar"], 0.16),
    (["Vegetables", "Fruits"], 0.30),
    (["Toilet Cleaner", "Floor Cleaner"], 0.14),
    (["Biscuits", "Snacks"], 0.15),
]


def generate_transactions(n_transactions: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    for tid in range(1, n_transactions + 1):
        basket: set[str] = set()

        # Inject at most a few correlated bundles
        for items, prob in BUNDLES:
            if rng.random() < prob:
                basket.update(items)

        # Ensure every basket has 2–6 products (realistic grocery size)
        target = rng.randint(2, 6)
        while len(basket) < target:
            basket.add(rng.choice(PRODUCTS))

        # Occasionally trim oversized baskets from overlapping bundles
        if len(basket) > 7:
            basket = set(rng.sample(sorted(basket), k=7))

        for product in sorted(basket):
            rows.append({"TransactionID": tid, "Product": product})

    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "household_transactions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate_transactions()
    df.to_csv(out, index=False)

    n_tx = df["TransactionID"].nunique()
    n_prod = df["Product"].nunique()
    print(f"Wrote {out}")
    print(f"Transactions: {n_tx} | Unique products: {n_prod} | Rows: {len(df)}")


if __name__ == "__main__":
    main()
