"""Apriori frequent itemsets, association rules, KPIs, and insights."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def frozenset_to_str(value: Any) -> str:
    """Pretty-print a frozenset / set of items."""
    if isinstance(value, (set, frozenset)):
        return ", ".join(sorted(str(x) for x in value))
    return str(value)


def run_apriori(
    onehot: pd.DataFrame,
    min_support: float = 0.05,
    max_len: int = 3,
) -> pd.DataFrame:
    """
    Discover frequent itemsets with the Apriori algorithm (mlxtend).

    max_len limits itemset size (default 3) so household demos stay interactive.

    Returns columns: itemsets (frozenset), support, length, itemset (str), rank
    """
    if onehot is None or onehot.empty:
        return pd.DataFrame(columns=["itemsets", "support", "length", "itemset", "rank"])

    # mlxtend expects 0/1 or boolean
    matrix = onehot.astype(bool)
    itemsets = apriori(
        matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    if itemsets.empty:
        return pd.DataFrame(columns=["itemsets", "support", "length", "itemset", "rank"])

    itemsets = itemsets.copy()
    itemsets["length"] = itemsets["itemsets"].apply(len)
    itemsets["itemset"] = itemsets["itemsets"].apply(lambda s: "{" + frozenset_to_str(s) + "}")
    itemsets = itemsets.sort_values("support", ascending=False).reset_index(drop=True)
    itemsets["rank"] = itemsets.index + 1
    return itemsets


def generate_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """Generate association rules and filter by confidence and lift."""
    cols = [
        "antecedents",
        "consequents",
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
        "rule",
    ]
    if frequent_itemsets is None or frequent_itemsets.empty:
        return pd.DataFrame(columns=cols)

    # Need at least one itemset with length >= 2 for rules
    if "length" in frequent_itemsets.columns and frequent_itemsets["length"].max() < 2:
        return pd.DataFrame(columns=cols)

    try:
        rules = association_rules(
            frequent_itemsets[["itemsets", "support"]],
            metric="confidence",
            min_threshold=min_confidence,
        )
    except ValueError:
        return pd.DataFrame(columns=cols)

    if rules.empty:
        return pd.DataFrame(columns=cols)

    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return pd.DataFrame(columns=cols)

    rules["antecedent"] = rules["antecedents"].apply(frozenset_to_str)
    rules["consequent"] = rules["consequents"].apply(frozenset_to_str)
    rules["rule"] = rules["antecedent"] + " → " + rules["consequent"]
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return rules.reset_index(drop=True)[cols]


def compute_kpis(
    onehot: pd.DataFrame,
    itemsets: pd.DataFrame,
    rules: pd.DataFrame,
) -> dict:
    """Dashboard KPI values."""
    top_combo = "—"
    if itemsets is not None and not itemsets.empty:
        multi = itemsets[itemsets["length"] >= 2] if "length" in itemsets.columns else itemsets
        if not multi.empty:
            top_combo = str(multi.iloc[0]["itemset"])
        else:
            top_combo = str(itemsets.iloc[0]["itemset"])

    avg_support = float(rules["support"].mean()) if rules is not None and not rules.empty else 0.0
    avg_confidence = (
        float(rules["confidence"].mean()) if rules is not None and not rules.empty else 0.0
    )
    max_lift = float(rules["lift"].max()) if rules is not None and not rules.empty else 0.0

    return {
        "total_transactions": int(len(onehot)) if onehot is not None else 0,
        "total_products": int(onehot.shape[1]) if onehot is not None and not onehot.empty else 0,
        "frequent_itemsets": int(len(itemsets)) if itemsets is not None else 0,
        "association_rules": int(len(rules)) if rules is not None else 0,
        "top_combination": top_combo,
        "average_support": avg_support,
        "average_confidence": avg_confidence,
        "maximum_lift": max_lift,
    }


def generate_insights(rules: pd.DataFrame, top_n: int = 5) -> list[str]:
    """Plain-language business insights from top association rules."""
    if rules is None or rules.empty:
        return [
            "No association rules found at the current thresholds. "
            "Try lowering minimum support, confidence, or lift."
        ]

    insights: list[str] = []
    top = rules.head(top_n)
    for _, row in top.iterrows():
        ant = row["antecedent"]
        cons = row["consequent"]
        lift = float(row["lift"])
        conf = float(row["confidence"])
        if lift >= 1.5:
            strength = "strong"
        elif lift >= 1.2:
            strength = "moderate"
        else:
            strength = "positive"
        insights.append(
            f"{ant} and {cons} have a {strength} purchasing relationship "
            f"(lift {lift:.2f}, confidence {conf:.0%})."
        )
        insights.append(
            f"Customers purchasing {ant} are more likely to also purchase {cons}."
        )

    # Deduplicate while preserving order; cap length
    seen: set[str] = set()
    unique: list[str] = []
    for text in insights:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique[: max(top_n, 5)]
