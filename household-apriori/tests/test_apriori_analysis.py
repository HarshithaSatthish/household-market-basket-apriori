"""Tests for Apriori analysis and recommendations."""

from __future__ import annotations

import pandas as pd

from src.apriori_analysis import (
    compute_kpis,
    generate_insights,
    generate_rules,
    run_apriori,
)
from src.preprocessing import to_onehot
from src.recommendations import recommend_products


def _sample_transactions() -> pd.DataFrame:
    rows = []
    # Strong Milk + Bread pattern
    for tid in range(1, 21):
        rows.append({"TransactionID": tid, "Product": "Milk"})
        rows.append({"TransactionID": tid, "Product": "Bread"})
        if tid % 2 == 0:
            rows.append({"TransactionID": tid, "Product": "Eggs"})
    for tid in range(21, 31):
        rows.append({"TransactionID": tid, "Product": "Detergent"})
        rows.append({"TransactionID": tid, "Product": "Dish Soap"})
    return pd.DataFrame(rows)


def test_run_apriori_finds_milk_bread():
    onehot = to_onehot(_sample_transactions())
    itemsets = run_apriori(onehot, min_support=0.2)
    assert not itemsets.empty
    labels = " ".join(itemsets["itemset"].astype(str))
    assert "Milk" in labels
    assert "Bread" in labels


def test_generate_rules_has_metrics():
    onehot = to_onehot(_sample_transactions())
    itemsets = run_apriori(onehot, min_support=0.15)
    rules = generate_rules(itemsets, min_confidence=0.3, min_lift=1.0)
    assert not rules.empty
    for col in ["support", "confidence", "lift", "antecedent", "consequent"]:
        assert col in rules.columns


def test_kpis_and_insights():
    onehot = to_onehot(_sample_transactions())
    itemsets = run_apriori(onehot, min_support=0.15)
    rules = generate_rules(itemsets, min_confidence=0.3, min_lift=1.0)
    kpis = compute_kpis(onehot, itemsets, rules)
    assert kpis["total_transactions"] == 30
    assert kpis["association_rules"] >= 1
    insights = generate_insights(rules)
    assert len(insights) >= 1


def test_no_itemsets_high_support():
    onehot = to_onehot(_sample_transactions())
    itemsets = run_apriori(onehot, min_support=0.99)
    assert itemsets.empty
    rules = generate_rules(itemsets, min_confidence=0.5, min_lift=1.0)
    assert rules.empty


def test_recommend_milk():
    onehot = to_onehot(_sample_transactions())
    itemsets = run_apriori(onehot, min_support=0.15)
    rules = generate_rules(itemsets, min_confidence=0.3, min_lift=1.0)
    recs = recommend_products("Milk", rules)
    assert not recs.empty
    assert "Bread" in set(recs["recommended_product"])
