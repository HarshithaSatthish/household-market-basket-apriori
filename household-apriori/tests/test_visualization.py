"""Smoke tests for visualization helpers."""

from __future__ import annotations

import pandas as pd

from src.visualization import (
    chart_rule_network,
    chart_rules_by_confidence,
    chart_rules_by_lift,
    chart_support_confidence_scatter,
    chart_top_itemsets,
    chart_top_products,
)


def test_charts_handle_empty():
    empty = pd.DataFrame()
    assert chart_top_products(empty) is not None
    assert chart_top_itemsets(empty) is not None
    assert chart_rules_by_confidence(empty) is not None
    assert chart_rules_by_lift(empty) is not None
    assert chart_support_confidence_scatter(empty) is not None
    assert chart_rule_network(empty) is not None
