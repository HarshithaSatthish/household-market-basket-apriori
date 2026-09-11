"""Tests for preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    DataValidationError,
    clean_transactions,
    dataset_summary,
    to_onehot,
    validate_schema,
)


def test_validate_schema_ok():
    df = pd.DataFrame({"TransactionID": [1], "Product": ["Milk"]})
    validate_schema(df)


def test_validate_schema_missing_column():
    df = pd.DataFrame({"TransactionID": [1], "Item": ["Milk"]})
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_clean_drops_duplicates_and_nulls():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 1, 1, 2, None],
            "Product": ["Milk", "Milk", "Bread", "Eggs", "Tea"],
        }
    )
    cleaned, stats = clean_transactions(df)
    assert stats["dropped_duplicates"] >= 1
    assert "Milk" in set(cleaned["Product"])
    assert cleaned["TransactionID"].isna().sum() == 0


def test_to_onehot_shape():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 1, 2, 2],
            "Product": ["Milk", "Bread", "Milk", "Eggs"],
        }
    )
    matrix = to_onehot(df)
    assert matrix.shape[0] == 2
    assert set(matrix.columns) >= {"Milk", "Bread", "Eggs"}
    assert bool(matrix.loc["1", "Milk"]) is True
    assert bool(matrix.loc["2", "Bread"]) is False


def test_dataset_summary_counts():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 1, 2],
            "Product": ["Milk", "Bread", "Milk"],
        }
    )
    summary = dataset_summary(df)
    assert summary["n_transactions"] == 2
    assert summary["n_products"] == 2
