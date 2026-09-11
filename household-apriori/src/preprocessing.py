"""Load, validate, clean, and one-hot encode household transaction data."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

REQUIRED_COLUMNS = {"TransactionID", "Product"}
PathOrBuffer = Union[str, Path, BinaryIO]


class DataValidationError(Exception):
    """Raised when the dataset is empty or missing required columns."""


def load_transactions(path_or_buffer: PathOrBuffer) -> pd.DataFrame:
    """Load a CSV with TransactionID and Product columns."""
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception as exc:  # noqa: BLE001 — surface as validation error for UI
        raise DataValidationError(f"Could not read CSV: {exc}") from exc

    if df.empty:
        raise DataValidationError("The dataset is empty. Please provide transaction rows.")

    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure required columns exist."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            "Missing required column(s): "
            + ", ".join(sorted(missing))
            + ". Expected columns: TransactionID, Product."
        )


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop missing values and duplicate rows; return cleaned frame + stats."""
    validate_schema(df)
    working = df.copy()
    working["TransactionID"] = working["TransactionID"].astype(str).str.strip()
    working["Product"] = working["Product"].astype(str).str.strip()

    null_before = int(working.isna().any(axis=1).sum())
    # Also treat empty strings as missing
    empty_mask = (working["TransactionID"] == "") | (working["Product"] == "") | (
        working["Product"].str.lower() == "nan"
    )
    null_before += int(empty_mask.sum())
    working = working.loc[~empty_mask].dropna(subset=["TransactionID", "Product"])

    dupes = int(working.duplicated().sum())
    working = working.drop_duplicates()

    if working.empty:
        raise DataValidationError(
            "No valid transactions remain after cleaning missing values and duplicates."
        )

    stats = {
        "rows_before": len(df),
        "rows_after": len(working),
        "dropped_missing": null_before,
        "dropped_duplicates": dupes,
        "n_transactions": int(working["TransactionID"].nunique()),
        "n_products": int(working["Product"].nunique()),
    }
    return working.reset_index(drop=True), stats


def to_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format transactions into a boolean one-hot matrix.

    Rows = TransactionID, columns = Product, values = True/False.
    """
    cleaned, _ = clean_transactions(df)
    basket = (
        cleaned.groupby(["TransactionID", "Product"])
        .size()
        .unstack(fill_value=0)
        .astype(bool)
    )
    basket = basket.reindex(sorted(basket.columns), axis=1)
    return basket


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return explorer-friendly statistics for a long-format transaction frame."""
    cleaned, stats = clean_transactions(df)
    top = (
        cleaned["Product"]
        .value_counts()
        .head(15)
        .rename_axis("Product")
        .reset_index(name="Frequency")
    )
    missing_report = {
        col: int(df[col].isna().sum()) if col in df.columns else 0
        for col in ["TransactionID", "Product"]
    }
    tx_sizes = cleaned.groupby("TransactionID")["Product"].nunique()
    return {
        **stats,
        "top_products": top,
        "missing_values": missing_report,
        "avg_items_per_transaction": float(tx_sizes.mean()) if len(tx_sizes) else 0.0,
        "min_items_per_transaction": int(tx_sizes.min()) if len(tx_sizes) else 0,
        "max_items_per_transaction": int(tx_sizes.max()) if len(tx_sizes) else 0,
    }
