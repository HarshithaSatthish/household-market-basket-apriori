"""Product recommendations from association rules."""

from __future__ import annotations

import pandas as pd


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.6:
        return "High confidence"
    if confidence >= 0.4:
        return "Medium confidence"
    return "Emerging association"


def recommend_products(
    selected: str,
    rules: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend products associated with `selected` using matching association rules.

    Matches rules where the selected product appears in the antecedent.
    Ranks by lift, then confidence.
    """
    columns = [
        "recommended_product",
        "antecedent",
        "rule",
        "support",
        "confidence",
        "lift",
        "strength",
        "explanation",
    ]
    if not selected or rules is None or rules.empty:
        return pd.DataFrame(columns=columns)

    selected = selected.strip()

    def _items(ant: str) -> set[str]:
        return {p.strip() for p in str(ant).split(",") if p.strip()}

    matched = rules.loc[rules["antecedent"].apply(lambda ant: selected in _items(ant))].copy()
    if matched.empty:
        return pd.DataFrame(columns=columns)

    # Prefer exact antecedent match (Milk → Bread) over multi-item antecedents
    matched["exact_match"] = matched["antecedent"].apply(lambda ant: _items(ant) == {selected})
    matched = matched.sort_values(
        ["exact_match", "lift", "confidence"],
        ascending=[False, False, False],
    )

    rows: list[dict] = []
    seen: set[str] = set()

    for _, row in matched.iterrows():
        for product in str(row["consequent"]).split(","):
            product = product.strip()
            if not product or product == selected or product in seen:
                continue
            seen.add(product)
            conf = float(row["confidence"])
            lift = float(row["lift"])
            strength = _confidence_label(conf)
            if lift >= 1.5:
                assoc = "Strong association"
            elif lift >= 1.2:
                assoc = "Clear association"
            else:
                assoc = "Mild association"
            explanation = (
                f"When customers buy {row['antecedent']}, they also buy {product} "
                f"with confidence {conf:.0%} and lift {lift:.2f} ({assoc.lower()})."
            )
            rows.append(
                {
                    "recommended_product": product,
                    "antecedent": row["antecedent"],
                    "rule": row["rule"],
                    "support": float(row["support"]),
                    "confidence": conf,
                    "lift": lift,
                    "strength": f"{strength} · {assoc}",
                    "explanation": explanation,
                }
            )
            if len(rows) >= top_n:
                break
        if len(rows) >= top_n:
            break

    return pd.DataFrame(rows, columns=columns)
