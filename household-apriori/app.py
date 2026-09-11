"""
Household Market Basket Analysis using Apriori
Interactive Streamlit dashboard for association rule mining.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apriori_analysis import (  # noqa: E402
    compute_kpis,
    generate_insights,
    generate_rules,
    run_apriori,
)
from src.preprocessing import (  # noqa: E402
    DataValidationError,
    clean_transactions,
    dataset_summary,
    load_transactions,
    to_onehot,
)
from src.recommendations import recommend_products  # noqa: E402
from src.visualization import (  # noqa: E402
    chart_rule_network,
    chart_rules_by_confidence,
    chart_rules_by_lift,
    chart_support_confidence_scatter,
    chart_top_itemsets,
    chart_top_products,
)

SAMPLE_PATH = ROOT / "data" / "household_transactions.csv"

st.set_page_config(
    page_title="Household Market Basket Analysis | Apriori",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-family: "Source Serif 4", Georgia, serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f766e;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: linear-gradient(145deg, #f0fdfa 0%, #ecfeff 55%, #f8fafc 100%);
        border: 1px solid #99f6e4;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 118, 110, 0.06);
        min-height: 110px;
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #0f766e;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #134e4a;
        line-height: 1.2;
        word-break: break-word;
    }
    .metric-help {
        background: #f8fafc;
        border-left: 4px solid #0d9488;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0 1.25rem 0;
        color: #334155;
        font-size: 0.95rem;
    }
    .insight-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        color: #1e293b;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f766e 0%, #115e59 45%, #134e4a 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f0fdfa !important;
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: #ccfbf1 !important;
    }
</style>
"""


def kpi_card(label: str, value: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


@st.cache_data(show_spinner=False)
def _load_sample() -> pd.DataFrame:
    return load_transactions(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def _run_pipeline(
    raw_hash: str,
    raw_df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> dict:
    """Cached analysis pipeline. raw_hash busts cache when data changes."""
    _ = raw_hash
    cleaned, clean_stats = clean_transactions(raw_df)
    onehot = to_onehot(cleaned)
    itemsets = run_apriori(onehot, min_support=min_support)
    rules = generate_rules(itemsets, min_confidence=min_confidence, min_lift=min_lift)
    kpis = compute_kpis(onehot, itemsets, rules)
    insights = generate_insights(rules)
    summary = dataset_summary(raw_df)
    return {
        "cleaned": cleaned,
        "clean_stats": clean_stats,
        "onehot": onehot,
        "itemsets": itemsets,
        "rules": rules,
        "kpis": kpis,
        "insights": insights,
        "summary": summary,
    }


def dataframe_hash(df: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(df, index=True).sum())


def render_sidebar() -> tuple[pd.DataFrame | None, float, float, float, str | None]:
    st.sidebar.markdown("### Household Market Basket")
    st.sidebar.caption("Apriori association rule mining")

    source = st.sidebar.radio("Data source", ["Sample dataset", "Upload CSV"], index=0)
    error: str | None = None
    raw: pd.DataFrame | None = None

    if source == "Sample dataset":
        try:
            raw = _load_sample()
            st.sidebar.success(f"Sample loaded · {raw['TransactionID'].nunique()} baskets")
        except Exception as exc:  # noqa: BLE001
            error = f"Could not load sample dataset: {exc}"
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV (TransactionID, Product)", type=["csv"])
        if uploaded is not None:
            try:
                raw = load_transactions(uploaded)
                st.sidebar.success("Upload accepted")
            except DataValidationError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                error = f"Invalid CSV: {exc}"
        else:
            error = "Upload a CSV file with columns TransactionID and Product."

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Apriori thresholds")
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.30, 0.08, 0.01)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.05, 1.00, 0.40, 0.05)
    min_lift = st.sidebar.slider("Minimum Lift", 0.5, 3.0, 1.0, 0.1)
    st.sidebar.caption(
        "Results refresh when data or thresholds change. "
        "Itemsets are limited to size ≤ 3 for interactive demos."
    )
    return raw, min_support, min_confidence, min_lift, error


def tab_dashboard(result: dict) -> None:
    kpis = result["kpis"]
    st.markdown("#### Key performance indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total Transactions", f"{kpis['total_transactions']:,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Total Products", f"{kpis['total_products']:,}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Frequent Itemsets", f"{kpis['frequent_itemsets']:,}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Association Rules", f"{kpis['association_rules']:,}"), unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(kpi_card("Top Combination", kpis["top_combination"]), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("Avg Support", f"{kpis['average_support']:.3f}"), unsafe_allow_html=True)
    with c7:
        st.markdown(kpi_card("Avg Confidence", f"{kpis['average_confidence']:.3f}"), unsafe_allow_html=True)
    with c8:
        st.markdown(kpi_card("Maximum Lift", f"{kpis['maximum_lift']:.3f}"), unsafe_allow_html=True)

    st.markdown("#### Business insights")
    for text in result["insights"]:
        st.markdown(f'<div class="insight-item">💡 {text}</div>', unsafe_allow_html=True)

    st.markdown("#### How stores can use these results")
    st.markdown(
        """
- **Product placement** — put strongly associated items near each other  
- **Bundle creation** — package high-lift pairs (e.g., Detergent + Dish Soap)  
- **Cross-selling** — suggest consequents at checkout  
- **Personalized recommendations** — use the Recommendations tab  
- **Promotional offers** — discount one item to lift sales of its partner  
- **Inventory planning** — stock associated products together for demand spikes  
"""
    )


def tab_explorer(result: dict, raw: pd.DataFrame) -> None:
    summary = result["summary"]
    st.markdown("#### Dataset preview")
    st.dataframe(raw.head(50), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", summary["n_transactions"])
    c2.metric("Unique products", summary["n_products"])
    c3.metric("Avg items / basket", f"{summary['avg_items_per_transaction']:.1f}")

    st.markdown("#### Data quality")
    st.write(
        {
            "Missing TransactionID": summary["missing_values"].get("TransactionID", 0),
            "Missing Product": summary["missing_values"].get("Product", 0),
            "Dropped missing rows": summary["dropped_missing"],
            "Dropped duplicates": summary["dropped_duplicates"],
            "Rows after cleaning": summary["rows_after"],
        }
    )

    st.markdown("#### Most frequently purchased products")
    st.dataframe(summary["top_products"], use_container_width=True)
    st.plotly_chart(chart_top_products(result["cleaned"]), use_container_width=True)

    with st.expander("One-hot encoded matrix (sample)"):
        st.caption("Each row is a transaction; 1/True means the product was purchased.")
        st.dataframe(result["onehot"].head(20).astype(int), use_container_width=True)


def tab_apriori(result: dict) -> None:
    st.markdown(
        '<div class="metric-help"><b>Support</b> — how often an itemset appears '
        "across all transactions. Example: support 0.20 means the combination "
        "appears in 20% of baskets.</div>",
        unsafe_allow_html=True,
    )
    itemsets = result["itemsets"]
    if itemsets.empty:
        st.warning(
            "No frequent itemsets found. Try lowering **Minimum Support** in the sidebar."
        )
        return

    display = itemsets[["rank", "itemset", "length", "support"]].rename(
        columns={"length": "Number of products", "itemset": "Itemset", "support": "Support", "rank": "Rank"}
    )
    st.dataframe(display, use_container_width=True, height=420)
    st.download_button(
        "Download frequent itemsets (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="frequent_itemsets.csv",
        mime="text/csv",
    )


def tab_rules(result: dict) -> None:
    st.markdown(
        '<div class="metric-help"><b>Confidence</b> — given the antecedent was bought, '
        "how often the consequent was also bought.<br>"
        "<b>Lift</b> — how much more likely the combination is versus chance. "
        "Lift &gt; 1 means a positive association.</div>",
        unsafe_allow_html=True,
    )
    rules = result["rules"]
    if rules.empty:
        st.warning(
            "No association rules found. Lower **Minimum Confidence** or **Minimum Lift**, "
            "or reduce **Minimum Support** so more itemsets are discovered."
        )
        return

    sort_by = st.selectbox(
        "Sort rules by",
        ["Highest Lift", "Highest Confidence", "Highest Support"],
    )
    key = {"Highest Lift": "lift", "Highest Confidence": "confidence", "Highest Support": "support"}[
        sort_by
    ]
    sorted_rules = rules.sort_values(key, ascending=False)
    display = sorted_rules[["antecedent", "consequent", "support", "confidence", "lift"]].rename(
        columns={
            "antecedent": "Antecedents",
            "consequent": "Consequents",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
        }
    )
    st.dataframe(display, use_container_width=True, height=420)
    st.download_button(
        "Download association rules (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="association_rules.csv",
        mime="text/csv",
    )


def tab_recommendations(result: dict) -> None:
    products = sorted(result["cleaned"]["Product"].unique())
    selected = st.selectbox("Select a household product", products)
    recs = recommend_products(selected, result["rules"])
    if recs.empty:
        st.info(
            f"No recommendations for **{selected}** at current thresholds. "
            "Try another product or lower confidence/lift."
        )
        return

    st.markdown(f"### Recommended products with **{selected}**")
    for i, row in recs.iterrows():
        st.markdown(
            f"**{i + 1}. {row['recommended_product']}** — {row['strength']}  \n"
            f"{row['explanation']}"
        )
    st.dataframe(recs, use_container_width=True)
    st.download_button(
        "Download recommendations (CSV)",
        data=recs.to_csv(index=False).encode("utf-8"),
        file_name=f"recommendations_{selected.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


def tab_visualizations(result: dict) -> None:
    st.plotly_chart(chart_top_products(result["cleaned"]), use_container_width=True)
    st.plotly_chart(chart_top_itemsets(result["itemsets"]), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_rules_by_confidence(result["rules"]), use_container_width=True)
    with c2:
        st.plotly_chart(chart_rules_by_lift(result["rules"]), use_container_width=True)
    st.plotly_chart(chart_support_confidence_scatter(result["rules"]), use_container_width=True)


def tab_network(result: dict) -> None:
    st.caption("Nodes = products · Edges = association rules · Thicker edges = stronger lift")
    st.plotly_chart(chart_rule_network(result["rules"]), use_container_width=True)


def tab_how_apriori() -> None:
    st.markdown("### How Apriori works (beginner walkthrough)")
    st.markdown(
        """
```
Transaction Data
    ↓
Data Cleaning
    ↓
Transaction Encoding (One-Hot)
    ↓
Generate 1-itemsets
    ↓
Calculate Support
    ↓
Prune Infrequent Itemsets
    ↓
Generate Larger Itemsets (candidates)
    ↓
Repeat Until No More Frequent Itemsets
    ↓
Generate Association Rules
    ↓
Calculate Confidence & Lift
    ↓
Rank Strong Rules
    ↓
Product Recommendations
```
"""
    )
    st.markdown("### Apriori pseudocode")
    st.code(
        """
1. Find frequent 1-itemsets (items meeting min support)
2. Set k = 2
3. Generate candidate k-itemsets from frequent (k-1)-itemsets
4. Calculate support for each candidate
5. Remove candidates below minimum support
6. Keep frequent k-itemsets
7. Increment k
8. Repeat until no frequent itemsets remain
9. Generate association rules from frequent itemsets
10. Calculate confidence and lift for each rule
11. Display / rank strong rules (above min confidence & lift)
""".strip(),
        language="text",
    )
    st.markdown("### Why Apriori suits household product combinations")
    st.markdown(
        """
Household shopping is a classic **market basket** problem: each receipt is a transaction,
and we want interpretable “customers who buy X also buy Y” rules.

**Apriori** is suitable because:
- It is the standard teaching algorithm for association rule mining  
- The **anti-monotone property** (subsets of frequent itemsets must be frequent) keeps search manageable for moderate product catalogs  
- Support, confidence, and lift are easy to explain in a viva  
- Results map directly to retail actions (bundles, shelf placement, cross-sell)  

This app uses **mlxtend’s Apriori** for reliable computation while this page explains every conceptual step for academic presentation.
"""
    )
    st.markdown("### Viva concept checklist")
    st.markdown(
        """
| Concept | Where you see it |
|---------|------------------|
| Data Mining / Market Basket Analysis | Entire project |
| Association Rule Mining | Association Rules tab |
| Apriori Algorithm | This page + Apriori Analysis |
| Frequent Itemsets / Candidate Generation | Apriori Analysis + pseudocode |
| Support / Confidence / Lift | Metrics boxes + rules table |
| Transaction Database | Dataset Explorer |
| One-Hot Encoding | Dataset Explorer expander |
| Preprocessing | Cleaning stats |
| Product Recommendation | Recommendations tab |
"""
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="main-title">Household Market Basket Analysis using Apriori</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Discover frequent household product combinations with '
        "association rule mining — support, confidence, lift, and recommendations.</div>",
        unsafe_allow_html=True,
    )

    raw, min_support, min_confidence, min_lift, error = render_sidebar()
    if error and raw is None:
        st.error(error)
        st.info("Use the sample dataset or upload a valid CSV to begin.")
        tab_how_apriori()
        return
    if error:
        st.error(error)

    assert raw is not None
    try:
        with st.spinner("Running Apriori analysis…"):
            result = _run_pipeline(
                dataframe_hash(raw),
                raw,
                min_support,
                min_confidence,
                min_lift,
            )
    except DataValidationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")
        return

    tabs = st.tabs(
        [
            "Dashboard",
            "Dataset Explorer",
            "Apriori Analysis",
            "Association Rules",
            "Recommendations",
            "Visualizations",
            "Rule Network",
            "How Apriori Works",
        ]
    )
    with tabs[0]:
        tab_dashboard(result)
    with tabs[1]:
        tab_explorer(result, raw)
    with tabs[2]:
        tab_apriori(result)
    with tabs[3]:
        tab_rules(result)
    with tabs[4]:
        tab_recommendations(result)
    with tabs[5]:
        tab_visualizations(result)
    with tabs[6]:
        tab_network(result)
    with tabs[7]:
        tab_how_apriori()


if __name__ == "__main__":
    main()
