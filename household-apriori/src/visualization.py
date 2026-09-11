"""Plotly visualizations for market basket analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def chart_top_products(transactions: pd.DataFrame, n: int = 10) -> go.Figure:
    """Top-N most frequently purchased products."""
    if transactions is None or transactions.empty or "Product" not in transactions.columns:
        return _empty_figure("No product frequency data available.")

    counts = transactions["Product"].value_counts().head(n).reset_index()
    counts.columns = ["Product", "Frequency"]
    fig = px.bar(
        counts,
        x="Frequency",
        y="Product",
        orientation="h",
        title=f"Top {n} Most Frequently Purchased Products",
        color="Frequency",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def chart_top_itemsets(itemsets: pd.DataFrame, n: int = 10) -> go.Figure:
    """Top frequent itemsets by support."""
    if itemsets is None or itemsets.empty:
        return _empty_figure("No frequent itemsets to chart. Lower minimum support.")

    data = itemsets.head(n).copy()
    label_col = "itemset" if "itemset" in data.columns else "itemsets"
    fig = px.bar(
        data,
        x="support",
        y=label_col,
        orientation="h",
        title=f"Top {min(n, len(data))} Frequent Itemsets by Support",
        color="support",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def chart_rules_by_confidence(rules: pd.DataFrame, n: int = 15) -> go.Figure:
    if rules is None or rules.empty:
        return _empty_figure("No association rules to chart. Adjust confidence/lift.")

    data = rules.sort_values("confidence", ascending=False).head(n)
    fig = px.bar(
        data,
        x="confidence",
        y="rule",
        orientation="h",
        title="Association Rules Ranked by Confidence",
        color="confidence",
        color_continuous_scale="Tealgrn",
        hover_data=["support", "lift"],
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def chart_rules_by_lift(rules: pd.DataFrame, n: int = 15) -> go.Figure:
    if rules is None or rules.empty:
        return _empty_figure("No association rules to chart. Adjust confidence/lift.")

    data = rules.sort_values("lift", ascending=False).head(n)
    fig = px.bar(
        data,
        x="lift",
        y="rule",
        orientation="h",
        title="Association Rules Ranked by Lift",
        color="lift",
        color_continuous_scale="Sunset",
        hover_data=["support", "confidence"],
    )
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def chart_support_confidence_scatter(rules: pd.DataFrame) -> go.Figure:
    if rules is None or rules.empty:
        return _empty_figure("No rules for scatter plot.")

    fig = px.scatter(
        rules,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_name="rule",
        hover_data={
            "antecedent": True,
            "consequent": True,
            "support": ":.3f",
            "confidence": ":.3f",
            "lift": ":.3f",
        },
        title="Support vs Confidence (bubble size = Lift)",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=480, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def chart_rule_network(rules: pd.DataFrame, max_edges: int = 40) -> go.Figure:
    """Interactive network of product association rules."""
    if rules is None or rules.empty:
        return _empty_figure("No association rules available for the network view.")
    if nx is None:
        return _empty_figure("networkx is required for the rule network chart.")

    subset = rules.sort_values("lift", ascending=False).head(max_edges)
    graph = nx.DiGraph()
    for _, row in subset.iterrows():
        ants = [p.strip() for p in str(row["antecedent"]).split(",") if p.strip()]
        cons = [p.strip() for p in str(row["consequent"]).split(",") if p.strip()]
        for a in ants:
            for c in cons:
                weight = float(row["lift"])
                if graph.has_edge(a, c):
                    graph[a][c]["weight"] = max(graph[a][c]["weight"], weight)
                    graph[a][c]["confidence"] = max(
                        graph[a][c]["confidence"], float(row["confidence"])
                    )
                else:
                    graph.add_edge(
                        a,
                        c,
                        weight=weight,
                        confidence=float(row["confidence"]),
                        support=float(row["support"]),
                    )

    if graph.number_of_edges() == 0:
        return _empty_figure("Could not build a network from the current rules.")

    pos = nx.spring_layout(graph, seed=42, k=1.2)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_widths: list[float] = []
    edge_hover: list[str] = []

    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(max(1.0, min(6.0, data["weight"] * 1.5)))
        edge_hover.append(
            f"{u} → {v}<br>Lift: {data['weight']:.2f}<br>"
            f"Confidence: {data['confidence']:.2f}<br>Support: {data['support']:.3f}"
        )

    # Draw edges as separate traces so width can vary
    fig = go.Figure()
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        width = max(1.0, min(6.0, data["weight"] * 1.5))
        opacity = min(0.9, 0.25 + data["weight"] / 4)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=f"rgba(13, 148, 136, {opacity:.2f})"),
                hoverinfo="text",
                text=edge_hover[i],
                showlegend=False,
            )
        )

    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    degrees = [graph.degree(n) for n in graph.nodes()]
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            marker=dict(
                size=[14 + d * 3 for d in degrees],
                color="#0f766e",
                line=dict(width=1, color="#134e4a"),
            ),
            hoverinfo="text",
            hovertext=[f"{n} (connections: {graph.degree(n)})" for n in graph.nodes()],
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Association Rule Network (stronger lift = thicker edges)",
        showlegend=False,
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
