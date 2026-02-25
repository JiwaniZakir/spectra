"""Streamlit dashboard for Spectra RAG evaluation and optimization.

Launch with:
    streamlit run src/spectra/dashboard/app.py
or:
    spectra  (if installed via pip)
"""

from __future__ import annotations

import json
from typing import Any


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Dashboard dependencies are required. Install with: "
            "pip install spectra-rag[dashboard]"
        ) from exc

    st.set_page_config(
        page_title="Spectra RAG Dashboard",
        page_icon="<>",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Spectra -- RAG Evaluation & Optimization")
    st.markdown(
        "Systematic evaluation of retrieval strategies with automated "
        "A/B testing, quality metrics, and Pareto-optimal pipeline selection."
    )

    # ---- Sidebar ----
    st.sidebar.header("Configuration")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Strategy Comparison",
            "Metrics Dashboard",
            "A/B Testing",
            "Pipeline Optimizer",
            "Pareto Frontier",
        ],
    )

    # ---- Strategy Comparison ----
    if page == "Strategy Comparison":
        _render_strategy_comparison(st, pd, px)

    elif page == "Metrics Dashboard":
        _render_metrics_dashboard(st, pd, px)

    elif page == "A/B Testing":
        _render_ab_testing(st, pd, px, go)

    elif page == "Pipeline Optimizer":
        _render_optimizer(st, pd, px)

    elif page == "Pareto Frontier":
        _render_pareto(st, pd, px, go)


def _render_strategy_comparison(st: Any, pd: Any, px: Any) -> None:
    """Render the strategy comparison page."""
    st.header("Retrieval Strategy Comparison")

    strategies = [
        {"Strategy": "Dense (Bi-encoder)", "Type": "Dense", "Latency": "Low", "Quality": "Medium", "Best For": "General retrieval"},
        {"Strategy": "BM25", "Type": "Sparse", "Latency": "Very Low", "Quality": "Medium", "Best For": "Keyword matching"},
        {"Strategy": "SPLADE", "Type": "Sparse", "Latency": "Medium", "Quality": "High", "Best For": "Learned sparse retrieval"},
        {"Strategy": "Hybrid (RRF)", "Type": "Hybrid", "Latency": "Low", "Quality": "High", "Best For": "Combining dense + sparse"},
        {"Strategy": "HyDE", "Type": "Generative", "Latency": "High", "Quality": "High", "Best For": "Zero-shot retrieval"},
        {"Strategy": "Self-RAG", "Type": "Iterative", "Latency": "Very High", "Quality": "Very High", "Best For": "Adaptive retrieval"},
        {"Strategy": "ColBERT", "Type": "Late Interaction", "Latency": "Medium", "Quality": "High", "Best For": "Fine-grained matching"},
        {"Strategy": "Multi-hop", "Type": "Iterative", "Latency": "High", "Quality": "High", "Best For": "Complex questions"},
        {"Strategy": "Graph-RAG", "Type": "Graph", "Latency": "High", "Quality": "High", "Best For": "Entity-rich domains"},
        {"Strategy": "IRCoT", "Type": "Iterative", "Latency": "Very High", "Quality": "Very High", "Best For": "Multi-step reasoning"},
        {"Strategy": "Chain-of-Note", "Type": "Generative", "Latency": "High", "Quality": "High", "Best For": "Robust retrieval"},
        {"Strategy": "Cross-encoder Reranker", "Type": "Reranking", "Latency": "Medium", "Quality": "Very High", "Best For": "Second-stage reranking"},
    ]
    df = pd.DataFrame(strategies)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Upload Evaluation Results")
    uploaded = st.file_uploader("Upload a JSON evaluation file", type=["json"])
    if uploaded:
        data = json.loads(uploaded.read())
        st.json(data)


def _render_metrics_dashboard(st: Any, pd: Any, px: Any) -> None:
    """Render the metrics dashboard page."""
    st.header("Evaluation Metrics")

    metrics_info = [
        {"Metric": "Faithfulness", "Range": "[0, 1]", "Description": "Claims supported by context"},
        {"Metric": "Answer Relevance", "Range": "[0, 1]", "Description": "Semantic similarity to query"},
        {"Metric": "Context Relevance", "Range": "[0, 1]", "Description": "Fraction of relevant contexts"},
        {"Metric": "Coherence", "Range": "[0, 1]", "Description": "Logical flow of answer"},
        {"Metric": "Context Recall", "Range": "[0, 1]", "Description": "Ground truth coverage"},
        {"Metric": "Context Precision", "Range": "[0, 1]", "Description": "Precision of retrieved docs"},
        {"Metric": "Answer Correctness", "Range": "[0, 1]", "Description": "F1 + semantic similarity"},
        {"Metric": "Latency Score", "Range": "[0, 1]", "Description": "Normalized inverse latency"},
    ]
    st.dataframe(pd.DataFrame(metrics_info), use_container_width=True, hide_index=True)

    st.subheader("Metric Visualization")
    st.markdown("Upload evaluation results or run a pipeline to see metric visualizations here.")

    # Demo radar chart with placeholder data
    demo_data = {
        "Metric": ["Faithfulness", "Answer Relevance", "Context Relevance", "Coherence",
                    "Context Recall", "Context Precision", "Answer Correctness", "Latency Score"],
        "Dense": [0.75, 0.80, 0.70, 0.85, 0.65, 0.72, 0.78, 0.90],
        "Hybrid": [0.82, 0.85, 0.78, 0.87, 0.75, 0.80, 0.83, 0.85],
        "HyDE": [0.88, 0.82, 0.85, 0.86, 0.80, 0.78, 0.85, 0.60],
    }
    demo_df = pd.DataFrame(demo_data)

    fig = px.line_polar(
        demo_df.melt(id_vars="Metric", var_name="Strategy", value_name="Score"),
        r="Score",
        theta="Metric",
        color="Strategy",
        line_close=True,
        title="Strategy Comparison (Demo Data)",
    )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)


def _render_ab_testing(st: Any, pd: Any, px: Any, go: Any) -> None:
    """Render the A/B testing page."""
    st.header("A/B Testing")

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Pipeline A", ["Dense", "BM25", "Hybrid", "HyDE", "Self-RAG"], key="pipeline_a")
    with col2:
        st.selectbox("Pipeline B", ["Dense", "BM25", "Hybrid", "HyDE", "Self-RAG"], index=2, key="pipeline_b")

    st.selectbox("Statistical Test", ["Welch's t-test", "Mann-Whitney U", "Bootstrap", "Paired t-test"])
    st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)

    if st.button("Run A/B Test"):
        st.info("Connect your evaluation data to run A/B tests. See the examples/ directory for usage.")

    st.subheader("Interpreting Results")
    st.markdown("""
    - **p-value < alpha**: Statistically significant difference detected.
    - **Cohen's d**: Effect size (small: 0.2, medium: 0.5, large: 0.8).
    - **Confidence Interval**: Range of plausible values for the true difference.
    """)


def _render_optimizer(st: Any, pd: Any, px: Any) -> None:
    """Render the pipeline optimizer page."""
    st.header("Pipeline Optimizer (Optuna)")

    st.number_input("Number of Trials", min_value=10, max_value=500, value=50)
    st.selectbox("Objective Metric", [
        "Faithfulness", "Answer Relevance", "Context Relevance",
        "Coherence", "Answer Correctness",
    ])
    st.multiselect(
        "Search Space: Retrieval Strategies",
        ["Dense", "BM25", "Hybrid"],
        default=["Dense", "BM25", "Hybrid"],
    )
    st.multiselect(
        "Search Space: Chunking Strategies",
        ["Fixed", "Recursive", "Semantic", "Document-Aware"],
        default=["Fixed", "Recursive"],
    )

    if st.button("Start Optimization"):
        st.info("Connect your document corpus and evaluation queries to run optimization.")

    st.subheader("Optimization History")
    st.markdown("Optimization trial history and hyperparameter importance will appear here after a run.")


def _render_pareto(st: Any, pd: Any, px: Any, go: Any) -> None:
    """Render the Pareto frontier page."""
    st.header("Pareto Frontier Analysis")

    st.markdown(
        "The Pareto frontier shows pipeline configurations where no other "
        "configuration is better on **all** objectives simultaneously."
    )

    # Demo Pareto frontier
    import numpy as np
    rng = np.random.default_rng(42)
    n = 30
    quality = rng.uniform(0.5, 1.0, n)
    latency = rng.uniform(0.1, 1.0, n)

    # Simple Pareto check
    is_pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and quality[j] >= quality[i] and latency[j] <= latency[i]:
                if quality[j] > quality[i] or latency[j] < latency[i]:
                    dominated = True
                    break
        is_pareto.append(not dominated)

    df = pd.DataFrame({
        "Quality": quality,
        "Latency": latency,
        "Pareto Optimal": ["Yes" if p else "No" for p in is_pareto],
    })

    fig = px.scatter(
        df,
        x="Latency",
        y="Quality",
        color="Pareto Optimal",
        color_discrete_map={"Yes": "#2ecc71", "No": "#95a5a6"},
        title="Pareto Frontier: Quality vs Latency (Demo)",
    )

    # Add frontier line
    pareto_df = df[df["Pareto Optimal"] == "Yes"].sort_values("Latency")
    fig.add_trace(
        go.Scatter(
            x=pareto_df["Latency"],
            y=pareto_df["Quality"],
            mode="lines",
            name="Frontier",
            line=dict(color="#2ecc71", dash="dash"),
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Frontier Configurations")
    st.dataframe(pareto_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
