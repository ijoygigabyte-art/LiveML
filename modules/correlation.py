
import pandas as pd
import numpy as np
import plotly.figure_factory as ff


def compute_correlation(df, num_cols):
    """Compute Pearson correlation matrix and extract top pairs."""
    corr = df[num_cols].corr(method='pearson')

    # Extract upper triangle pairs
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = corr.where(mask).unstack().dropna().sort_values()
    top_neg = pairs.head(5).reset_index()
    top_neg.columns = ['Feature 1', 'Feature 2', 'Correlation']
    top_pos = pairs.tail(5).sort_values(ascending=False).reset_index()
    top_pos.columns = ['Feature 1', 'Feature 2', 'Correlation']

    return corr, top_pos, top_neg


def render(df, feature_cols):
    st.markdown("## 🔗 Problem 2: Correlation Analysis")
    st.markdown("---")

    # Auto-detect numerical columns
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    available = [c for c in all_numeric if c in feature_cols or c in df.columns]

    num_cols = st.multiselect(
        "📐 Numerical columns for correlation",
        options=all_numeric,
        default=[c for c in all_numeric if c in feature_cols][:10],
        key="p2_nums"
    )

    if len(num_cols) < 2:
        st.warning("Select at least 2 numerical columns.")
        return

    corr, top_pos, top_neg = compute_correlation(df, num_cols)

    # Interactive Plotly Heatmap
    st.markdown("### 🗺️ Pearson Correlation Heatmap")
    z = corr.values.tolist()
    labels = corr.columns.tolist()

    fig = ff.create_annotated_heatmap(
        z=z, x=labels, y=labels,
        annotation_text=np.around(corr.values, 2).tolist(),
        colorscale='RdBu_r',
        showscale=True,
        zmid=0
    )
    fig.update_layout(
        height=500 + len(num_cols) * 15,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA')
    )
    # Fix axis order
    fig.update_layout(yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig, use_container_width=True)

    # Top correlations table
    st.markdown("### 📊 Strongest Correlations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Top 5 Positive")
        st.dataframe(
            top_pos.style.format({'Correlation': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    with col2:
        st.markdown("#### 🔴 Top 5 Negative")
        st.dataframe(
            top_neg.style.format({'Correlation': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

    # Interpretation
    if len(top_pos) > 0:
        f1, f2, val = top_pos.iloc[0]
        st.info(f"**Strongest positive:** `{f1}` ↔ `{f2}` (r = {val:.4f}) — "
                f"these features move together. Consider checking for redundancy or multicollinearity.")
    if len(top_neg) > 0:
        f1, f2, val = top_neg.iloc[0]
        st.info(f"**Strongest negative:** `{f1}` ↔ `{f2}` (r = {val:.4f}) — "
                f"these features move in opposite directions, potentially useful for diverse model input.")
