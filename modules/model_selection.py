
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def run_model_selection(df, target_col, feature_cols, cat_cols, num_cols, reduced_features, interact_a, interact_b):
    """Build Full, Reduced, and Interaction OLS models, return comparison data."""
    df_work = df.copy()

    if cat_cols:
        df_work = pd.get_dummies(df_work, columns=cat_cols, drop_first=True, dtype=float)
    if num_cols:
        scaler = StandardScaler()
        df_work[num_cols] = scaler.fit_transform(df_work[num_cols])

    y = df_work[target_col]
    X_all = df_work.drop(target_col, axis=1).select_dtypes(include=[np.number, np.float64]).astype(float)

    # Model 1: Full
    X_full = sm.add_constant(X_all)
    model_full = sm.OLS(y, X_full).fit()

    # Model 2: Reduced
    available_reduced = [c for c in reduced_features if c in X_all.columns]
    if len(available_reduced) < 1:
        available_reduced = X_all.columns[:3].tolist()
    X_red = sm.add_constant(X_all[available_reduced])
    model_reduced = sm.OLS(y, X_red).fit()

    # Model 3: Interaction
    X_inter = X_all[available_reduced].copy()
    if interact_a in X_inter.columns and interact_b in X_inter.columns:
        X_inter[f'{interact_a}_x_{interact_b}'] = X_inter[interact_a] * X_inter[interact_b]
    X_inter_sm = sm.add_constant(X_inter)
    model_interaction = sm.OLS(y, X_inter_sm).fit()

    models = {
        'Full Model': model_full,
        'Reduced Model': model_reduced,
        'Interaction Model': model_interaction
    }

    comparison = []
    for name, m in models.items():
        comparison.append({
            'Model': name,
            '# Features': m.df_model,
            'AIC': m.aic,
            'BIC': m.bic,
            'Adj R²': m.rsquared_adj
        })
    comparison_df = pd.DataFrame(comparison)
    return comparison_df, models


def render(df, target_col, feature_cols):
    st.markdown("## 🏆 Problem 5: Model Selection Using AIC and BIC")
    st.markdown(f"**Target:** `{target_col}` &nbsp;|&nbsp; **Features:** {len(feature_cols)} columns")
    st.markdown("---")

    if not np.issubdtype(df[target_col].dtype, np.number):
        st.error(f"⚠️ Target `{target_col}` must be numeric for regression.")
        return

    # Detect column types
    possible_cats = [c for c in feature_cols if df[c].dtype == 'object' or df[c].nunique() < 15]
    possible_nums = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number) and df[c].nunique() >= 15]
    all_numeric_features = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

    # User picks reduced features
    st.markdown("### 🔧 Configuration")
    col1, col2 = st.columns(2)

    with col1:
        reduced_features = st.multiselect(
            "📉 Reduced model features (pick 3–4)",
            options=all_numeric_features,
            default=all_numeric_features[:3] if len(all_numeric_features) >= 3 else all_numeric_features,
            key="p5_reduced"
        )

    with col2:
        interact_options = reduced_features if reduced_features else all_numeric_features[:3]
        interact_a = st.selectbox("🔗 Interaction Feature A", options=interact_options, key="p5_ia")
        interact_b = st.selectbox("🔗 Interaction Feature B",
                                   options=[c for c in interact_options if c != interact_a],
                                   key="p5_ib")

    with st.spinner("🔄 Fitting 3 OLS models..."):
        comparison_df, models = run_model_selection(
            df, target_col, feature_cols, possible_cats, possible_nums,
            reduced_features, interact_a, interact_b
        )

    # Comparison Table with highlighting
    st.markdown("### 📊 Model Comparison")

    def highlight_best(col):
        """Green for best value in each metric column."""
        styles = [''] * len(col)
        if col.name in ('AIC', 'BIC'):
            best_idx = col.idxmin()
            styles[best_idx] = 'background-color: #27ae60; color: white; font-weight: bold'
        elif col.name == 'Adj R²':
            best_idx = col.idxmax()
            styles[best_idx] = 'background-color: #27ae60; color: white; font-weight: bold'
        return styles

    st.dataframe(
        comparison_df.style
            .apply(highlight_best)
            .format({'AIC': '{:,.2f}', 'BIC': '{:,.2f}', 'Adj R²': '{:.4f}', '# Features': '{:.0f}'}),
        use_container_width=True, hide_index=True
    )

    # Best model callouts
    best_aic = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']
    best_bic = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']
    best_r2 = comparison_df.loc[comparison_df['Adj R²'].idxmax(), 'Model']

    c1, c2, c3 = st.columns(3)
    c1.success(f"**Best AIC:** {best_aic}")
    c2.success(f"**Best BIC:** {best_bic}")
    c3.success(f"**Best Adj R²:** {best_r2}")

    # Grouped bar chart
    st.markdown("### 📈 AIC & BIC Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='AIC', x=comparison_df['Model'], y=comparison_df['AIC'],
        marker_color='#3498db', text=comparison_df['AIC'].apply(lambda x: f'{x:,.0f}'),
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name='BIC', x=comparison_df['Model'], y=comparison_df['BIC'],
        marker_color='#e74c3c', text=comparison_df['BIC'].apply(lambda x: f'{x:,.0f}'),
        textposition='auto'
    ))
    fig.update_layout(
        barmode='group', height=450,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        xaxis=dict(gridcolor='#2D3348'),
        yaxis=dict(gridcolor='#2D3348', title='Information Criterion Value'),
        legend=dict(font=dict(size=13))
    )
    st.plotly_chart(fig, use_container_width=True)

    # AIC vs BIC explanation
    st.markdown("### 💬 AIC vs BIC: Why They May Disagree")
    st.info(
        "**AIC** (Akaike) penalizes complexity with a fixed term: `2k`. "
        "**BIC** (Bayesian) penalizes with `k·ln(n)`, which grows with sample size.\n\n"
        f"With n = {len(df):,} rows, BIC's penalty per parameter is "
        f"`ln({len(df):,}) ≈ {np.log(len(df)):.1f}` vs AIC's fixed `2`. "
        "This means **BIC strongly favors simpler models in large datasets**, "
        "and may select a different (smaller) model than AIC."
    )
