import streamlit as st
import pandas as pd
import numpy as np

# ─── Page Config ───
st.set_page_config(
    page_title="BikeSmart ML Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #FFE66D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .stMetric {
        background: #1A1F2E;
        border: 1px solid #2D3348;
        border-radius: 12px;
        padding: 15px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 🚲 BikeSmart ML")
    st.markdown("---")

    # File Upload
    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        # Cache the dataframe so we don't consume the file stream on every rerun
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
            # Reset run state on new file upload
            st.session_state['run'] = False
        else:
            df = st.session_state['df']

        st.success(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

        with st.expander("📋 Preview & Dtypes", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
            st.caption("Column Types:")
            st.dataframe(
                df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}),
                use_container_width=True
            )

        st.markdown("---")

        # Column Drop
        cols_to_drop = st.multiselect(
            "🗑️ Columns to Drop",
            options=df.columns.tolist(),
            help="Select columns to exclude from analysis"
        )

        remaining_cols = [c for c in df.columns if c not in cols_to_drop]

        # Target Column
        target_col = st.selectbox(
            "🎯 Target Column",
            options=remaining_cols,
            help="The variable you want to predict"
        )

        # Feature Columns (auto-fill: everything except target and dropped)
        default_features = [c for c in remaining_cols if c != target_col]
        feature_cols = st.multiselect(
            "📊 Feature Columns",
            options=remaining_cols,
            default=default_features,
            help="Select feature columns for modeling"
        )

        st.markdown("---")

        # Problem Selector
        problem = st.radio(
            "🧪 Select Analysis",
            [
                "Problem 1: Imbalanced Classification",
                "Problem 2: Correlation Analysis",
                "Problem 3: Multiple Linear Regression",
                "Problem 4: Gradient Descent",
                "Problem 5: AIC/BIC Model Selection"
            ]
        )

        run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

        # Store selections in session state
        if run_btn:
            cleaned_df = df.drop(columns=cols_to_drop, errors='ignore')
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['target_col'] = target_col
            st.session_state['feature_cols'] = feature_cols
            st.session_state['problem'] = problem
            st.session_state['run'] = True

# ─── Main Area ───
st.markdown('<p class="main-header">🚲 BikeSmart ML Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent analytics for bike-sharing operations</p>', unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.markdown("""
    ### What you can do:
    | Analysis | Description |
    |---|---|
    | **Problem 1** | Imbalanced Classification — baseline vs SMOTE/Tomek resampling |
    | **Problem 2** | Correlation Analysis — interactive Pearson heatmap |
    | **Problem 3** | Multiple Linear Regression — OLS with 4 assumption checks |
    | **Problem 4** | Gradient Descent — manual optimization with learning rate curves |
    | **Problem 5** | AIC/BIC Model Selection — full vs reduced vs interaction models |
    """)

elif st.session_state.get('run'):
    problem = st.session_state.get('problem', '')
    cleaned_df = st.session_state['cleaned_df']
    target_col = st.session_state['target_col']
    feature_cols = st.session_state['feature_cols']

    if "Problem 1" in problem:
        from modules.classification import render
        render(cleaned_df, target_col, feature_cols)
    elif "Problem 2" in problem:
        from modules.correlation import render
        render(cleaned_df, feature_cols)
    elif "Problem 3" in problem:
        from modules.regression import render
        render(cleaned_df, target_col, feature_cols)
    elif "Problem 4" in problem:
        from modules.gradient_descent import render
        render(cleaned_df, target_col, feature_cols)
    elif "Problem 5" in problem:
        from modules.model_selection import render
        render(cleaned_df, target_col, feature_cols)
else:
    st.info("👈 Configure your analysis in the sidebar and click **Run Analysis**.")
