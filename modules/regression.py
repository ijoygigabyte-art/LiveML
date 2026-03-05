
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_regression(df, target_col, feature_cols, cat_cols, num_cols):
    """Fit OLS model, compute assumption diagnostics, and return results."""
    df_work = df.copy()

    # One-hot encode categoricals
    if cat_cols:
        df_work = pd.get_dummies(df_work, columns=cat_cols, drop_first=True)

    # Standardize numericals
    if num_cols:
        scaler = StandardScaler()
        df_work[num_cols] = scaler.fit_transform(df_work[num_cols])

    y = df_work[target_col]
    X_features = [c for c in df_work.columns if c != target_col]
    X = df_work[X_features]
    X_sm = sm.add_constant(X).astype(float)

    model = sm.OLS(y, X_sm).fit()
    y_pred = model.predict(X_sm)
    residuals = model.resid

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # VIF (skip constant)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_sm.columns[1:]  # skip const
    vif_values = []
    for i in range(1, len(X_sm.columns)):
        try:
            v = variance_inflation_factor(X_sm.values, i)
            vif_values.append(v)
        except Exception:
            vif_values.append(np.nan)
    vif_data["VIF"] = vif_values
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    return model, y, y_pred, residuals, mse, rmse, vif_data


def render(df, target_col, feature_cols):
    st.markdown("## 📐 Problem 3: Multiple Linear Regression & Assumptions")
    st.markdown(f"**Target:** `{target_col}` &nbsp;|&nbsp; **Features:** {len(feature_cols)} columns")
    st.markdown("---")

    # Check target is numeric
    if not np.issubdtype(df[target_col].dtype, np.number):
        st.error(f"⚠️ Target `{target_col}` is not numeric. Regression requires a continuous target.")
        return

    # Let user specify categoricals and numericals
    col1, col2 = st.columns(2)
    possible_cats = [c for c in feature_cols if df[c].dtype == 'object' or df[c].nunique() < 15]
    possible_nums = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number) and df[c].nunique() >= 15]

    with col1:
        cat_cols = st.multiselect("🏷️ Categorical columns (to one-hot encode)",
                                   options=possible_cats, default=possible_cats, key="p3_cats")
    with col2:
        num_cols = st.multiselect("📏 Numerical columns (to standardize)",
                                   options=possible_nums, default=possible_nums, key="p3_nums")

    with st.spinner("🔄 Fitting OLS Model & computing diagnostics..."):
        model, y, y_pred, residuals, mse, rmse, vif_data = run_regression(
            df, target_col, feature_cols, cat_cols, num_cols
        )

    # Model evaluation metrics
    st.markdown("### 📊 Model Evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{model.rsquared:.4f}")
    m2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
    m3.metric("MSE", f"{mse:,.2f}")
    m4.metric("RMSE", f"{rmse:,.2f}")

    # OLS Summary
    with st.expander("📜 Full OLS Summary", expanded=False):
        st.text(str(model.summary()))

    # Assumption Plots — 2x2 grid
    st.markdown("### 🔍 Assumption Checks")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Linearity
    axes[0, 0].scatter(y_pred, y, alpha=0.2, s=8, color='#3498db')
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=2)
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Observed')
    axes[0, 0].set_title('1. Linearity (Observed vs Predicted)')

    # 2. Homoscedasticity
    axes[0, 1].scatter(y_pred, residuals, alpha=0.2, s=8, color='#e74c3c')
    axes[0, 1].axhline(0, color='black', lw=2)
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('2. Homoscedasticity (Residuals vs Fitted)')

    # 3. Normality — Histogram
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='#2ecc71')
    axes[1, 0].set_title('3a. Normality (Histogram of Residuals)')

    # 4. Normality — Q-Q
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('3b. Normality (Q-Q Plot)')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # VIF Table
    st.markdown("### 🧮 Multicollinearity (VIF)")

    def highlight_vif(val):
        if isinstance(val, (int, float)) and val > 10:
            return 'background-color: #FF4444; color: white'
        return ''

    st.dataframe(
        vif_data.head(15).style.applymap(highlight_vif, subset=['VIF']).format({'VIF': '{:.2f}'}),
        use_container_width=True, hide_index=True
    )

    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        st.warning(f"⚠️ {len(high_vif)} features have VIF > 10, indicating multicollinearity. "
                   f"Consider removing or combining: {', '.join(high_vif['Feature'].head(5).tolist())}")
    else:
        st.success("✅ All VIF values are below 10 — no severe multicollinearity detected.")
