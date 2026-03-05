
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.combine import SMOTETomek


def run_classification(df, target_col, feature_cols, cat_cols):
    """Run baseline and resampled classification, return all metrics and confusion matrices."""
    df_work = df.copy()

    # One-hot encode selected categoricals
    if cat_cols:
        df_work = pd.get_dummies(df_work, columns=cat_cols, drop_first=True)
        # Update feature list with new dummy columns
        new_features = [c for c in df_work.columns if c != target_col]
    else:
        new_features = feature_cols

    X = df_work[new_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df_work[target_col]

    class_dist = y.value_counts(normalize=True) * 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Baseline ---
    rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train, y_train)
    y_pred_base = rf_baseline.predict(X_test)

    base_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_base),
        'Precision': precision_score(y_test, y_pred_base, zero_division=0),
        'Recall': recall_score(y_test, y_pred_base, zero_division=0),
        'F1-score': f1_score(y_test, y_pred_base, zero_division=0)
    }
    base_cm = confusion_matrix(y_test, y_pred_base)

    # --- Resampled ---
    smt = SMOTETomek(random_state=42, n_jobs=-1)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

    rf_resampled = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_resampled.fit(X_train_res, y_train_res)
    y_pred_res = rf_resampled.predict(X_test)

    res_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_res),
        'Precision': precision_score(y_test, y_pred_res, zero_division=0),
        'Recall': recall_score(y_test, y_pred_res, zero_division=0),
        'F1-score': f1_score(y_test, y_pred_res, zero_division=0)
    }
    res_cm = confusion_matrix(y_test, y_pred_res)

    return class_dist, base_metrics, base_cm, res_metrics, res_cm


def render(df, target_col, feature_cols):
    st.markdown("## 📊 Problem 1: Imbalanced Classification")
    st.markdown(f"**Target:** `{target_col}` &nbsp;|&nbsp; **Features:** {len(feature_cols)} columns")
    st.markdown("---")

    # Check binary target
    unique = df[target_col].nunique()
    if unique != 2:
        st.error(f"⚠️ Target column `{target_col}` has {unique} unique values. Classification requires a binary (2-class) target.")
        st.info("💡 Tip: Pick a binary column, or create one (e.g., threshold a continuous variable).")
        return

    # Let user pick categoricals to encode
    possible_cats = [c for c in feature_cols if df[c].dtype == 'object' or df[c].nunique() < 15]
    cat_cols = st.multiselect(
        "🏷️ Categorical columns to one-hot encode",
        options=possible_cats,
        default=possible_cats,
        key="p1_cats"
    )

    with st.spinner("🔄 Training baseline & resampled models... (this may take a minute)"):
        class_dist, base_metrics, base_cm, res_metrics, res_cm = run_classification(
            df, target_col, feature_cols, cat_cols
        )

    # Class distribution
    st.markdown("### Class Distribution")
    col1, col2 = st.columns(2)
    for val, pct in class_dist.items():
        with col1 if val == class_dist.index[0] else col2:
            st.metric(f"Class {val}", f"{pct:.1f}%")

    # Metrics side-by-side
    st.markdown("### 📈 Model Comparison")
    col_base, col_res = st.columns(2)

    with col_base:
        st.markdown("#### Baseline Random Forest")
        for metric, value in base_metrics.items():
            st.metric(metric, f"{value:.4f}")

    with col_res:
        st.markdown("#### After SMOTETomek")
        for metric, value in res_metrics.items():
            delta = value - base_metrics[metric]
            st.metric(metric, f"{value:.4f}", delta=f"{delta:+.4f}")

    # Confusion Matrices side-by-side
    st.markdown("### 🔢 Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title('Baseline RF')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        st.pyplot(fig1)
        plt.close(fig1)

    with cm_col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(res_cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2)
        ax2.set_title('Resampled RF (SMOTETomek)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        st.pyplot(fig2)
        plt.close(fig2)

    # Automatic comparison text
    st.markdown("### 💬 Analysis")
    recall_delta = res_metrics['Recall'] - base_metrics['Recall']
    prec_delta = res_metrics['Precision'] - base_metrics['Precision']

    if recall_delta > 0:
        st.success(f"✅ **Recall improved by {recall_delta*100:.1f}%** after resampling, "
                   f"meaning the model catches more of the minority class.")
    elif recall_delta < 0:
        st.warning(f"⚠️ Recall decreased by {abs(recall_delta)*100:.1f}% after resampling.")
    else:
        st.info("Recall unchanged after resampling.")

    if prec_delta < 0:
        st.info(f"ℹ️ Precision dropped by {abs(prec_delta)*100:.1f}% — a typical trade-off when "
                f"resampling boosts recall at the cost of more false positives.")
