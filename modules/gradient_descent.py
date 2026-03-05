
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def compute_cost(X, y, theta):
    n = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1 / (2 * n)) * np.sum(errors ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
    n = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / n) * X.T.dot(errors)
        theta = theta - alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history


def run_gd(df, target_col, feature_cols, cat_cols, num_cols, learning_rates, iterations):
    """Run gradient descent for multiple learning rates."""
    df_work = df.copy()

    if cat_cols:
        df_work = pd.get_dummies(df_work, columns=cat_cols, drop_first=True, dtype=float)
    if num_cols:
        scaler = StandardScaler()
        df_work[num_cols] = scaler.fit_transform(df_work[num_cols])

    y = df_work[target_col].values.astype(np.float64)
    # Exclude non-numeric target or strings but include the newly created float categorical cols
    X = df_work.drop(target_col, axis=1).select_dtypes(include=[np.number, np.float64]).values.astype(np.float64)

    # Standardize target for stable convergence
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std

    # Add bias column
    X = np.column_stack([np.ones(X.shape[0]), X])
    initial_theta = np.zeros(X.shape[1])

    results = {}
    for alpha in learning_rates:
        theta = initial_theta.copy()
        theta_final, cost_history = gradient_descent(X, y_scaled, theta, alpha, iterations)
        results[alpha] = {
            'cost_history': cost_history,
            'final_cost': cost_history[-1],
            'theta': theta_final
        }
    return results


def render(df, target_col, feature_cols):
    st.markdown("## ⚙️ Problem 4: Gradient Descent in Real-Life Regression")
    st.markdown(f"**Target:** `{target_col}` &nbsp;|&nbsp; **Features:** {len(feature_cols)} columns")
    st.markdown("---")

    if not np.issubdtype(df[target_col].dtype, np.number):
        st.error(f"⚠️ Target `{target_col}` must be numeric for regression.")
        return

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        iterations = st.slider("🔄 Iterations", 100, 5000, 1000, step=100, key="p4_iter")
    with col2:
        st.markdown("**Learning Rates**")
        lr1 = st.number_input("α₁", value=0.1, format="%.4f", key="p4_lr1")
        lr2 = st.number_input("α₂", value=0.01, format="%.4f", key="p4_lr2")
        lr3 = st.number_input("α₃", value=0.001, format="%.4f", key="p4_lr3")

    learning_rates = [lr1, lr2, lr3]

    # Detect column types
    possible_cats = [c for c in feature_cols if df[c].dtype == 'object' or df[c].nunique() < 15]
    possible_nums = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number) and df[c].nunique() >= 15]

    with st.spinner("🔄 Running gradient descent..."):
        results = run_gd(df, target_col, feature_cols, possible_cats, possible_nums,
                         learning_rates, iterations)

    # Final cost metrics
    st.markdown("### 📉 Final Costs")
    cols = st.columns(len(learning_rates))
    for i, alpha in enumerate(learning_rates):
        cols[i].metric(f"α = {alpha}", f"{results[alpha]['final_cost']:.6f}")

    # Interactive cost curve
    st.markdown("### 📈 Cost vs. Iterations")
    colors = ['#FF6B6B', '#2ecc71', '#3498db', '#FFE66D', '#9b59b6']
    fig = go.Figure()

    for i, alpha in enumerate(learning_rates):
        fig.add_trace(go.Scatter(
            x=list(range(iterations)),
            y=results[alpha]['cost_history'],
            mode='lines',
            name=f'α = {alpha} (final: {results[alpha]["final_cost"]:.4f})',
            line=dict(color=colors[i % len(colors)], width=2.5)
        ))

    fig.update_layout(
        xaxis_title='Iterations',
        yaxis_title='Cost (MSE / 2)',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        legend=dict(font=dict(size=13)),
        xaxis=dict(gridcolor='#2D3348'),
        yaxis=dict(gridcolor='#2D3348')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mathematical derivation
    with st.expander("📝 Mathematical Derivation", expanded=False):
        st.markdown("#### MSE Cost Function")
        st.latex(r"J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)})^2")

        st.markdown("#### Gradient (Partial Derivative)")
        st.latex(r"\frac{\partial J}{\partial \theta_j} = \frac{1}{n} \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}")

        st.markdown("#### Update Rule")
        st.latex(r"\theta_j := \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}")

        st.markdown("---")
        st.markdown("#### Why No Local Minima?")
        st.info("The MSE cost function for linear regression is a **convex quadratic** "
                "(its Hessian Xᵀ X is positive semi-definite). Convex functions have a single "
                "global minimum — so gradient descent is guaranteed to converge to the optimal solution "
                "given a suitable learning rate.")
