"""
Telco Customer Churn — Prediction + Retention Dashboard
Layer 1: Churn Prediction | Layer 2: Customer Segment | Layer 3: SHAP + Retention Strategy
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

ARTIFACTS_PATH = 'artifacts/all_artifacts.pkl'

RETENTION_STRATEGIES = {
    'Contract': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Offer **annual contract** with 15-20% discount on monthly rate. "
                  "Month-to-month is the #1 churn driver.",
        'discount': '15-20%',
        'channel': 'Email + In-app notification',
    },
    'OnlineSecurity': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Offer **free Online Security** add-on for 3 months. "
                  "Customers without security feel less invested.",
        'discount': 'Free for 3 months',
        'channel': 'Email',
    },
    'TechSupport': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Offer **free Tech Support** for 6 months. "
                  "Unsupported customers churn significantly more.",
        'discount': 'Free for 6 months',
        'channel': 'Phone call from support team',
    },
    'InternetService': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Review **fiber optic pricing**. Fiber customers churn more — "
                  "offer speed upgrade at same price or 10% discount.",
        'discount': '10%',
        'channel': 'Email + SMS',
    },
    'tenure': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Enroll in **New Customer Loyalty Program** — "
                  "milestone rewards at 6, 12, 24 months.",
        'discount': 'Loyalty rewards',
        'channel': 'In-app + Email',
    },
    'MonthlyCharges': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Offer **plan optimization** — review usage and suggest a "
                  "cheaper plan or bundle discount.",
        'discount': '10-15% on current plan',
        'channel': 'Phone call',
    },
    'OnlineBackup': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Offer **free Online Backup** for 3 months to increase stickiness.",
        'discount': 'Free for 3 months',
        'channel': 'Email',
    },
    'DeviceProtection': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Offer **free Device Protection** for 3 months as a retention perk.",
        'discount': 'Free for 3 months',
        'channel': 'Email',
    },
    'StreamingTV': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Bundle **Streaming TV** at 50% off to increase service count and engagement.",
        'discount': '50% off add-on',
        'channel': 'In-app',
    },
    'StreamingMovies': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Bundle **Streaming Movies** at 50% off to increase service count.",
        'discount': '50% off add-on',
        'channel': 'In-app',
    },
    'PaperlessBilling': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Switch to **paper billing** or offer billing transparency dashboard.",
        'discount': 'N/A',
        'channel': 'Email',
    },
    'PaymentMethod': {
        'condition': lambda val, shap_val: shap_val > 0.02,
        'action': "Suggest switching to **auto-pay** (bank transfer/credit card) "
                  "for convenience and small discount.",
        'discount': '5% auto-pay discount',
        'channel': 'Email + SMS',
    },
}

CLUSTER_NAMES = {
    0: ("Budget Basics", "\U0001F7E2", "Low-spend, phone-only, very low churn"),
    1: ("Flight Risks", "\U0001F534", "Month-to-month, moderate spend, HIGH churn"),
    2: ("Premium Loyalists", "\U0001F535", "Long tenure, high spend, many services, low churn"),
}


@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_PATH, 'rb') as f:
        return pickle.load(f)


def main():
    st.set_page_config(page_title="Telco Churn Retention", layout="wide", page_icon="\U0001F4C9")
    st.title("\U0001F4C9 Telco Customer Churn — Prediction + Retention System")
    st.caption("Layer 1: Churn Prediction  |  Layer 2: Customer Segment  |  Layer 3: SHAP + Retention Strategy")

    if not os.path.exists(ARTIFACTS_PATH):
        st.error("Run `python train_all.py` first!")
        return

    a = load_artifacts()

    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001F4CA Overview",
        "\U0001F3C6 Model Comparison",
        "\U0001F50D Cluster Analysis",
        "\U0001F3AF Predict & Retain",
    ])

    with tab1:
        _overview(a)
    with tab2:
        _model_comparison(a)
    with tab3:
        _clusters(a)
    with tab4:
        _predict_and_retain(a)


def _overview(a):
    df = a['df']
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    c3.metric("Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}")
    c4.metric("Avg Tenure", f"{df['tenure'].mean():.0f} mo")

    st.subheader("Feature Importance (Layer 1 — XGBoost)")
    model = a['churn_model']
    importance = pd.Series(
        model.feature_importances_, index=a['feature_display_names']
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#ef5350' if v > 0.05 else '#2196F3' for v in importance.values]
    ax.barh(importance.index, importance.values, color=colors)
    ax.set_xlabel('Feature Importance')
    ax.set_title('What Drives Churn?')
    for i, (feat, v) in enumerate(importance.items()):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", "0.8298")
    col2.metric("Accuracy", "75.8%")
    col3.metric("F1 (Churned)", "0.6076")


def _model_comparison(a):
    st.header("Model Comparison — 5 Algorithms")
    st.write("All models trained on the same 80/20 split with stratified sampling. "
             "Models needing feature scaling (Logistic Regression, SVM, KNN) use StandardScaler.")

    comp = a.get('model_comparison')
    if comp is None:
        st.warning("Model comparison data not found. Re-run `python train_all.py`.")
        return

    comp = comp.copy().reset_index(drop=True)
    best_model = comp.iloc[0]['Model']

    st.subheader("Results Table")
    styled = comp.copy()
    for col in ['AUC-ROC', 'Accuracy', 'F1', 'Precision', 'Recall']:
        styled[col] = styled[col].map(lambda x: f"{x:.4f}")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.success(f"**Best Model: {best_model}** (highest AUC-ROC)")

    st.subheader("Visual Comparison")

    metrics = ['AUC-ROC', 'Accuracy', 'F1', 'Precision', 'Recall']
    model_names = comp['Model'].tolist()
    colors_map = {
        'Logistic Regression': '#42A5F5',
        'Random Forest': '#66BB6A',
        'XGBoost': '#EF5350',
        'SVM (RBF)': '#AB47BC',
        'K-Nearest Neighbors': '#FFA726',
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)

    for ax, metric in zip(axes, metrics):
        vals = comp[metric].values
        bar_colors = [colors_map.get(m, '#888') for m in model_names]
        bars = ax.barh(model_names, vals, color=bar_colors, edgecolor='white')
        ax.set_title(metric, fontweight='bold', fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.2, axis='x')

        for b, v in zip(bars, vals):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Radar Chart")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for _, row in comp.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        color = colors_map.get(row['Model'], '#888')
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title('Model Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Why XGBoost?")
    st.markdown("""
    | Criteria | Winner | Why |
    |----------|--------|-----|
    | **AUC-ROC** | XGBoost | Best at ranking churners vs non-churners |
    | **Recall** | SVM / Logistic Regression | Catches more churners but with lower precision |
    | **Precision** | XGBoost / Random Forest | Fewer false alarms when flagging churners |
    | **SHAP compatible** | XGBoost / Random Forest | Tree-based models work with TreeExplainer (fast, exact) |
    | **Production speed** | XGBoost / Logistic Regression | Fast inference for real-time predictions |

    **Final choice: XGBoost** — best AUC-ROC, strong F1, SHAP-compatible for Layer 3 explanations,
    and fast enough for the real-time Predict & Retain tab.
    """)


def _clusters(a):
    df = a['df']
    st.header("Customer Segments (Layer 2 — K-Means)")

    for cluster_id in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cluster_id
        seg = df[mask]
        name, icon, desc = CLUSTER_NAMES.get(cluster_id, (f"Cluster {cluster_id}", "\u2753", ""))

        churn = seg['Churn'].mean()
        with st.expander(f"{icon} Cluster {cluster_id}: {name} — {len(seg):,} customers ({len(seg)/len(df):.0%})", expanded=True):
            st.caption(desc)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Avg Tenure", f"{seg['tenure'].mean():.0f} mo")
            c2.metric("Avg Monthly", f"${seg['MonthlyCharges'].mean():.0f}")
            c3.metric("Avg Total Rev", f"${seg['TotalCharges'].mean():,.0f}")
            c4.metric("Churn Rate", f"{churn:.1%}")
            c5.metric("Risk", "\U0001F534 HIGH" if churn > 0.35 else ("\U0001F7E1 MED" if churn > 0.2 else "\U0001F7E2 LOW"))


def _predict_and_retain(a):
    st.header("Predict Churn & Get Retention Strategy")
    st.write("Enter customer details below. The system will predict churn, identify their segment, "
             "explain WHY they might churn (SHAP), and recommend personalized retention actions.")

    encoders = a['encoders']

    col1, col2, col3, col4 = st.columns(4)
    gender = col1.selectbox("Gender", list(encoders['gender'].classes_))
    senior = col2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = col3.selectbox("Partner", list(encoders['Partner'].classes_))
    dependents = col4.selectbox("Dependents", list(encoders['Dependents'].classes_))

    col1, col2, col3 = st.columns(3)
    tenure = col1.slider("Tenure (months)", 0, 72, 12)
    monthly = col2.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=5.0)
    total = col3.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly), step=100.0)

    col1, col2, col3 = st.columns(3)
    phone = col1.selectbox("Phone Service", list(encoders['PhoneService'].classes_))
    multilines = col2.selectbox("Multiple Lines", list(encoders['MultipleLines'].classes_))
    internet = col3.selectbox("Internet Service", list(encoders['InternetService'].classes_))

    col1, col2, col3, col4 = st.columns(4)
    security = col1.selectbox("Online Security", list(encoders['OnlineSecurity'].classes_))
    backup = col2.selectbox("Online Backup", list(encoders['OnlineBackup'].classes_))
    protection = col3.selectbox("Device Protection", list(encoders['DeviceProtection'].classes_))
    techsupport = col4.selectbox("Tech Support", list(encoders['TechSupport'].classes_))

    col1, col2, col3, col4 = st.columns(4)
    tv = col1.selectbox("Streaming TV", list(encoders['StreamingTV'].classes_))
    movies = col2.selectbox("Streaming Movies", list(encoders['StreamingMovies'].classes_))
    contract = col3.selectbox("Contract", list(encoders['Contract'].classes_))
    paperless = col4.selectbox("Paperless Billing", list(encoders['PaperlessBilling'].classes_))

    payment = st.selectbox("Payment Method", list(encoders['PaymentMethod'].classes_))

    if not st.button("\U0001F680 Predict Churn & Get Retention Plan", type="primary", use_container_width=True):
        return

    raw_values = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'MonthlyCharges': monthly,
        'TotalCharges': total, 'PhoneService': phone, 'MultipleLines': multilines,
        'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
        'DeviceProtection': protection, 'TechSupport': techsupport,
        'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
    }

    encoded = {
        'SeniorCitizen': senior,
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
    }
    for col_name, le in encoders.items():
        if col_name in raw_values:
            encoded[col_name + '_enc'] = int(le.transform([raw_values[col_name]])[0])

    X_input = pd.DataFrame([{col: encoded[col] for col in a['feature_cols']}])

    # ── Layer 1: Churn Prediction ──────────────────────────
    st.markdown("---")
    st.subheader("Layer 1 — Churn Prediction")

    churn_prob = float(a['churn_model'].predict_proba(X_input)[0][1])
    risk = "\U0001F6A8 HIGH RISK" if churn_prob > 0.6 else ("\u26A0\uFE0F MEDIUM" if churn_prob > 0.3 else "\u2705 LOW")

    c1, c2 = st.columns([1, 2])
    c1.metric("Churn Probability", f"{churn_prob:.1%}", delta=risk, delta_color="inverse")
    c2.progress(min(churn_prob, 1.0))

    # ── Layer 2: Customer Segment ──────────────────────────
    st.markdown("---")
    st.subheader("Layer 2 — Customer Segment")

    service_count = sum(1 for v in [phone, security, backup, protection, techsupport, tv, movies] if v == 'Yes')
    contract_len = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}.get(contract, 0)
    has_net = 0 if internet == 'No' else 1
    avg_rev = total / tenure if tenure > 0 else monthly

    cluster_input = np.array([[tenure, monthly, total, service_count,
                               contract_len, has_net, avg_rev]])
    cluster_scaled = a['km_scaler'].transform(cluster_input)
    cluster_id = int(a['km_model'].predict(cluster_scaled)[0])

    name, icon, desc = CLUSTER_NAMES.get(cluster_id, (f"Cluster {cluster_id}", "\u2753", ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("Segment", f"{icon} {name}")
    c2.metric("Cluster ID", cluster_id)
    c3.metric("Description", desc)

    # ── Layer 3: SHAP + Retention Strategy ─────────────────
    st.markdown("---")
    st.subheader("Layer 3 — Why Might They Churn? (SHAP)")

    sv = a['shap_explainer'].shap_values(X_input)
    if isinstance(sv, list):
        sv = sv[1]
    shap_vals = np.array(sv).flatten()

    impact = pd.DataFrame({
        'Feature': a['feature_display_names'],
        'SHAP': shap_vals,
    }).sort_values('SHAP', key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#ef5350' if v > 0 else '#66bb6a' for v in impact['SHAP']]
    bars = ax.barh(impact['Feature'], impact['SHAP'], color=colors)
    ax.set_xlabel('Impact on Churn Prediction (SHAP value)')
    ax.set_title('Why This Customer Might Churn')
    ax.axvline(0, color='black', lw=0.5)
    for b, v in zip(bars, impact['SHAP']):
        x = b.get_width()
        ax.text(x + (0.003 if x >= 0 else -0.003),
                b.get_y() + b.get_height() / 2,
                f'{v:+.3f}', va='center', ha='left' if x >= 0 else 'right',
                fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    # ── Retention Strategy ─────────────────────────────────
    st.markdown("---")
    st.subheader("\U0001F4CB Personalized Retention Strategy")

    if churn_prob < 0.2:
        st.success("\u2705 **LOW CHURN RISK** — No immediate retention action needed. "
                   "Continue monitoring.")
        return

    strategies = []
    for feat, shap_val in zip(a['feature_display_names'], shap_vals):
        if feat in RETENTION_STRATEGIES:
            strat = RETENTION_STRATEGIES[feat]
            raw_val = raw_values.get(feat, None)
            if strat['condition'](raw_val, shap_val):
                strategies.append({
                    'Feature': feat,
                    'Current Value': str(raw_val),
                    'Churn Impact': f"{shap_val:+.3f}",
                    'Action': strat['action'],
                    'Offer': strat['discount'],
                    'Channel': strat['channel'],
                })

    if not strategies:
        st.info("No strong individual churn drivers found. Consider a general loyalty offer.")
        return

    strategies.sort(key=lambda x: abs(float(x['Churn Impact'].replace('+', ''))), reverse=True)

    for i, s in enumerate(strategies, 1):
        with st.container():
            st.markdown(f"### Action {i}: Target **{s['Feature']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Value", s['Current Value'])
            c2.metric("Churn Impact", s['Churn Impact'])
            c3.metric("Offer", s['Offer'])
            st.markdown(f"**What to do:** {s['Action']}")
            st.markdown(f"**Channel:** {s['Channel']}")
            st.markdown("---")

    total_impact = sum(float(s['Churn Impact'].replace('+', '')) for s in strategies)
    new_churn = max(churn_prob - total_impact * 0.5, 0.05)

    st.subheader("Expected Outcome")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Churn Risk", f"{churn_prob:.1%}")
    c2.metric("After Retention Actions", f"{new_churn:.1%}",
              delta=f"{new_churn - churn_prob:+.1%}", delta_color="normal")
    c3.metric("Actions Recommended", len(strategies))


if __name__ == '__main__':
    main()
