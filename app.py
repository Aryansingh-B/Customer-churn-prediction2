import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=os.getenv("APP_TITLE", "Churn Intelligence Dashboard"),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7c83fd;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 4px;
    }
    .risk-high { color: #FF4B4B; font-weight: bold; }
    .risk-medium { color: #FFA500; font-weight: bold; }
    .risk-low { color: #00C851; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Artifacts (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_churn_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

@st.cache_data
def load_predictions():
    return pd.read_csv("models/test_predictions.csv")

@st.cache_data
def load_metrics():
    with open("models/metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_importance():
    return pd.read_csv("models/feature_importance.csv")

@st.cache_data
def load_raw_data():
    df = pd.read_csv("data/telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Churn Intelligence")
    st.markdown("---")

    st.markdown("### 🔍 Filters")
    raw_df = load_raw_data()

    contract_filter = st.multiselect(
        "Contract Type",
        options=raw_df["Contract"].unique().tolist(),
        default=raw_df["Contract"].unique().tolist()
    )

    internet_filter = st.multiselect(
        "Internet Service",
        options=raw_df["InternetService"].unique().tolist(),
        default=raw_df["InternetService"].unique().tolist()
    )

    tenure_range = st.slider(
        "Tenure (months)",
        min_value=int(raw_df["tenure"].min()),
        max_value=int(raw_df["tenure"].max()),
        value=(0, 72)
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    metrics = load_metrics()
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    st.metric("Precision", f"{metrics['precision']:.4f}")
    st.metric("Recall", f"{metrics['recall']:.4f}")

    st.markdown("---")
    st.caption("Built with XGBoost + Streamlit")


# ─────────────────────────────────────────────
# Apply Filters
# ─────────────────────────────────────────────
filtered_df = raw_df[
    raw_df["Contract"].isin(contract_filter) &
    raw_df["InternetService"].isin(internet_filter) &
    raw_df["tenure"].between(tenure_range[0], tenure_range[1])
]


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("Real-time churn analysis, risk segmentation, and retention levers.")
st.markdown("---")


# ─────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────
churners = filtered_df[filtered_df["Churn"] == 1]
non_churners = filtered_df[filtered_df["Churn"] == 0]
churn_rate = filtered_df["Churn"].mean()
clv_at_risk = churners["TotalCharges"].sum()
avg_tenure_churn = churners["tenure"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Customers", f"{len(filtered_df):,}")
with col2:
    st.metric("Churned", f"{len(churners):,}", delta=f"{churn_rate:.1%} rate", delta_color="inverse")
with col3:
    st.metric("Revenue at Risk", f"₹{clv_at_risk:,.0f}", delta="CLV at risk", delta_color="inverse")
with col4:
    st.metric("Avg Tenure (Churners)", f"{avg_tenure_churn:.1f} mo")
with col5:
    retention_opp = churners["MonthlyCharges"].mean() * 12
    st.metric("Avg Annual Value/Churner", f"₹{retention_opp:,.0f}")

st.markdown("---")


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview",
    "🎯 Segment Analysis",
    "⚠️ Risk Scoring",
    "🔮 Predict Single Customer"
])


# ══════════════════════════════════════════════
# TAB 1: Overview
# ══════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    # Churn by Contract Type
    with col_a:
        contract_churn = filtered_df.groupby("Contract")["Churn"].agg(["sum", "count"]).reset_index()
        contract_churn["churn_rate"] = contract_churn["sum"] / contract_churn["count"]
        fig = px.bar(
            contract_churn, x="Contract", y="churn_rate",
            color="churn_rate",
            color_continuous_scale="RdYlGn_r",
            title="Churn Rate by Contract Type",
            labels={"churn_rate": "Churn Rate", "Contract": "Contract Type"},
            text=contract_churn["churn_rate"].apply(lambda x: f"{x:.1%}")
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Churn by Internet Service
    with col_b:
        internet_churn = filtered_df.groupby("InternetService")["Churn"].agg(["sum", "count"]).reset_index()
        internet_churn["churn_rate"] = internet_churn["sum"] / internet_churn["count"]
        fig = px.pie(
            internet_churn, values="sum", names="InternetService",
            title="Churners by Internet Service",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Monthly Charges Distribution
    with col_c:
        fig = px.histogram(
            filtered_df, x="MonthlyCharges", color=filtered_df["Churn"].map({0: "No Churn", 1: "Churned"}),
            barmode="overlay", nbins=40,
            title="Monthly Charges: Churned vs Retained",
            labels={"color": "Status"},
            color_discrete_map={"Churned": "#FF4B4B", "No Churn": "#00C851"},
            opacity=0.75
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Tenure vs Churn
    with col_d:
        fig = px.histogram(
            filtered_df, x="tenure", color=filtered_df["Churn"].map({0: "No Churn", 1: "Churned"}),
            barmode="overlay", nbins=30,
            title="Tenure Distribution: Churned vs Retained",
            labels={"color": "Status"},
            color_discrete_map={"Churned": "#FF4B4B", "No Churn": "#00C851"},
            opacity=0.75
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("### 🔑 Top Predictors of Churn")
    fi_df = load_feature_importance().head(15)
    fig = px.bar(
        fi_df, x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        title="Top 15 Features by XGBoost Importance",
        labels={"importance": "Importance Score", "feature": "Feature"}
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2: Segment Analysis
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🎯 Churn by Customer Segment")

    col_a, col_b = st.columns(2)

    with col_a:
        # Payment Method
        pm_churn = filtered_df.groupby("PaymentMethod")["Churn"].agg(["sum", "count"]).reset_index()
        pm_churn["churn_rate"] = pm_churn["sum"] / pm_churn["count"]
        pm_churn["PaymentMethod"] = pm_churn["PaymentMethod"].str.replace(" (automatic)", "", regex=False)
        fig = px.bar(
            pm_churn, x="churn_rate", y="PaymentMethod",
            orientation="h",
            color="churn_rate",
            color_continuous_scale="RdYlGn_r",
            title="Churn Rate by Payment Method",
            text=pm_churn["churn_rate"].apply(lambda x: f"{x:.1%}")
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Senior Citizen
        senior_churn = filtered_df.groupby("SeniorCitizen")["Churn"].agg(["sum", "count"]).reset_index()
        senior_churn["churn_rate"] = senior_churn["sum"] / senior_churn["count"]
        senior_churn["SeniorCitizen"] = senior_churn["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior Citizen"})
        fig = px.bar(
            senior_churn, x="SeniorCitizen", y="churn_rate",
            color="SeniorCitizen",
            title="Senior vs Non-Senior Churn Rate",
            text=senior_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
            color_discrete_sequence=["#7c83fd", "#FF4B4B"]
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: Churn rate by Contract × Internet
    st.markdown("### 🔥 Churn Heatmap: Contract × Internet Service")
    heatmap_data = filtered_df.groupby(["Contract", "InternetService"])["Churn"].mean().unstack()
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale="RdYlGn_r",
        text_auto=".1%",
        title="Churn Rate by Contract Type and Internet Service",
        labels={"color": "Churn Rate"}
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Retention Levers
    st.markdown("### 💡 Actionable Retention Levers")
    col1, col2, col3 = st.columns(3)

    month_to_month = filtered_df[filtered_df["Contract"] == "Month-to-month"]
    m2m_churn = month_to_month["Churn"].mean()

    with col1:
        st.info(f"""
        **📋 Month-to-Month Contracts**
        
        Churn rate: **{m2m_churn:.1%}**
        
        Customers: {len(month_to_month):,}
        
        → Offer 3-month free upgrade to Annual plan
        """)

    no_security = filtered_df[filtered_df["OnlineSecurity"] == "No"]
    ns_churn = no_security["Churn"].mean()
    with col2:
        st.warning(f"""
        **🔒 No Online Security**
        
        Churn rate: **{ns_churn:.1%}**
        
        Customers: {len(no_security):,}
        
        → Bundle security at 50% discount for 6 months
        """)

    fiber_users = filtered_df[filtered_df["InternetService"] == "Fiber optic"]
    fiber_churn = fiber_users["Churn"].mean()
    with col3:
        st.error(f"""
        **🌐 Fiber Optic Users**
        
        Churn rate: **{fiber_churn:.1%}**
        
        Customers: {len(fiber_users):,}
        
        → Priority support + speed guarantee SLA
        """)


# ══════════════════════════════════════════════
# TAB 3: Risk Scoring
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### ⚠️ Customer Risk Score Distribution")

    pred_df = load_predictions()

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(
            pred_df, x="Churn_Probability", nbins=40,
            color_discrete_sequence=["#7c83fd"],
            title="Distribution of Churn Probabilities",
            labels={"Churn_Probability": "Churn Probability"}
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Risk segments
        pred_df["Risk_Segment"] = pd.cut(
            pred_df["Churn_Probability"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]
        )
        risk_counts = pred_df["Risk_Segment"].value_counts().reset_index()
        risk_counts.columns = ["Segment", "Count"]
        fig = px.pie(
            risk_counts, values="Count", names="Segment",
            title="Customers by Risk Segment",
            color_discrete_map={
                "🟢 Low Risk": "#00C851",
                "🟡 Medium Risk": "#FFA500",
                "🔴 High Risk": "#FF4B4B"
            },
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.markdown("### 📉 Model Confusion Matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"]
    )
    fig = px.imshow(
        cm_df, text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix (Test Set)",
        labels={"color": "Count"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4: Predict Single Customer
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 🔮 Predict Churn for a New Customer")
    st.markdown("Fill in customer details to get an instant churn risk score.")

    model = load_model()
    scaler = load_scaler()
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        with col2:
            st.markdown("**📦 Services**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("**💳 Billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges (₹)", 18.0, 120.0, 65.0, step=0.5)
            total_charges = monthly_charges * tenure

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        # Build raw input dict
        raw_input = {
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "PhoneService": 1 if phone_service == "Yes" else 0,
            "PaperlessBilling": 1 if paperless == "Yes" else 0,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "AvgMonthlySpend": total_charges / (tenure + 1),
            "CLV": monthly_charges * tenure,
            "NumServices": sum([
                phone_service == "Yes",
                multiple_lines == "Yes",
                internet_service != "No",
                online_security == "Yes",
                tech_support == "Yes"
            ])
        }

        # OHE columns — set all to 0 first, then flip the right ones
        ohe_cols = [f for f in feature_names if f not in raw_input]
        for col in ohe_cols:
            raw_input[col] = 0

        # Flip relevant OHE flags
        def set_ohe(prefix, value):
            key = f"{prefix}_{value}"
            if key in raw_input:
                raw_input[key] = 1

        set_ohe("MultipleLines", multiple_lines)
        set_ohe("InternetService", internet_service)
        set_ohe("OnlineSecurity", online_security)
        set_ohe("TechSupport", tech_support)
        set_ohe("Contract", contract)
        set_ohe("PaymentMethod", payment_method)

        # Build DataFrame with exact feature columns
        input_df = pd.DataFrame([raw_input])[feature_names]

        # Scale numeric features
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "CLV", "NumServices"]
        numeric_cols = [c for c in numeric_cols if c in input_df.columns]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        prob = model.predict_proba(input_df)[0][1]

        # Display result
        st.markdown("---")
        if prob >= 0.70:
            risk_label, risk_color = "🔴 HIGH RISK", "#FF4B4B"
            advice = "Immediate intervention recommended. Offer contract upgrade + loyalty discount."
        elif prob >= 0.40:
            risk_label, risk_color = "🟡 MEDIUM RISK", "#FFA500"
            advice = "Monitor closely. Consider proactive outreach within 30 days."
        else:
            risk_label, risk_color = "🟢 LOW RISK", "#00C851"
            advice = "Customer appears stable. Focus on upsell opportunities."

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Probability %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 40], "color": "#1a3a2a"},
                        {"range": [40, 70], "color": "#3a3000"},
                        {"range": [70, 100], "color": "#3a0000"}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": prob * 100
                    }
                }
            ))
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col_r2:
            st.markdown(f"## {risk_label}")
            st.markdown(f"**Churn Probability: {prob:.1%}**")
            st.markdown(f"**💡 Recommendation:** {advice}")
            st.markdown(f"**Estimated Annual Value at Risk:** ₹{monthly_charges * 12:,.0f}")

            if contract == "Month-to-month":
                st.warning("⚠️ Month-to-month contract is a top churn predictor. Consider offering annual plan.")
            if online_security == "No":
                st.info("🔒 No online security — bundle discount could improve retention.")
            if internet_service == "Fiber optic":
                st.warning("🌐 Fiber optic churn is elevated. Ensure SLA is met.")