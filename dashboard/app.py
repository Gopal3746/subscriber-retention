from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

st.set_page_config(page_title="Subscription Retention & LTV", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    return {
        "scores": pd.read_csv(ART / "customer_scores.csv"),
        "shap": pd.read_csv(ART / "shap_global.csv"),
        "survival": pd.read_csv(ART / "survival_curves.csv"),
        "ltv": pd.read_csv(ART / "ltv_by_segment.csv"),
        "metrics": json.loads((ART / "metrics.json").read_text()),
    }

data = load_data()
scores, shap_df, survival, ltv, metrics = data["scores"], data["shap"], data["survival"], data["ltv"], data["metrics"]

st.title("Subscription Retention & LTV Analysis")
st.caption("Retention economics for 7,043 telecom subscribers: churn risk, survival-adjusted LTV, and explainable customer prioritization.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{metrics['rows']:,}")
c2.metric("Observed churn", f"{metrics['observed_churn_rate']:.1%}")
c3.metric("XGBoost ROC-AUC", f"{metrics['holdout_metrics']['xgboost']['roc_auc']:.3f}")
c4.metric("At-risk LTV gap", f"{metrics['at_risk_ltv_discount_vs_healthy']:.0%} lower")

st.subheader("Retention curves")
st.info("The source is a single customer snapshot, so the app uses Kaplan–Meier survival cohorts by customer segment rather than fabricating signup-month history.")
dimension = st.selectbox("Cohort dimension", sorted(survival["cohort_dimension"].unique()), index=0)
view = survival[survival["cohort_dimension"] == dimension].copy()
fig = px.line(view, x="month", y="survival", color="cohort_value", labels={"survival":"Retention probability", "month":"Month N", "cohort_value":dimension})
fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
st.plotly_chart(fig, use_container_width=True)

heat = view.pivot_table(index="cohort_value", columns="month", values="survival", aggfunc="first")
# Thin columns for readable heatmap while preserving early-month detail.
cols = [c for c in heat.columns if c <= 12 or c % 6 == 0]
fig_h = px.imshow(heat[cols], aspect="auto", zmin=0, zmax=1, labels=dict(x="Month N", y=dimension, color="Retention"))
st.plotly_chart(fig_h, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Global churn drivers")
    top_n = st.slider("Features to show", 5, 15, 8)
    imp = shap_df.head(top_n).sort_values("mean_abs_shap")
    fig_s = px.bar(imp, x="mean_abs_shap", y="source_feature", orientation="h", labels={"mean_abs_shap":"Mean |SHAP|", "source_feature":"Feature"})
    st.plotly_chart(fig_s, use_container_width=True)
with right:
    st.subheader("Survival-adjusted LTV")
    risk_ltv = ltv[ltv["segment_dimension"] == "risk_segment"].copy()
    fig_l = px.bar(risk_ltv, x="segment", y="estimated_ltv_72m", text_auto="$.3s", labels={"estimated_ltv_72m":"Estimated LTV ($)", "segment":"Risk segment"})
    st.plotly_chart(fig_l, use_container_width=True)

st.subheader("Top at-risk customers")
contract_filter = st.multiselect("Contract", sorted(scores["Contract"].unique()), default=sorted(scores["Contract"].unique()))
risk = scores[scores["Contract"].isin(contract_filter)].sort_values("churn_probability", ascending=False).head(20).copy()
risk["churn_probability"] = risk["churn_probability"].map(lambda x: f"{x:.1%}")
st.dataframe(risk[["customerID", "Contract", "tenure", "MonthlyCharges", "service_count", "churn_probability", "shap_explanation"]], use_container_width=True, hide_index=True)

st.subheader("Stakeholder findings")
st.markdown(f"""
**Contracts are the clearest retention lever.** Month-to-month customers churn at **{metrics['month_to_month_churn_rate']:.1%}**, versus **{metrics['two_year_churn_rate']:.1%}** for two-year contracts. SHAP also ranks contract type as the strongest global model signal, so retention programs should prioritize migration paths from month-to-month plans before adding broad discounts.

**Early-tenure customers need intervention sooner.** Customers in months 0–6 show **{metrics['new_customer_0_6_churn_rate']:.1%}** observed churn, versus **{metrics['mature_49_72_churn_rate']:.1%}** for customers with 49–72 months of tenure. That suggests onboarding, support activation, and first-renewal experiences deserve more attention than blanket lifecycle messaging.

**Payment behavior identifies a concentrated risk pool.** Electronic-check customers show **{metrics['electronic_check_churn_rate']:.1%}** observed churn. This is not proof that the payment method causes churn, but it is a practical segmentation signal for targeted research and retention experiments.

**Risk translates into economics, not just classification.** Under a 72-month restricted Kaplan–Meier horizon, the model-defined at-risk quartile has estimated LTV of **${metrics['at_risk_ltv_72m']:,.0f}**, roughly **{metrics['at_risk_ltv_discount_vs_healthy']:.0%} lower** than the low-risk quartile (**${metrics['healthy_ltv_72m']:,.0f}**). This converts churn probabilities into a prioritization metric stakeholders can use.
""")
