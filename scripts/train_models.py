import json
import pandas as pd
import numpy as np

from retention_ltv.config import ARTIFACTS, RAW_CSV, SURVIVAL_HORIZON_MONTHS
from retention_ltv.features import prepare_customers, model_columns
from retention_ltv.modeling import evaluate_models, save_model, shap_outputs, train_deployment_model
from retention_ltv.survival import build_survival_curves, ltv_by_segment

ARTIFACTS.mkdir(parents=True, exist_ok=True)
df = prepare_customers(pd.read_csv(RAW_CSV))
results, _, _ = evaluate_models(df)

pipe, categorical = train_deployment_model(df)
X = df[model_columns(df)]
df["churn_probability"] = pipe.predict_proba(X)[:, 1]
q25, q75 = df["churn_probability"].quantile([0.25, 0.75])
df["risk_segment"] = np.select(
    [df["churn_probability"] <= q25, df["churn_probability"] >= q75],
    ["Healthy", "At-risk"],
    default="Middle",
)

global_shap, transformed_shap, explanations = shap_outputs(pipe, X, categorical)
df["shap_explanation"] = explanations

survival = build_survival_curves(df, horizon=SURVIVAL_HORIZON_MONTHS)
ltv_risk = ltv_by_segment(df, "risk_segment", SURVIVAL_HORIZON_MONTHS)
ltv_risk["segment_dimension"] = "risk_segment"
ltv_contract = ltv_by_segment(df, "Contract", SURVIVAL_HORIZON_MONTHS)
ltv_contract["segment_dimension"] = "Contract"
ltv = pd.concat([ltv_risk, ltv_contract], ignore_index=True)

scores_cols = [
    "customerID", "Contract", "tenure", "tenure_bucket", "PaymentMethod",
    "InternetService", "MonthlyCharges", "service_count", "churn", "churn_probability",
    "risk_segment", "shap_explanation"
]
df[scores_cols].sort_values("churn_probability", ascending=False).to_csv(ARTIFACTS / "customer_scores.csv", index=False)
global_shap.to_csv(ARTIFACTS / "shap_global.csv", index=False)
transformed_shap.to_csv(ARTIFACTS / "shap_transformed.csv", index=False)
survival.to_csv(ARTIFACTS / "survival_curves.csv", index=False)
ltv.to_csv(ARTIFACTS / "ltv_by_segment.csv", index=False)
save_model(pipe, ARTIFACTS / "churn_xgboost.joblib")

# Business summary metrics
contract = df.groupby("Contract")["churn"].agg(["size", "mean"]).reset_index()
payment = df.groupby("PaymentMethod")["churn"].agg(["size", "mean"]).reset_index().sort_values("mean", ascending=False)
tenure = df.groupby("tenure_bucket", observed=True)["churn"].agg(["size", "mean"]).reset_index()
healthy = ltv_risk.loc[ltv_risk["segment"] == "Healthy", "estimated_ltv_72m"].iloc[0]
at_risk = ltv_risk.loc[ltv_risk["segment"] == "At-risk", "estimated_ltv_72m"].iloc[0]
ltv_gap = (healthy - at_risk) / healthy

summary = {
    "rows": int(len(df)),
    "observed_churn_rate": float(df["churn"].mean()),
    "holdout_metrics": results,
    "strongest_shap_features": global_shap.head(8).to_dict(orient="records"),
    "month_to_month_churn_rate": float(contract.loc[contract["Contract"] == "Month-to-month", "mean"].iloc[0]),
    "two_year_churn_rate": float(contract.loc[contract["Contract"] == "Two year", "mean"].iloc[0]),
    "electronic_check_churn_rate": float(payment.loc[payment["PaymentMethod"] == "Electronic check", "mean"].iloc[0]),
    "new_customer_0_6_churn_rate": float(tenure.loc[tenure["tenure_bucket"] == "0-6", "mean"].iloc[0]),
    "mature_49_72_churn_rate": float(tenure.loc[tenure["tenure_bucket"] == "49-72", "mean"].iloc[0]),
    "healthy_ltv_72m": float(healthy),
    "at_risk_ltv_72m": float(at_risk),
    "at_risk_ltv_discount_vs_healthy": float(ltv_gap),
}
(ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
