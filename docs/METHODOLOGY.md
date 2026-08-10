# Methodology notes

## What the dataset can and cannot support

The IBM Telco Customer Churn sample is a **single customer-level snapshot** with tenure, current plan/service attributes, charges, and whether the customer churned. It is not a longitudinal billing-event table. Therefore this project does **not** invent calendar signup dates or claim true acquisition-month retention cohorts.

Instead, retention is modeled as a censoring-aware survival problem. Churned customers are treated as events at observed tenure; active customers are right-censored at observed tenure. Kaplan–Meier curves are calculated for meaningful business cohorts such as contract type, internet service, and model risk segment.

If the project is later connected to monthly subscription events, the same DuckDB layer can be extended to a true `signup_month × month_number` cohort matrix.

## LTV definition

For a segment, the project estimates restricted expected tenure as the area under the Kaplan–Meier survival curve through month 72. A deliberately simple revenue LTV is then:

`estimated_ltv_72m = average_monthly_charge × restricted_expected_tenure_months`

This is a gross-revenue estimate, not contribution margin, and it excludes discount rate, CAC, upsell, and servicing costs.

## Modeling

A stratified 80/20 split with random state 42 is used for holdout evaluation. Logistic regression is the interpretable baseline. XGBoost is the nonlinear comparison model. The final deployment model is refit on all available rows after holdout evaluation; SHAP values from that refit power the dashboard explanations.

## Causal language

All findings are observational. For example, the project says electronic-check customers *show higher churn*, not that electronic checks *cause churn*.
