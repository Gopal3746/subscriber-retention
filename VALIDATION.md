# Validation

## Reproduced analytics

- Rows: 7,043
- Observed churn: 26.537%
- Logistic regression ROC-AUC: 0.844744
- XGBoost ROC-AUC: 0.848161
- Month-to-month churn: 42.710%
- Two-year churn: 2.832%
- At-risk 72-month LTV: $1,363.87
- Healthy 72-month LTV: $3,292.21
- At-risk LTV discount vs healthy: 58.573%

## Tests

`PYTHONPATH=src pytest -q` -> **2 passed** in the build environment.

Model training, SHAP generation, survival/LTV calculation, CSV artifact generation, and the static dashboard preview were executed successfully.
