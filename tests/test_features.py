import pandas as pd
from retention_ltv.features import prepare_customers


def test_feature_engineering_counts_services():
    df = pd.DataFrame({
        "TotalCharges": ["100.0"], "Churn": ["Yes"], "tenure": [5],
        "PaymentMethod": ["Credit card (automatic)"], "MonthlyCharges": [50.0],
        "PhoneService": ["Yes"], "MultipleLines": ["No"], "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"], "DeviceProtection": ["Yes"], "TechSupport": ["No"],
        "StreamingTV": ["No"], "StreamingMovies": ["Yes"],
    })
    out = prepare_customers(df)
    assert out.loc[0, "service_count"] == 4
    assert out.loc[0, "churn"] == 1
    assert out.loc[0, "tenure_bucket"] == "0-6"
    assert out.loc[0, "automatic_payment"] == 1
