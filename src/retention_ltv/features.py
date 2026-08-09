from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def prepare_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the IBM Telco snapshot and add model/business features."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["churn"] = (out["Churn"] == "Yes").astype(int)
    out["service_count"] = out[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    out["automatic_payment"] = out["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    out["monthly_charge_band"] = pd.cut(
        out["MonthlyCharges"],
        bins=[-np.inf, 35, 70, 90, np.inf],
        labels=["<=35", "35-70", "70-90", ">90"],
    ).astype(str)
    return out


def model_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"customerID", "Churn", "churn"}
    return [c for c in df.columns if c not in excluded]


def source_feature_from_transformed(name: str, categorical_columns: list[str]) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    if name.startswith("cat__"):
        raw = name.replace("cat__", "", 1)
        for col in sorted(categorical_columns, key=len, reverse=True):
            if raw == col or raw.startswith(col + "_"):
                return col
    return name


def humanize_transformed_feature(name: str, categorical_columns: list[str]) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    raw = name.replace("cat__", "", 1)
    for col in sorted(categorical_columns, key=len, reverse=True):
        prefix = col + "_"
        if raw.startswith(prefix):
            return f"{col}={raw[len(prefix):]}"
    return raw
