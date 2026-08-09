from __future__ import annotations

import numpy as np
import pandas as pd


def kaplan_meier(durations, events, horizon: int = 72) -> pd.DataFrame:
    """Kaplan-Meier estimator with integer monthly durations.

    Active customers are right-censored at observed tenure; churned customers
    are treated as events at their observed tenure month.
    """
    durations = np.asarray(durations, dtype=int)
    events = np.asarray(events, dtype=int)
    rows = [{"month": 0, "at_risk": int(len(durations)), "events": 0, "survival": 1.0}]
    survival = 1.0
    for month in range(1, horizon + 1):
        at_risk = int((durations >= month).sum())
        n_events = int(((durations == month) & (events == 1)).sum())
        if at_risk > 0:
            survival *= (1.0 - n_events / at_risk)
        rows.append({
            "month": month,
            "at_risk": at_risk,
            "events": n_events,
            "survival": float(survival),
        })
    return pd.DataFrame(rows)


def restricted_mean_survival_time(curve: pd.DataFrame, horizon: int = 72) -> float:
    """Area under the monthly survival curve up to a fixed horizon."""
    values = curve.loc[curve["month"] < horizon, "survival"].to_numpy(float)
    return float(values.sum())


def build_survival_curves(df: pd.DataFrame, dimensions=("Contract", "InternetService", "risk_segment"), horizon=72):
    rows = []
    for dimension in dimensions:
        if dimension not in df.columns:
            continue
        for value, group in df.groupby(dimension, dropna=False):
            curve = kaplan_meier(group["tenure"], group["churn"], horizon)
            curve["cohort_dimension"] = dimension
            curve["cohort_value"] = str(value)
            curve["cohort_size"] = len(group)
            rows.append(curve)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ltv_by_segment(df: pd.DataFrame, segment_col: str, horizon: int = 72) -> pd.DataFrame:
    rows = []
    for segment, group in df.groupby(segment_col, dropna=False):
        curve = kaplan_meier(group["tenure"], group["churn"], horizon)
        expected_tenure = restricted_mean_survival_time(curve, horizon)
        monthly_charge = float(group["MonthlyCharges"].mean())
        rows.append({
            "segment": str(segment),
            "customers": int(len(group)),
            "observed_churn_rate": float(group["churn"].mean()),
            "avg_monthly_charge": monthly_charge,
            "expected_tenure_months_rmst": expected_tenure,
            "estimated_ltv_72m": monthly_charge * expected_tenure,
            "survival_12m": float(curve.loc[curve["month"] == 12, "survival"].iloc[0]),
            "survival_24m": float(curve.loc[curve["month"] == 24, "survival"].iloc[0]),
            "survival_48m": float(curve.loc[curve["month"] == 48, "survival"].iloc[0]),
        })
    return pd.DataFrame(rows)
