from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE, TEST_SIZE
from .features import humanize_transformed_feature, model_columns, source_feature_from_transformed


def build_preprocessor(X: pd.DataFrame):
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return preprocessor, categorical


def estimators():
    return {
        "logistic_regression": LogisticRegression(max_iter=2500, class_weight="balanced", C=1.0),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.025,
            min_child_weight=5,
            reg_lambda=4,
            subsample=0.80,
            colsample_bytree=0.90,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }


def score_binary(y_true, probability, threshold=0.5):
    pred = (np.asarray(probability) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def evaluate_models(df: pd.DataFrame):
    cols = model_columns(df)
    X = df[cols]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    results = {}
    fitted = {}
    for name, estimator in estimators().items():
        prep, _ = build_preprocessor(X_train)
        pipe = Pipeline([("preprocessor", prep), ("model", estimator)])
        pipe.fit(X_train, y_train)
        probability = pipe.predict_proba(X_test)[:, 1]
        results[name] = score_binary(y_test, probability)
        fitted[name] = pipe
    return results, fitted, (X_train, X_test, y_train, y_test)


def train_deployment_model(df: pd.DataFrame):
    X = df[model_columns(df)]
    y = df["churn"]
    prep, categorical = build_preprocessor(X)
    model = estimators()["xgboost"]
    pipe = Pipeline([("preprocessor", prep), ("model", model)])
    pipe.fit(X, y)
    return pipe, categorical


def shap_outputs(pipe: Pipeline, X: pd.DataFrame, categorical_columns: list[str]):
    prep = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    transformed = prep.transform(X)
    feature_names = prep.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(transformed))

    mean_abs = np.abs(shap_values).mean(axis=0)
    transformed_importance = pd.DataFrame({
        "transformed_feature": feature_names,
        "mean_abs_shap": mean_abs,
    })
    transformed_importance["source_feature"] = transformed_importance["transformed_feature"].map(
        lambda x: source_feature_from_transformed(x, categorical_columns)
    )
    global_importance = (
        transformed_importance.groupby("source_feature", as_index=False)["mean_abs_shap"]
        .sum().sort_values("mean_abs_shap", ascending=False)
    )

    top_explanations = []
    for row in shap_values:
        idx = np.argsort(np.abs(row))[::-1][:3]
        parts = []
        for i in idx:
            label = humanize_transformed_feature(str(feature_names[i]), categorical_columns)
            direction = "+risk" if row[i] > 0 else "-risk"
            parts.append(f"{label} ({direction}, SHAP {row[i]:+.2f})")
        top_explanations.append("; ".join(parts))
    return global_importance, transformed_importance, top_explanations


def save_model(pipe: Pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
