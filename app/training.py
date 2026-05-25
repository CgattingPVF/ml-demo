from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import ConvergenceWarning


@dataclass
class TrainConfig:
    task_type: str
    test_size: float = 0.2
    random_state: int = 42


def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _candidate_models(task_type: str, random_state: int) -> dict[str, BaseEstimator]:
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
        }

    return {
        "elastic_net": ElasticNet(random_state=random_state),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=random_state),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            random_state=random_state
        ),
    }


def _classification_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_weighted": float(
            precision_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
    }
    if probabilities is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        except Exception:
            pass
    return metrics


def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _primary_metric(task_type: str) -> str:
    return "f1_weighted" if task_type == "classification" else "r2"


def _try_feature_importance(model_pipeline: Pipeline, top_n: int = 25) -> list[dict[str, Any]]:
    model = model_pipeline.named_steps["model"]
    preprocessor = model_pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return []

    importances: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return []

    ranked = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_n]

    return [{"feature": str(name), "importance": float(score)} for name, score in ranked]


def train_and_select_best(
    x: pd.DataFrame,
    y: pd.Series,
    config: TrainConfig,
) -> dict[str, Any]:
    stratify = y if config.task_type == "classification" and y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    preprocessor = _build_preprocessor(x)
    models = _candidate_models(config.task_type, config.random_state)
    primary = _primary_metric(config.task_type)

    results: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(x_train, y_train)
            preds = pipeline.predict(x_test)

        probs = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
        if config.task_type == "classification":
            metrics = _classification_metrics(y_test, preds, probs)
        else:
            metrics = _regression_metrics(y_test, preds)

        result = {
            "model_name": model_name,
            "metrics": metrics,
            "primary_metric_name": primary,
            "primary_metric_value": float(metrics.get(primary, -np.inf)),
        }
        results.append(result)

        if best_payload is None or result["primary_metric_value"] > best_payload["result"]["primary_metric_value"]:
            best_payload = {
                "result": result,
                "pipeline": pipeline,
                "feature_importance": _try_feature_importance(pipeline),
            }

    if best_payload is None:
        raise RuntimeError("Model training failed. No candidates completed.")

    return {
        "results": results,
        "best_model_name": best_payload["result"]["model_name"],
        "best_metrics": best_payload["result"]["metrics"],
        "best_primary_metric_name": best_payload["result"]["primary_metric_name"],
        "best_primary_metric_value": best_payload["result"]["primary_metric_value"],
        "best_pipeline": best_payload["pipeline"],
        "feature_importance": best_payload["feature_importance"],
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
