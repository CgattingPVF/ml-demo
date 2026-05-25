from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd


DATE_HINTS = ("date", "time", "created", "updated", "due", "completed")


def _looks_like_datetime(series: pd.Series, sample_size: int = 800) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.astype(str).head(sample_size)
    if sample.str.contains(r"\d", regex=True).mean() < 0.7:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True, utc=False)
    success_ratio = parsed.notna().mean()
    return success_ratio >= 0.7


def expand_datetime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    expanded_columns: list[str] = []

    for col in out.columns:
        lowered = str(col).lower()
        series = out[col]

        if not (
            any(token in lowered for token in DATE_HINTS)
            or pd.api.types.is_datetime64_any_dtype(series)
            or _looks_like_datetime(series)
        ):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)
        if parsed.notna().mean() < 0.5:
            continue

        out[f"{col}_year"] = parsed.dt.year
        out[f"{col}_month"] = parsed.dt.month
        out[f"{col}_day"] = parsed.dt.day
        out[f"{col}_weekday"] = parsed.dt.weekday
        expanded_columns.extend(
            [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday"]
        )
        out = out.drop(columns=[col])

    return out, expanded_columns


def drop_leaky_and_sparse_columns(
    df: pd.DataFrame,
    target_column: str,
    sparse_threshold: float = 0.98,
    high_cardinality_threshold: int = 5000,
) -> tuple[pd.DataFrame, list[str]]:
    x = df.drop(columns=[target_column]).copy()
    dropped: list[str] = []

    for col in list(x.columns):
        missing_ratio = x[col].isna().mean()
        if missing_ratio >= sparse_threshold:
            dropped.append(col)
            x = x.drop(columns=[col])
            continue

        if pd.api.types.is_object_dtype(x[col]) and x[col].nunique(dropna=True) >= high_cardinality_threshold:
            dropped.append(col)
            x = x.drop(columns=[col])

    return x, dropped


def infer_task_type(y: pd.Series, forced: str | None = None) -> str:
    if forced in {"classification", "regression"}:
        return forced

    if pd.api.types.is_numeric_dtype(y):
        unique = y.nunique(dropna=True)
        if unique <= 20:
            return "classification"
        return "regression"

    return "classification"


def sanitize_training_frame(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    working = df.copy()
    working = working.loc[:, ~working.columns.duplicated()]
    working = working.dropna(subset=[target_column])

    expanded, expanded_cols = expand_datetime_features(working)
    features_df, dropped_cols = drop_leaky_and_sparse_columns(expanded, target_column)
    target = expanded[target_column].copy()

    info = {
        "dropped_columns": dropped_cols,
        "expanded_datetime_columns": expanded_cols,
        "feature_columns": list(features_df.columns),
    }
    return features_df, target, info


def baseline_signature(features_df: pd.DataFrame, max_categories: int = 25) -> dict:
    signature: dict[str, dict] = {}
    for col in features_df.columns:
        series = features_df[col]
        if pd.api.types.is_numeric_dtype(series):
            signature[col] = {
                "type": "numeric",
                "mean": float(series.mean(skipna=True)) if not series.dropna().empty else 0.0,
                "std": float(series.std(skipna=True)) if not series.dropna().empty else 0.0,
                "q1": float(series.quantile(0.25)) if not series.dropna().empty else 0.0,
                "q3": float(series.quantile(0.75)) if not series.dropna().empty else 0.0,
            }
        else:
            counts = (
                series.fillna("MISSING")
                .astype(str)
                .value_counts(normalize=True)
                .head(max_categories)
                .to_dict()
            )
            signature[col] = {
                "type": "categorical",
                "top_distribution": {str(k): float(v) for k, v in counts.items()},
            }
    return signature


def align_input_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in required_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[list(required_columns)]
