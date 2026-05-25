from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _categorical_drift_score(
    baseline_distribution: dict[str, float],
    current_distribution: dict[str, float],
) -> float:
    keys = set(baseline_distribution) | set(current_distribution)
    if not keys:
        return 0.0
    total_variation = 0.5 * sum(
        abs(float(baseline_distribution.get(k, 0.0)) - float(current_distribution.get(k, 0.0)))
        for k in keys
    )
    return float(total_variation)


def evaluate_drift(
    baseline_signature: dict[str, dict[str, Any]],
    current_features: pd.DataFrame,
    alert_threshold: float = 0.35,
) -> dict[str, Any]:
    per_feature: list[dict[str, Any]] = []

    for col, baseline in baseline_signature.items():
        if col not in current_features.columns:
            per_feature.append(
                {
                    "feature": col,
                    "feature_type": baseline.get("type", "unknown"),
                    "drift_score": 1.0,
                    "status": "missing_in_current_data",
                }
            )
            continue

        series = current_features[col]
        if baseline.get("type") == "numeric":
            mean_curr = float(series.mean(skipna=True)) if not series.dropna().empty else 0.0
            std_base = float(baseline.get("std", 0.0))
            mean_base = float(baseline.get("mean", 0.0))
            shift = abs(mean_curr - mean_base) / max(std_base, 1e-6)
            score = float(min(1.0, shift / 3.0))
            per_feature.append(
                {
                    "feature": col,
                    "feature_type": "numeric",
                    "drift_score": score,
                    "baseline_mean": mean_base,
                    "current_mean": mean_curr,
                    "status": "drifted" if score >= alert_threshold else "stable",
                }
            )
        else:
            curr_dist = (
                series.fillna("MISSING")
                .astype(str)
                .value_counts(normalize=True)
                .head(25)
                .to_dict()
            )
            score = _categorical_drift_score(
                baseline.get("top_distribution", {}),
                curr_dist,
            )
            per_feature.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "drift_score": float(score),
                    "status": "drifted" if score >= alert_threshold else "stable",
                }
            )

    scores = [item["drift_score"] for item in per_feature]
    overall_score = float(np.mean(scores)) if scores else 0.0
    drifted = [item for item in per_feature if item["status"] in {"drifted", "missing_in_current_data"}]

    return {
        "overall_drift_score": overall_score,
        "alert_threshold": float(alert_threshold),
        "is_alert": overall_score >= alert_threshold,
        "drifted_feature_count": len(drifted),
        "total_features_checked": len(per_feature),
        "top_drifted_features": sorted(
            drifted,
            key=lambda x: float(x["drift_score"]),
            reverse=True,
        )[:25],
        "per_feature": per_feature,
    }

