from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from app.data_loader import dataset_profile, load_dataset, normalize_columns
from app.features import (
    align_input_columns,
    baseline_signature,
    expand_datetime_features,
    infer_task_type,
    sanitize_training_frame,
)
from app.monitoring import evaluate_drift
from app.registry import load_registered_model, register_experiment
from app.training import TrainConfig, train_and_select_best


ACTIVE_STATUS_TOKENS = ("booked", "planned", "awaiting", "rts", "request", "sent")
CANCELLED_STATUS_TOKENS = ("cancel", "postponed")
COMPLETED_STATUS_TOKENS = ("down", "completed", "complete", "closed", "handed over")


@lru_cache(maxsize=2)
def _cached_normalized_dataset(path_str: str) -> pd.DataFrame:
    return normalize_columns(load_dataset(path_str))


def _load_normalized_dataset(path: str | Path) -> pd.DataFrame:
    resolved = str(Path(path).expanduser().resolve())
    return _cached_normalized_dataset(resolved).copy()


def _resolve_target(normalized_df: pd.DataFrame, target_column: str) -> str:
    normalized_target = target_column.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized_target in normalized_df.columns:
        return normalized_target
    if target_column in normalized_df.columns:
        return target_column
    raise ValueError(f"Target column '{target_column}' not found.")


def list_target_candidates(df: pd.DataFrame, max_items: int = 150) -> list[str]:
    candidates = []
    for col in df.columns:
        unique = df[col].nunique(dropna=True)
        if unique < 2:
            continue
        if df[col].isna().mean() > 0.7:
            continue
        candidates.append(col)
    return candidates[:max_items]


def profile_data(path: str | Path) -> dict[str, Any]:
    df = _load_normalized_dataset(path)
    profile = dataset_profile(df)
    profile["target_candidates"] = list_target_candidates(df)
    profile["columns"] = list(df.columns)
    return profile


def train_model(
    *,
    dataset_path: str | Path,
    target_column: str,
    task_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    df = _load_normalized_dataset(dataset_path)
    resolved_target = _resolve_target(df, target_column)

    x, y, prep_info = sanitize_training_frame(df, resolved_target)
    chosen_task = infer_task_type(y, forced=task_type)

    # Convert target into numeric when task is regression but values are strings.
    if chosen_task == "regression" and not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors="coerce")
        mask = y.notna()
        x = x.loc[mask]
        y = y.loc[mask]

    result = train_and_select_best(
        x=x,
        y=y,
        config=TrainConfig(task_type=chosen_task, test_size=test_size, random_state=random_state),
    )

    metadata = {
        "training_summary": {
            "results": result["results"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "task_type": chosen_task,
        },
        "feature_importance": result["feature_importance"],
        "feature_columns": prep_info["feature_columns"],
        "prep_info": prep_info,
        "baseline_signature": baseline_signature(x[prep_info["feature_columns"]]),
    }

    registry_record = register_experiment(
        dataset_path=str(Path(dataset_path).expanduser().resolve()),
        target_column=resolved_target,
        task_type=chosen_task,
        total_rows=int(df.shape[0]),
        total_columns=int(df.shape[1]),
        best_model_name=result["best_model_name"],
        metric_name=result["best_primary_metric_name"],
        metric_value=result["best_primary_metric_value"],
        metadata=metadata,
        fitted_pipeline=result["best_pipeline"],
    )

    return {
        "experiment": registry_record,
        "best_model_name": result["best_model_name"],
        "best_metrics": result["best_metrics"],
        "all_results": result["results"],
        "feature_importance": result["feature_importance"],
    }


def _prepare_inference_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    normalized = normalize_columns(df)
    expanded, _ = expand_datetime_features(normalized)
    return align_input_columns(expanded, feature_columns)


def predict_records(experiment_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    pipeline, experiment = load_registered_model(experiment_id)
    metadata = experiment["metadata"]
    feature_columns = metadata["feature_columns"]

    input_df = pd.DataFrame(records)
    inference_df = _prepare_inference_frame(input_df, feature_columns)

    predictions = pipeline.predict(inference_df)
    response: dict[str, Any] = {
        "experiment_id": experiment_id,
        "rows": len(inference_df),
        "predictions": [value.item() if isinstance(value, np.generic) else value for value in predictions],
    }

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(inference_df)
        response["confidence"] = [float(np.max(row)) for row in probs]

    return response


def batch_predict(experiment_id: str, dataset_path: str | Path) -> pd.DataFrame:
    pipeline, experiment = load_registered_model(experiment_id)
    metadata = experiment["metadata"]
    feature_columns = metadata["feature_columns"]

    raw = load_dataset(dataset_path)
    prepared = _prepare_inference_frame(raw, feature_columns)
    preds = pipeline.predict(prepared)

    output = raw.copy()
    output["prediction"] = preds
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(prepared)
        output["prediction_confidence"] = np.max(probs, axis=1)
    return output


def drift_check(experiment_id: str, current_dataset_path: str | Path) -> dict[str, Any]:
    _, experiment = load_registered_model(experiment_id)
    baseline = experiment["metadata"].get("baseline_signature", {})
    feature_columns = experiment["metadata"]["feature_columns"]

    current = load_dataset(current_dataset_path)
    prepared = _prepare_inference_frame(current, feature_columns)
    return evaluate_drift(baseline_signature=baseline, current_features=prepared)


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    as_text = series.astype(str).str.replace(",", "", regex=False)
    as_text = as_text.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(as_text, errors="coerce")


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace(
        {
            "0000-00-00 00:00:00": pd.NA,
            "0000-00-00": pd.NA,
            "nan": pd.NA,
            "NaT": pd.NA,
            "None": pd.NA,
            "": pd.NA,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(cleaned, errors="coerce", dayfirst=True, utc=False)


def _parse_duration_to_hours(series: pd.Series) -> pd.Series:
    parsed = pd.to_timedelta(series, errors="coerce")
    return parsed.dt.total_seconds() / 3600.0


def _status_masks(status_series: pd.Series) -> dict[str, pd.Series]:
    status_lower = status_series.fillna("").astype(str).str.lower()
    active_mask = status_lower.apply(lambda value: any(token in value for token in ACTIVE_STATUS_TOKENS))
    cancelled_mask = status_lower.apply(
        lambda value: any(token in value for token in CANCELLED_STATUS_TOKENS)
    )
    completed_mask = status_lower.apply(
        lambda value: any(token in value for token in COMPLETED_STATUS_TOKENS)
    )
    # If a status is explicitly cancelled, do not count as active/completed.
    active_mask = active_mask & (~cancelled_mask)
    completed_mask = completed_mask & (~cancelled_mask)
    return {
        "active": active_mask,
        "cancelled": cancelled_mask,
        "completed": completed_mask,
    }


def _safe_sum(series: pd.Series) -> float:
    clean = _clean_numeric_series(series)
    return float(clean.sum(skipna=True))


def _safe_mean(series: pd.Series) -> float:
    clean = _clean_numeric_series(series)
    if clean.dropna().empty:
        return 0.0
    return float(clean.mean(skipna=True))


def _value_counts_dict(series: pd.Series, top_n: int = 10) -> dict[str, int]:
    counts = series.fillna("Unknown").astype(str).value_counts().head(top_n)
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def _split_entity_values(series: pd.Series) -> pd.Series:
    values: list[str] = []
    for raw_value in series.dropna().astype(str):
        for item in raw_value.split("|"):
            cleaned = item.strip()
            if cleaned:
                values.append(cleaned)
    return pd.Series(values, dtype="object")


def _partner_mask(df: pd.DataFrame, partner_name: str) -> pd.Series:
    target_name = partner_name.strip().lower()
    if "related_contacts" not in df.columns or not target_name:
        return pd.Series([False] * len(df), index=df.index)
    return df["related_contacts"].fillna("").astype(str).apply(
        lambda value: target_name in [part.strip().lower() for part in value.split("|") if part.strip()]
    )


def _pct(value: float) -> float:
    return float(round(max(0.0, min(100.0, float(value) * 100)), 1))


def _risk_level(percent: float, high: float, medium: float) -> str:
    if percent >= high:
        return "High"
    if percent >= medium:
        return "Medium"
    return "Low"


def _valid_date_count(series: pd.Series) -> int:
    if series.empty:
        return 0
    return int(_parse_datetime_series(series).notna().sum())


def _postcode_outward(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text.split()[0]


def _lead_time_days(job_record: dict[str, Any]) -> int | None:
    start_raw = job_record.get("date") or job_record.get("start_date")
    due_raw = job_record.get("due_date") or job_record.get("install_date")
    if not start_raw or not due_raw:
        return None

    start = _parse_datetime_series(pd.Series([start_raw])).iloc[0]
    due = _parse_datetime_series(pd.Series([due_raw])).iloc[0]
    if pd.isna(start) or pd.isna(due):
        return None
    return int((due - start).days)


def _job_price(job_record: dict[str, Any]) -> float | None:
    for key in ("budget_revenue", "job_price", "price", "quoted_price"):
        if key in job_record and job_record[key] not in {None, ""}:
            value = _clean_numeric_series(pd.Series([job_record[key]])).iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def _non_empty_text_mask(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    mask = pd.Series([False] * len(frame), index=frame.index)
    for column in columns:
        if column in frame.columns:
            mask = mask | frame[column].fillna("").astype(str).str.strip().ne("")
    return mask


def _team_event_counts(frame: pd.DataFrame) -> tuple[int, int]:
    hse_mask = _non_empty_text_mask(frame, ("c_hscategory",))
    if "c_hsincidentdate" in frame.columns:
        hse_mask = hse_mask | _parse_datetime_series(frame["c_hsincidentdate"]).notna()
    if "c_riddorreportable" in frame.columns:
        hse_mask = hse_mask | (_clean_numeric_series(frame["c_riddorreportable"]) > 0).fillna(False)

    damage_mask = _non_empty_text_mask(
        frame,
        ("c_damagestatus", "c_damagecategory", "c_damagedescription"),
    )
    return int(hse_mask.sum()), int(damage_mask.sum())


def _team_leaderboard(
    df: pd.DataFrame,
    *,
    selected_scaffolder: str,
    job_type: str,
    outward: str,
    price: float | None,
    limit: int = 8,
) -> dict[str, Any]:
    if "related_contacts" not in df.columns:
        return {"summary": {}, "teams": []}

    team_counts = _split_entity_values(df["related_contacts"]).value_counts()
    if team_counts.empty:
        return {"summary": {}, "teams": []}

    selected_lower = selected_scaffolder.strip().lower()
    candidate_names = [str(name) for name in team_counts.head(120).index.tolist()]
    selected_name = next(
        (str(name) for name in team_counts.index.tolist() if str(name).strip().lower() == selected_lower),
        selected_scaffolder,
    )
    if selected_name and selected_lower not in {name.strip().lower() for name in candidate_names}:
        candidate_names.append(selected_name)

    rows: list[dict[str, Any]] = []
    job_type_lower = job_type.strip().lower()
    minimum_jobs = 8

    for team in candidate_names:
        team_df = df[_partner_mask(df, team)].copy()
        total_jobs = int(len(team_df))
        if total_jobs < minimum_jobs and team.strip().lower() != selected_lower:
            continue

        status_masks = (
            _status_masks(team_df["status"])
            if "status" in team_df.columns
            else {
                "cancelled": pd.Series([False] * total_jobs, index=team_df.index),
                "completed": pd.Series([False] * total_jobs, index=team_df.index),
            }
        )
        completed_jobs = int(status_masks["completed"].sum())
        cancelled_jobs = int(status_masks["cancelled"].sum())
        hse_events, damage_events = _team_event_counts(team_df)

        similar_mask = pd.Series([True] * total_jobs, index=team_df.index)
        job_type_matches = 0
        if job_type_lower and "c_scaffoldpurpose" in team_df.columns:
            purpose_mask = team_df["c_scaffoldpurpose"].fillna("").astype(str).str.lower().eq(job_type_lower)
            job_type_matches = int(purpose_mask.sum())
            if job_type_matches >= 3:
                similar_mask = similar_mask & purpose_mask

        postcode_matches = 0
        if outward and "c_homeownerpostcode" in team_df.columns:
            postcode_mask = (
                team_df["c_homeownerpostcode"]
                .fillna("")
                .astype(str)
                .map(_postcode_outward)
                .eq(outward)
            )
            postcode_matches = int(postcode_mask.sum())
            if postcode_matches >= 3:
                similar_mask = similar_mask & postcode_mask

        revenue = (
            _clean_numeric_series(team_df["budget_revenue"])
            if "budget_revenue" in team_df.columns
            else pd.Series(dtype=float)
        )
        median_price = float(revenue.median()) if not revenue.dropna().empty else None
        price_delta_pct = None
        price_fit = 0.62
        if price is not None and median_price and median_price > 0:
            price_delta_pct = ((price - median_price) / median_price) * 100
            price_fit = max(0.0, 1.0 - min(abs(price_delta_pct) / 45.0, 1.0))
            price_mask = revenue.between(price * 0.75, price * 1.25)
            if int(price_mask.sum()) >= 3:
                similar_mask = similar_mask & price_mask.fillna(False)

        similar_jobs = int(similar_mask.sum())
        completion_rate = completed_jobs / total_jobs if total_jobs else 0.0
        cancellation_rate = cancelled_jobs / total_jobs if total_jobs else 0.0
        hse_rate = hse_events / total_jobs if total_jobs else 0.0
        damage_rate = damage_events / total_jobs if total_jobs else 0.0
        job_type_fit = min(job_type_matches / 18.0, 1.0) if job_type_lower else 0.5
        postcode_fit = min(postcode_matches / 10.0, 1.0) if outward else 0.5
        similar_fit = min(similar_jobs / 10.0, 1.0)
        safety_fit = 1.0 - min(hse_rate / 0.12, 1.0)
        damage_fit = 1.0 - min(damage_rate / 0.08, 1.0)

        fit_score = (
            completion_rate * 31.0
            + (1.0 - cancellation_rate) * 17.0
            + safety_fit * 12.0
            + damage_fit * 8.0
            + job_type_fit * 12.0
            + postcode_fit * 7.0
            + similar_fit * 8.0
            + price_fit * 5.0
        )

        if similar_jobs >= 10 and postcode_matches >= 3:
            reason = "Strong match across job type, area and price band."
        elif similar_jobs >= 10 and price_delta_pct is not None:
            reason = "Strong job-type and price-band match; limited area history."
        elif similar_jobs >= 10:
            reason = "Strong job-type history; limited area evidence."
        elif job_type_matches >= 18:
            reason = "Strong job-type history; confirm area and commercial fit."
        elif completion_rate >= 0.86 and cancellation_rate <= 0.08:
            reason = "Reliable completion record with low cancellation pressure."
        elif total_jobs < 20:
            reason = "Limited history; use tighter booking and handover checks."
        else:
            reason = "Viable team with mixed matching signals."

        rows.append(
            {
                "team": team,
                "fit_score": round(float(fit_score), 1),
                "total_jobs": total_jobs,
                "similar_jobs": similar_jobs,
                "job_type_matches": job_type_matches,
                "postcode_matches": postcode_matches,
                "completion_rate_pct": _pct(completion_rate),
                "cancellation_rate_pct": _pct(cancellation_rate),
                "hse_rate_pct": _pct(hse_rate),
                "damage_rate_pct": _pct(damage_rate),
                "median_price": round(median_price, 2) if median_price is not None and not pd.isna(median_price) else None,
                "price_delta_pct": round(float(price_delta_pct), 1) if price_delta_pct is not None else None,
                "is_selected": team.strip().lower() == selected_lower,
                "reason": reason,
            }
        )

    ranked = sorted(rows, key=lambda item: item["fit_score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index

    selected_row = next((row for row in ranked if row["is_selected"]), None)
    display_rows = ranked[:limit]
    if selected_row and selected_row not in display_rows:
        display_rows = display_rows[: max(limit - 1, 0)] + [selected_row]
        display_rows = sorted(display_rows, key=lambda item: item["rank"])

    best_row = ranked[0] if ranked else None
    summary = {
        "teams_compared": len(ranked),
        "minimum_jobs": minimum_jobs,
        "best_team": best_row["team"] if best_row else None,
        "best_fit_score": best_row["fit_score"] if best_row else None,
        "selected_rank": selected_row["rank"] if selected_row else None,
        "selected_fit_score": selected_row["fit_score"] if selected_row else None,
    }
    return {"summary": summary, "teams": display_rows}


def _build_profile_card(df: pd.DataFrame, profile_type: str, name: str) -> dict[str, Any]:
    if profile_type == "client":
        key_col = "contact_name"
    elif profile_type == "partner":
        key_col = "related_contacts"
    else:
        raise ValueError("profile_type must be 'client' or 'partner'.")

    if key_col not in df.columns:
        raise ValueError(f"Column '{key_col}' is not available in this dataset.")

    if profile_type == "partner":
        entity_mask = _partner_mask(df, name)
    else:
        target_name = name.strip().lower()
        entity_mask = df[key_col].fillna("").astype(str).str.lower() == target_name

    filtered = df[entity_mask].copy()
    if filtered.empty:
        raise ValueError(f"No records found for {profile_type} '{name}'.")

    total_jobs = int(len(filtered))
    status_masks = _status_masks(filtered["status"]) if "status" in filtered.columns else {
        "active": pd.Series([False] * total_jobs, index=filtered.index),
        "cancelled": pd.Series([False] * total_jobs, index=filtered.index),
        "completed": pd.Series([False] * total_jobs, index=filtered.index),
    }
    active_jobs = int(status_masks["active"].sum())
    cancelled_jobs = int(status_masks["cancelled"].sum())
    completed_jobs = int(status_masks["completed"].sum())

    start_dates = _parse_datetime_series(filtered["date"]) if "date" in filtered.columns else pd.Series(dtype="datetime64[ns]")
    due_dates = _parse_datetime_series(filtered["due_date"]) if "due_date" in filtered.columns else pd.Series(dtype="datetime64[ns]")
    completed_dates = (
        _parse_datetime_series(filtered["completed_date"])
        if "completed_date" in filtered.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    cycle_days = (completed_dates - start_dates).dt.days

    now = pd.Timestamp.utcnow().tz_localize(None)
    open_not_completed = ~status_masks["completed"]
    overdue_open_jobs = int(((due_dates < now) & open_not_completed).sum()) if not due_dates.empty else 0

    budget_revenue = _safe_sum(filtered["budget_revenue"]) if "budget_revenue" in filtered.columns else 0.0
    actual_revenue = _safe_sum(filtered["actual_revenue"]) if "actual_revenue" in filtered.columns else 0.0
    budget_cost = _safe_sum(filtered["budget_cost"]) if "budget_cost" in filtered.columns else 0.0
    actual_cost = _safe_sum(filtered["actual_cost"]) if "actual_cost" in filtered.columns else 0.0
    budget_margin = budget_revenue - budget_cost
    actual_margin = actual_revenue - actual_cost
    revenue_variance_pct = (
        ((actual_revenue - budget_revenue) / budget_revenue) * 100 if abs(budget_revenue) > 1e-9 else 0.0
    )
    margin_variance_pct = (
        ((actual_margin - budget_margin) / abs(budget_margin)) * 100 if abs(budget_margin) > 1e-9 else 0.0
    )

    total_to_do_hours = (
        _parse_duration_to_hours(filtered["total_to_do"]).dropna()
        if "total_to_do" in filtered.columns
        else pd.Series(dtype=float)
    )
    total_done_hours = (
        _parse_duration_to_hours(filtered["total_done"]).dropna()
        if "total_done" in filtered.columns
        else pd.Series(dtype=float)
    )

    top_statuses = _value_counts_dict(filtered["status"], top_n=12) if "status" in filtered.columns else {}
    top_budget_types = _value_counts_dict(filtered["budget_type"], top_n=6) if "budget_type" in filtered.columns else {}
    top_postcodes = (
        _value_counts_dict(filtered["c_homeownerpostcode"], top_n=10)
        if "c_homeownerpostcode" in filtered.columns
        else {}
    )

    recent_columns = [col for col in ["project_no", "project_name", "status", "date", "due_date"] if col in filtered.columns]
    recent_jobs = filtered.sort_index(ascending=False).head(10)[recent_columns] if recent_columns else filtered.head(10)

    return {
        "profile_type": profile_type,
        "name": name,
        "entity_column": key_col,
        "snapshot": {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "records_analyzed": total_jobs,
            "date_range": {
                "first_job_date": start_dates.min().date().isoformat() if not start_dates.dropna().empty else None,
                "last_job_date": start_dates.max().date().isoformat() if not start_dates.dropna().empty else None,
            },
        },
        "portfolio": {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "cancelled_jobs": cancelled_jobs,
            "completion_rate_pct": round((completed_jobs / total_jobs) * 100, 2) if total_jobs else 0.0,
            "cancellation_rate_pct": round((cancelled_jobs / total_jobs) * 100, 2) if total_jobs else 0.0,
            "overdue_open_jobs": overdue_open_jobs,
        },
        "financials": {
            "budget_revenue_total": round(budget_revenue, 2),
            "actual_revenue_total": round(actual_revenue, 2),
            "budget_cost_total": round(budget_cost, 2),
            "actual_cost_total": round(actual_cost, 2),
            "budget_margin_total": round(budget_margin, 2),
            "actual_margin_total": round(actual_margin, 2),
            "revenue_variance_pct": round(revenue_variance_pct, 2),
            "margin_variance_pct": round(margin_variance_pct, 2),
        },
        "delivery": {
            "avg_cycle_time_days": round(float(cycle_days.dropna().mean()), 2) if not cycle_days.dropna().empty else None,
            "median_cycle_time_days": round(float(cycle_days.dropna().median()), 2) if not cycle_days.dropna().empty else None,
            "avg_total_to_do_hours": round(float(total_to_do_hours.mean()), 2) if not total_to_do_hours.empty else None,
            "avg_total_done_hours": round(float(total_done_hours.mean()), 2) if not total_done_hours.empty else None,
            "top_statuses": top_statuses,
            "top_budget_types": top_budget_types,
            "top_postcodes": top_postcodes,
        },
        "recent_jobs": recent_jobs.fillna("").astype(str).to_dict(orient="records"),
    }


def generate_profile_card(
    *,
    dataset_path: str | Path,
    profile_type: str,
    name: str,
) -> dict[str, Any]:
    df = _load_normalized_dataset(dataset_path)
    return _build_profile_card(df, profile_type=profile_type, name=name)


def list_profile_entities(
    *,
    dataset_path: str | Path,
    profile_type: str,
    limit: int = 200,
) -> list[str]:
    df = _load_normalized_dataset(dataset_path)
    key_col = "contact_name" if profile_type == "client" else "related_contacts"
    if key_col not in df.columns:
        return []
    if profile_type == "partner":
        entity_series = _split_entity_values(df[key_col])
    else:
        entity_series = (
            df[key_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )

    entities = entity_series.value_counts().head(limit).index.tolist()
    return [str(item) for item in entities]


def scaffolder_job_predictions(
    *,
    dataset_path: str | Path,
    job_record: dict[str, Any],
) -> dict[str, Any]:
    df = _load_normalized_dataset(dataset_path)
    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in job_record.items()}
    scaffolder = str(normalized.get("related_contacts", "")).strip()
    if not scaffolder:
        raise ValueError("A scaffold partner company is required for scaffolder-focused predictions.")

    partner_df = df[_partner_mask(df, scaffolder)].copy()
    if partner_df.empty:
        raise ValueError(f"No historical jobs found for scaffolder '{scaffolder}'.")

    job_type = str(
        normalized.get("c_scaffoldpurpose")
        or normalized.get("job_type")
        or normalized.get("scaffold_purpose")
        or ""
    ).strip()
    postcode = str(
        normalized.get("c_homeownerpostcode")
        or normalized.get("postcode")
        or normalized.get("homeowner_postcode")
        or ""
    ).strip()
    outward = _postcode_outward(postcode)
    price = _job_price(normalized)
    cost = None
    if normalized.get("budget_cost") not in {None, ""}:
        parsed_cost = _clean_numeric_series(pd.Series([normalized.get("budget_cost")])).iloc[0]
        if pd.notna(parsed_cost):
            cost = float(parsed_cost)

    similar = partner_df.copy()
    if job_type and "c_scaffoldpurpose" in similar.columns:
        job_type_mask = similar["c_scaffoldpurpose"].fillna("").astype(str).str.lower() == job_type.lower()
        if int(job_type_mask.sum()) >= 10:
            similar = similar[job_type_mask]

    if outward and "c_homeownerpostcode" in similar.columns:
        postcode_mask = (
            similar["c_homeownerpostcode"]
            .fillna("")
            .astype(str)
            .map(_postcode_outward)
            .eq(outward)
        )
        if int(postcode_mask.sum()) >= 10:
            similar = similar[postcode_mask]

    if price is not None and "budget_revenue" in similar.columns:
        revenue = _clean_numeric_series(similar["budget_revenue"])
        price_mask = revenue.between(price * 0.75, price * 1.25)
        if int(price_mask.sum()) >= 10:
            similar = similar[price_mask]

    status = partner_df["status"].fillna("").astype(str).str.lower() if "status" in partner_df.columns else pd.Series(dtype=str)
    similar_status = similar["status"].fillna("").astype(str).str.lower() if "status" in similar.columns else pd.Series(dtype=str)

    partner_total = max(int(len(partner_df)), 1)
    similar_total = max(int(len(similar)), 1)
    global_total = max(int(len(df)), 1)

    partner_completed = status.apply(lambda value: any(token in value for token in COMPLETED_STATUS_TOKENS)).sum()
    partner_cancelled = status.apply(lambda value: any(token in value for token in CANCELLED_STATUS_TOKENS)).sum()
    similar_completed = similar_status.apply(lambda value: any(token in value for token in COMPLETED_STATUS_TOKENS)).sum()
    similar_cancelled = similar_status.apply(lambda value: any(token in value for token in CANCELLED_STATUS_TOKENS)).sum()

    hse_events = 0
    global_hse_events = 0
    if "c_hscategory" in partner_df.columns:
        hse_events += int(partner_df["c_hscategory"].fillna("").astype(str).str.strip().ne("").sum())
        global_hse_events += int(df["c_hscategory"].fillna("").astype(str).str.strip().ne("").sum())
    if "c_hsincidentdate" in partner_df.columns:
        hse_events = max(hse_events, _valid_date_count(partner_df["c_hsincidentdate"]))
        global_hse_events = max(global_hse_events, _valid_date_count(df["c_hsincidentdate"]))
    if "c_riddorreportable" in partner_df.columns:
        hse_events += int((_clean_numeric_series(partner_df["c_riddorreportable"]) > 0).sum())
        global_hse_events += int((_clean_numeric_series(df["c_riddorreportable"]) > 0).sum())

    damage_events = 0
    global_damage_events = 0
    for col in ("c_damagestatus", "c_damagecategory", "c_damagedescription"):
        if col in partner_df.columns:
            damage_events = max(damage_events, int(partner_df[col].fillna("").astype(str).str.strip().ne("").sum()))
            global_damage_events = max(global_damage_events, int(df[col].fillna("").astype(str).str.strip().ne("").sum()))

    global_completed = (
        df["status"].fillna("").astype(str).str.lower().apply(
            lambda value: any(token in value for token in COMPLETED_STATUS_TOKENS)
        ).sum()
        if "status" in df.columns
        else 0
    )
    global_cancelled = (
        df["status"].fillna("").astype(str).str.lower().apply(
            lambda value: any(token in value for token in CANCELLED_STATUS_TOKENS)
        ).sum()
        if "status" in df.columns
        else 0
    )

    completion_rate = ((similar_completed + partner_completed + global_completed) / (similar_total + partner_total + global_total))
    cancellation_rate = ((similar_cancelled + partner_cancelled + global_cancelled) / (similar_total + partner_total + global_total))
    hse_rate = ((hse_events + global_hse_events) / (partner_total + global_total))
    damage_rate = ((damage_events + global_damage_events) / (partner_total + global_total))

    lead_days = _lead_time_days(normalized)
    lead_penalty = 0.0
    if lead_days is not None:
        if lead_days < 0:
            lead_penalty = 0.35
        elif lead_days <= 2:
            lead_penalty = 0.18
        elif lead_days <= 5:
            lead_penalty = 0.08

    high_risk_terms = ("roof", "chimney", "edge", "gable", "height", "flue", "render", "brickwork")
    job_type_penalty = 0.06 if any(term in job_type.lower() for term in high_risk_terms) else 0.0

    hse_percent = _pct(hse_rate + job_type_penalty + max(0.0, lead_penalty / 3))
    damage_percent = _pct(damage_rate + (0.04 if "roof" in job_type.lower() else 0.0))
    completion_percent = _pct(completion_rate - min(0.25, lead_penalty))
    cancellation_percent = _pct(cancellation_rate + min(0.22, lead_penalty))

    avg_revenue = _safe_mean(partner_df["budget_revenue"]) if "budget_revenue" in partner_df.columns else 0.0
    median_revenue = float(_clean_numeric_series(partner_df["budget_revenue"]).median()) if "budget_revenue" in partner_df.columns else 0.0
    margin_risk = 0.0
    if price is not None and median_revenue:
        if price < median_revenue * 0.75:
            margin_risk += 0.22
        elif price < median_revenue * 0.9:
            margin_risk += 0.1
    if price is not None and cost is not None and price > 0:
        margin_pct = (price - cost) / price
        if margin_pct < 0:
            margin_risk += 0.45
        elif margin_pct < 0.15:
            margin_risk += 0.25
        elif margin_pct < 0.25:
            margin_risk += 0.1
    margin_percent = _pct(margin_risk)

    programme_percent = _pct(
        cancellation_rate
        + lead_penalty
        + (0.08 if partner_total >= 500 and (partner_completed / partner_total) < 0.8 else 0.0)
    )
    failure_percent = round(
        max(
            100.0 - completion_percent,
            cancellation_percent,
            programme_percent * 0.85,
        ),
        1,
    )

    def prediction(name: str, value: float, level: str, detail: str) -> dict[str, Any]:
        return {
            "name": name,
            "score_pct": round(value, 1),
            "level": level,
            "detail": detail,
        }

    predictions = [
        prediction(
            "Failure / non-completion risk",
            failure_percent,
            _risk_level(failure_percent, high=25, medium=12),
            "Combined risk that the job does not complete cleanly, using completion, cancellation and programme signals.",
        ),
        prediction(
            "HSE / incident risk",
            hse_percent,
            _risk_level(hse_percent, high=8, medium=3),
            "Based on scaffolder incident history, RIDDOR flags, job type and short-notice pressure.",
        ),
        prediction(
            "Completion likelihood",
            completion_percent,
            "Strong" if completion_percent >= 82 else "Watch" if completion_percent >= 65 else "Weak",
            "Likelihood the job follows the historic completed/scaffold-down pattern for this scaffolder and similar jobs.",
        ),
        prediction(
            "Cancellation / postponement risk",
            cancellation_percent,
            _risk_level(cancellation_percent, high=18, medium=8),
            "Uses the scaffolder's historical cancellation/postponement pattern, adjusted for lead time.",
        ),
        prediction(
            "Programme risk",
            programme_percent,
            _risk_level(programme_percent, high=22, medium=10),
            "Highlights risk from short lead times, historic non-completion and cancellation patterns.",
        ),
        prediction(
            "Damage / claim risk",
            damage_percent,
            _risk_level(damage_percent, high=5, medium=2),
            "Based on historic damage status, category and description records for this scaffolder.",
        ),
        prediction(
            "Price / margin pressure",
            margin_percent,
            _risk_level(margin_percent, high=35, medium=15),
            "Compares input price/cost against historic scaffolder job values and implied margin.",
        ),
    ]

    recommendations: list[str] = []
    if hse_percent >= 3:
        recommendations.append("Request RAMS early and check handover/photo evidence before install.")
    if programme_percent >= 10:
        recommendations.append("Confirm scaffolder availability and homeowner access before committing dates.")
    if cancellation_percent >= 8:
        recommendations.append("Keep a backup scaffolder option live until booking is confirmed.")
    if margin_percent >= 15:
        recommendations.append("Review price/cost before issue; this job may carry margin pressure.")
    if not recommendations:
        recommendations.append("Proceed with standard booking controls and normal handover checks.")

    leaderboard = _team_leaderboard(
        df,
        selected_scaffolder=scaffolder,
        job_type=job_type,
        outward=outward,
        price=price,
    )

    return {
        "scaffolder": scaffolder,
        "job_inputs": {
            "postcode": postcode,
            "postcode_area": outward,
            "job_type": job_type,
            "lead_time_days": lead_days,
            "job_price": price,
            "budget_cost": cost,
        },
        "predictions": predictions,
        "recommendations": recommendations,
        "evidence": {
            "scaffolder_jobs": int(len(partner_df)),
            "similar_jobs_used": int(len(similar)),
            "hse_events": int(hse_events),
            "damage_events": int(damage_events),
            "avg_scaffolder_price": round(avg_revenue, 2),
            "median_scaffolder_price": round(median_revenue, 2) if not pd.isna(median_revenue) else 0.0,
            "completion_rate_pct": _pct(partner_completed / partner_total),
            "cancellation_rate_pct": _pct(partner_cancelled / partner_total),
        },
        "team_match_summary": leaderboard["summary"],
        "team_leaderboard": leaderboard["teams"],
    }


def portal_reference_data(*, dataset_path: str | Path) -> dict[str, Any]:
    df = _load_normalized_dataset(dataset_path)

    def values_for(column: str, limit: int = 500) -> list[str]:
        if column not in df.columns:
            return []
        values = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(limit)
            .index.tolist()
        )
        return [str(value) for value in values]

    return {
        "businesses": list_profile_entities(
            dataset_path=dataset_path,
            profile_type="client",
            limit=10000,
        ),
        "scaffolders": list_profile_entities(
            dataset_path=dataset_path,
            profile_type="partner",
            limit=10000,
        ),
        "project_managers": values_for("project_manager", 500),
        "budget_types": values_for("budget_type", 20),
        "current_statuses": values_for("status", 50),
        "scaffold_purposes": values_for("c_scaffoldpurpose", 200),
        "scaffold_locations": values_for("c_locationofscaffold", 500),
        "postcodes": values_for("c_homeownerpostcode", 2000),
    }


def get_job_intake_template(experiment_id: str, max_fields: int = 18) -> dict[str, Any]:
    _, experiment = load_registered_model(experiment_id)
    metadata = experiment["metadata"]
    dataset_path = experiment["dataset_path"]
    df = _load_normalized_dataset(dataset_path)
    feature_columns: list[str] = metadata.get("feature_columns", [])
    feature_importance = metadata.get("feature_importance", [])

    # Map expanded date features (e.g. date_year/date_month/...) to base date input names.
    date_base_candidates: set[str] = set()
    for col in feature_columns:
        for suffix in ("_year", "_month", "_day", "_weekday"):
            if col.endswith(suffix):
                date_base_candidates.add(col[: -len(suffix)])

    priority_columns = [
        "project_name",
        "contact_name",
        "related_contacts",
        "project_manager",
        "date",
        "due_date",
        "budget_type",
        "budget_revenue",
        "budget_cost",
        "actual_revenue",
        "actual_cost",
        "status",
        "description",
        "tags",
        "c_homeownerpostcode",
        "c_scaffoldpurpose",
    ]

    selected_fields: list[str] = []
    for col in priority_columns:
        if col in df.columns:
            selected_fields.append(col)

    for base_col in sorted(date_base_candidates):
        if base_col in df.columns and base_col not in selected_fields:
            selected_fields.append(base_col)

    # Include top raw features that appear in model feature space.
    if feature_importance:
        for item in feature_importance:
            feature_name = str(item.get("feature", ""))
            if "__" in feature_name:
                continue
            if feature_name in df.columns and feature_name not in selected_fields:
                selected_fields.append(feature_name)

    for col in feature_columns:
        if col in df.columns and col not in selected_fields:
            selected_fields.append(col)
        if len(selected_fields) >= max_fields:
            break

    selected_fields = selected_fields[:max_fields]
    fields: list[dict[str, Any]] = []

    for col in selected_fields:
        series = df[col] if col in df.columns else pd.Series(dtype=object)
        lower_col = col.lower()

        if pd.api.types.is_numeric_dtype(series):
            field_type = "number"
            default_value = _safe_mean(series)
            suggestions: list[Any] = []
        elif any(token in lower_col for token in ("date", "time")):
            field_type = "date"
            default_value = None
            suggestions = []
        else:
            field_type = "text"
            top_values = series.dropna().astype(str).value_counts().head(8).index.tolist()
            if len(top_values) >= 2 and series.nunique(dropna=True) <= 50:
                field_type = "category"
            suggestions = [str(v) for v in top_values]
            default_value = suggestions[0] if suggestions else ""

        fields.append(
            {
                "field": col,
                "type": field_type,
                "default": default_value,
                "suggested_values": suggestions,
                "required": col in {"contact_name", "project_manager", "date", "due_date"},
            }
        )

    return {
        "experiment_id": experiment_id,
        "target_column": experiment["target_column"],
        "task_type": experiment["task_type"],
        "dataset_path": dataset_path,
        "recommended_fields": fields,
    }


def predict_new_job_with_profiles(
    *,
    experiment_id: str,
    job_record: dict[str, Any],
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction = predict_records(experiment_id=experiment_id, records=[job_record])
    _, experiment = load_registered_model(experiment_id)

    profile_source = (
        str(Path(dataset_path).expanduser().resolve())
        if dataset_path
        else experiment["dataset_path"]
    )
    normalized_input = {str(k).strip().lower().replace(" ", "_"): v for k, v in job_record.items()}
    client_name = str(normalized_input.get("contact_name", "")).strip()
    partner_name = str(normalized_input.get("related_contacts", "")).strip()

    output: dict[str, Any] = {
        "prediction": prediction,
        "profiles": {},
    }

    if partner_name:
        output["operational_predictions"] = scaffolder_job_predictions(
            dataset_path=profile_source,
            job_record=job_record,
        )

    if client_name:
        try:
            output["profiles"]["client"] = generate_profile_card(
                dataset_path=profile_source,
                profile_type="client",
                name=client_name,
            )
        except Exception as exc:
            output["profiles"]["client_error"] = str(exc)

    if partner_name:
        try:
            output["profiles"]["partner"] = generate_profile_card(
                dataset_path=profile_source,
                profile_type="partner",
                name=partner_name,
            )
        except Exception as exc:
            output["profiles"]["partner_error"] = str(exc)

    return output
