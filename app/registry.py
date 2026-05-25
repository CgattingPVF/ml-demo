from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib

from app.config import EXPERIMENT_DB_PATH, MODELS_DIR


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(EXPERIMENT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_registry() -> None:
    with _connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                target_column TEXT NOT NULL,
                task_type TEXT NOT NULL,
                total_rows INTEGER NOT NULL,
                total_columns INTEGER NOT NULL,
                best_model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                model_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def register_experiment(
    *,
    dataset_path: str,
    target_column: str,
    task_type: str,
    total_rows: int,
    total_columns: int,
    best_model_name: str,
    metric_name: str,
    metric_value: float,
    metadata: dict[str, Any],
    fitted_pipeline: Any,
) -> dict[str, Any]:
    init_registry()

    experiment_id = str(uuid4())
    created_at = datetime.now(tz=timezone.utc).isoformat()

    model_path = MODELS_DIR / f"{experiment_id}.joblib"
    joblib.dump(fitted_pipeline, model_path)

    record = {
        "id": experiment_id,
        "created_at": created_at,
        "dataset_path": dataset_path,
        "target_column": target_column,
        "task_type": task_type,
        "total_rows": int(total_rows),
        "total_columns": int(total_columns),
        "best_model_name": best_model_name,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "model_path": str(model_path),
        "metadata_json": json.dumps(metadata),
    }

    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO experiments (
                id, created_at, dataset_path, target_column, task_type,
                total_rows, total_columns, best_model_name, metric_name,
                metric_value, model_path, metadata_json
            ) VALUES (
                :id, :created_at, :dataset_path, :target_column, :task_type,
                :total_rows, :total_columns, :best_model_name, :metric_name,
                :metric_value, :model_path, :metadata_json
            )
            """,
            record,
        )
        conn.commit()

    output = record.copy()
    output["metadata"] = metadata
    output.pop("metadata_json", None)
    return output


def list_experiments(limit: int = 100) -> list[dict[str, Any]]:
    init_registry()
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM experiments
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        results.append(payload)
    return results


def get_experiment(experiment_id: str) -> dict[str, Any]:
    init_registry()
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Experiment '{experiment_id}' not found.")
    payload = dict(row)
    payload["metadata"] = json.loads(payload.pop("metadata_json"))
    return payload


def load_registered_model(experiment_id: str) -> tuple[Any, dict[str, Any]]:
    experiment = get_experiment(experiment_id)
    model_path = Path(experiment["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")
    model = joblib.load(model_path)
    return model, experiment

