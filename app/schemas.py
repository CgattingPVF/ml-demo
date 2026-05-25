from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProfileRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to CSV/XLSX dataset")


class TrainRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to CSV/XLSX dataset")
    target_column: str = Field(..., description="Target column to predict")
    task_type: Literal["classification", "regression"] | None = Field(
        default=None,
        description="Set explicitly or leave empty for auto-infer",
    )
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = Field(default=42)


class PredictRequest(BaseModel):
    experiment_id: str
    records: list[dict[str, Any]]


class DriftRequest(BaseModel):
    experiment_id: str
    current_dataset_path: str


class IntakeTemplateRequest(BaseModel):
    experiment_id: str
    max_fields: int = Field(default=18, ge=5, le=60)


class NewJobPredictRequest(BaseModel):
    experiment_id: str
    job_record: dict[str, Any]
    dataset_path: str | None = Field(
        default=None,
        description="Optional profile source dataset path. Defaults to model training dataset.",
    )


class ProfileCardRequest(BaseModel):
    dataset_path: str
    profile_type: Literal["client", "partner"]
    name: str


class ProfileEntitiesRequest(BaseModel):
    dataset_path: str
    profile_type: Literal["client", "partner"]
    limit: int = Field(default=200, ge=5, le=2000)
