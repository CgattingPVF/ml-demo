from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import UPLOADS_DIR
from app.platform import (
    batch_predict,
    drift_check,
    generate_profile_card,
    get_job_intake_template,
    list_profile_entities,
    predict_new_job_with_profiles,
    predict_records,
    profile_data,
    train_model,
)
from app.registry import get_experiment, list_experiments
from app.schemas import (
    DriftRequest,
    IntakeTemplateRequest,
    NewJobPredictRequest,
    PredictRequest,
    ProfileCardRequest,
    ProfileEntitiesRequest,
    ProfileRequest,
    TrainRequest,
)


app = FastAPI(
    title="PVF Machine Learning Platform API",
    version="1.0.0",
    description=(
        "API for dataset profiling, automated model training, model registry, "
        "prediction, and drift monitoring."
    ),
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "PVF ML Platform API is running."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/profile")
def profile(request: ProfileRequest) -> dict:
    try:
        return profile_data(request.dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train(request: TrainRequest) -> dict:
    try:
        return train_model(
            dataset_path=request.dataset_path,
            target_column=request.target_column,
            task_type=request.task_type,
            test_size=request.test_size,
            random_state=request.random_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/experiments")
def experiments(limit: int = 100) -> list[dict]:
    return list_experiments(limit=limit)


@app.get("/experiments/{experiment_id}")
def experiment(experiment_id: str) -> dict:
    try:
        return get_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        return predict_records(
            experiment_id=request.experiment_id,
            records=request.records,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/drift")
def drift(request: DriftRequest) -> dict:
    try:
        return drift_check(
            experiment_id=request.experiment_id,
            current_dataset_path=request.current_dataset_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/intake/template")
def intake_template(request: IntakeTemplateRequest) -> dict:
    try:
        return get_job_intake_template(
            experiment_id=request.experiment_id,
            max_fields=request.max_fields,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/intake/predict")
def intake_predict(request: NewJobPredictRequest) -> dict:
    try:
        return predict_new_job_with_profiles(
            experiment_id=request.experiment_id,
            job_record=request.job_record,
            dataset_path=request.dataset_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/profiles/card")
def profile_card(request: ProfileCardRequest) -> dict:
    try:
        return generate_profile_card(
            dataset_path=request.dataset_path,
            profile_type=request.profile_type,
            name=request.name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/profiles/entities")
def profile_entities(request: ProfileEntitiesRequest) -> dict:
    try:
        entities = list_profile_entities(
            dataset_path=request.dataset_path,
            profile_type=request.profile_type,
            limit=request.limit,
        )
        return {"profile_type": request.profile_type, "entities": entities}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/batch-predict")
async def batch_predict_endpoint(
    experiment_id: str = Form(...),
    data_file: UploadFile = File(...),
) -> FileResponse:
    upload_id = str(uuid.uuid4())
    upload_ext = Path(data_file.filename or "uploaded.csv").suffix or ".csv"
    upload_path = UPLOADS_DIR / f"{upload_id}{upload_ext}"

    content = await data_file.read()
    upload_path.write_bytes(content)

    try:
        prediction_df = batch_predict(experiment_id=experiment_id, dataset_path=upload_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_path = UPLOADS_DIR / f"{upload_id}_predictions.csv"
    prediction_df.to_csv(output_path, index=False)
    return FileResponse(
        path=output_path,
        media_type="text/csv",
        filename=f"{experiment_id}_predictions.csv",
    )
