from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any
import json

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import DEFAULT_DATASET_PATH
from app.platform import (
    generate_profile_card,
    get_job_intake_template,
    list_profile_entities,
    portal_reference_data,
    predict_new_job_with_profiles,
    profile_data,
    scaffolder_job_predictions,
    train_model,
)
from app.registry import get_experiment, list_experiments


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
SETTINGS_PATH = BASE_DIR / "reports" / "app_settings.json"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000


app = FastAPI(
    title="PVF Machine Learning Platform",
    version="2.0.0",
    description="Python + HTML/CSS/JS web application launcher via app.py",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _experiment_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "created_at": record["created_at"],
        "dataset_path": record["dataset_path"],
        "target_column": record["target_column"],
        "task_type": record["task_type"],
        "best_model_name": record["best_model_name"],
        "metric_name": record["metric_name"],
        "metric_value": record["metric_value"],
        "total_rows": record["total_rows"],
        "total_columns": record["total_columns"],
    }


def _load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _resolve_active_experiment_id() -> str:
    settings = _load_settings()
    saved_id = settings.get("active_experiment_id")
    if saved_id:
        try:
            get_experiment(saved_id)
            return saved_id
        except Exception:
            pass

    experiments = list_experiments(limit=1)
    if not experiments:
        raise HTTPException(status_code=400, detail="No model experiments found. Train a model in /dev first.")
    fallback_id = experiments[0]["id"]
    settings["active_experiment_id"] = fallback_id
    _save_settings(settings)
    return fallback_id


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "worker.html",
        {
            "request": request,
            "default_dataset_path": str(DEFAULT_DATASET_PATH),
        },
    )


@app.get("/dev", response_class=HTMLResponse)
def dev_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "dev.html",
        {
            "request": request,
            "default_dataset_path": str(DEFAULT_DATASET_PATH),
        },
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/profile")
def api_profile(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_path = str(payload.get("dataset_path", "")).strip()
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required.")
    try:
        return profile_data(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/train")
def api_train(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_path = str(payload.get("dataset_path", "")).strip()
    target_column = str(payload.get("target_column", "")).strip()
    task_type = payload.get("task_type")
    test_size = float(payload.get("test_size", 0.2))
    random_state = int(payload.get("random_state", 42))

    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required.")
    if not target_column:
        raise HTTPException(status_code=400, detail="target_column is required.")
    if task_type not in {None, "", "classification", "regression"}:
        raise HTTPException(status_code=400, detail="task_type must be classification, regression, or empty.")

    try:
        return train_model(
            dataset_path=dataset_path,
            target_column=target_column,
            task_type=(task_type or None),
            test_size=test_size,
            random_state=random_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/experiments")
def api_experiments(limit: int = 100) -> list[dict[str, Any]]:
    records = list_experiments(limit=limit)
    return [_experiment_summary(item) for item in records]


@app.get("/api/experiments/{experiment_id}")
def api_experiment_details(experiment_id: str) -> dict[str, Any]:
    try:
        return get_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/active-model")
def api_active_model() -> dict[str, Any]:
    experiment_id = _resolve_active_experiment_id()
    experiment = get_experiment(experiment_id)
    return {
        "active_experiment_id": experiment_id,
        "experiment": _experiment_summary(experiment),
    }


@app.post("/api/active-model")
def api_set_active_model(payload: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(payload.get("experiment_id", "")).strip()
    if not experiment_id:
        raise HTTPException(status_code=400, detail="experiment_id is required.")

    try:
        experiment = get_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings = _load_settings()
    settings["active_experiment_id"] = experiment_id
    _save_settings(settings)
    return {
        "active_experiment_id": experiment_id,
        "experiment": _experiment_summary(experiment),
    }


@app.post("/api/intake/template")
def api_intake_template(payload: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(payload.get("experiment_id", "")).strip()
    max_fields = int(payload.get("max_fields", 18))
    if not experiment_id:
        raise HTTPException(status_code=400, detail="experiment_id is required.")

    try:
        return get_job_intake_template(
            experiment_id=experiment_id,
            max_fields=max_fields,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/intake/predict")
def api_intake_predict(payload: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(payload.get("experiment_id", "")).strip()
    job_record = payload.get("job_record")
    dataset_path = payload.get("dataset_path")

    if not experiment_id:
        raise HTTPException(status_code=400, detail="experiment_id is required.")
    if not isinstance(job_record, dict) or not job_record:
        raise HTTPException(status_code=400, detail="job_record must be a non-empty object.")

    try:
        return predict_new_job_with_profiles(
            experiment_id=experiment_id,
            job_record=job_record,
            dataset_path=dataset_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/worker/predict")
def api_worker_predict(payload: dict[str, Any]) -> dict[str, Any]:
    job_record = payload.get("job_record")
    dataset_path = payload.get("dataset_path")
    if not isinstance(job_record, dict) or not job_record:
        raise HTTPException(status_code=400, detail="job_record must be a non-empty object.")

    experiment_id = _resolve_active_experiment_id()

    try:
        return predict_new_job_with_profiles(
            experiment_id=experiment_id,
            job_record=job_record,
            dataset_path=dataset_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/scaffolder/predict")
def api_scaffolder_predict(payload: dict[str, Any]) -> dict[str, Any]:
    job_record = payload.get("job_record")
    dataset_path = payload.get("dataset_path")
    if not dataset_path:
        dataset_path = str(DEFAULT_DATASET_PATH)
    if not isinstance(job_record, dict) or not job_record:
        raise HTTPException(status_code=400, detail="job_record must be a non-empty object.")

    try:
        return scaffolder_job_predictions(
            dataset_path=dataset_path,
            job_record=job_record,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/profiles/entities")
def api_profile_entities(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_path = str(payload.get("dataset_path", "")).strip()
    profile_type = str(payload.get("profile_type", "")).strip().lower()
    limit = int(payload.get("limit", 200))

    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required.")
    if profile_type not in {"client", "partner"}:
        raise HTTPException(status_code=400, detail="profile_type must be client or partner.")

    try:
        entities = list_profile_entities(
            dataset_path=dataset_path,
            profile_type=profile_type,
            limit=limit,
        )
        return {"profile_type": profile_type, "entities": entities}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/portal/reference-data")
def api_portal_reference_data(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_path = str(payload.get("dataset_path", "")).strip()
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required.")

    try:
        return portal_reference_data(dataset_path=dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/profiles/card")
def api_profile_card(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_path = str(payload.get("dataset_path", "")).strip()
    profile_type = str(payload.get("profile_type", "")).strip().lower()
    name = str(payload.get("name", "")).strip()

    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required.")
    if profile_type not in {"client", "partner"}:
        raise HTTPException(status_code=400, detail="profile_type must be client or partner.")
    if not name:
        raise HTTPException(status_code=400, detail="name is required.")

    try:
        return generate_profile_card(
            dataset_path=dataset_path,
            profile_type=profile_type,
            name=name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _list_listening_socket_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    port_hex = f"{port:04X}"
    for table_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table_path.exists():
            continue
        try:
            lines = table_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue

            local_address = parts[1]
            state = parts[3]
            inode = parts[9]
            try:
                _, local_port_hex = local_address.split(":")
            except ValueError:
                continue

            if state == "0A" and local_port_hex.upper() == port_hex:
                inodes.add(inode)

    return inodes


def _pids_listening_on_port(port: int) -> set[int]:
    socket_inodes = _list_listening_socket_inodes(port)
    if not socket_inodes:
        return set()

    pids: set[int] = set()
    proc_root = Path("/proc")

    for entry in proc_root.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue

        fd_dir = entry / "fd"
        if not fd_dir.exists():
            continue

        try:
            for fd_entry in fd_dir.iterdir():
                try:
                    target = os.readlink(fd_entry)
                except OSError:
                    continue

                if not target.startswith("socket:[") or not target.endswith("]"):
                    continue

                inode = target[8:-1]
                if inode in socket_inodes:
                    pids.add(int(entry.name))
                    break
        except (PermissionError, FileNotFoundError, OSError):
            continue

    return pids


def _read_process_cmdline(pid: int) -> str:
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_exit(pid: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _is_process_alive(pid):
            return True
        time.sleep(0.1)
    return not _is_process_alive(pid)


def _terminate_pid(pid: int) -> bool:
    if not _is_process_alive(pid):
        return True

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    if _wait_for_process_exit(pid, timeout_seconds=4.0):
        return True

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    return _wait_for_process_exit(pid, timeout_seconds=2.0)


def _close_existing_server_instances(host: str, port: int) -> None:
    current_pid = os.getpid()
    listener_pids = _pids_listening_on_port(port) - {current_pid}
    if not listener_pids:
        return

    for pid in sorted(listener_pids):
        cmdline = _read_process_cmdline(pid) or "<unknown process>"
        print(f"Stopping existing process on {host}:{port} -> PID {pid}: {cmdline}")
        if not _terminate_pid(pid):
            raise RuntimeError(
                f"Could not stop PID {pid} already listening on {host}:{port}. "
                f"Process: {cmdline}"
            )

    remaining_pids = _pids_listening_on_port(port) - {current_pid}
    if remaining_pids:
        details = ", ".join(
            f"PID {pid}: {_read_process_cmdline(pid) or '<unknown process>'}"
            for pid in sorted(remaining_pids)
        )
        raise RuntimeError(f"Address {host}:{port} is still in use by {details}")


if __name__ == "__main__":
    _close_existing_server_instances(host=SERVER_HOST, port=SERVER_PORT)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, reload=False)
