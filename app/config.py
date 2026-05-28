from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "projects_export_gdpr_safe.xlsx"

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"

EXPERIMENT_DB_PATH = REPORTS_DIR / "experiments.db"

for directory in (DATA_DIR, MODELS_DIR, UPLOADS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
