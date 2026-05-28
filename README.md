# PVF Machine Learning Platform

Production-style ML platform scaffolded around a GDPR-reduced training workbook (`data/projects_export_gdpr_safe.xlsx`) with:

- Dataset profiling and target recommendation
- Automated feature engineering (missing value handling, categorical encoding, datetime expansion)
- Multi-model training and best-model selection
- Persistent model registry (SQLite + serialized models)
- New job intake form templates and single-job predictive scoring
- Detailed client and partner profile cards (portfolio, financial, delivery insights)
- HTML/CSS/JS web frontend powered by Python backend
- Worker UX at `/` for scaffolder-focused job risk: failure, completion, HSE, and damage risk
- ML management UX at `/dev` for model training and control
- Entity mapping used in the worker flow:
  - `Business` = dataset `contact_name`
  - `Scaffold Partner Company` = dataset `related_contacts`
- Worker dropdowns are populated from the GDPR-safe workbook, not the active model sample.

## Project Structure

```text
app/                # Shared ML/business logic
app.py              # Main launcher (FastAPI + HTML/CSS/JS)
templates/          # HTML templates
static/             # CSS/JS assets
models/             # Saved trained models
reports/            # Registry database
uploads/            # Uploaded scoring files
data/projects_export_gdpr_safe.xlsx # GDPR-reduced training workbook
data/projects_export_gdpr_safe_manifest.json # Reduction/audit notes
scripts/create_gdpr_safe_dataset.py # Rebuilds the safe workbook from an external raw export
```

## Quick Start

1. Create environment and install dependencies:

```bash
cd "/home/claytongatting/ML Demo"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Launch the web application:

```bash
python app.py
```

3. Open worker interface: `http://localhost:8000/`
4. Open ML management interface: `http://localhost:8000/dev`

## API Endpoints

- `GET /api/health` - Health status
- `GET /api/active-model` - Show active production model for worker screen
- `POST /api/active-model` - Set active production model
- `POST /api/profile` - Dataset profiling
- `POST /api/train` - Train and register best model
- `GET /api/experiments` - List experiments
- `GET /api/experiments/{experiment_id}` - Experiment details
- `POST /api/intake/template` - Dynamic new-job intake template
- `POST /api/intake/predict` - Predict a new job and return partner/client cards
- `POST /api/worker/predict` - Worker-friendly prediction endpoint (uses active model)
- `POST /api/scaffolder/predict` - Scaffolder-focused HSE/completion/programme/damage/price predictions
- `POST /api/portal/reference-data` - Full worker dropdown/reference data
- `POST /api/profiles/entities` - List client/partner names for profile cards
- `POST /api/profiles/card` - Generate detailed client/partner profile card

## Notes About The Dataset

- The app defaults to `data/projects_export_gdpr_safe.xlsx`.
- The raw export (`projects_export.csv`) is intentionally not kept in the codebase.
- The GDPR-safe workbook keeps scaffolder/team company names in `related_contacts` so training and selection remain identifiable.
- Direct homeowner/contact PII, free-text notes, photo/image fields, exact project identifiers, and exact full postcodes are removed or reduced.
- The active model is trained on `status_group`, a coarse non-PII status target derived from the raw status values.
- To rebuild the workbook from a local raw export, run:

```bash
python scripts/create_gdpr_safe_dataset.py /path/to/projects_export.csv
```

## Suggested PVF Next Steps

1. Define 1-2 business-critical targets first (e.g., job margin risk, completion delay, cancellation risk).
2. Add role-based auth and SSO before organization-wide rollout.
3. Containerize (`Dockerfile`) and deploy to Azure/AWS with managed PostgreSQL for enterprise scale.
