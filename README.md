# PVF Machine Learning Platform

Production-style ML platform scaffolded around your dataset export (`projects_export.csv`) with:

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
- Worker dropdowns are populated from the full dataset export, not the active model sample.

## Project Structure

```text
app/                # Shared ML/business logic
app.py              # Main launcher (FastAPI + HTML/CSS/JS)
templates/          # HTML templates
static/             # CSS/JS assets
models/             # Saved trained models
reports/            # Registry database
uploads/            # Uploaded scoring files
projects_export.csv # Source dataset (UTF-16 tab-delimited export)
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

## Notes About Your Dataset

- The provided file appears to be **tab-delimited UTF-16** (not plain UTF-8 CSV).
- Loader auto-detects encoding and delimiter, so no manual preprocessing is required.
- With 70k+ rows and many custom fields, first training runs can take several minutes.

## Suggested PVF Next Steps

1. Define 1-2 business-critical targets first (e.g., job margin risk, completion delay, cancellation risk).
2. Add role-based auth and SSO before organization-wide rollout.
3. Containerize (`Dockerfile`) and deploy to Azure/AWS with managed PostgreSQL for enterprise scale.
