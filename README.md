# MedicaidGuard Cloud Deploy

Production-style FastAPI inference service for Medicaid fraud detection, designed for Cloud Run deployment and CI/CD automation.

## Endpoints

- `POST /predict`: single transaction prediction
- `POST /predict/batch`: batch predictions up to configured max size
- `GET /health`: model and runtime health
- `GET /metrics`: aggregated inference metrics

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Open docs at `http://localhost:8000/docs`.

## Docker

```bash
docker build -t medicaidguard-api .
docker run --rm -p 8000:8000 medicaidguard-api
```

## Tests

```bash
pytest --cov=app --cov-report=term-missing -v
```

## Deployment

- Manual deploy script: `scripts/deploy_cloudrun.sh`
- GCS model upload script: `scripts/upload_model_gcs.sh`
- GitHub Actions workflow: `.github/workflows/deploy-medicaidguard.yml`
