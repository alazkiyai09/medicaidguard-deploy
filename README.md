# MedicaidGuard Deploy

Production ML fraud detection inference service (XGBoost + FastAPI) prepared for GCP Cloud Run and CI/CD.

## Live Demo (Deployment Placeholder)

- API base URL: `https://medicaidguard-api-xxxxx-as.a.run.app`
- Swagger docs: `https://medicaidguard-api-xxxxx-as.a.run.app/docs`
- Status: `pending deployment`

## Endpoints

- `POST /predict`
- `POST /predict/batch`
- `GET /health`
- `GET /metrics`

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Docs: `http://localhost:8000/docs`

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

- Script: `scripts/deploy_cloudrun.sh`
- Cloud Build config: `cloudbuild.yaml`
- GitHub Actions workflow: `.github/workflows/deploy-medicaidguard.yml`

Update live links above once deployed.
