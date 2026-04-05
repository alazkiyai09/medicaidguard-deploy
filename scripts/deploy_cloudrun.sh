#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-asia-southeast1}"
SERVICE_NAME="${SERVICE_NAME:-medicaidguard-api}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-false}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost,http://127.0.0.1}"
API_KEY_SECRET="${API_KEY_SECRET:-service-api-key}"
MODEL_PATH="${MODEL_PATH:-/app/model/model.pkl}"
MODEL_SOURCE="${MODEL_SOURCE:-local}"
MODEL_SHA256="${MODEL_SHA256:-$(python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("model/model_metadata.json").read_text(encoding="utf-8"))
print(payload.get("sha256", ""))
PY
)}"
ENV_VARS="^@^MODEL_PATH=${MODEL_PATH}@MODEL_SOURCE=${MODEL_SOURCE}@MODEL_SHA256=${MODEL_SHA256}@ALLOWED_ORIGINS=${ALLOWED_ORIGINS}"

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "Set PROJECT_ID env var before running this script."
  exit 1
fi

deploy_args=(
  "${SERVICE_NAME}"
  --source .
  --region "${REGION}"
  --platform managed
  --memory 1Gi
  --cpu 1
  --min-instances 0
  --max-instances 3
  --timeout 300
  --concurrency 80
  --set-env-vars "${ENV_VARS}"
  --update-secrets "API_KEY=${API_KEY_SECRET}:latest"
)

if [[ "${ALLOW_UNAUTHENTICATED,,}" == "true" ]]; then
  deploy_args+=(--allow-unauthenticated)
fi

gcloud run deploy "${deploy_args[@]}"

echo "Deployed URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format 'value(status.url)'
