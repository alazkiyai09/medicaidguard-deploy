#!/usr/bin/env bash
set -euo pipefail

BUCKET="${BUCKET:-medicaidguard-models}"
MODEL_DIR="${MODEL_DIR:-model}"
VERSION="${VERSION:-v1.0.0}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}"
  exit 1
fi

gsutil cp "${MODEL_DIR}/model.pkl" "gs://${BUCKET}/models/${VERSION}/model.pkl"
gsutil cp "${MODEL_DIR}/feature_names.json" "gs://${BUCKET}/models/${VERSION}/feature_names.json"
gsutil cp "${MODEL_DIR}/model_metadata.json" "gs://${BUCKET}/models/${VERSION}/model_metadata.json"

echo "Uploaded model artifacts to gs://${BUCKET}/models/${VERSION}/"
