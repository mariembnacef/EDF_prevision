#!/usr/bin/env bash
set -euo pipefail

# 1. Lancer MLflow Server (utilise GUNICORN_CMD_ARGS pour augmenter le timeout)
exec mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port 5000 &
  
# 2. Petit délai pour s’assurer que MLflow est UP
sleep 5

# 3. Exécuter le pipeline Prefect
exec python /app/pre_main.py

