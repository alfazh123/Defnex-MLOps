#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Checking vLLM serving dependency..."
if ! ./scripts/check-vllm.sh 30 2; then
    echo "WARNING: vLLM health check failed. Starting anyway — inference/deployment will be unavailable."
    echo "Set SERVING_BACKEND=mock to skip this check in local dev."
fi

echo "Starting backend..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
