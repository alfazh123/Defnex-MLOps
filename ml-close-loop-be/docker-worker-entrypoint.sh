#!/bin/sh
set -e

echo "Checking vLLM serving dependency..."
if ! ./scripts/check-vllm.sh 30 2; then
    echo "WARNING: vLLM health check failed. Starting anyway — training GPU handoff may be affected."
    echo "Set SERVING_BACKEND=mock to skip this check in local dev."
fi

echo "Starting training worker..."
exec python -m app.workers.training_worker
