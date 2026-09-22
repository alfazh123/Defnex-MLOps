#!/bin/sh
# scripts/check-vllm.sh — Startup health check for external vLLM dependency (issue #162).
#
# Validates that the vLLM serving backend configured via VLLM_URL is reachable before
# the application starts. This prevents silent failures where the backend starts but
# can't reach the serving stack, leaving inference/deployment broken with no alert.
#
# Usage:
#   ./scripts/check-vllm.sh [max_retries] [retry_interval]
#
# Exit codes:
#   0 — vLLM is reachable and healthy
#   1 — vLLM is unreachable after retries (or SERVING_BACKEND != vllm, check skipped)
#   2 — Configuration error (missing VLLM_URL)

set -e

SERVING_BACKEND="${SERVING_BACKEND:-mock}"
VLLM_URL="${VLLM_URL:-}"
MAX_RETRIES="${1:-30}"
RETRY_INTERVAL="${2:-2}"

if [ "$SERVING_BACKEND" != "vllm" ]; then
    echo "[check-vllm] SERVING_BACKEND=$SERVING_BACKEND — skipping vLLM health check."
    exit 0
fi

if [ -z "$VLLM_URL" ]; then
    echo "[check-vllm] ERROR: VLLM_URL is not set but SERVING_BACKEND=vllm."
    echo "[check-vllm] Set VLLM_URL to the vLLM server endpoint (e.g. http://host:8001)."
    exit 2
fi

echo "[check-vllm] Checking vLLM at $VLLM_URL (max ${MAX_RETRIES} retries, ${RETRY_INTERVAL}s interval)..."

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    if wget -q -O /dev/null --timeout=5 "${VLLM_URL}/health" 2>/dev/null; then
        echo "[check-vllm] vLLM is healthy at $VLLM_URL (attempt ${attempt}/${MAX_RETRIES})"
        exit 0
    fi
    echo "[check-vllm] vLLM not ready (attempt ${attempt}/${MAX_RETRIES}), retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL" 2>/dev/null || true
done

echo "[check-vllm] FATAL: vLLM unreachable at $VLLM_URL after ${MAX_RETRIES} attempts."
echo "[check-vllm] Possible causes:"
echo "  - defnex-vllm container is not running (check: docker ps | grep defnex-vllm)"
echo "  - VLLM_URL points to wrong host/port (current: $VLLM_URL)"
echo "  - Network connectivity issue (if using host gateway, check Docker bridge)"
echo "  - vLLM is still starting (increase max_retries or retry_interval)"
echo ""
echo "Recovery steps:"
echo "  1. Check if defnex-vllm is running: docker ps | grep defnex-vllm"
echo "  2. Check vLLM health manually: curl -s ${VLLM_URL}/health"
echo "  3. If using external vLLM, ensure the defnex-mlops-experiment stack is up"
echo "  4. Check docker-compose.yml VLLM_URL setting matches your setup"
exit 1
