#!/bin/sh
set -e

echo "Starting training worker..."
exec python -m app.workers.training_worker
