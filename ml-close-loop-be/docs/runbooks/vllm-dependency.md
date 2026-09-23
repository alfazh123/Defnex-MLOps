# External vLLM Dependency Runbook

**Issue:** #162 — harden or integrate the external defnex-vllm / legacy-experiment dependency
**Status:** Pinned and documented (not integrated into main Compose)

---

## Architecture

The main `ml-close-loop-be` Docker Compose stack depends on a **separate, externally-managed** vLLM serving instance (`defnex-vllm`). This is NOT part of this repository's Compose stack.

```
┌─────────────────────────────────────────────────────────────────┐
│ ml-close-loop-be (this repo)                                    │
│ ┌───────────┐  ┌──────────┐  ┌──────┐                         │
│ │  backend   │  │  worker  │  │ minio│                         │
│ │  :8000     │  │  (GPU)   │  │ :9000│                         │
│ └─────┬──────┘  └────┬─────┘  └──────┘                         │
│       │              │                                         │
│       │  VLLM_URL    │  volume mounts                          │
│       ▼              ▼                                         │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ Network: ml-close-loop-be_default                           ││
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
              │                              │
              │ http://172.17.0.1:8001       │ /models, /opt/training-venv
              │ (Docker bridge gateway)      │ (host path mounts)
              ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ defnex-mlops-experiment (EXTERNAL — separate Compose project)   │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │  defnex-vllm  (vllm/vllm-openai:latest)                    ││
│ │  Port: 8001:8000                                            ││
│ │  Model: Qwen/Qwen2.5-0.5B-Instruct                         ││
│ │  Network: defnex-mlops-experiment_default                    ││
│ └─────────────────────────────────────────────────────────────┘│
│ ┌─────────────────────────────────────────────────────────────┐│
│ │  gpu_controller.py (host-side, manages defnex-vllm)         ││
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Pinned External Dependencies

| What | Where (host path) | Used by | Purpose |
|------|-------------------|---------|---------|
| `defnex-vllm` container | `172.17.0.1:8001` | backend, worker | Model serving (vLLM runtime API) |
| Training outputs | `/home/ubuntu/defnex-mlops-experiment/outputs` | worker (as `/models`) | Adapter artifacts shared with vLLM |
| Training venv | `/home/ubuntu/defnex-mlops-experiment/.venv` | worker (as `/opt/training-venv`) | Unsloth + ML dependencies |

## How to Verify the Dependency

```bash
# 1. Check if defnex-vllm is running
docker ps | grep defnex-vllm

# 2. Check vLLM health
curl -s http://172.17.0.1:8001/health

# 3. Check from inside the backend container
docker compose exec backend wget -qO- http://172.17.0.1:8001/health

# 4. Run the startup check script manually
./scripts/check-vllm.sh
```

## Startup Behavior

Both `docker-entrypoint.sh` and `docker-worker-entrypoint.sh` run `scripts/check-vllm.sh` before starting:

- **SERVING_BACKEND=mock**: Check is skipped entirely (no vLLM needed).
- **SERVING_BACKEND=vllm**: Checks up to 30 retries with 2s interval (60s total). On failure:
  - Backend: **warns but starts anyway** — inference/deploy endpoints will fail with clear errors.
  - Worker: **warns but starts anyway** — training GPU handoff may be affected.

This is intentionally non-blocking to avoid preventing the app from starting during local dev or when the external vLLM is temporarily down.

## Troubleshooting

### "vLLM unreachable" on startup

1. Check the external project is running:
   ```bash
   docker ps | grep defnex-vllm
   ```

2. Verify network connectivity:
   ```bash
   curl -s http://172.17.0.1:8001/health
   ```

3. If the external project was restarted, the container IP may have changed. The host gateway IP (`172.17.0.1`) is stable, but the internal Docker IP of `defnex-vllm` may not be. Ensure port mapping `8001:8000` is still active.

### Port conflict on 8001

If you see `Bind for 0.0.0.0:8001 failed: port is already allocated`:
- The external `defnex-vllm` is already using port 8001.
- Do NOT enable the `gpu` profile (`docker compose --profile gpu up`) — it will conflict.
- Keep using `VLLM_URL=http://172.17.0.1:8001` to reach the existing vLLM.

### Training venv or outputs path changed

If the external project's paths changed:
1. Update the volume mounts in `docker-compose.yml` (worker service).
2. Update `ARTIFACT_STORAGE_DIR` if the artifacts path changed.
3. Update `GPU_CONTROL_DIR` if the GPU control directory changed.
4. Test: run a training job end-to-end.

## Migration Path (Future)

When ready to eliminate the external dependency:
1. Enable the `gpu` profile in docker-compose.yml (remove `profiles: ["gpu"]` or pass `--profile gpu`).
2. Set `VLLM_URL=http://serving:8000` (Compose internal DNS).
3. Move the training venv installation into the worker Dockerfile.
4. Move the outputs volume mount to a shared Compose volume.
5. Remove the host-path volume mounts from the worker service.

This requires the training venv to be reproducible inside a Docker image (issue #166).
