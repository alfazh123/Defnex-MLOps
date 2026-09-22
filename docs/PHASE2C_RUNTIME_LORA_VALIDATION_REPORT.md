# PHASE 2C RUNTIME LoRA VALIDATION REPORT

**Date:** 2026-09-16
**Status:** RUNTIME LoRA NOT VERIFIED

---

## Objective

Prove whether the Phase 2B LoRA artifact can be loaded and used by the existing `defnex-vllm` instance for inference via the runtime LoRA API (`/v1/load_lora_adapter`).

## Current vLLM Configuration

| Parameter | Value |
|-----------|-------|
| Container | `defnex-vllm` |
| Image | `vllm/vllm-openai:v0.28.0` |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| `--enable-lora` | Yes |
| `--max-loras` | 1 |
| `--max-lora-rank` | 16 |
| `--lora-modules` | `mlops-lora=/models/qwen2.5-0.5b-mlops-lora` |
| `--dtype` | bfloat16 |
| `--max-model-len` | 1024 |
| `--gpu-memory-utilization` | 0.15 |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | **NOT SET** in running container |
| Port | 8001 |
| Health | 200 |

**Full command line (from `docker inspect`):**

```
--model Qwen/Qwen2.5-0.5B-Instruct
--dtype bfloat16
--max-model-len 1024
--gpu-memory-utilization 0.15
--enable-lora
--max-loras 1
--max-lora-rank 16
--lora-modules mlops-lora=/models/qwen2.5-0.5b-mlops-lora
```

## Artifact

| Property | Value |
|----------|-------|
| Model ID | `smoke-llm-v3` |
| Version | 1 |
| Training run | `run-9e66b6` |
| Artifact URI | `file:///models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1` |
| Host path | `/home/ubuntu/defnex-mlops-experiment/outputs/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1/` |
| Base model (metadata.json) | `Qwen/Qwen2.5-0.5B-Instruct` |

### Artifact Visibility Inside Container

**CONFIRMED** — the artifact exists at `/models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1/` inside `defnex-vllm` with all required files:

- `adapter_model.safetensors` (2,175,168 bytes) — **readable**
- `adapter_config.json` — **readable**
- `tokenizer.json` — **readable**
- `tokenizer_config.json` — **readable**
- `chat_template.jinja` — **readable**
- `metadata.json` — **readable**

## Adapter Configuration

Extracted from `adapter_config.json`:

| Parameter | Value | Expected |
|-----------|-------|----------|
| `peft_type` | `LORA` | `LORA` ✓ |
| `r` | 8 | 8 ✓ |
| `lora_alpha` | 8 | 8 ✓ |
| `target_modules` | `["q_proj", "v_proj"]` | `q_proj, v_proj` ✓ |
| `lora_dropout` | 0.0 | — |
| `bias` | `"none"` | — |
| `use_dora` | false | — |
| `task_type` | `CAUSAL_LM` | — |
| `base_model_name_or_path` | `unsloth/Qwen2.5-0.5B-Instruct` | ⚠️ Namespace normalization |

### Base Model Namespace

The adapter records `base_model_name_or_path` as `unsloth/Qwen2.5-0.5B-Instruct` (Unsloth training namespace) while the running model is `Qwen/Qwen2.5-0.5B-Instruct` (standard HuggingFace). This is a **known Unsloth namespace normalization artifact**. vLLM resolves base model weights by architecture match (`Qwen2ForCausalLM`), not by exact name string — this would likely not cause a load failure, but cannot be proven without actual adapter load.

## Runtime Load Mechanism

### API Endpoint Probe

```
POST /v1/load_lora_adapter → 404 Not Found
POST /v1/unload_lora_adapter → 404 Not Found
```

### Root Cause

**vLLM 0.28.0 does NOT support the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints.** These were introduced in **vLLM ≥0.34.0**.

The project code (`app/services/serving.py:182-183`) expects these endpoints and requires:
- vLLM launched with `--enable-lora`
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` environment variable

The running `defnex-vllm` container:
1. Uses vLLM **0.28.0** (too old for dynamic LoRA API)
2. Does **NOT** have `VLLM_ALLOW_RUNTIME_LORA_UPDATING` set
3. Was started **externally** (not via docker-compose.yml which has this env var for the `serving` profile)

### Current LoRA Support

LoRA adapters are defined **statically at startup** via `--lora-modules`. The only adapter available is:
- `mlops-lora` → `/models/qwen2.5-0.5b-mlops-lora`

There is no mechanism to add adapters at runtime in vLLM 0.28.0.

## Baseline Inference

### Base Model (`Qwen/Qwen2.5-0.5B-Instruct`)

```
POST http://localhost:8001/v1/chat/completions
Body: {"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 20, "temperature": 0.0}

HTTP Status: 200
Response: "Hello! How can I assist you today?"
Model returned: "Qwen/Qwen2.5-0.5B-Instruct"
```

### Existing Adapter (`mlops-lora`)

```
POST http://localhost:8001/v1/chat/completions
Body: {"model": "mlops-lora", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 20, "temperature": 0.0}

HTTP Status: 200
Response: "Hello! How can I assist you today?"
Model returned: "mlops-lora"
```

Both the base model and existing adapter serve traffic successfully.

## New Adapter Load

**NOT PERFORMED.**

The `/v1/load_lora_adapter` endpoint returns 404. There is no supported runtime mechanism in vLLM 0.28.0 to load the `smoke-llm-v3` adapter without restarting the vLLM instance.

## New Adapter Inference

**NOT PERFORMED.**

Since the adapter could not be loaded, inference with `smoke-llm-v3` was not attempted.

## GPU State

```
NVIDIA H100 PCIe
Memory: 78,964 MiB / 81,559 MiB used
Utilization: 0%
```

Multiple GPU processes running (tenant workloads). No unrelated processes terminated. No GPU reset. No intervention.

## Safety

- No GPU processes killed or terminated
- No GPU reset performed
- No docker containers restarted
- No files modified inside the vLLM container
- No training runs created or executed
- All artifact reads were read-only
- vLLM health confirmed post-test: **200**

## Final Classification

### RUNTIME LoRA NOT VERIFIED

| State | Status |
|-------|--------|
| Artifact exists | ✅ Confirmed |
| Artifact visible to vLLM | ✅ Confirmed (all files readable inside container) |
| Adapter registered with vLLM | ❌ Not possible — no dynamic load API |
| Adapter loaded by vLLM | ❌ Not possible — requires vLLM restart with new `--lora-modules` |
| Adapter used for inference | ❌ Not possible — adapter not loaded |

The artifact is **physically present** and **visible** to the vLLM container, but the runtime mechanism to load it at runtime does not exist in the installed vLLM version.

## Limitations / Follow-up

### Blocking Issue: vLLM Version Too Old

The installed vLLM **0.28.0** does not support dynamic LoRA loading via API. To enable runtime LoRA:

1. **Upgrade vLLM to ≥0.34.0** — which introduced `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`
2. **Set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`** in the vLLM container environment
3. **Increase `--max-loras`** from 1 to ≥2 (current setting allows only 1 adapter at a time; loading a new one would require unloading the existing `mlops-lora` first)

### Alternative: Static Load via Restart

Without a vLLM upgrade, the adapter can be loaded by restarting `defnex-vllm` with:

```bash
--lora-modules mlops-lora=/models/qwen2.5-0.5b-mlops-lora,smoke-llm-v3-v1=/models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1
--max-loras 2
```

This was **NOT performed** per safety constraints (shared GPU, no restart without explicit instruction).

### Project Code Gap

`app/services/serving.py:VLLMServingBackend.deploy()` calls `POST /v1/load_lora_adapter` which returns 404 on vLLM 0.28.0. The project code was written for a newer vLLM version than what is currently deployed.

---

## FOLLOW-UP: Compose Serving Validation (2026-09-22)

**See full report:** [`PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md`](PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md)

### Resolution

The blocking issue was resolved by using the compose `serving` service (docker-compose.yml) instead of the existing `defnex-vllm` container:

| Parameter | defnex-vllm (old) | Compose serving (new) |
|-----------|-------------------|----------------------|
| vLLM version | 0.28.0 | **0.30.0** |
| Runtime LoRA API | Not available (404) | **Available** (`/v1/load_lora_adapter`) |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | Not set | **true** |
| `--max-loras` | 1 | **4** |

### Updated Findings

1. The `/v1/load_lora_adapter` endpoint was introduced in vLLM **0.29.0+** (not 0.34.0 as previously estimated). The remote `vllm/vllm-openai:latest` tag resolved to v0.30.0 as of 2026-09-22.

2. The runtime LoRA API is conditionally registered only when `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` is set in the container environment.

3. The full artifact→load→inference pipeline was verified end-to-end:
   - Artifact visible via `/models:ro` mount
   - Adapter registered via `POST /v1/load_lora_adapter`
   - Adapter appeared in `/v1/models` list
   - Inference with adapter returned HTTP 200
   - Adapter unloaded via `POST /v1/unload_lora_adapter`

### Updated Classification

**Previous (2026-09-16):** RUNTIME LoRA NOT VERIFIED
**Updated (2026-09-22):** RUNTIME LoRA E2E VERIFIED (via compose serving with vLLM 0.30.0)
