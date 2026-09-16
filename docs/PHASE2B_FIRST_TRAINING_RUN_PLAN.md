# Phase 2B — First Real Training Run Plan (FINAL)

**Date:** 2026-09-15 (updated 2026-09-16)
**Status:** THIRD SMOKE TRAINING RUN: PASS ✅
**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Mode:** Real training on shared H100 GPU — smoke test (5 steps, 10 examples)

---

## 1. Final JSON Config (Ready to Submit)

```json
{
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "hf_dataset": "/app/data/datasets/smoke_train.jsonl",
  "format_type": "text",
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "max_seq_length": 128,
  "max_steps": 5,
  "lora_r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.0,
  "target_modules": ["q_proj", "v_proj"],
  "epochs": 1,
  "learning_rate": 2e-5,
  "optim": "adamw_8bit",
  "random_seed": 42
}
```

## 2. Dataset Source and Number of Examples

**Source:** Local JSONL fixture at `/app/data/datasets/smoke_train.jsonl` (host: `ml-close-loop-be/data/datasets/smoke_train.jsonl`)
**Number of examples:** 10
**Format:** `{"text": "..."}` per line — compatible with `dataset_text_field="text"`

### Fixture Content (10 examples)

```jsonl
{"text": "Example 0: What is 0+0? The answer is 0."}
{"text": "Example 1: What is 1+1? The answer is 2."}
{"text": "Example 2: What is 2+2? The answer is 4."}
{"text": "Example 3: What is 3+3? The answer is 6."}
{"text": "Example 4: What is 4+4? The answer is 8."}
{"text": "Example 5: What is 5+5? The answer is 10."}
{"text": "Example 6: What is 6+6? The answer is 12."}
{"text": "Example 7: What is 7+7? The answer is 14."}
{"text": "Example 8: What is 8+8? The answer is 16."}
{"text": "Example 9: What is 9+9? The answer is 18."}
```

## 3. Timeout Status

**`training_timeout_seconds`:** ✅ SET to 600 (10 minutes) — no code changes required.

- **Location:** `app/config.py:243` — `training_timeout_seconds: int = 0`
- **Set via:** Env var `TRAINING_TIMEOUT_SECONDS=600` in docker-compose.yml worker environment
- **Enforcement:** `threading.Timer` kills subprocess after 600s (`training_provider.py:225-242`)
- **Verified:** `echo $TRAINING_TIMEOUT_SECONDS` → `600` inside worker container

## 4. Code Changes Applied

### Change 1: Local File Dataset Support (run_training.py:102-108)

```python
import os

hf_dataset = config.get("hf_dataset")
if os.path.isfile(hf_dataset):
    dataset = load_dataset("json", data_files=hf_dataset, split="train")
else:
    dataset = load_dataset(hf_dataset)
```

**Verified:** `load_dataset("json", data_files="/app/data/datasets/smoke_train.jsonl", split="train")` → 10 rows, columns: `['text']`

### Change 2: `max_steps` Config Wiring (run_training.py:119)

```python
max_steps=int(config.get("max_steps", -1)),
```

**Verified:** `grep -n "max_steps" /app/app/training/run_training.py` → line 119

## 5. What IS Verified (All Green)

### Safety Chain — All Verified ✅

| Step | Code Location | Verified |
|------|--------------|----------|
| GPU lock acquired | `training_worker.py:173` | ✅ `with gpu_lock(...)` |
| coordinator.cycle() entered | `training_worker.py:178` | ✅ `with coordinator.cycle()` |
| stop vLLM | `gpu_orchestrator.py:200,367-383` | ✅ writes request.json, polls response.json |
| VRAM threshold check | `gpu_orchestrator.py:216-222` | ✅ free >= threshold, deadline enforced |
| Training subprocess runs | `training_worker.py:202` | ✅ `job_runner.run(db, training_run)` |
| Adapter saved | `run_training.py:143-144` | ✅ `model.save_pretrained(staging)` |
| Restart vLLM (finally) | `gpu_orchestrator.py:231-232` | ✅ `control.start()` in finally block |
| Health check (finally) | `gpu_orchestrator.py:235-239` | ✅ `control.health_check()` |
| GPU lock released | Context exit | ✅ automatic |

### Artifact Output — Verified ✅

| Check | Result |
|-------|--------|
| `model.save_pretrained(staging)` creates adapter | ✅ Produces `adapter_model.safetensors` + `adapter_config.json` |
| `tokenizer.save_pretrained(staging)` creates tokenizer files | ✅ Produces `tokenizer.json` + `tokenizer_config.json` |
| `finalize_version()` moves to immutable dir | ✅ `shutil.move(staging → {ARTIFACT_STORAGE_DIR}/{model_id}/{name}/)` |
| Artifact URI format | ✅ `file:///models/artifacts/{model_id}/{name}/` |
| defnex-vllm can read artifact | ✅ Same `/models` mount, `--lora-modules` reads from there |
| Existing adapter format compatible | ✅ `qwen2.5-0.5b-mlops-lora/` has same file structure |

### Worker GPU — Verified ✅

| Check | Result |
|-------|--------|
| `/dev/nvidia*` in worker | ✅ `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` |
| `torch.cuda.is_available()` | ✅ True |
| GPU device | ✅ NVIDIA H100 PCIe |
| Unsloth import | ✅ OK |

### Training Venv — Verified ✅

| Package | Version | Status |
|---------|---------|--------|
| Python | 3.12.3 | ✅ |
| torch | 2.11.0+cu130 | ✅ |
| unsloth | 2026.9.2 | ✅ |
| trl | 0.24.0 | ✅ |
| transformers | 5.5.0 | ✅ |
| datasets | 4.3.0 | ✅ |

### Configuration — Verified ✅

| Setting | Value | Source |
|---------|-------|--------|
| `SERVING_CONTROL` | `file_signal` | docker-compose.yml |
| `GPU_CONTROL_DIR` | `/models/.gpu-control` | docker-compose.yml |
| `VRAM_FREE_THRESHOLD_MB` | `8000` | docker-compose.yml |
| `ARTIFACT_STORAGE_DIR` | `/models/artifacts` | docker-compose.yml |
| `TRAINING_PYTHON` | `/opt/training-venv/bin/python` | docker-compose.yml |
| `TRAINING_TIMEOUT_SECONDS` | `600` | docker-compose.yml (NEW) |

### Fixture — Verified ✅

| Check | Result |
|-------|--------|
| File exists at `/app/data/datasets/smoke_train.jsonl` | ✅ |
| 10 lines, each `{"text": "..."}` | ✅ |
| `load_dataset("json", data_files=..., split="train")` works | ✅ 10 rows |
| Columns match `dataset_text_field="text"` | ✅ |

### Tests — Verified ✅

| Test Suite | Result |
|------------|--------|
| `test_file_signaling.py` | ✅ (included) |
| `test_gpu_orchestration.py` | ✅ (included) |
| **Total** | **156 passed** |

---

## 6. Execution Steps (After This Report)

1. Create training run via API with config from Section 1
2. Monitor: `docker compose logs -f worker` + `watch -n 5 nvidia-smi`
3. Verify completion: training run status = COMPLETED
4. Verify artifact: `ls /home/ubuntu/defnex-mlops-experiment/outputs/artifacts/`
5. Verify vLLM restarted: `curl -s http://localhost:8001/health`
6. Verify 14 tenant processes still alive: `nvidia-smi --query-compute-apps=pid --format=csv`

---

## 7. Final Status

**THIRD SMOKE TRAINING RUN: PASS**

### Training Run History

| Run | Date | Status | Root Cause |
|-----|------|--------|------------|
| First smoke | 2026-09-15 | FAILED | EOS token mismatch, import order |
| Second smoke (run-a47b70) | 2026-09-16 01:13 | FAILED | Missing on_init_end callback |
| Third smoke attempt 1 (run-f06ff2) | 2026-09-16 01:40 | FAILED | Missing on_train_begin callback |
| Third smoke attempt 2 (run-21a878) | 2026-09-16 01:46 | FAILED | Missing on_epoch_begin callback |
| Third smoke attempt 3 (run-2c2b60) | 2026-09-16 01:55 | FAILED | Missing C compiler for Triton |
| Third smoke attempt 4 (run-9e66b6) | 2026-09-16 02:13 | **PASS** | -- |

### Bugs Fixed During Third Smoke Run

1. Missing on_train_begin -- Added no-op to _ProgressCallback
2. Missing on_epoch_begin -- Added ALL TrainerCallback lifecycle hooks as no-ops
3. Missing C compiler for Triton -- Added build-essential to Dockerfile

### Verified End-to-End

- TrainingRun run-9e66b6 created and completed
- Worker claimed job, GPU lock acquired
- File signal stopped defnex-vllm (VRAM freed to 17272 MiB)
- Qwen2.5-0.5B loaded, LoRA applied (540,672 params)
- SFTTrainer initialized, trainer.train() executed, 5 steps completed
- Adapter artifact: adapter_model.safetensors (2.1 MB) + adapter_config.json
- ModelVersion: smoke-llm-v3 v1 (REGISTERED)
- defnex-vllm restarted, health = HTTP 200
- GPU lock released, VRAM returned to 78964 MiB baseline

### All Blockers Resolved

1. Local file dataset support -- applied and verified
2. max_steps config wiring -- applied and verified
3. TRAINING_TIMEOUT_SECONDS=600 -- set and verified
4. Smoke fixture -- 10 examples at /app/data/datasets/smoke_train.jsonl
5. Worker GPU -- /dev/nvidia*, torch.cuda=True, Unsloth imports
6. Tests -- 156 passed, no regressions
7. Worker recreated -- all new config active
8. EOS token -- using tokenizer.eos_token
9. Import order -- unsloth before trl/datasets/transformers
10. _ProgressCallback -- all lifecycle hooks implemented
11. C compiler -- build-essential in Docker image for Triton
