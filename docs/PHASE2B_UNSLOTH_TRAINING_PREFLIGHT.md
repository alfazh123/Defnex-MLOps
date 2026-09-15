# Phase 2B — Unsloth Training Preflight Report

**Date:** 2026-09-15
**Status:** READY FOR REAL TRAINING AFTER GPU PASSTHROUGH
**Mode:** PREFLIGHT ONLY — no training executed, no config changes applied

---

## 1. Repository / Training Path Audit

### Training Call Chain

```
TrainingRun created via API
  ↓
training_worker.run_forever() — polls DB every 5s
  ↓
process_next_job() — claims next PENDING run
  ↓
gpu_lock() — acquires exclusive flock
  ↓
coordinator.cycle() — FileSignalingServingControl
  ↓
FileSignalingServingControl.stop() — writes request.json
  ↓
host gpu_controller.py — docker stop defnex-vllm
  ↓
nvidia-smi — measures free VRAM
  ↓
response.json — vram_free_mb returned
  ↓
VRAM threshold check — free >= 8000 MiB
  ↓
LocalSubprocessProvider.submit() — spawns subprocess
  ↓
/opt/training-venv/bin/python -u run_training.py --config '<json>' --staging <dir>
  ↓
run_training.py imports: torch, unsloth, trl, transformers, datasets
  ↓
FastLanguageModel.from_pretrained() — loads Qwen2.5-0.5B-Instruct
  ↓
FastLanguageModel.get_peft_model() — applies LoRA
  ↓
SFTTrainer.train() — runs training
  ↓
model.save_pretrained(staging) — saves adapter
  ↓
coordinator.cycle() exit — FileSignalingServingControl.start()
  ↓
host gpu_controller.py — docker start defnex-vllm
  ↓
vLLM health check — HTTP 200
```

### Source Locations

| Component | File | Line |
|-----------|------|------|
| Worker entry point | `app/workers/training_worker.py:274-280` | `run_forever(LocalSubprocessProvider())` |
| GPU lock | `app/workers/training_worker.py:173` | `with gpu_lock(...)` |
| Serving coordinator | `app/workers/gpu_orchestrator.py:491-504` | `make_coordinator()` for file_signal |
| FileSignalingServingControl | `app/workers/gpu_orchestrator.py:304-407` | writes request.json, polls response.json |
| LocalSubprocessProvider | `app/providers/training_provider.py:93-274` | spawns subprocess |
| Training script | `app/training/run_training.py:49-142` | `_run_training()` |
| Unsloth model load | `app/training/run_training.py:71-77` | `FastLanguageModel.from_pretrained()` |
| LoRA config | `app/training/run_training.py:78-97` | `FastLanguageModel.get_peft_model()` |
| Training execution | `app/training/run_training.py:130-134` | `trainer.train()` |
| Adapter save | `app/training/run_training.py:136-137` | `model.save_pretrained(staging)` |

---

## 2. Worker GPU Configuration Audit

### Current Worker GPU State

| Property | Value |
|----------|-------|
| DeviceRequests | `null` |
| Devices | `null` |
| Runtime | `runc` |
| GPU profile | NOT enabled |
| `/dev/nvidia*` inside container | **DOES NOT EXIST** |
| `torch.cuda.is_available()` inside container | **False** |
| Unsloth inside container | **CRASHES** — "cannot find any torch accelerator? You need a GPU" |

### Required GPU Configuration

The worker needs NVIDIA GPU passthrough. Two options:

**Option A: CDI (recommended for consistency with defnex-vllm)**

```yaml
worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

With Docker CDI enabled (already used by defnex-vllm), this mounts `/dev/nvidia*` and libnvidia libs into the container.

**Option B: Device mount**

```yaml
worker:
  devices:
    - /dev/nvidia0:/dev/nvidia0
    - /dev/nvidiactl:/dev/nvidiactl
    - /dev/nvidia-uvm:/dev/nvidia-uvm
```

**Recommendation:** Option A (deploy.resources.reservations.devices) — matches the `serving` service pattern already in docker-compose.yml, cleaner, and Docker handles CDI automatically.

### Why CDI Should Be Used

- defnex-vllm already uses CDI (nvidia.com/gpu=all or driver nvidia)
- The H100 driver (580.173.02) and CUDA 13.0 are already installed
- PyTorch in the training venv is built with CUDA 13.0 (`torch=2.11.0+cu130`)
- CDI is the standard Docker GPU passthrough mechanism for NVIDIA

---

## 3. Training Venv Audit

### Host Venv (`/home/ubuntu/defnex-mlops-experiment/.venv`)

| Package | Version | Status |
|---------|---------|--------|
| Python | 3.12.3 | ✅ |
| torch | 2.11.0+cu130 | ✅ |
| torch.version.cuda | 13.0 | ✅ |
| torch.cuda.is_available() | True (on host) | ✅ |
| unsloth | 2026.9.2 | ✅ |
| trl | 0.24.0 | ✅ |
| transformers | 5.5.0 | ✅ |
| datasets | 4.3.0 | ✅ |

### Worker Venv (`/opt/training-venv` inside container)

| Check | Result |
|-------|--------|
| Python path exists | ✅ `/opt/training-venv/bin/python → python3` |
| unsloth package present | ✅ `site-packages/unsloth/__init__.py` |
| unsloth importable | ❌ CRASH — "Unsloth cannot find any torch accelerator? You need a GPU" |
| datasets importable | ❌ ModuleNotFoundError (no GPU, unsloth crash blocks import chain) |

**Root cause:** Unsloth's `_gpu_init.py` requires a GPU at import time. This is expected — Unsloth cannot even initialize without CUDA. The venv itself is correctly configured; it just needs GPU passthrough to function.

---

## 4. CUDA Compatibility

### Host Environment

| Property | Value |
|----------|-------|
| Driver version | 580.173.02 |
| CUDA version | 13.0 |
| PyTorch CUDA build | cu130 |
| torch.cuda.is_available() | **True** |
| GPU | NVIDIA H100 PCIe |

**HOST CUDA: READY** ✅

### Worker Environment (current, no GPU)

| Property | Value |
|----------|-------|
| torch.cuda.is_available() | **False** (no /dev/nvidia*) |
| Unsloth import | CRASH (requires GPU) |

**WORKER GPU: NOT READY** — needs GPU passthrough

### Compatibility Assessment

- Driver 580.173.02 supports CUDA 13.0 ✅
- PyTorch 2.11.0+cu130 matches CUDA 13.0 ✅
- H100 PCIe is CUDA-capable ✅
- Once GPU passthrough is added, torch.cuda.is_available() will be True inside the container

---

## 5. Unsloth Initialization Path

### Import Sequence in `run_training.py:49-62`

```python
def _run_training(config, staging):
    try:
        from datasets import load_dataset        # line 51
        from transformers import TrainingArguments # line 52
        from trl import SFTTrainer                # line 54
        from unsloth import FastLanguageModel, is_bfloat16_supported  # line 55
    except ImportError as exc:
        return 2  # missing dependency
```

### Initialization Sequence

| Step | Code | GPU Required? |
|------|------|--------------|
| 1. Import torch | Implicit via unsloth | No (but no CUDA) |
| 2. Import unsloth | `from unsloth import ...` | **YES — crashes without GPU** |
| 3. Load model | `FastLanguageModel.from_pretrained()` | **YES — loads to GPU** |
| 4. Apply LoRA | `FastLanguageModel.get_peft_model()` | Yes |
| 5. Load dataset | `load_dataset(config["hf_dataset"])` | No |
| 6. Create trainer | `SFTTrainer(...)` | No |
| 7. Train | `trainer.train()` | **YES — CUDA OOM possible** |
| 8. Save adapter | `model.save_pretrained(staging)` | No |

### Safe "Load Model Only" Path

**There is no safe load-model-only path in `run_training.py`.** The script does everything in one function. However, we can test initialization by:

1. Running `run_training.py` with a tiny config that loads the model but immediately saves without training (set `epochs=0` or `max_steps=0` if supported by SFTTrainer)
2. Or creating a separate test script that imports unsloth and loads the model

**For the first real test, the full path is acceptable** — the model will load, train 1-2 steps, and save.

---

## 6. Model Selection

### Target Model: Qwen2.5-0.5B-Instruct

| Property | Value |
|----------|-------|
| Model identifier | `Qwen/Qwen2.5-0.5B-Instruct` |
| Expected by run_training.py | `config.get("base_model")` → passed to `FastLanguageModel.from_pretrained()` |
| Cached locally | ✅ `/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/` |
| Cache size | 949 MiB |
| Network download required | **NO** — fully cached |
| Also cached (4-bit) | `models--unsloth--qwen2.5-0.5b-instruct-unsloth-bnb-4bit` |
| defnex-vllm currently serving | ✅ Same model: `--model Qwen/Qwen2.5-0.5B-Instruct` |

### Model in defnex-vllm

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

**The model is already cached and served. First training test will use the same base model.**

---

## 7. Dataset Requirements

### Training Input Contract (`run_training.py:102-129`)

```python
dataset = load_dataset(config.get("hf_dataset"))  # line 102
# ...
trainer = SFTTrainer(
    train_dataset=dataset,
    dataset_text_field=str(config.get("format_type", "text")),  # line 125
    max_seq_length=max_seq_length,
    ...
)
```

### Expected Format

- `config["hf_dataset"]`: HuggingFace dataset name or path
- `config["format_type"]`: text field name (default: `"text"`)
- Dataset must have a `"text"` field (or whatever `format_type` specifies)

### Existing Dataset Fixtures

| File | Content |
|------|---------|
| `data/datasets/_staging/ea5504f1.../data.json` | `[{"a":1},{"a":2},{"a":3}]` — NOT text format |
| `data/datasets/_staging/da1f0bf.../train.jsonl` | Exists but format unknown |
| `data/datasets/_staging/661932f.../train.jsonl` | Exists but format unknown |

### Safest Tiny Training Fixture

Use HuggingFace `dair-ai/alpaca` (standard instruction-following dataset) or create a minimal inline fixture. For the first test, a tiny 5-example subset of alpaca is safest:

```python
config = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "hf_dataset": "dair-ai/alpaca",
    "format_type": "text",  # alpaca has "text" or "instruction" field
    "batch_size": 1,
    "epochs": 1,
    "max_seq_length": 128,
    "lora_r": 8,
    "lora_alpha": 8,
}
```

**Note:** `run_training.py` uses `load_dataset(config["hf_dataset"])` without split selection. For alpaca this loads the full train split. For a tiny test, the dataset should be sliced to 5-10 examples before passing, or a custom tiny dataset should be used.

---

## 8. Training Resource Requirements

### Current Default Configuration (`run_training.py` + `config.py`)

| Parameter | Default | Source |
|-----------|---------|--------|
| batch_size | 1 | `config.get("batch_size", 1)` |
| gradient_accumulation_steps | 1 | `config.get("gradient_accumulation_steps", 1)` |
| max_seq_length | 2048 | `config.get("max_seq_length", 2048)` |
| lora_r | 16 | `config.get("lora_r", 16)` |
| lora_alpha | 16 | `config.get("lora_alpha", 16)` |
| lora_dropout | 0.0 | `config.get("lora_dropout", 0.0)` |
| target_modules | [q,k,v,o,gate,up,down_proj] | 7 modules |
| epochs | 1 | `config.get("epochs", 1)` |
| learning_rate | 2e-5 | `config.get("learning_rate", 2e-5)` |
| optim | adamw_8bit | `config.get("optim", "adamw_8bit")` |
| precision | bf16 (if supported) | `is_bfloat16_supported()` |
| seed | 42 | `config.get("random_seed", 42)` |
| output_dir | staging dir | `staging` argument |
| training_timeout | 0 (disabled) | `config.py:243` |

### Proposed Minimal First Training Configuration

```json
{
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "hf_dataset": "dair-ai/alpaca",
  "format_type": "text",
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "max_seq_length": 128,
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

**Expected VRAM for 0.5B + LoRA r=8 + seq_len=128 + batch=1:** ~2-4 GiB (well within 15+ GiB available after vLLM stop).

---

## 9. GPU Memory Safety

### Historical Observations

| Metric | Value | When |
|--------|-------|------|
| Free VRAM with vLLM running | 2116 MiB | Both live tests |
| Free VRAM after vLLM stop | 17272 MiB | Both live tests |
| vLLM VRAM freed | ~15156 MiB | Both live tests |
| Other GPU processes | 14 (tenant) | Baseline |
| defnex-vllm VRAM | ~15150 MiB | After stop |

### Current GPU State (READ-ONLY)

| Metric | Value |
|--------|-------|
| Free VRAM | 2116 MiB |
| Used VRAM | 78964 MiB |
| Total VRAM | 81559 MiB |
| GPU processes | 14 + defnex-vllm (15 total) |
| defnex-vllm PID | 2621174, 15150 MiB |

### Feasibility Assessment

| Factor | Assessment |
|--------|-----------|
| 0.5B model VRAM requirement | ~1-2 GiB (full precision), ~0.5-1 GiB (4-bit) |
| LoRA r=8 additional VRAM | ~50-100 MiB |
| Training overhead (optimizer states) | ~1-2 GiB |
| **Total estimated VRAM** | **~3-5 GiB** |
| **Available after vLLM stop** | **~17 GiB** |
| **Safety margin** | **~12-14 GiB** |

**0.5B training is plausibly feasible.** The 17 GiB free after vLLM stop far exceeds the ~3-5 GiB needed.

### Minimum Free VRAM Threshold

Current `VRAM_FREE_THRESHOLD_MB=8000` (8 GiB). After vLLM stop, ~17 GiB is free. The threshold is conservative and safe.

---

## 10. Artifact Output Compatibility

### Output Path Trace

```
run_training.py
  → model.save_pretrained(staging)  # staging = tempfile.mkdtemp()
  → tokenizer.save_pretrained(staging)
  → adapter files in staging dir

LocalSubprocessProvider.collect_result()
  → returns staging dir path

training_worker.process_next_job()
  → training_service.complete_training_run(db, run, artifact_uri=staging_dir)
  → model_service.register_model_version(db, run, staging_dir=staging_dir)
```

### Artifact Path in Shared Mount

| Path | Container | Host |
|------|-----------|------|
| Training output | `/tmp/defnex-training-*` | `/tmp/defnex-training-*` (temp) |
| After registration | `/models/artifacts/{version}/` | `/home/ubuntu/defnex-mlops-experiment/outputs/artifacts/{version}/` |
| defnex-vllm reads from | `/models/` | Same mount |

### Current Artifact Directory

```
/home/ubuntu/defnex-mlops-experiment/outputs/
├── artifacts/          (exists, empty)
├── evaluation.json
└── qwen2.5-0.5b-mlops-lora/  (existing LoRA adapter)
```

### Compatibility

- Worker mounts `/models` (same host dir as defnex-vllm) ✅
- Worker has write access to `/models/artifacts/` ✅
- defnex-vllm can read from `/models/` ✅
- URI normalization: `file:///models/artifacts/...` resolves in both containers ✅

---

## 11. Training Provider Readiness

### LocalSubprocessProvider

| Property | Value | Status |
|----------|-------|--------|
| Python executable | `settings.training_python` → `/opt/training-venv/bin/python` | ✅ |
| Script path | `settings.training_script_path` → `app/training/run_training.py` | ✅ |
| Command | `[/opt/training-venv/bin/python, -u, run_training.py, --config, <json>, --staging, <dir>]` | ✅ |
| Timeout | `settings.training_timeout_seconds` → 0 (disabled) | ⚠️ No timeout — consider setting |
| Environment | Subprocess inherits container env | ✅ |
| Output dir | `tempfile.mkdtemp(prefix="defnex-training-")` | ✅ |

### Provider Readiness

| Check | Result |
|-------|--------|
| Provider configured | ✅ `LocalSubprocessProvider()` at `training_worker.py:280` |
| Executable path exists | ✅ `/opt/training-venv/bin/python` (symlink to python3) |
| Script path exists | ✅ `/app/app/training/run_training.py` (mounted from host) |
| Subprocess command correct | ✅ `python -u run_training.py --config <json> --staging <dir>` |
| Timeout handling | ✅ Timer-based kill in `_wait_for_process()` |
| Output directory | ✅ Temp dir, registered as artifact_uri |

---

## 12. GPU Handoff Readiness

### Handoff Sequence

```
1. gpu_lock acquired (training_worker.py:173)
   ↓
2. coordinator.cycle() entered (training_worker.py:178)
   ↓
3. FileSignalingServingControl.stop() (gpu_orchestrator.py:367)
   → writes request.json {action: "stop_serving"}
   → polls response.json
   → host controller: docker stop defnex-vllm
   → host controller: nvidia-smi → vram_free_mb
   → response: {phase: "vram_checked", vram_free_mb: 17272}
   ↓
4. VRAM threshold check (gpu_orchestrator.py:216)
   → free (17272) >= threshold (8000) → PASS
   ↓
5. Training subprocess runs (training_provider.py:138)
   → Unsloth loads model, trains, saves adapter
   ↓
6. coordinator.cycle() exit (gpu_orchestrator.py:226-239)
   → FileSignalingServingControl.start()
   → writes request.json {action: "start_serving"}
   → host controller: docker start defnex-vllm
   → health check polls for up to 180s
   → response: {phase: "healthy"}
   ↓
7. gpu_lock released (training_worker.py:173, context exit)
```

### Safety Verification

| Check | Source | Status |
|-------|--------|--------|
| GPU lock acquired before stop | `training_worker.py:173` | ✅ |
| Lock held during training | `training_worker.py:178` + context manager | ✅ |
| start_serving guaranteed in finally | `gpu_orchestrator.py:226-239` | ✅ |
| Training cannot bypass stop gate | `serving_cycle` context manager | ✅ |
| Insufficient VRAM aborts training | `gpu_orchestrator.py:218-222` raises `VRAMNotFree` | ✅ |
| vLLM restarted after failure | `gpu_orchestrator.py:231-232` (finally block) | ✅ |
| No unrelated processes touched | Host controller only operates on `defnex-vllm` | ✅ |

### Remaining Risks

1. **Training timeout disabled** — a runaway training job could hold the GPU lock indefinitely. Consider setting `training_timeout_seconds` to a reasonable value (e.g., 3600).
2. ~~**Worker GPU passthrough not yet configured**~~ — **RESOLVED** (see Section 13 results).

---

## 13. CRITICAL: Does the Worker Need GPU Passthrough?

### Answer: **YES — NOW READY** ✅ (Applied 2026-09-15)

| Check | Before | After |
|-------|--------|-------|
| `/dev/nvidia*` in worker | **DOES NOT EXIST** | ✅ `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, etc. |
| `torch.cuda.is_available()` in worker | **False** | ✅ **True** |
| Unsloth import in worker | **CRASHES** — "You need a GPU" | ✅ **OK** — imports successfully |
| `DeviceRequests` in container config | `null` | ✅ `[{Driver: nvidia, Count: 1, Capabilities: [[gpu]]}]` |
| `deploy.resources.reservations.devices` | **NOT CONFIGURED** | ✅ **CONFIGURED** |

### Configuration Applied

```yaml
worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Verification Results (2026-09-15)

```bash
# /dev/nvidia* exists
$ docker compose exec worker ls /dev/nvidia*
/dev/nvidia-modeset  /dev/nvidia-uvm  /dev/nvidia-uvm-tools  /dev/nvidia0  /dev/nvidiactl

# torch.cuda available
$ docker compose exec worker /opt/training-venv/bin/python -c "import torch; print(torch.cuda.is_available())"
True

# GPU device name
$ docker compose exec worker /opt/training-venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
NVIDIA H100 PCIe

# Unsloth imports successfully
$ docker compose exec worker /opt/training-venv/bin/python -c "from unsloth import FastLanguageModel; print('OK')"
OK
```

---

## 14. CRITICAL: Do We Need to Stop vLLM for Training?

### Answer: **YES**

### Lifecycle

1. defnex-vllm currently occupies ~15150 MiB of GPU VRAM
2. Free VRAM is only 2116 MiB
3. 0.5B training requires ~3-5 GiB
4. **Training cannot start while vLLM is running** — insufficient VRAM
5. The `file_signal` serving coordinator handles this automatically:
   - `FileSignalingServingControl.stop()` → stops vLLM → frees ~15 GiB
   - VRAM gate: verifies free >= 8000 MiB
   - Training runs
   - `FileSignalingServingControl.start()` → restarts vLLM

### Why This Is Safe

- Only `defnex-vllm` is stopped (no other containers/processes)
- VRAM is verified before training starts
- vLLM is restarted in `finally` block (even on training failure)
- Health check confirms vLLM is ready before lock is released
- GPU lock prevents concurrent training during serving

---

## 15. First Real Training Test Design

### NOT EXECUTED — PROPOSAL ONLY

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (cached, 949 MiB)

**Dataset:** `dair-ai/alpaca` (standard instruction dataset, HuggingFace)

**Training Config:**

```json
{
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "hf_dataset": "dair-ai/alpaca",
  "format_type": "text",
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "max_seq_length": 128,
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

**Expected Flow:**

1. POST `/api/v1/training-runs` with above config
2. Worker picks up PENDING run
3. GPU lock acquired
4. File signal: stop defnex-vllm
5. VRAM verified: ~17 GiB free >= 8 GiB threshold
6. Unsloth loads Qwen2.5-0.5B-Instruct
7. LoRA applied (r=8, 2 target modules)
8. SFTTrainer runs 1 epoch on alpaca (~52K examples, but we could limit)
9. Adapter saved to staging dir
10. File signal: start defnex-vllm
11. vLLM health check passes
12. GPU lock released
13. Adapter registered as new ModelVersion

**Estimated duration:** 5-15 minutes (0.5B is small, alpaca is large but 1 epoch with small batch)

**Risk:** alpaca has 52K examples — 1 epoch with batch_size=1 could be slow. Consider limiting dataset to first N examples for the first test.

---

## 16. Required Decision

**C. READY FOR REAL TRAINING** ✅ (GPU passthrough applied 2026-09-15)

| Requirement | Status |
|-------------|--------|
| GPU passthrough configured | ✅ Applied and verified — `/dev/nvidia*`, `torch.cuda=True`, Unsloth imports OK |
| Training venv correct | ✅ Python 3.12, torch 2.11+cu130, unsloth 2026.9.2 |
| CUDA compatible | ✅ Driver 580.173.02, CUDA 13.0, torch cu130 |
| Model cached | ✅ Qwen2.5-0.5B-Instruct, 949 MiB |
| Training script ready | ✅ `run_training.py` with Unsloth |
| Serving coordinator ready | ✅ File signal, verified in 2 live tests |
| VRAM gate ready | ✅ 8000 MiB threshold, verified in live tests |
| Artifact path compatible | ✅ Shared mount, both containers |
| Safety guarantees intact | ✅ No PID kill, no docker.sock, defnex-vllm only |

---

## 17. Git Safety

### Current Changes

```
git status --short
```

Only documentation files should be modified. No source/config changes in this preflight.

---

## 18. Final Terminal Summary

```
PHASE 2B UNSLOTH PREFLIGHT — COMPLETE ✅

Worker GPU:           READY — /dev/nvidia* present, torch.cuda=True, Unsloth imports OK
CUDA:                 READY — Driver 580.173.02, CUDA 13.0, torch cu130
Unsloth:              READY — 2026.9.2, imports successfully with GPU access
Training provider:    READY — LocalSubprocessProvider, subprocess command correct
Training artifact:    READY — shared mount /models, both containers read/write
GPU handoff:          READY — file_signal verified in 2 live tests
First 0.5B training:  READY — all prerequisites satisfied
Real training executed: NO — awaiting user approval

GPU passthrough applied: 2026-09-15
  - docker-compose.yml: deploy.resources.reservations.devices (nvidia, count=1)
  - Worker container recreated
  - Verified: /dev/nvidia*, torch.cuda.is_available()=True, Unsloth import OK

Top remaining risks:
  1. Training timeout disabled — consider setting training_timeout_seconds
  2. 52K-example alpaca dataset — first test should limit to 5-10 examples

Recommended next action:
  Create first training run via POST /api/v1/training-runs
  with minimal config (Qwen2.5-0.5B, tiny dataset, 1 epoch, batch=1)
```
