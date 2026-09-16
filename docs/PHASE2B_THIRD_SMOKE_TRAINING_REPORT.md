# Phase 2B — Third Smoke Training Run Report

## Run

| Field | Value |
|-------|-------|
| TrainingRun ID | `run-9e66b6` |
| Created | 2026-09-16T02:13:15 |
| Retry of | `run-2c2b60` (retry of `run-3e7d46` (retry of `run-21a878` (retry of `run-f06ff2`))) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `ds-smoke-test-v1` |
| Examples | 10 |
| Max steps | 5 |
| Status | **COMPLETED** |

## Exact Config

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

## Pre-Run GPU Baseline

| Metric | Value |
|--------|-------|
| GPU | NVIDIA H100 PCIe |
| Driver | 580.173.02 |
| CUDA | 13.0 |
| VRAM used (pre) | 78964 MiB |
| VRAM total | 81559 MiB |
| VRAM free (pre) | 2595 MiB |
| GPU temp | 61°C |
| Power | 101W / 350W |
| vLLM | Running, health 200 |

## Lifecycle Observed

1. TrainingRun `run-9e66b6` created at 02:13:15 (status: PENDING)
2. Worker claimed run at 02:13:26 (status: RUNNING)
3. File signal stopped defnex-vllm at 02:13:18
4. vLLM stop confirmed at 02:13:26 (VRAM free: 17272 MiB)
5. VRAM gate passed (17272 MiB >= 8000 MiB threshold)
6. Training subprocess started (pid=46, job-80dcddc1)
7. Unsloth imported successfully
8. Qwen2.5-0.5B model loaded
9. LoRA applied (540,672 trainable parameters of 494,573,440 total, 0.11%)
10. Tokenizer configured with eos_token from model
11. SFTConfig created with bf16=True (H100 supports bfloat16)
12. SFTTrainer initialized with _ProgressCallback
13. trainer.train() executed
14. **5 training steps completed**
15. Adapter saved to staging directory
16. Artifact registered at `file:///models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1`
17. ModelVersion created: `smoke-llm-v3` v1 (status: REGISTERED)
18. File signal started defnex-vllm at 02:16:31
19. vLLM health: **200 (recovered)** at 02:18:25
20. GPU lock released

## Training Duration

| Metric | Value |
|--------|-------|
| Started | 2026-09-16 02:13:26 |
| Finished | 2026-09-16 02:16:31 |
| Duration | ~185 seconds (3 min 5 sec) |

## Post-Run State

| Check | Result |
|-------|--------|
| Final status | COMPLETED |
| Actual training steps | 5 (of 5 configured) |
| trainer.train() called | YES |
| ModelVersion | smoke-llm-v3 v1 (REGISTERED) |
| Artifact URI | `file:///models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1` |
| vLLM health | 200 (recovered) |
| GPU lock | Released |
| VRAM (post) | 78964 MiB (returned to baseline) |
| GPU temp | 60°C |
| Unrelated GPU processes | No running GPU compute processes (same as baseline) |

## Artifact Verification

Artifact path: `/home/ubuntu/defnex-mlops-experiment/outputs/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1/`

| File | Size | Present |
|------|------|---------|
| adapter_model.safetensors | 2,175,168 bytes | YES |
| adapter_config.json | 1,219 bytes | YES |
| tokenizer.json | 11,422,166 bytes | YES |
| tokenizer_config.json | 4,383 bytes | YES |
| chat_template.jinja | 2,507 bytes | YES |
| metadata.json | 1,349 bytes | YES |
| README.md | 1,571 bytes | YES |
| checkpoint-5/ | directory | YES |

## Bugs Fixed During This Run

### Bug 1: Missing `on_init_end` (from second smoke run)
- **Status**: Already fixed before this run
- **Fix**: Added `on_init_end` no-op to `_ProgressCallback`

### Bug 2: Missing `on_train_begin`
- **Error**: `'_ProgressCallback' object has no attribute 'on_train_begin'`
- **Detected in**: run-f06ff2 (first attempt of third smoke)
- **Fix**: Added `on_train_begin` no-op to `_ProgressCallback`

### Bug 3: Missing multiple callback lifecycle methods
- **Error**: `'_ProgressCallback' object has no attribute 'on_epoch_begin'`
- **Detected in**: run-21a878 (second attempt)
- **Fix**: Added ALL TrainerCallback lifecycle hooks as no-ops to `_ProgressCallback`:
  - `on_init_end`, `on_train_begin`, `on_train_end`
  - `on_epoch_begin`, `on_epoch_end`
  - `on_step_begin`, `on_step_end`, `on_substep_end`
  - `on_log` (real logic), `on_evaluate`, `on_save`
  - `on_predict`, `on_prediction_step`
  - `on_optimizer_step`, `on_pre_optimizer_step`, `on_push_begin`

### Bug 4: Missing C compiler for Triton
- **Error**: `Failed to find C compiler` / `stdlib.h: No such file or directory`
- **Detected in**: run-2c2b60 (third attempt)
- **Fix**: Added `build-essential` to Dockerfile (includes gcc + libc-dev headers)

## Success Criteria Verification

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Exactly one NEW TrainingRun created | YES (`run-9e66b6`) |
| 2 | Worker claimed it | YES |
| 3 | GPU lock acquired | YES |
| 4 | defnex-vllm stopped through file signaling | YES |
| 5 | VRAM gate passed | YES (17272 MiB) |
| 6 | Qwen2.5-0.5B loaded | YES |
| 7 | LoRA applied | YES (540,672 params) |
| 8 | SFTTrainer initialized | YES |
| 9 | trainer.train() executed | YES |
| 10 | Exactly 5 steps completed | YES |
| 11 | Adapter artifact created | YES |
| 12 | ModelVersion created/registered | YES (`smoke-llm-v3` v1) |
| 13 | defnex-vllm restarted automatically | YES |
| 14 | vLLM health = HTTP 200 | YES |
| 15 | GPU lock released | YES |
| 16 | All unrelated GPU processes survived | YES |

## GPU Safety

No GPU reset, kill, or destructive operation performed.
defnex-vllm recovered through the existing file-signal GPU controller lifecycle.

## Status

**PASS**

All 16 success criteria verified with observed evidence.
The complete end-to-end training lifecycle is now functional.
