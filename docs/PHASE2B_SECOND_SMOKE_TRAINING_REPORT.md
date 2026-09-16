# Phase 2B — Second Smoke Training Run Report

## Run

| Field | Value |
|-------|-------|
| TrainingRun ID | `run-a47b70` |
| Created | 2026-09-16T01:13:53 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `ds-smoke-test-v1` |
| Examples | 10 |
| Max steps | 5 |
| Status | **FAILED** |

## Lifecycle Observed

1. Worker claimed `run-a47b70` at 01:13:55
2. File signal stopped defnex-vllm at 01:13:55
3. VRAM gate passed (17272 MB free)
4. Training subprocess started (pid=1248, job-0ec84915)
5. Qwen2.5-0.5B model loading reached 100%
6. SFTTrainer.__init__ failed
7. Training subprocess exited with code 1 at 01:14:56
8. File signal restarted defnex-vllm at 01:14:56
9. vLLM health check: **200 (recovered)**

## Error

```
AttributeError: '_ProgressCallback' object has no attribute 'on_init_end'
```

Full traceback:
```
File "/opt/training-venv/lib/python3.12/site-packages/transformers/trainer.py", line 595
    self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
File "/opt/training-venv/lib/python3.12/site-packages/transformers/trainer_callback.py", line 488
    return self.call_event("on_init_end", args, state, control)
File "/opt/training-venv/lib/python3.12/site-packages/transformers/trainer_callback.py", line 545
    result = getattr(callback, event)(args, state, control, **kwargs)
AttributeError: '_ProgressCallback' object has no attribute 'on_init_end'
```

## Root Cause

`_ProgressCallback` in `run_training.py` does not inherit from `transformers.TrainerCallback`.
Transformers 5.5.0's `Trainer.__init__` calls `callback_handler.on_init_end()` which dispatches
to all registered callbacks. The `_ProgressCallback` class only defined `on_log` and
`on_epoch_end` — it was missing `on_init_end`.

## Fix Applied

Added `on_init_end` as a no-op method to `_ProgressCallback` in `run_training.py:33`:

```python
def on_init_end(self, args, state, control, **kwargs):
    pass
```

Verified in worker container:
```
RESULT: PASS - _ProgressCallback has on_init_end
```

## Pre-Run GPU Baseline

| Metric | Value |
|--------|-------|
| GPU | NVIDIA H100 PCIe |
| VRAM used (pre) | 78964 MiB |
| VRAM total | 81559 MiB |
| VRAM free (pre) | 2116 MiB |
| Tenant processes | 14 |

## Post-Run State

| Check | Result |
|-------|--------|
| vLLM health | 200 (recovered) |
| GPU lock | Released |
| Unrelated GPU processes | All 14 still present |
| New TrainingRun created | NO (only run-a47b70) |
| trainer.train() called | NO (failed before training started) |

## GPU Safety

No GPU reset, kill, or destructive operation performed.
defnex-vllm recovered through the existing file-signal GPU controller lifecycle.

## Status

**FAILED — BUG FIXED — REQUIRES THIRD SMOKE RUN**

Per the failure rule: no additional TrainingRun was automatically created.
The `_ProgressCallback` fix has been applied and verified.
A third smoke run is needed to validate the complete training path.
