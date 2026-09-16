# Phase 2B — First Training Run Failure Report

## Failure

The first real smoke-training attempt failed during SFTTrainer initialization.

## Error

ValueError: The specified eos_token (<|EOS_TOKEN|>) is not found in the vocabulary of the given processing_class (Qwen2Tokenizer).

## Root Cause

The production training path supplied the invalid placeholder:

<|EOS_TOKEN|>

The Qwen2Tokenizer does not contain this token.

## Actual Tokenizer

eos_token = <|im_end|>
eos_token_id = 151645

The token <|EOS_TOKEN|> is not present in the tokenizer vocabulary.

## Fix

The training path was changed to use:

tokenizer.eos_token

for the SFT configuration.

For Qwen2.5-0.5B-Instruct, this resolves to:

<|im_end|>

No new tokenizer token was added and the vocabulary was not resized.


## Import Order Fix

Unsloth must be imported before trl, transformers, peft. The original code imported
`datasets` and `trl` before `unsloth`, producing a runtime warning and potential
patching issues. The import order in `_run_training()` was corrected to import
`unsloth` first.

Before:
```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
```

After:
```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
```

## Verification

A non-training verification was performed using the production training environment.

The verification successfully:

1. Imported unsloth first, then datasets, then trl (correct order).
2. Loaded Qwen2.5-0.5B-Instruct.
3. Applied LoRA.
4. Loaded the smoke dataset.
5. Loaded exactly 10 examples.
6. Created SFTConfig with eos_token set from tokenizer.eos_token.
7. Created SFTTrainer successfully.

trainer.train() was NOT called.

## Dataset

Path:

/app/data/datasets/smoke_train.jsonl

Example count:

10

## Focused Tests

156 tests passed (focused subset: file_signaling, gpu_orchestration, training_worker, training_provider, training_service).

## Training Rerun

No.

## Second TrainingRun

Not created.

## GPU Safety

No GPU reset or destructive GPU operation was performed as part of the EOS fix verification.

The existing GPU handoff architecture was not changed.

## Current Status

READY FOR SECOND SMOKE RUN
