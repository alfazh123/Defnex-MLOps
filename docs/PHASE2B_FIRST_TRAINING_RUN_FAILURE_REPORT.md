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

## Verification

A non-training verification was performed using the production training environment.

The verification successfully:

1. Loaded Qwen2.5-0.5B-Instruct.
2. Applied LoRA.
3. Loaded the smoke dataset.
4. Loaded exactly 10 examples.
5. Created SFTConfig with eos_token set from tokenizer.eos_token.
6. Created SFTTrainer successfully.

trainer.train() was NOT called.

## Dataset

Path:

/app/data/datasets/smoke_train.jsonl

Example count:

10

## Focused Tests

185 tests passed.

## Training Rerun

No.

## Second TrainingRun

Not created.

## GPU Safety

No GPU reset or destructive GPU operation was performed as part of the EOS fix verification.

The existing GPU handoff architecture was not changed.

## Current Status

READY FOR SECOND SMOKE RUN
