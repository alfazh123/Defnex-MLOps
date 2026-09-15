"""Standalone Unsloth training script (issue #38).

Executes REAL Unsloth fine-tuning in a SEPARATE venv/subprocess. The app/serving venv never
imports Unsloth (serving and training venvs are kept apart per the project constraint); the
`UnslothTrainingRunner` spawns this with:

    python -u run_training.py --config '<json>' --staging <dir>

Subprocess contract (issue #38):
  * stdout is newline-delimited JSON:
        {"event": "progress", "epoch": 1, "step": 100, "train_loss": 0.5, "eval_loss": 0.4}
        {"event": "done", "files": ["adapter_model.safetensors", ...]}
  * exit code 0 on success; non-zero with a stderr reason (e.g. a CUDA OOM) on failure.
  * the trained LoRA adapter is written into `--staging`. The worker later moves the staging
    directory into its immutable per-version location and records metadata.

Only ever run against a real GPU in the training venv — never during tests/CI (the runner is
tested against a fake script instead).
"""

import argparse
import json
import sys


def _emit(**payload) -> None:
    print(json.dumps(payload), flush=True)


class _ProgressCallback:
    """Stream step/epoch/loss progress to stdout for the runner to persist (issue #38)."""

    def __init__(self):
        self._epoch = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        _emit(
            event="progress",
            epoch=self._epoch,
            step=state.global_step,
            train_loss=logs.get("loss") if logs else None,
            eval_loss=logs.get("eval_loss") if logs else None,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        self._epoch += 1


def _run_training(config: dict, staging: str) -> int:
    try:
        from datasets import load_dataset

        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as exc:
        print(
            f"training environment missing a dependency: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    base_model = config.get("base_model") or config.get("model_name")
    if not base_model:
        print("missing base_model in config", file=sys.stderr, flush=True)
        return 2

    max_seq_length = int(config.get("max_seq_length", 2048))
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=config.get("load_in_4bit", False)
            or config.get("peft_method") == "qlora",
            trust_remote_code=config.get("trust_remote_code", False),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(config.get("lora_r", 16)),
            target_modules=config.get("target_modules")
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=int(config.get("lora_alpha", 16)),
            lora_dropout=float(config.get("lora_dropout", 0.0)),
            use_gradient_checkpointing=config.get("gradient_checkpointing", False),
            use_rslora=config.get("use_rslora", False)
            or config.get("peft_method") == "rslora",
        )
    except Exception as exc:  # CUDA OOM surfaces here as torch.cuda.OutOfMemoryError
        print(f"unable to load model for training: {exc}", file=sys.stderr, flush=True)
        return 3

    import os

    hf_dataset = config.get("hf_dataset")
    if os.path.isfile(hf_dataset):
        dataset = load_dataset("json", data_files=hf_dataset, split="train")
    else:
        dataset = load_dataset(hf_dataset)

    training_args = SFTConfig(
        output_dir=staging,
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        warmup_ratio=config.get("warmup_ratio"),
        num_train_epochs=int(config.get("epochs", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        max_steps=int(config.get("max_steps", -1)),
        logging_steps=1,
        optim=str(config.get("optim", "adamw_8bit")),
        weight_decay=float(config.get("weight_decay", 0.001)),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "linear")),
        seed=int(config.get("random_seed", 42)),
        report_to=[],
        dataset_text_field=str(config.get("format_type", "text")),
        max_length=max_seq_length,
        packing=bool(config.get("packing", False)),
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.eos_token,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=[_ProgressCallback()],
    )
    try:
        trainer.train()
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr, flush=True)
        return 4

    model.save_pretrained(staging)
    tokenizer.save_pretrained(staging)
    _emit(
        event="done",
        files=["adapter_model.safetensors", "adapter_config.json"],
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON training_config dict")
    parser.add_argument("--staging", required=True, help="output staging directory")
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as exc:
        print(f"invalid --config json: {exc}", file=sys.stderr, flush=True)
        return 2

    return _run_training(config, args.staging)


if __name__ == "__main__":
    sys.exit(main())
