"""Seed the MLOps backend with a realistic 4-dataset / 4-model closed-loop for UI preview.

Runs against the local SQLite DB via the app's own service layer (deterministic, no worker
timing). Idempotent: pass --reset to wipe existing dataset/training/model/junk data first.
Auth users are preserved.

Usage (from repo root, inside the venv):
    .venv/bin/python seed.py --reset
"""

import sys

from sqlalchemy import delete

from app.db.session import SessionLocal
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.schemas.model import (
    EvalLossTrend,
    GeneralDomainRegressionCheck,
    QualitativeComparison,
    EvaluationUpdateRequest,
)
from app.schemas.promotion import DecisionCreateRequest
from app.schemas.eval_set import EvalSetVersionCreateRequest
from app.models import deployment as deployment_models
from app.models import promotion as promotion_models
from app.models.validation import ValidationReport
from app.models.model import Model, ModelVersion
from app.models.training import TrainingRun
from app.models.dataset import Dataset, DatasetVersion
from app.services import (
    dataset_service,
    validation_service,
    training_service,
    model_service,
    promotion_service,
    deployment_service,
    eval_set_service,
)

ADMIN = "mockadmin"


def reset(db):
    """Wipe dataset/training/model data (junk + prior seeds). Keeps auth users intact."""
    for model in (
        promotion_models.PromotionDecision,
        deployment_models.Deployment,
        ModelVersion,
        Model,
        TrainingRun,
        ValidationReport,
        DatasetVersion,
        Dataset,
    ):
        db.execute(delete(model))
    db.commit()


def row_counts():
    return {
        "ds-support-indo-v1": {1: 12450, 2: 15200},
        "ds-legal-qa-v1": {1: 8600},
        "ds-orca-reasoning-v1": {1: 18200},
        "ds-ultrachat-general-v1": {1: 28600},
    }


def sample_records(n=120, src="ds-support-indo-v1"):
    """Build `n` valid records so the validation gate reports a non-zero VALID count.
    Storage of real record content doesn't exist yet, so this stands in for the (empty)
    intake pipeline just to make the validation report/pages display real numbers."""
    recs = []
    for i in range(n):
        assistant = (
            "Silakan hubungi layanan pelanggan kami melalui aplikasi pada menu bantuan, "
            "atau kunjungi pusat bantuan di situs resmi untuk mengajukan keluhan dan "
            "menyampaikan masukan yang lebih lanjut kepada tim kami."
        )
        recs.append(
            {
                "id": f"rec-{src}-{i}",
                "messages": [
                    {"role": "user", "content": f"Pertanyaan contoh nomor {i}?"},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {"source_dataset": src, "source_id": f"src-{i}"},
            }
        )
    return recs


def sample_eval_records(n=20):
    """Golden/eval-set records (issue #43) for the eval gate (issue #128). Content must not
    overlap any validated training record's user content (H8 leakage), so this uses a distinct
    question template from `sample_records`."""
    recs = []
    for i in range(n):
        recs.append(
            {
                "id": f"eval-rec-{i}",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Pertanyaan evaluasi golden-set nomor {i}?",
                    },
                    {
                        "role": "assistant",
                        "content": "Jawaban rujukan untuk pertanyaan evaluasi ini.",
                    },
                ],
                "metadata": {"source_id": f"eval-src-{i}"},
            }
        )
    return recs


def make_dataset(
    db, dataset_id, source, fmt, versions, created_by=ADMIN, commit_date="2026-08-10"
):
    for v in versions:
        req = DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset=source,
            source_commit_or_snapshot_date=commit_date,
            source_format=fmt,
        )
        dv = dataset_service.create_dataset_version(db, dataset_id, req)
        dv.row_count = row_counts()[dataset_id][v]
        dv.cleaning_steps_applied = [
            "dedupe_by_normalized_pair",
            "strip_markdown_links_keep_label",
            "cut_at_boilerplate_markers",
        ]
        dv.created_by = created_by
        db.flush()
        # validation gate (PASS: no leakage, sample valid records -> non-zero VALID count)
        validation_service.validate_dataset_version(
            db, dv, records=sample_records(120, dataset_id)
        )
    db.commit()


def make_run(
    db, run_id, dataset_id, version, model_id, base_model, loss_cfg, triggered_by=ADMIN
):
    req = TrainingRunCreateRequest(
        dataset_id=dataset_id,
        dataset_version=version,
        model_id=model_id,
        base_model=base_model,
        training_config=TrainingConfig(
            hf_dataset=dataset_id,
            format_type=loss_cfg["format_type"],
            learning_rate=loss_cfg["lr"],
            epochs=loss_cfg["epochs"],
            lora_r=16,
            lora_alpha=16,
            max_seq_length=2048,
        ),
        triggered_by=triggered_by,
    )
    dv = dataset_service.get_dataset_version(db, dataset_id, version)
    run = training_service.create_training_run(db, dv, req)
    training_service.start_training_run(db, run)
    training_service.complete_training_run(
        db,
        run,
        artifact_uri=f"s3://defnex-mlops/artifacts/runs/{run_id}/adapter_model.bin",
    )
    run.train_loss = loss_cfg["train_loss"]
    run.eval_loss = loss_cfg["eval_loss"]
    run.current_epoch = loss_cfg["epochs"]
    run.current_step = loss_cfg["steps"]
    db.flush()
    model_service.register_model_version(db, run)
    db.commit()
    return run


def make_evaluation(
    db,
    model_id,
    version,
    trend,
    quality,
    regression,
    evaluator=ADMIN,
    eval_set_id=None,
    eval_set_version=None,
):
    mv = model_service.get_model_version(db, model_id, version)
    model_service.submit_evaluation(
        db,
        mv,
        EvaluationUpdateRequest(
            eval_set_id=eval_set_id,
            eval_set_version=eval_set_version,
            eval_loss_trend=EvalLossTrend(
                previous_version_eval_loss=trend["prev"],
                this_version_eval_loss=trend["this"],
            ),
            qualitative_comparison=QualitativeComparison(
                question_table_version=1,
                wins=quality["wins"],
                losses=quality["losses"],
                ties=0,
                total=quality["wins"] + quality["losses"],
            ),
            general_domain_regression_check=GeneralDomainRegressionCheck(
                checked=True,
                regressions_found=regression,
            ),
        ),
    )
    db.commit()
    return mv


def make_decision(db, model_id, version, decision, rationale, decided_by=ADMIN):
    mv = model_service.get_model_version(db, model_id, version)
    promotion_service.create_decision(
        db,
        mv,
        DecisionCreateRequest(
            decision=decision, decided_by=decided_by, rationale=rationale
        ),
    )
    db.commit()


def deploy(db, model_id, version):
    mv = model_service.get_model_version(db, model_id, version)
    deployment_service.deploy(db, mv)
    db.commit()


def main():
    if "--reset" not in sys.argv:
        print("pass --reset to wipe existing dataset/training/model data first")
    db = SessionLocal()
    try:
        if "--reset" in sys.argv:
            reset(db)

        # Datasets (each v1 PROCESSED + validated; support gets v2 for versioned intake)
        make_dataset(
            db,
            "ds-support-indo-v1",
            "databricks/databricks-dolly-15k",
            "sharegpt",
            [1, 2],
        )
        make_dataset(db, "ds-legal-qa-v1", "tatsu-lab/alpaca", "alpaca", [1])
        make_dataset(db, "ds-orca-reasoning-v1", "Open-Orca/OpenOrca", "chatml", [1])
        make_dataset(
            db,
            "ds-ultrachat-general-v1",
            "HuggingFaceH4/ultrachat_200k",
            "sharegpt",
            [1],
        )

        # Training runs -> registered model versions
        make_run(
            db,
            "run-support-01",
            "ds-support-indo-v1",
            1,
            "defnex-support-llm",
            "unsloth/Qwen2.5-7B-Instruct",
            dict(
                format_type="chatml",
                lr=2e-4,
                epochs=3,
                train_loss=0.428,
                eval_loss=0.492,
                steps=934,
            ),
        )
        make_run(
            db,
            "run-legal-01",
            "ds-legal-qa-v1",
            1,
            "defnex-legal-llm",
            "unsloth/Qwen2.5-7B-Instruct",
            dict(
                format_type="alpaca",
                lr=2e-4,
                epochs=2,
                train_loss=0.511,
                eval_loss=0.578,
                steps=672,
            ),
        )
        make_run(
            db,
            "run-orca-01",
            "ds-orca-reasoning-v1",
            1,
            "defnex-orca-reasoner",
            "unsloth/Qwen3-8B-Instruct",
            dict(
                format_type="chatml",
                lr=1.5e-4,
                epochs=3,
                train_loss=0.372,
                eval_loss=0.815,
                steps=1428,
            ),
        )
        make_run(
            db,
            "run-ultra-01",
            "ds-ultrachat-general-v1",
            1,
            "defnex-general-llm",
            "unsloth/Llama-3.2-3B-Instruct",
            dict(
                format_type="chatml",
                lr=2e-4,
                epochs=1,
                train_loss=0.601,
                eval_loss=0.664,
                steps=566,
            ),
        )

        # Golden/eval set (issue #43) -- required by the promotion eval gate (issue #128).
        eval_version = eval_set_service.create_eval_set_version(
            db,
            "golden-eval-v1",
            EvalSetVersionCreateRequest(records=sample_eval_records()),
        )
        db.commit()

        # Evaluations (3 good -> EVALUATED, 1 weak stays via REJECT below)
        make_evaluation(
            db,
            "defnex-support-llm",
            1,
            dict(prev=0.87, this=0.49),
            dict(wins=78, losses=22),
            [],
            eval_set_id="golden-eval-v1",
            eval_set_version=eval_version.version,
        )
        make_evaluation(
            db,
            "defnex-general-llm",
            1,
            dict(prev=0.71, this=0.66),
            dict(wins=55, losses=45),
            [],
            eval_set_id="golden-eval-v1",
            eval_set_version=eval_version.version,
        )
        make_evaluation(
            db,
            "defnex-orca-reasoner",
            1,
            dict(prev=0.40, this=0.82),
            dict(wins=30, losses=70),
            [],
            eval_set_id="golden-eval-v1",
            eval_set_version=eval_version.version,
        )

        # Decisions + deploy
        make_decision(
            db,
            "defnex-support-llm",
            1,
            "PROMOTED",
            "Eval loss turun 0.87->0.49, menang 78/100, tanpa regresi domain umum.",
        )
        deploy(db, "defnex-support-llm", 1)
        make_decision(
            db,
            "defnex-general-llm",
            1,
            "PROMOTED",
            "Loss stabil menurun, kualitas sedikit lebih baik dari base, tanpa regresi.",
        )
        make_decision(
            db,
            "defnex-orca-reasoner",
            1,
            "REJECTED",
            "Eval loss naik (0.40->0.82) dan kalah 30/100; overfit pada subset reasoning.",
        )
        # defnex-legal-llm intentionally left REGISTERED (no eval yet)

        print(
            "done. 4 datasets, 4 training runs, 4 models (1 DEPLOYED, 1 PROMOTED, 1 REJECTED, 1 REGISTERED)"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
