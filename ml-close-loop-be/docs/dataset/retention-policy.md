# Dataset / Model Artifact Retention Policy

Issue #132. Written to close a gap named in that issue: PRD §43 states only a *negative*
retention rule ("don't delete rollback artifacts") with no positive policy for when
rejected/superseded data may be deleted.

## Governing rule (never overridden)

> Artifacts required for rollback must not be deleted merely because a model is no longer
> active. (PRD §43, `DEFNEX_MLOps_Multi_Server_Architecture_v2_PRD.md:2122-2124`)

Every minimum below is a floor, not a target: an artifact that could still be needed for
rollback stays, regardless of how old it is. Concretely, a `ModelVersion` in `PROMOTED`,
`DEPLOYED`, or `RETIRED` status (`app/services/model_service.py`) is a rollback candidate and
is out of scope for deletion entirely — only `REJECTED` and `ARCHIVED` versions are ever
eligible, and `ARCHIVED` already exists precisely as a reversible "soft-hide that keeps the
artifact intact for rollback" state (`app/services/model_service.py:312-334`).

## Scope: what "artifact type" covers here

| Artifact type | Status(es) eligible for retention aging | Config setting |
|---|---|---|
| Dataset version, validation-rejected | `DatasetVersion` rows whose only validation report has `gate_decision == "FAIL"` and were never committed (`app/services/validation_service.py:229-236`, `app/api/intake_validate.py:191-192`) | `retention_rejected_dataset_days` (default 90) |
| Dataset version, superseded | An older `DatasetVersion` for the same `dataset_id` once a newer version has been committed (`dataset_service._allocate_version`, `app/models/dataset.py:23-60`) | `retention_superseded_dataset_days` (default 180) |
| Model version, rejected | `ModelVersion.status == "REJECTED"` (`app/services/promotion_service.py`) | `retention_rejected_model_version_days` (default 90) |
| Model version, archived | `ModelVersion.status == "ARCHIVED"` (`app/services/model_service.py:319-334`) | `retention_archived_model_version_days` (default 365) |

Abandoned intake attempts — a `DatasetVersion` created at `POST /datasets/intake/validate` time
(`app/api/intake_validate.py:88-100`) whose staging was never committed — are also, in effect,
"rejected dataset" candidates under the same `retention_rejected_dataset_days` window; they are
called out here because nothing currently marks or cleans them up.

## Minimum retention windows

Defaults live in `app/config.py` (`retention_*` settings) and `.env.example`. They are **not**
values named anywhere in the PRD — PRD §43 gives no positive number — so treat them as
conservative, operator-overridable defaults, not a governance-approved figure. Raise or lower
them via environment variables before any deletion tooling is ever built against them.

- `RETENTION_REJECTED_DATASET_DAYS=90`
- `RETENTION_SUPERSEDED_DATASET_DAYS=180`
- `RETENTION_REJECTED_MODEL_VERSION_DAYS=90`
- `RETENTION_ARCHIVED_MODEL_VERSION_DAYS=365`

## Deletion: explicitly out of scope for this issue

This codebase does not implement any code path that deletes a dataset version, model version,
or artifact bytes. The settings above define eligibility windows only — no scheduled job, admin
endpoint, or CLI reads them yet. That is a deliberate scope decision, not an oversight:

- Issue #132's acceptance criteria make an audit-logged deletion path conditional ("if
  implemented at all"), not mandatory.
- Batch 9 (issues #129/#130) is scoped to build the real `audit_service.py`. Building a
  deletion path now would mean inventing a one-off, throwaway audit mechanism ahead of that
  work — exactly what issue #132 says not to preempt.

**Whoever implements deletion later must, at minimum:**

1. Only ever delete artifacts strictly older than the relevant `retention_*` window, and never
   an artifact in a rollback-eligible status (`PROMOTED`/`DEPLOYED`/`RETIRED`, or a dataset
   version that is the current `canonical_file_uri` source for training).
2. Require an explicit authorization step before deletion executes — mirroring the existing
   `require_admin` pattern used for promotion/deploy (`app/api/deps.py`), not an unattended
   sweep with no approver.
3. Write an auditable record of the deletion (who/when/what/why) before or atomically with the
   delete, reusing Batch 9's `audit_service.py` once it exists. Until then, do not build a
   parallel ad-hoc audit table for this alone — that duplicates state (see project rule: one
   source of truth per fact) and creates a second thing to migrate to the real service later.

## PII-screening (related, different mechanism)

Issue #132 also adds a dataset-intake PII warning: `app/services/validation_service.py`'s
`_pii_warnings_for_record` scans record text for email/ID-number/phone-number patterns and
records it in `ValidationReport.warnings_summary` / `dataset_statistics.pii_screening`. It never
blocks a commit and is unrelated to retention/deletion — see that module's docstring for
detail, not repeated here to avoid two sources of truth for the same behavior.
