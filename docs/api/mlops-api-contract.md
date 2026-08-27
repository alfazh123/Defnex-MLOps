# Cross-cutting — MLOps Closed Loop API Contract

Scope: translates `docs/architecture/mlops-architecture.md` (WBS 1.1) and
`docs/dataset/dataset-lifecycle-and-schema.md` / `docs/dataset/validation-rules.md` (WBS 2.1/2.2) /
`docs/registry/model-artifact-versioning-lineage.md` / `docs/registry/model-promotion-approval-workflow.md`
(WBS 3.2/3.3) into the single Backend API contract those five documents already assume exists
(`mlops-architecture.md` §1.5: "the full request/response contract... is WBS-assigned separately"). This
document **does not** redesign anything those five documents already decided — every schema, enum, and
lifecycle rule below is a direct pass-through. Where this contract must introduce something none of the five
documents defined (HTTP verbs/paths, error envelope, a `training_run` entity), it is marked **[NEW]** and kept
to the minimum shape needed for the stated purpose.

Labels: **[PROJECT]** = confirmed from this vault's own notes, **[REC]** = engineering recommendation carried
over from a source WBS doc, **[NEW]** = introduced by this document (no prior WBS doc defined it),
**[UNKNOWN]** = needs confirmation.

**Hard constraints carried from the task and from `mlops-architecture.md` §1.2/§1.4:**
- Frontend talks **only** to this Backend API. Never to Unsloth Core, never to the training runner, never
  directly to dataset/artifact storage.
- Model Registry state (`model` resource, WBS 3.2) and Deployment state (`current_deployed_version`, WBS 3.2
  §4) are exposed as **separate resources** below, even where one action updates both.
- No numeric thresholds are introduced anywhere in this contract that weren't already defined upstream —
  every threshold-shaped field here is typed but its value-space is left to whatever WBS 2.2/3.3 eventually
  resolve (both already mark their own thresholds `[UNKNOWN]`).
- No Kafka/Airflow/MLflow/Kubernetes. This is a single Backend service with a REST/JSON API.

---

## 1. Entities and ID relationships

| Entity                                | ID shape                                       | Defined in              | Notes                                                                                                                                                                                                                        |
| -------------------------------------- | ------------------------------------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset version                       | `dataset_id` (string slug) + `version`         | WBS 2.1 §2              | See §9.1 — **inconsistency**: source docs disagree on whether `version` is an integer or a `"v1"`-prefixed string. This contract uses **integer**, flagged, not silently fixed.                                           |
| Validation report                     | `dataset_id` + `version` + `run_at`            | WBS 2.1 §2, WBS 2.2 §5  | No separate report version counter — multiple reports can exist per dataset version (re-runs).                                                                                                                             |
| Training run                          | `training_run_id` (string/UUID)                | **[NEW]** — see §9.2    | None of the five source docs define a schema for "a training run in progress." `mlops-architecture.md` §1.2 lists "training job records" as backend-owned state but never schematizes it — this contract fills that gap. |
| Model version                         | `model_id` (string slug) + `version` (integer) | WBS 3.2 §2              | Created only once a training run reaches `COMPLETED` (§4).                                                                                                                                                                  |
| Evaluation                            | attached to a model version, no separate ID    | WBS 3.2 §5              | Sub-resource of a model version.                                                                                                                                                                                             |
| Promotion/rejection/rollback decision | `decision_id` (string)                         | WBS 3.3 §8              | Referenced by a model version's `promotion_decision_ref`.                                                                                                                                                                   |
| Deployment pointer                    | `current_deployed_version`, per `model_id`     | WBS 3.2 §4              | Registry-level, not per-version — see §9.3 for a gap this raises.                                                                                                                                                           |

```
dataset_id/version ──validated by──> validation report ──gates──> train/ + eval/{holdout,benchmark}
        │                                                                    │
        └──────────────────────consumed by (dataset_id, version)────────────┘
                                          │
                                          ▼
                                  training_run (NEW entity)
                                          │  on COMPLETED, backend calls WBS 3.2 Register (internal)
                                          ▼
                              model_id/version  (status: REGISTERED)
                                          │  evaluation payload submitted (WBS 3.2 §5)
                                          ▼
                              model_id/version  (status: EVALUATED)
                                          │  promotion decision (WBS 3.3 §8)
                                          ▼
                    PROMOTED ──deploy──> DEPLOYED ──supersede/rollback──> RETIRED
                    REJECTED (terminal)
```

---

## 2. Endpoint table (concise)

| #   | Method + Path                                                             | Area                                   | MVP?                                     |
| --- | ------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------- |
| 1   | `POST /datasets/{dataset_id}/versions`                                    | Dataset intake                         | **MVP**                                  |
| 2   | `GET /datasets`                                                           | Dataset intake                         | MVP                                      |
| 3   | `GET /datasets/{dataset_id}/versions`                                     | Dataset intake                         | MVP                                      |
| 4   | `GET /datasets/{dataset_id}/versions/{version}`                           | Dataset intake                         | MVP                                      |
| 5   | `POST /datasets/{dataset_id}/versions/{version}/validate`                 | Validation                             | MVP                                      |
| 6   | `GET /datasets/{dataset_id}/versions/{version}/validation-reports`        | Validation                             | MVP                                      |
| 7   | `GET /datasets/{dataset_id}/versions/{version}/validation-reports/latest` | Validation                             | MVP                                      |
| 8   | `POST /datasets/{dataset_id}/versions/{version}/split`                    | Validation → dataset lifecycle         | MVP                                      |
| 9   | `POST /datasets/{dataset_id}/benchmark-sets`                              | Validation (benchmark set, WBS 2.1 §3) | Optional                                 |
| 10  | `POST /training-runs`                                                     | Training                               | MVP                                      |
| 11  | `GET /training-runs/{training_run_id}`                                    | Training                               | MVP                                      |
| 12  | `GET /training-runs`                                                      | Training                               | Optional                                 |
| 13  | `GET /models`                                                             | Model registry                         | MVP                                      |
| 14  | `GET /models/{model_id}/versions/{version}`                               | Model registry                         | MVP                                      |
| 15  | `GET /models/{model_id}/versions/{version}/lineage`                       | Model registry                         | Optional (derivable from #14 + #6 + #11) |
| 16  | `POST /models/{model_id}/versions/{version}/evaluation`                   | Evaluation                             | MVP                                      |
| 17  | `GET /models/{model_id}/versions/{version}/evaluation`                    | Evaluation                             | MVP                                      |
| 18  | `POST /models/{model_id}/versions/{version}/decisions`                    | Promotion/rejection                    | MVP                                      |
| 19  | `GET /models/{model_id}/versions/{version}/decisions`                     | Promotion/rejection                    | Optional (audit trail)                   |
| 20  | `POST /models/{model_id}/versions/{version}/deploy`                       | Deployment                             | MVP                                      |
| 21  | `GET /models/{model_id}/deployment`                                       | Deployment                             | MVP                                      |
| 22  | `POST /models/{model_id}/rollback`                                        | Deployment (rollback, WBS 3.3 §9)      | MVP                                      |
| 23  | `POST /feedback`                                                          | Feedback / closed loop                 | Optional — see §8                        |

**[NEW]** Common error envelope (no source doc defines HTTP conventions, since none pre-date this contract):
```json
{"error": {"code": "VALIDATION_INCOMPLETE", "message": "human-readable explanation"}}
```
Standard codes used throughout: `400` malformed request body, `404` unknown `dataset_id`/`model_id`/
`version`/`training_run_id`, `409` state-conflict (e.g. promoting a version not in `EVALUATED`), `422`
semantically invalid but well-formed request (e.g. incomplete evaluation payload).

---

## 3. Detailed endpoint specs

### 3.1 Dataset intake / dataset versions

**`POST /datasets/{dataset_id}/versions`** — creates a new dataset version by intake (WBS 2.1 §1: source →
intake → normalization → canonical ChatML → processed, run as one backend-orchestrated job).
- Request: `{"source_type": "huggingface", "source_dataset": "HuggingFaceH4/no_robots", "source_commit_or_snapshot_date": "2026-08-01", "source_format": "chatml"}`. **[REC]** `source_type: "file_upload"` is modeled as a valid enum value for future use but is **not required for MVP** — both Phase A datasets (WBS 2.1 §5) are HF-sourced, so file upload is optional/TBD (§10).
- Response `201`: `{"dataset_id": "no_robots", "version": 1, "status": "PROCESSING", "manifest": {"source_dataset": "...", "source_format": "chatml", "seed": null, "row_count": null, "created_at": "..."}}`.
- `status` **[NEW]** enum: `PENDING → PROCESSING → PROCESSED → FAILED`. Not defined in WBS 2.1 (which describes directories, not a job-status enum) — added here because intake/normalization/cleaning is a backend job the frontend needs to poll, per `mlops-architecture.md` §1.2's "dataset upload UI" requiring *some* status view.
- Relationship to IDs: `dataset_id` + `version` become the pair every downstream resource (validation, training run, model registry `dataset_id`/`dataset_version`) references.
- MVP: **required**.

**`GET /datasets`** — list known `dataset_id`s with their latest version + status. MVP.

**`GET /datasets/{dataset_id}/versions`** — list all versions for a `dataset_id`, each with `status` and (once available) `manifest`. MVP.

**`GET /datasets/{dataset_id}/versions/{version}`** — full manifest (WBS 2.1 §2 fields:
`source_url_or_hf_id`, `source_commit_or_snapshot_date`, `source_format`, `seed`, `row_count`,
`cleaning_steps_applied`, `created_at`, `created_by`) + current `status`. `404` if unknown. MVP.

### 3.2 Dataset validation and validation reports

**`POST /datasets/{dataset_id}/versions/{version}/validate`** — runs WBS 2.2's rule set (H1-H9, W1-W5, Q1-Q4)
against `processed/{dataset_id}/{version}/` and writes a report. **[REC]** Runs **synchronously** for Phase A
scale (hundreds to low-thousands of rows per WBS 2.1 §5) — an async job pattern is unnecessary complexity
at this size; revisit only if real dataset sizes grow past what a synchronous HTTP request can handle.
**[UNKNOWN]** whether this should be auto-triggered immediately after `PROCESSED` status (no source doc says
either way) — modeled here as an explicit, separate action so the frontend's "validation report view"
(`mlops-architecture.md` §1.2) has something concrete to trigger and poll.
- Request: `{}` (no body needed — operates on the already-processed version) or optionally `{"rule_set_version": "2.2.0"}` to pin a rule set.
- Response `201`: the full WBS 2.2 §5 dataset-level report object, verbatim schema (`record_count`, `status_counts`, `warnings_summary`, `dataset_statistics`, `gate_decision`, `gate_reason`).
- `409` if `dataset_id`/`version` is not yet `PROCESSED`.
- MVP: **required**.

**`GET /datasets/{dataset_id}/versions/{version}/validation-reports`** — list all report runs (WBS 2.1 §2: a
report's identity is `{dataset_id}/{version}` + `run_at`, not a separate version). MVP.

**`GET /datasets/{dataset_id}/versions/{version}/validation-reports/latest`** — convenience for the most
recent report; `404` if none run yet. MVP.

### 3.3 Dataset split (train/eval)

**`POST /datasets/{dataset_id}/versions/{version}/split`** — triggers the automatic holdout split (WBS 2.1
§3: fixed-seed random split, ratio a per-dataset configurable default, not a hard rule) and writes
`train/`/`eval/holdout/`. **[REC]** Precondition: latest validation report's `gate_decision == "PASS"` —
this enforces WBS 2.2 §6's flow ("VALID records proceed to train/eval split... Otherwise FAIL, not
promoted") without inventing a new rule; `409` if the latest report is `FAIL` or none exists.
- Request: `{"holdout_ratio": 0.05}` — **[UNKNOWN]** no default ratio is mandated (WBS 2.1 §3 explicitly calls
  95/5 "a default candidate only, not a hard requirement"); caller must supply one, or the backend applies
  0.05 as a documented, overridable default matching what practice actually used at ~2,000-row scale.
- Response `201`: `{"train_row_count": ..., "eval_holdout_row_count": ...}`.
- MVP: **required** (blocks training run creation, §3.4).

**`POST /datasets/{dataset_id}/benchmark-sets`** — registers a fixed, hand-curated benchmark question set
(WBS 2.1 §3: "manual, curated independently of `processed/`... versioned the same way"). **[UNKNOWN]**
whether a benchmark set has its own independent `dataset_id`/`version` or is nested under its paired training
dataset — WBS 2.1 §3 doesn't fully disambiguate this (§9.4). Modeled here as its own `dataset_id` namespace
(e.g. `legal_qa_benchmark_v1`) to avoid guessing a nesting relationship the source doc doesn't specify.
- MVP: **optional** — Phase A's one fixed 20-question set (already documented as static content in
  `Praktik/20 Aug 2026...md`) can be seeded directly by the backend without needing this endpoint on day one.

### 3.4 Training run creation and status

**`POST /training-runs`** — **[NEW]** creates a training run. This is the pipeline stage none of the five
source documents schematize (`mlops-architecture.md` §1.2 only says the backend "owns... training job
records"; WBS 3.2's lineage starts at `REGISTERED`, i.e. *after* this step).
- Request:
  ```json
  {
    "dataset_id": "no_robots",
    "dataset_version": 1,
    "model_id": "qwen-sft-domain-x",
    "training_config": {
      "base_model": "Qwen/Qwen3.8-27B",
      "peft_method": "dora",
      "load_in_4bit": false,
      "lora_r": 16,
      "lora_alpha": 16,
      "learning_rate": null,
      "epochs": 2,
      "max_seq_length": 4096
    },
    "triggered_by": "UNKNOWN — fill from actual user/session identity"
  }
  ```
  `training_config` fields are exactly WBS 3.2 §6's sample `training_config` object, reused verbatim, not
  redefined — this contract does not add a schema WBS 3.2 didn't already imply.
- Response `201`: `{"training_run_id": "run-8f2a...", "status": "PENDING"}`.
- `409` if `dataset_id`/`version` has not completed `split` (§3.3), or `model_id` doesn't exist yet (first
  version) — **[UNKNOWN]** whether `model_id` must be pre-registered or can be created implicitly on first
  training run; this contract assumes implicit creation (simplest, no separate "register a model line"
  endpoint needed), flagged in §10.
- Relationship to IDs: this is the **only** place `dataset_id`/`dataset_version` and `model_id` are joined
  before a model registry record exists.
- MVP: **required**.

**`GET /training-runs/{training_run_id}`** — status/progress. Per `mlops-architecture.md` §2 feasibility
table ("training process itself reports status/metrics to backend"):
- Response: `{"training_run_id": "...", "status": "RUNNING", "current_epoch": 1, "current_step": 120, "train_loss": 1.21, "eval_loss": 1.33, "model_id": null, "model_version": null}`.
- `status` **[NEW]** enum: `PENDING → RUNNING → COMPLETED | FAILED`.
- On `COMPLETED`, `model_id`/`model_version` are populated once the backend's internal WBS 3.2 Register call
  succeeds — this is the **only** link this contract provides between a `training_run_id` and its resulting
  model version, since WBS 3.2's own schema (§9.2) has no `training_run_id` field to store the reverse link.
- MVP: **required**.

**`GET /training-runs`** — list/filter by `dataset_id`, `model_id`, or `status`. Optional for MVP.

### 3.5 Model artifacts and model versions

WBS 3.2 §7 already defines the registry interface (Register / Read-list / Status-update / Lineage) — this
section maps it to REST verbs, changing nothing about the underlying contract.

**Register** — **not a public endpoint.** Triggered internally when a training run reaches `COMPLETED`
(§3.4); the training runner writes the artifact to storage and the backend calls WBS 3.2's Register operation
server-side. No frontend action creates a model version directly, consistent with "frontend never calls
Unsloth Core / the training runner directly."

**`GET /models`** — list `model_id`s with their latest version + status. Filter by `status` (e.g.
`status=PROMOTED`) per WBS 3.2 §7's Read/list operation. MVP.

**`GET /models/{model_id}/versions/{version}`** — full record, WBS 3.2 §8 JSON Schema verbatim
(`status`, `base_model`, `dataset_id`, `dataset_version`, `dataset_validation_report_ref`, `training_config`,
`created_at`, `created_by`, `evaluation`, `artifacts[]`, `promotion_decision_ref`, `previous_model_id`) plus
one **[NEW]** API-only convenience field `training_run_id` (assembled from the backend's own training-run
store, not part of WBS 3.2's persisted schema — see §9.2). `404` if unknown. MVP.

**`GET /models/{model_id}/versions/{version}/lineage`** — WBS 3.2 §7's Lineage operation: the full chain
dataset version → training config → artifacts → evaluation → promotion decision → deployment record.
**[REC]** Optional for MVP since it's fully derivable by the frontend composing #14 (model record) +
validation report (via `dataset_validation_report_ref`) + decisions (§3.7) — worth adding once a
single-screen lineage view is actually built, not before.

### 3.6 Evaluation results

**`POST /models/{model_id}/versions/{version}/evaluation`** — submits or updates the three-signal evaluation
payload, WBS 3.2 §5 fields verbatim: `eval_loss_trend`, `qualitative_comparison`,
`general_domain_regression_check`. **[UNKNOWN]** caller identity is mixed per WBS 3.3 §7 — `eval_loss_trend`
could be system-submitted by the training runner at completion, while `qualitative_comparison` and
`general_domain_regression_check` require a human eval pass; this endpoint accepts partial payloads from
either source and merges them onto the record.
- Request (partial payloads allowed): `{"eval_loss_trend": {"previous_version_eval_loss": 0.91, "this_version_eval_loss": 0.842}}`.
- Response `200`: the full updated `evaluation` object + current `status`. **[REC]** Once all three fields are
  present, the backend auto-transitions `REGISTERED → EVALUATED` (WBS 3.3 §2 table's exact condition) —
  no separate "mark as evaluated" call is needed.
- `409` if the version is not currently `REGISTERED` or `EVALUATED` (e.g. already `PROMOTED`) — evaluation
  data is not meant to be edited after a decision has been made against it (WBS 3.3 §8's `evidence_snapshot`
  freezes it at decision time regardless, but editing post-decision would still be confusing).
- MVP: **required**.

**`GET /models/{model_id}/versions/{version}/evaluation`** — current evaluation object. MVP.

### 3.7 Promotion / rejection decisions

**`POST /models/{model_id}/versions/{version}/decisions`** — WBS 3.3 §8 decision record, verbatim schema.
- Request:
  ```json
  {
    "decision": "PROMOTED",
    "decided_by": "UNKNOWN — fill from actual decision-maker identity/role",
    "rationale": "All three signals aligned: eval_loss down, majority win on 20-question table, no general-domain regression found in this pass."
  }
  ```
  `decision` enum: `PROMOTED | REJECTED` at this endpoint (`ROLLBACK` is a distinct action, §3.8, since it
  operates model-wide, not on a single version). Note the task brief's "APPROVED" is `PROMOTED` here per WBS
  3.3 §10 Decision 3 — not a separate enum value.
- Response `201`: the full decision record including a backend-populated `evidence_snapshot` (frozen copy of
  the version's current `evaluation` object at decision time, per WBS 3.3 §8 Decision 4 — the caller does not
  supply this, the backend does, so it can't be tampered with or omitted).
- `409` if the version is not `EVALUATED`, or a decision already exists for this version (WBS 3.3 §2: no
  transition out of `REJECTED`; a `PROMOTED` version doesn't get re-decided either — a fix is a new version).
- **This endpoint must be human-triggered only** (WBS 3.3 §10 Decision 1 — no automated caller is permitted,
  since no numeric threshold exists to automate against). The backend should not expose any internal
  service-account path to this endpoint that bypasses the frontend approve/reject UI.
- MVP: **required**.

**`GET /models/{model_id}/versions/{version}/decisions`** — audit trail (normally 0 or 1 entries per version,
plus any later rollback entries referencing it via `rollback_of_version`). Optional for MVP.

### 3.8 Deployment status

**`POST /models/{model_id}/versions/{version}/deploy`** — WBS 3.3 §3's release gate (`PROMOTED → DEPLOYED`).
- Request: `{}`.
- Response `200`: `{"model_id": "...", "current_deployed_version": <version>, "previous_deployed_version": <int|null>}`. The previously-deployed version (if any) is marked `RETIRED` as a side effect (WBS 3.3 §4 supersession path) in the same call.
- `409` if the version is not `PROMOTED`.
- **[UNKNOWN]** Whether this is always a separate action from `POST .../decisions` or the frontend/backend
  chains them automatically for Phase A — WBS 3.3 §3 explicitly leaves this open pending the deployment stack
  lock (`mlops-architecture.md` §3 Decision 3). This contract keeps them as **two endpoints** so that
  splitting the workflow later needs no schema change, per WBS 3.3's own reasoning.
- **[UNKNOWN]** What actually happens inside this call (starting a vLLM/llama-server process, updating a
  config, etc.) is undefined — deployment stack is not locked. The endpoint's *contract shape* (what the
  frontend sends/receives) does not depend on that being resolved yet.
- MVP: **required** (shape only; internal implementation is a later, VM/stack-dependent concern).

**`GET /models/{model_id}/deployment`** — current deployment pointer: `{"model_id": "...", "current_deployed_version": <int|null>, "deployed_at": "...", "status": "DEPLOYED"}`. This is deliberately its **own** resource, separate from a model *version's* record, per the task's explicit "keep registry and deployment state conceptually separate" constraint and WBS 3.2 §4's own framing of `current_deployed_version` as registry-level, not per-record. MVP.

**`POST /models/{model_id}/rollback`** — WBS 3.3 §9's rollback action, model-scoped (not version-scoped,
since it changes the model-level deployment pointer).
- Request: `{"rollback_of_version": 5, "decided_by": "UNKNOWN — fill from actual decision-maker", "rationale": "UNKNOWN — must state the observed production problem"}`.
- Response `201`: a decision record with `decision: "ROLLBACK"`, `evidence_snapshot: null` (WBS 3.3 §9 —
  rollback responds to an observed production problem, not new offline eval data). Updates
  `current_deployed_version` and retires the version rolled back from.
- `409` if `rollback_of_version` is not `PROMOTED` or previously `DEPLOYED`.
- **Human-only**, same reasoning as §3.7 — no monitoring/alerting system exists in the vault to trigger this
  automatically (WBS 3.3 §1, §9).
- MVP: **required** — rollback is cheap to include given it reuses the decision schema entirely (WBS 3.3 §9
  Decision 2), and leaving it out would mean Phase A has no way to recover from a bad deploy at all.

### 3.9 Feedback / inference interaction

**[UNKNOWN]** None of the five source documents design this stage in detail. `mlops-architecture.md` §1.3's
flow diagram shows `Feedback capture → back to Dataset storage: raw/ (next iteration)` but no schema, and
§1.2 explicitly routes deployment/serving to talk **directly to inference clients**, not through this
Backend API — so **inference itself is intentionally not proxied here**, per architecture's own component
boundary. Only a minimal feedback-intake stub is included, as a placeholder for the closed loop's return path:

**`POST /feedback`** — **[NEW], optional, minimal placeholder only.**
- Request: `{"model_id": "...", "model_version": 5, "input": "...", "output": "...", "flag": "good|bad|correction", "correction": "string, optional", "submitted_by": "UNKNOWN"}`.
- Response `201`: `{"feedback_id": "..."}`.
- **[UNKNOWN]** How/whether this ever becomes a `raw/` dataset record (WBS 2.1's intake interface, §1) is not
  designed anywhere — this endpoint only captures the feedback event; turning it into a new dataset version
  is future work, explicitly out of scope for this contract.
- MVP: **optional** — nothing in the assigned WBS packages (1.1, 2.1, 2.2, 3.2, 3.3) requires the feedback
  loop to be functional for Phase A; the loop diagram is architectural context, not a Phase A deliverable.

---

## 4. JSON examples for the most important endpoints

**`POST /datasets/{dataset_id}/versions` → `201`**
```json
{
  "dataset_id": "no_robots",
  "version": 1,
  "status": "PROCESSING",
  "manifest": {
    "source_url_or_hf_id": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
    "seed": null,
    "row_count": null,
    "cleaning_steps_applied": [],
    "created_at": "2026-08-27T10:00:00Z",
    "created_by": "UNKNOWN — fill from actual user/session identity"
  }
}
```

**`GET /datasets/{dataset_id}/versions/{version}/validation-reports/latest` → `200`** (WBS 2.2 §5 schema, unchanged)
```json
{
  "dataset_id": "no_robots",
  "dataset_version": 1,
  "rule_set_version": "2.2.0",
  "run_at": "2026-08-27T10:15:00Z",
  "record_count": 500,
  "status_counts": {"VALID": 468, "INVALID": 21, "NEEDS_REVIEW": 11},
  "warnings_summary": {"W1_rhetorical_question": 4, "W4_citation_pattern_present": 32},
  "dataset_statistics": {"...": "see WBS 2.2 §5 for full shape"},
  "gate_decision": "PASS",
  "gate_reason": "No hard-error rate breach; no leakage found; NEEDS_REVIEW queue within threshold."
}
```
Note: `dataset_version` is shown here as the **integer** `1`, not the string `"v1"` from WBS 2.2's own
sample — see §9.1, this is the flagged inconsistency, resolved *for this contract only* in favor of the
integer form.

**`POST /training-runs` → `201`**, then **`GET /training-runs/{training_run_id}` → `200`** (mid-run, then completed)
```json
{"training_run_id": "run-8f2a1c", "status": "RUNNING", "current_epoch": 1, "current_step": 120, "train_loss": 1.21, "eval_loss": 1.33, "model_id": null, "model_version": null}
```
```json
{"training_run_id": "run-8f2a1c", "status": "COMPLETED", "current_epoch": 2, "current_step": 238, "train_loss": 0.98, "eval_loss": 0.842, "model_id": "qwen-sft-domain-x", "model_version": 5}
```

**`GET /models/{model_id}/versions/{version}` → `200`** (WBS 3.2 §6 schema, unchanged, + one `[NEW]` field)
```json
{
  "model_id": "qwen-sft-domain-x",
  "version": 5,
  "status": "EVALUATED",
  "training_run_id": "run-8f2a1c",
  "base_model": "Qwen/Qwen3.8-27B",
  "dataset_id": "no_robots",
  "dataset_version": 1,
  "dataset_validation_report_ref": "validation/no_robots/1/report.json",
  "training_config": {"peft_method": "dora", "load_in_4bit": false, "lora_r": 16, "lora_alpha": 16, "learning_rate": "UNKNOWN — fill from actual run config", "epochs": 2, "max_seq_length": 4096},
  "created_at": "2026-08-27T10:15:00Z",
  "created_by": "UNKNOWN — fill from actual run",
  "evaluation": {
    "eval_loss_trend": {"previous_version_eval_loss": 0.91, "this_version_eval_loss": 0.842},
    "qualitative_comparison": {"question_table_version": 1, "wins": 13, "losses": 5, "ties": 2, "total": 20},
    "general_domain_regression_check": {"checked": true, "regressions_found": []}
  },
  "artifacts": [{"type": "adapter", "uri": "UNKNOWN — persistent storage location not yet decided", "size_bytes": 104857600}],
  "promotion_decision_ref": null,
  "previous_model_id": null
}
```

**`POST /models/{model_id}/versions/{version}/decisions` → `201`**
```json
{
  "decision_id": "promo-qwen-sft-domain-x-v5-001",
  "model_id": "qwen-sft-domain-x",
  "version": 5,
  "decision": "PROMOTED",
  "decided_by": "UNKNOWN — fill from actual decision-maker identity/role",
  "decided_at": "2026-08-27T11:00:00Z",
  "evidence_snapshot": {
    "eval_loss_trend": {"previous_version_eval_loss": 0.91, "this_version_eval_loss": 0.842},
    "qualitative_comparison": {"question_table_version": 1, "wins": 13, "losses": 5, "ties": 2, "total": 20},
    "general_domain_regression_check": {"checked": true, "regressions_found": []}
  },
  "rationale": "All three signals aligned: eval_loss down, majority win on 20-question table, no general-domain regression found in this pass.",
  "rollback_of_version": null
}
```

**`GET /models/{model_id}/deployment` → `200`**
```json
{"model_id": "qwen-sft-domain-x", "current_deployed_version": 5, "deployed_at": "2026-08-27T11:05:00Z", "status": "DEPLOYED"}
```

---

## 5. Entity relationship / data-flow overview

```
[Frontend]
    │  REST/JSON, only this API — never Unsloth Core, never the training runner directly
    ▼
[Backend API]
    │
    ├── Dataset domain ──────────────────────────────────────────────────────────
    │     dataset_id/version ──(validate)──> validation report ──(gate PASS)──> split
    │                                                                              │
    ├── Training domain ───────────────────────────────────────────────────────── │
    │     training_run_id [NEW] ── consumes dataset_id/version + training_config <┘
    │            │  COMPLETED → internal WBS 3.2 Register call (not a public endpoint)
    │            ▼
    ├── Model registry domain (WBS 3.2) ───────────────────────────────────────────
    │     model_id/version: REGISTERED → EVALUATED → PROMOTED|REJECTED
    │            │  evaluation payload (WBS 3.2 §5)     │
    │            ▼                                       ▼
    ├── Promotion domain (WBS 3.3) ── decision_id (PROMOTED/REJECTED/ROLLBACK) ────
    │            │
    │            ▼
    └── Deployment domain (separate resource, WBS 3.2 §4 pointer) ────────────────
          current_deployed_version ── DEPLOYED ── RETIRED (supersede or rollback)

[Deployment/serving] ──inference──> [Inference clients]   (NOT through this Backend API — mlops-architecture.md §1.2)
```

Four domains, four resource families (`/datasets`, `/training-runs`, `/models` + `/evaluation` +
`/decisions`, `/models/{id}/deployment`) — deliberately kept as separate top-level resources rather than
nested into one mega-object, matching the same separation of concerns WBS 3.2/3.3 already established
between registry state and promotion/deployment state.

---

## 6. "Frontend can start now" — mockable immediately

**[PROJECT]** `mlops-architecture.md` §1.5 already states this directly: *"the frontend can be built entirely
against mocked responses for: dataset upload, dataset/validation status, training job creation & status,
evaluation result, model version list, promotion action, deployment status, feedback submission."* This
contract's endpoint groups map 1:1 onto that list — **every endpoint in §2 can be mocked today**, with zero
dependency on the VM (`mlops-architecture.md` §4) or the base-model/training questions:

- **§3.1–3.3 (dataset intake/validation/split)** — mock with the exact WBS 2.1/2.2 sample JSON already in
  those documents; the validation rule logic itself can even run for real today (it's pure data processing,
  no GPU needed) against the two Phase A datasets.
- **§3.4 (training runs)** — mock `PENDING → RUNNING → COMPLETED` status transitions with fixture
  `train_loss`/`eval_loss` numbers; the frontend's training-status UI needs no real GPU to be built and tested.
- **§3.5–3.7 (model registry, evaluation, decisions)** — mock with WBS 3.2/3.3's own sample JSON verbatim
  (already reused directly in §4 above); the promotion approve/reject UI can be fully built and tested against
  fixture `EVALUATED` records before any real model exists.
- **§3.8 (deployment)** — mock `current_deployed_version` transitions; the actual serving mechanics being
  undecided (deployment stack, `mlops-architecture.md` §3 Decision 3) does not block building the UI that
  calls this endpoint shape.
- **§3.9 (feedback)** — trivially mockable, and optional besides.

**Nothing in this contract requires waiting for the VM, the training runner, or a locked deployment stack** —
this mirrors exactly what WBS 1.1 §6 already concluded about WBS 2.1/2.2/3.2/3.3 themselves.

---

## 7. Backend implementation order

**[REC]** Ordered by (a) which WBS design it depends on being stable, and (b) VM-independence, following the
same logic WBS 1.1 §6 and WBS 3.2/3.3's own "recommended next steps" already used:

1. **Dataset domain (§3.1–3.3)** — WBS 2.1/2.2 are fully designed and require no VM; this is real,
   runnable logic today (HF download, cleaning rules, validation rules are pure Python/data operations).
2. **Model registry read/write + evaluation + decisions (§3.5–3.7)** — WBS 3.2/3.3 are fully designed and
   also VM-independent; can be built and tested end-to-end against **fixture** training-run outputs before
   any real training run exists, exactly as §6 describes.
3. **Training run domain (§3.4)** — the schema is defined here (this contract), but real execution is
   VM-blocked (`mlops-architecture.md` §4). Build the API surface and status-polling shape now (item 2's
   fixtures can already simulate `COMPLETED` runs feeding into the registry); wire it to the actual training
   runner only once the VM exists.
4. **Deployment domain (§3.8)** — blocked on `mlops-architecture.md` §3 Decision 3 (serving stack not
   locked). Build the API *shape* now (per §6, the frontend needs it to build the deployment status UI); defer
   the internal implementation until the stack is chosen.
5. **Feedback (§3.9)** — optional, no WBS package currently requires it; implement last, if at all, for
   Phase A.

---

## 8. Key decisions

**Decision 1 — Register (WBS 3.2) is not exposed as a public endpoint.** Why: the training runner/backend job
that completes training already has direct access to the registry internally; exposing Register over HTTP
would let something other than a completed training job create a model version, which nothing in the
architecture calls for. Evidence: [PROJECT] `mlops-architecture.md` §1.2 (Unsloth Core does not own state;
Backend/API is "the only integration point"). Alternatives: expose it as a public endpoint guarded by a
service-role check — rejected as unrequested complexity when an internal function call achieves the same
result for a single-service backend. Open question: none blocking.

**Decision 2 — `training_run` is a new, first-class entity with its own ID, not folded into the model
registry.** Why: `mlops-architecture.md` §1.2 explicitly lists "training job records" as backend-owned state
distinct from the model registry, and WBS 3.2's lineage only begins at `REGISTERED` (post-completion) — there
is no way to represent "a run in progress" using WBS 3.2's schema alone. Evidence: [PROJECT] architecture
§1.2 table; [NEW] gap identified by cross-referencing WBS 3.2 §2 (starts at `REGISTERED`). Alternatives: add a
`PENDING`/`RUNNING` pre-state to WBS 3.2's own status enum — rejected, this document may not modify WBS 3.2,
and conflating "a run exists" with "an artifact was produced" would blur exactly the boundary WBS 3.2 §1.6
draws. Open question: carried to §9.2 below — WBS 3.2 may want to add a `training_run_id` field of its own in
a future revision for a complete audit trail without needing this contract's assembled join.

**Decision 3 — Promotion and rollback decisions (§3.7, §3.8) have no service-account/system-caller path.**
Why: directly enforces WBS 3.3 §10 Decision 1 (must not be automated, since no numeric threshold exists to
automate against). Evidence: [PROJECT] `model-promotion-approval-workflow.md` §6, §10. Alternatives: allow an
internal automation path for future use — rejected now, would create a backdoor around a constraint the task
explicitly restated. Open question: none blocking.

**Decision 4 — Deployment (§3.8) is modeled as a separate resource (`/models/{id}/deployment`) from the
model version record (§3.5).** Why: direct task constraint ("keep Model Registry and Deployment state
conceptually separate") and matches WBS 3.2 §4's own framing of `current_deployed_version` as a registry-level
pointer, not a per-version field. Evidence: [PROJECT] WBS 3.2 §4, §9 Decision 4. Alternatives: embed
`is_deployed`/`deployed_at` directly on the model version resource — rejected, would re-merge the two
concepts the source documents and the task both explicitly keep apart. Open question: none blocking.

---

## 9. Inconsistencies discovered across the five source documents (flagged, not silently resolved)

**9.1 — `dataset_version` type disagreement.** `dataset-lifecycle-and-schema.md` §2 describes version as "a
plain incrementing integer," but its own prose examples use a `v`-prefix (`v1`, `v2`). `validation-rules.md`
§5/§7 types `dataset_version` as a **string** in both its sample JSON (`"dataset_version": "v1"`) and its
JSON Schema (`{"type": "string"}`). `model-artifact-versioning-lineage.md` §6/§8 types `dataset_version` as an
**integer** (`"dataset_version": 1`, `{"type": "integer"}`) and its own path-reference example
(`validation/no_robots/1/report.json`) uses the bare integer, not `v1`. **These two already-approved WBS
documents (2.2 and 3.2) are not mutually consistent on this field's type.** This contract uses the **integer**
form throughout (matching WBS 3.2, the more recently written of the two, and matching WBS 2.1's own stated
"plain incrementing integer" philosophy) — but this is a **pick, not a resolution**; WBS 2.1/2.2 should be
reconciled directly rather than treating this contract's choice as authoritative.

**9.2 — No source document schematizes "a training run in progress."** `mlops-architecture.md` §1.2 names
"training job records" as backend-owned state; WBS 3.2's registry only starts at `REGISTERED` (i.e., after a
training run has already produced an artifact). Nothing defines what a record looks like *during* training,
or how a completed run's ID relates back to the model version it produces. This contract introduces
`training_run_id` (§1, §3.4) to fill that gap and adds it to the model-version read response as an **[NEW]**
API-only field — but if this API contract is later ever used to generate WBS 3.2's actual database schema,
someone should decide whether `training_run_id` also belongs as a persisted field on the registry record
itself, not just an assembled API response.

**9.3 — The model registry's `promotion_decision_ref` is a single field, but WBS 3.3's decision schema
supports multiple decision types (PROMOTED/REJECTED/ROLLBACK) and rollback is model-scoped, not
version-scoped.** WBS 3.2 §6/§8 defines `promotion_decision_ref` as a single nullable string on a **version**
record. A rollback decision (WBS 3.3 §9) references the version being rolled back *to*, but conceptually
affects the model's deployment pointer, not any single version's own promotion outcome. Neither source
document states whether a rollback decision should *also* be written into the rolled-back-from version's
`promotion_decision_ref`, overwriting or supplementing whatever was there. This contract does not resolve
this — `POST /models/{model_id}/rollback` (§3.8) creates a decision record but this contract does not
prescribe writing it back onto any specific version's `promotion_decision_ref`. Flagged for WBS 3.2/3.3's
owners to jointly decide.

**9.4 — Benchmark set versioning relationship is ambiguous.** `dataset-lifecycle-and-schema.md` §3 says
`eval/benchmark/` is "versioned the same way (`{dataset_id}/{version}/manifest.json`)" without stating
whether a benchmark set's `dataset_id` is the *same* `dataset_id` as its paired training dataset (sharing the
same namespace) or an independent one. This contract (§3.3) assumes an independent `dataset_id` namespace for
benchmark sets — not stated as fact, just the least-assumption-heavy reading, flagged as **[UNKNOWN]** rather
than treated as confirmed.

**9.5 — `mlops-architecture.md` §1.3's flow diagram places Validation between `raw/` and `processed/`, while
WBS 2.1 §1 and WBS 2.2's own stated scope both place validation *after* `processed/` exists (gating the
`processed/` → `train/`/`eval/` transition, not the `raw/` → `processed/` one).** This contract follows the
more detailed and mutually-consistent WBS 2.1/2.2 sequencing (§3.1–3.3 above: intake → processed → validate →
split), since `mlops-architecture.md`'s diagram is an intentionally simplified illustration and explicitly
defers detailed design to WBS 2.1/2.2 ("designed separately"). Flagged rather than silently normalized,
since a reader of the architecture doc alone would design the pipeline differently.

---

## 10. Open questions

1. **[UNKNOWN]** §9.1 — reconcile `dataset_version`'s type between WBS 2.1/2.2 (string, `v`-prefixed) and
   WBS 3.2 (integer). This contract picked integer; not authoritative.
2. **[UNKNOWN]** Whether `model_id` must be explicitly pre-registered before the first training run
   (§3.4) or is created implicitly on first use — this contract assumes implicit creation.
3. **[UNKNOWN]** §9.3 — how (or whether) a rollback decision updates the rolled-back-from version's
   `promotion_decision_ref`.
4. **[UNKNOWN]** §9.4 — benchmark set `dataset_id` namespace relationship to its paired training dataset.
5. **[UNKNOWN]** File-upload dataset intake (`source_type: "file_upload"`, §3.1) — modeled but not required
   for MVP since both Phase A datasets are HF-sourced; needs real design once a non-HF source is needed.
6. **[UNKNOWN]** Auto-trigger vs. explicit-click for `POST .../validate` (§3.2) and `POST .../split` (§3.3) —
   no source doc specifies whether these should fire automatically on completion of the prior stage.
7. Carried over unchanged from WBS 1.1/2.1/2.2/3.2/3.3: VM environment, exact deployment stack
   (`mlops-architecture.md` §3 Decision 3), all numeric thresholds (validation gate, promotion signals) —
   none of these block finalizing this contract's *shape*, all block only its real backend implementation.

---

## 11. Recommended next steps

1. Resolve §9.1 (dataset_version type) directly between whoever owns WBS 2.1/2.2 and WBS 3.2 — this is the
   highest-leverage inconsistency since it affects every cross-reference between the dataset and model
   registry domains.
2. Build the dataset domain (§3.1–3.3) and model-registry-read/evaluation/decisions domain (§3.5–3.7) first,
   per §7 — both are fully specifiable today and unblock frontend development immediately per §6.
3. Decide §10.2 (implicit vs. explicit `model_id` creation) before backend work on §3.4 begins — it's a small
   decision but changes that endpoint's error-handling shape.
4. Once the deployment stack (`mlops-architecture.md` §3 Decision 3) is locked, revisit §3.8's internal
   implementation and resolve §9.3 (rollback's effect on `promotion_decision_ref`) alongside it.
