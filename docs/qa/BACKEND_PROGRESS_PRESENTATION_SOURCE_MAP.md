# Source Map — BACKEND_PROGRESS_PRESENTATION.pptx

Traceability table for every claim in `docs/qa/BACKEND_PROGRESS_PRESENTATION.pptx`
(13 slides). Status values:

- **VERIFIED** — read from source file in this repository (file/line cited).
- **MEASURED** — produced by running a command locally on 2026-09-25.
- **NOT VERIFIED** — not derivable from the repo; labeled as such in the slide itself.

Slide deck generated from `/tmp/opencode/build_deck.js` (pptxgenjs, 13.33x7.5in).
Validated by `pptx/scripts/office/validate.py` → `All validations PASSED!`.

---

| Slide | Claim | Source File | Evidence |
|---|---|---|---|
| 1 | Title "DEFNEX MLOps Backend — Current Progress & Technical Status" | `docs/qa/BACKEND_PROGRESS_PRESENTATION.pptx` | slide 1 |
| 1 | Scope = `ml-close-loop-be/` FastAPI backend, closed-loop ML lifecycle | `CLAUDE.md` §Repository shape | "Two-app monorepo. `ml-close-loop-be/` (FastAPI, the real work)" |
| 1 | Target architecture is PRD v2, repo is Phase 0 prototype being migrated | `CLAUDE.md` §Target architecture; `DEFNEX_MLOps_Multi_Server_Architecture_v2_PRD.md` | "The current codebase is the Phase 0 prototype being migrated toward it" |
| 2 | Starting state: 19 raw issues, prototype depth | `docs/qa/ISSUE_RECONCILIATION_AFTER_PHASE2ABC.md` | issue reconciliation table |
| 2 | Now: 18/18 signaling PASS; application-layer ladder closed | `docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md` | §13–§16, §19 |
| 2 | #167 READY TO CLOSE (supersedes earlier PARTIAL) | `docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md` (2026-09-23 19:12) supersedes `docs/qa/ISSUE_RECONCILIATION_AFTER_PHASE2ABC.md` (12:20) | later timestamp, direct measurement |
| 3 | 5 layer stack: FastAPI / Services / SQLAlchemy 2.0 / SQLite→PostgreSQL / MinIO | `CLAUDE.md` §Backend architecture; `ml-close-loop-be/pyproject.toml` | three-layer split documented; SQLAlchemy 2.0 Mapped ORM |
| 3 | SQLite dev, PostgreSQL authoritative target (PRD §23.1) | `CLAUDE.md`; PRD §23.1 | "PostgreSQL menjadi basis otoritatif (PRD §23.1); SQLite hanya untuk dev lokal" |
| 3 | MinIO/S3 object storage, adapter artifact path | `ml-close-loop-be/app/services/artifact_storage.py`; `docs/PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md` | `s3://` URI wiring; artifact `file:///models/artifacts/smoke-llm-v3/...` |
| 3 | `docker compose` services: backend, worker, minio, serving (gpu profile) | `ml-close-loop-be/docker-compose.yml` | service definitions; `serving` under `gpu` profile |
| 3 | **14 external GPU processes (shared H100)** | runtime observation on VM | `nvidia-smi` — **NOT REPRODUCIBLE FROM REPO**, labeled as observation |
| 3 | no docker.sock in worker | `ml-close-loop-be/docker-compose.yml` | worker service has no `/var/run/docker.sock` mount |
| 4 | Ladder `EVALUATED → STAGING → VALIDATED → DEPLOYED` | `app/services/promotion_service.py:27-28` | comment block "Promotion Ladder issues #69/#70, PRD §16.2, §41 P4" |
| 4 | Endpoint `/models/{model_id}/versions/{version}/deploy-staging` | `app/api/promotion.py:116` | route decorator |
| 4 | Endpoint `/models/{model_id}/versions/{version}/validate-staging` | `app/api/promotion.py:164` | route decorator |
| 4 | Endpoint `/models/{model_id}/versions/{version}/promote-production` | `app/api/promotion.py:205` | route decorator |
| 4 | Inference endpoint `/models/{model_id}/inference` | `app/api/inference.py:19` | route decorator |
| 4 | Promotion is human-triggered only, never auto on thresholds | `CLAUDE.md` §The closed loop | "promotion is human-triggered only — never add automatic promotion on numeric thresholds" |
| 4 | Deployment has no FK to model registry (deliberate separation) | `CLAUDE.md`; `app/services/deployment_service.py` | "`Deployment` deliberately has **no** foreign key to the model registry" |
| 5 | Two training execution paths (inline vs worker) | `CLAUDE.md` §The closed loop | path 1 `POST /api/v1/training-runs`, path 2 `app/workers/training_worker.py` |
| 5 | Worker is the only caller of `register_model_version` | `app/workers/training_worker.py` | "without the worker nothing ever creates a `ModelVersion`" (`CLAUDE.md`) |
| 5 | run-9e66b6: 16/16 checks, LoRA 540,672 params, 0.11% of 494,573,440 | `docs/PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md` | run report |
| 5 | ds-smoke-test-v1, durasi ±185s, VRAM gate 17272 ≥ 8000 MiB | `docs/PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md`; `docs/PHASE2B_POST_RUN_VALIDATION_REPORT.md` | run + post-run validation |
| 5 | First run failed before fix | `docs/PHASE2B_FIRST_TRAINING_RUN_FAILURE_REPORT.md` | failure report |
| 5 | Serial training concurrency 1, GPU `flock` | `CLAUDE.md` constraints; `app/workers/gpu_lock.py` | "Concurrency training tetap 1 (serial) ... GPU lock `flock`" |
| 6 | H100 80GB shared, 78964/2116 MiB baseline, 14 external procs, driver 580.173.02, CUDA 13.0 | runtime `nvidia-smi` on VM | **NOT REPRODUCIBLE FROM REPO** — labeled as live observation |
| 6 | vLLM via Docker + NVIDIA CDI; Unsloth in separate venv | `CLAUDE.md` constraints | "Serving: vLLM via Docker + NVIDIA CDI. Training: Unsloth native di venv terpisah" |
| 6 | Signaling health-check fix (timeout) | `docs/PHASE2A_HEALTH_TIMEOUT_FIX_REPORT.md` | fix report |
| 6 | 18/18 Phase 2A signaling checks | `docs/PHASE2A_SECOND_LIVE_SIGNALING_TEST_REPORT.md`; `docs/PHASE2A_FINAL_QA_REPORT.md` | live signaling test |
| 7 | Three layers: backend → vLLM → GPU/CDI | `ml-close-loop-be/docker-compose.yml`; `docs/PHASE2C_RUNTIME_LORA_VALIDATION_REPORT.md` | serving profile + runtime report |
| 7 | `parse_json=False` guard for plain-text 200 from vLLM | `app/services/serving.py:237,266` | `parse_json=False` passed on vLLM client calls |
| 7 | `parse_json` flag defined at | `app/services/http_retry.py:66` (first client), `:127` (second) | `parse_json: bool = True,` |
| 7 | Root cause: vLLM returns plain-text 200 that `resp.json()` cannot parse | `docs/PHASE2C_HTTP_RESPONSE_FIX_REPORT.md` | fix report |
| 7 | vLLM 0.28.0 `defnex-vllm` external returns 404 for LoRA path (negative result) | `docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md` | negative test result |
| 7 | Compose `serving` runs vLLM 0.30.0; active `VLLM_URL` still `http://172.17.0.1:8001` | `ml-close-loop-be/docker-compose.yml` | env wiring — **open decision**, listed as gap |
| 8 | Application-layer E2E rehearsal 16/16 | `docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md` §13–§16 | V2 report, 2026-09-23 19:12 |
| 8 | Earlier rehearsal FAILED (JSONDecodeError at deploy-staging) — superseded | `docs/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_REPORT.md` (13:08) | first attempt, pre-fix; do not cite as current status |
| 8 | Fix: `resp.json()` no longer raises on plain-text 200 | `app/services/http_retry.py:75-76`, `:136-137` | `if parse_json:` guard before `return resp.json()` |
| 9 | `pytest tests/ -q` → **1014 passed, 27 failed, 3 skipped** | local run 2026-09-25, `ml-close-loop-be/` | **MEASURED** |
| 9 | Coverage **92.0%** (5160 stmts, 411 missed), gate `--cov-fail-under=80` | `coverage report` on `.coverage`, 2026-09-25; `ml-close-loop-be/pyproject.toml` | **MEASURED**; an earlier 43.78% figure came from a collect-only run and is wrong |
| 9 | 27 failures are all `tests/test_file_signaling.py`, import `gpu_controller` from hard-coded `/home/ubuntu/defnex-mlops-experiment/gpu_controller` | `ml-close-loop-be/tests/test_file_signaling.py` | portability defect, not a feature regression |
| 9 | ruff check + format | `.github/workflows/ci.yml:15` (`lint` job) | CI job |
| 9 | pytest + coverage | `.github/workflows/ci.yml:29` (`test` job) | CI job |
| 10 | pip-audit + bandit with `.bandit-baseline.json` (27 pre-existing) | `.github/workflows/ci.yml:49` (`security` job); `ml-close-loop-be/.bandit-baseline.json` | CI job + baseline file |
| 10 | gitleaks secret scan (fetch-depth 0) | `.github/workflows/ci.yml:69` (`secret-scan` job) | CI job |
| 10 | Trivy container scan, report-only (exit-code 0) | `.github/workflows/ci.yml:81` (`container-scan` job) | CI job |
| 10 | Frontend typecheck | `.github/workflows/ci.yml:98` (`frontend` job) | CI job |
| 10 | Dependabot: pip, npm, docker (be+fe), github-actions; weekly; open-PR limit 10 | `.github/dependabot.yml` | ecosystem entries |
| 10 | Merged PRs #190–#206 as evidence | git history (`git log --oneline`) | merged to main |
| 11 | 5 required distinctions (E2E vs unit, worker vs inline, staging vs production, observation vs inference, known vs unknown) | `docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md`; `CLAUDE.md` §The closed loop | both sources contrast the paired concepts |
| 11 | RBAC 5 roles, `has_permission` | `app/rbac.py:42` (`ROLE_PERMISSIONS`), `:51` (`has_permission`) | role→permission map |
| 11 | `require_admin` dependency | `app/api/deps.py:127` | dependency function |
| 11 | Audit log append-only | `app/models/audit_log.py:9` (`class AuditLog`); `app/services/audit_service.py:46` (`AuditLog(`) | model + recorder |
| 11 | `openapi.yaml` is frozen contract asserted by tests | `CLAUDE.md` §Constraints; `ml-close-loop-be/tests/test_contract.py` | "Change the implementation to match the spec, not the reverse" |
| 11 | `conftest.py` is FROZEN | `ml-close-loop-be/tests/conftest.py` | marked FROZEN in `CLAUDE.md` §Constraints |
| 12 | Status matrix: VERIFIED / PARTIAL / IMPLEMENTED-NOT-LIVE-VERIFIED / GATED & BACKLOG | derived from `docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md` + `docs/qa/ISSUE_RECONCILIATION_AFTER_PHASE2ABC.md` | categories are evidence-based, not code-presence-based |
| 12 | #167 VERIFIED (READY TO CLOSE) | `docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md` | supersedes PARTIAL in reconciliation |
| 12 | #174 secrets backend DECISION-GATED | `docs/qa/ISSUE_RECONCILIATION_AFTER_PHASE2ABC.md`; PRD §21, §43 | `secret://…` reference model still undecided |
| 12 | Ubuntu / CPU / RAM / disk | — | **NOT VERIFIED** — not present in repo; slide says so explicitly |
| 13 | Issue backlog #161–#179 with status + next action | `docs/qa/ISSUE_RECONCILIATION_AFTER_PHASE2ABC.md` | issue table (di-superseded for #167) |
| 13 | Next engineering priorities 1–5 | derived from `docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md` §13–§16, `docs/PHASE2B_POST_RUN_VALIDATION_REPORT.md` (Follow-ups), `docker-compose.yml`, `tests/test_file_signaling.py` | each item traces to a recorded gap |
| 13 | #163 systemd unit, #166 pinned training venv | `docs/qa/ISSUE_RECONCILIATION_AFTER_PHASE2ABC.md` | issue list |

---

## Claims intentionally marked NOT VERIFIED on the slides

| Claim | Why not verified |
|---|---|
| Ubuntu version, CPU model, RAM, disk size | Not recorded anywhere in the repo or its docs |
| Production-readiness / go-live | Evidence covers staging + live rehearsal only; no production deployment record |
| "14 external GPU processes", H100 baseline figures | Live `nvidia-smi` observation from one VM session, not reproducible from repo content |

## Evidence file index

```
docs/qa/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_V2_REPORT.md   # primary, supersedes others
docs/qa/ISSUE_RECONCILIATION_AFTER_PHASE2ABC.md                 # issue table (#167 superseded)
docs/PHASE2C_APPLICATION_LAYER_LIVE_REHEARSAL_REPORT.md         # first (failed) run — historical
docs/PHASE2C_HTTP_RESPONSE_FIX_REPORT.md                        # root cause resp.json()
docs/PHASE2C_RUNTIME_LORA_VALIDATION_REPORT.md
docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md                  # negative result (v0.28.0 404)
docs/PHASE2A_SECOND_LIVE_SIGNALING_TEST_REPORT.md
docs/PHASE2A_FINAL_QA_REPORT.md
docs/PHASE2A_HEALTH_TIMEOUT_FIX_REPORT.md
docs/PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md
docs/PHASE2B_POST_RUN_VALIDATION_REPORT.md
docs/PHASE2B_FIRST_TRAINING_RUN_FAILURE_REPORT.md
ml-close-loop-be/docs/CURRENT_RUNTIME_ARCHITECTURE.md
DEFNEX_MLOps_Multi_Server_Architecture_v2_PRD.md                # authoritative target
```
