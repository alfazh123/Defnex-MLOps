# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

Two-app monorepo. `ml-close-loop-be/` (FastAPI, the real work) and `ml-close-loop-fe/`
(React Router 8 app, still close to the starter template — only `home` and `chat` routes exist).
Almost every command below runs from `ml-close-loop-be/`.

`ml-close-loop-be/scripts/ralph/` holds an autonomous-agent harness (PRD + progress log +
`CLAUDE.md` for that agent). It is not instructions for interactive sessions, but
`scripts/ralph/progress.txt` has a `## Codebase Patterns` section worth reading before large changes.

## Target architecture (PRD v2)

`DEFNEX_MLOps_Multi_Server_Architecture_v2_PRD.md` (repo root) is the **authoritative target**
architecture. The current codebase is the Phase 0 prototype being migrated toward it;
`AUDIT_MLOPS_v2.md` records the gaps per area. Do not implement anything that contradicts
the PRD without a written decision. Key targets (details live in the PRD):

- Multi-server control plane: FastAPI + PostgreSQL + Redis + Celery + MinIO. Training runs on a
  GPU VPS (Unsloth) or Colab runner — never inside the HTTP handler (PRD §10.1).
- Environments `staging` and `production`; no model goes straight from training to production
  (PRD §41 P4). Promotion is authorized; rollback keeps prior immutable artifacts.
- Stable abstractions that must not be coupled to one implementation:
  `TrainingProvider`, `ArtifactStore`, `SecretStore`, `JobManager`, `ComputeResource`,
  `InferenceTarget`, `ModelRegistry` (PRD §48).

## Commands (from `ml-close-loop-be/`)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,intake]"
alembic upgrade head
uvicorn app.main:app --reload            # :8000, docs at /docs

.venv/bin/pytest tests/ -q               # pytest.ini_options enforce --cov-fail-under=80
.venv/bin/pytest tests/test_promotion_api.py::test_name -q --no-cov   # single test
ruff check . && ruff format --check .    # CI pins ruff==0.15.13

alembic revision --autogenerate -m "msg" && alembic upgrade head
.venv/bin/python seed.py --reset         # populate SQLite with a full 4-model closed loop for UI work
```

Docker: `cp .env.example .env && docker compose up --build` — `backend` runs migrations then
uvicorn; `worker` runs `python -m app.workers.training_worker`. Unsloth Studio is commented out
in `docker-compose.yml` (needs an NVIDIA GPU).

CI (`.github/workflows/ci.yml`) gates on ruff → pytest → pip-audit, all against `ml-close-loop-be/`.

## Backend architecture

Strict three-layer split; keep it:

- `app/api/*.py` — routers only. Resolve dependencies, call a service, `db.commit()`, return a schema.
  Services never commit (they `db.flush()`); the router owns the transaction boundary.
- `app/services/*.py` — all business rules and state transitions. Raise `ValueError` on an illegal
  transition; the router translates it into an `APIError` with the right code (e.g. `409
  DECISION_NOT_ALLOWED`).
- `app/models/` SQLAlchemy 2.0 `Mapped[...]` ORM, `app/schemas/` Pydantic request/response.

Errors: always raise `app.api.errors.APIError(status, CODE, message)`. It pre-wraps the body in the
`{"error": {"code", "message"}}` envelope that `openapi.yaml` and the frontend expect; the handler in
`main.py` passes that dict through untouched. Plain `HTTPException` breaks the contract.

Routing: every router is mounted under a single `/api/v1` `APIRouter` in `main.py`, and
`ApiVersionRedirectMiddleware` 301-redirects legacy unprefixed paths. Adding a new top-level resource
means adding its prefix to `_LEGACY_PREFIXES` too.

Auth: `get_current_user` / `require_admin` in `app/api/deps.py`, plus `get_pagination` and
`get_filters` — reuse those `Depends` rather than re-parsing `page`/`size`/`status`/`search`.

## The closed loop (spans several files)

`Dataset → DatasetVersion → ValidationReport → TrainingRun → ModelVersion → PromotionDecision → Deployment`

Statuses are plain strings guarded by per-service transition tables, not enums:

- `TrainingRun`: `PENDING → RUNNING → COMPLETED|FAILED` (`training_service._VALID_TRANSITIONS`).
- `ModelVersion`: `REGISTERED → EVALUATED` (auto, once all three eval signals are present in
  `model_service.evaluate`) `→ PROMOTED|REJECTED → DEPLOYED → RETIRED`.
- `PromotionDecision` covers `PROMOTED`/`REJECTED`/`ROLLBACK` in one table; promotion is
  human-triggered only — never add automatic promotion on numeric thresholds.
- `Deployment` deliberately has **no** foreign key to the model registry; registry and deployment
  are kept as separate concepts. `deployment_service.deploy` retires the previous `DEPLOYED` version.

Two training execution paths exist and are easy to confuse:

1. `POST /api/v1/training-runs` calls `unsloth_client.start_training` inline and moves the run to
   `RUNNING`. Real path; failures are logged, not raised.
2. `app/workers/training_worker.py` polls the oldest `PENDING` run with a swappable `TrainingRunner`
   (currently `MockTrainingRunner`) — and it is the **only** caller of
   `model_service.register_model_version`, so without the worker nothing ever creates a
   `ModelVersion` and the loop never closes.

`app/worker.py` is unrelated: a stdlib inbox/outbox file poller kept as a broker-free job stub.

External boundaries are `Protocol`s with mock implementations because the real stack is undecided:
`services/serving.py` (`ServingBackend`), `services/artifact_storage.py`, `workers/training_worker.TrainingRunner`.
Swap the implementation, don't change the callers.

## Constraints

- `openapi.yaml` is the frozen contract with the frontend, asserted by `tests/test_contract.py`.
  Change the implementation to match the spec, not the reverse.
- `tests/conftest.py` is marked FROZEN — shared fixtures (`client`, `db_session`, `admin_token`,
  `user_token`, `count_queries`) are edited only with team review. `client` binds a per-test
  in-memory SQLite engine and exposes it as `client.engine`.
- Stack target per PRD v2: FastAPI + PostgreSQL + Redis + Celery + MinIO (PRD §10.1, §23, §32).
  Still no Kubernetes, Kafka, Airflow, MLflow, or Vault (PRD §4 Non-Goals / §41 Principle 9).
  The backend worker is currently stdlib polling (`app/workers/training_worker.py`) — when moving
  to Celery, route through a `JobManager` abstraction, never couple the domain to Celery directly.
- Don't invent evaluation metrics, numeric thresholds, or environment ladders that aren't already in
  the code or `openapi.yaml`; existing docstrings cite their source doc section for exactly this reason.

## Constraints proyek (jangan dilanggar tanpa persetujuan eksplisit)

- GPU adalah H100 80GB **shared** dengan tenant lain di luar kendali kita. Jangan
  menjalankan training, `docker compose up` untuk service GPU, atau perintah apa
  pun yang menyentuh GPU kecuali issue secara eksplisit menyuruhnya dan
  menyebutkan langkah verifikasi VRAM sebelum-sesudah. Jangan membunuh proses GPU
  tak dikenal atau me-reset GPU yang dipakai tenant lain (PRD §18.4 / §43).
- Stack target per PRD v2: PostgreSQL + Redis + Celery + MinIO. Prototype saat ini
  masih broker-free (stdlib polling), jadi jangan tulis kode baru yang **mengunci**
  arsitektur ke tanpa-broker, dan migrasi ke Celery/PostgreSQL wajib lewat
  `JobManager` / layer abstraction yang sudah ada — jangan menyambung domain
  langsung ke Celery.
- DB: SQLAlchemy 2.0 ORM + Alembic. PostgreSQL menjadi basis otoritatif (PRD §23.1);
  SQLite hanya untuk dev lokal. Setiap perubahan skema wajib lewat migrasi
  baru, jangan edit migrasi lama yang sudah ada.
- Concurrency training tetap 1 (serial) di tahap MVP: GPU lock `flock` di satu titik
  eksekusi masih sah (PRD §18.2), tapi operasi deployment yang menyentuh GPU yang
  sama harus ikut mekanisme lock yang sama (PRD §18.1), dan lock mustahil dibiarkan
  menggantung — timeout/cleanup via `finally` (PRD §18.4).
- peft_method yang diterima saat ini: `lora`, `qlora`. `dora`/`qdora` DITOLAK
  di validasi request sampai ada keputusan governance tertulis — jangan pernah
  memetakannya diam-diam ke LoRA biasa.
- Serving: vLLM via Docker + NVIDIA CDI. Training: Unsloth native di venv
  terpisah dari venv serving.
- Versioning adapter: `{project}-{base_model}-v{N}`, N dari database
  (MAX+1 di dalam transaksi dengan row lock atau unique constraint + retry),
  bukan dari menghitung folder.
- Status "production saat ini" harus punya SATU sumber kebenaran. Jangan
  membuat dua kolom/tabel yang bisa saling divergen untuk fakta yang sama.
- Wajib ada `staging` sebelum `production`: tidak ada model yang langsung promosi
  dari training ke production (PRD §16, §41 P4). Promosi butuh otorisasi; kandidat
  yang gagal tetap non-production; rollback harus mungkin memakai artifact
  immutabel yang lama (PRD §43).
- Artifact model/version bersifat immutabel; bytes besar disimpan di object storage
  (MinIO/S3), bukan di PostgreSQL (PRD §41 P2/P3). Artifact yang dipakai untuk
  rollback jangan dihapus hanya karena model tidak lagi aktif (PRD §43).
- Kredensial diperlakukan sebagai reference ke secret storage (`secret://…`),
  bukan data aplikasi; jangan pernah menaruh secret tertulis di kode/config
  versi (PRD §21, §43).

## Standar kualitas wajib untuk setiap issue

1. **Fase THINK dulu, fase IMPLEMENT kemudian.** Selalu keluarkan rencana
   tertulis (lihat format di Bagian B) sebelum menyentuh kode, dan berhenti
   menunggu persetujuan sebelum lanjut implementasi.
2. Kutip `file:baris` untuk setiap klaim tentang kode yang sudah ada. Jangan
   mendeskripsikan dari ingatan/asumsi.
3. Setiap perubahan perilaku wajib disertai unit test baru atau diperbarui.
   Test harus mencakup jalur gagal/edge case, bukan cuma happy path.
4. Setelah implementasi, **jalankan test sungguhan** (`pytest`) dan laporkan
   hasil asli (jumlah pass/fail, bukan estimasi). Jangan menandai selesai
   tanpa run test yang benar-benar dieksekusi di sesi ini.
5. Jangan diam-diam memperluas atau mempersempit scope issue. Kalau menemukan
   bug lain di luar scope saat investigasi, laporkan sebagai temuan terpisah,
   jangan langsung diperbaiki dalam PR yang sama.
6. Jangan diam-diam mendowngrade atau menyederhanakan fitur yang diminta
   (contoh kasus nyata: dora/qdora pernah dipetakan diam-diam ke LoRA tanpa
   pemberitahuan — ini pola yang harus dihindari, bukan hanya untuk DoRA).
7. Update README/openapi.yaml hanya untuk bagian yang benar-benar berubah,
   dan pastikan klaimnya bisa diverifikasi dari kode (tidak ada lagi klaim
   coverage/CI yang tidak didukung artefak nyata).

LOCAL DEVELOPMENT ONLY.

You may:
- inspect the repository
- read and modify source files
- run tests
- run local development servers
- run local Docker Compose
- create migrations
- inspect git history/status

Do NOT:
- access production servers
- SSH into company/shared infrastructure
- modify remote GPU servers
- kill GPU processes
- reset GPUs
- delete databases
- run rm -rf on project or unknown directories
- git push --force
- modify branches outside this repository
- expose or print secrets
- commit .env or credentials

Before any destructive or irreversible operation, STOP and ask for confirmation.