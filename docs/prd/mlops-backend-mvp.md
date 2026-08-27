# PRD — DEFNEX MLOps Backend MVP

## 1. Product Overview

**Product:** DEFNEX MLOps Backend

**Purpose:**  
Menyediakan backend/API untuk mengorkestrasi lifecycle MLOps prototype:

Dataset → Validation → Training → Evaluation → Model Registry → Promotion → Deployment.

Backend menjadi orchestration layer antara frontend dan komponen ML seperti Unsloth Core.

### High-Level Flow

```text
Frontend
    │
    │ REST API
    ▼
┌─────────────────────────────┐
│       FastAPI Backend       │
│                             │
│ Dataset                     │
│ Validation                  │
│ Training                    │
│ Evaluation                  │
│ Model Registry              │
│ Promotion                   │
│ Deployment                  │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
    SQLite          Job Worker
                        │
                        ▼
                   Unsloth Core

Target: **functional MLOps prototype dalam scope 1 bulan**.

---

# 2. Goals

## Primary Goals

1. Menyediakan REST API yang mengikuti `docs/api/openapi.yaml`.
2. Menyediakan persistence layer menggunakan SQLite untuk development.
3. Menggunakan SQLAlchemy ORM agar database dapat dimigrasikan ke PostgreSQL nantinya.
4. Mengelola dataset lifecycle.
5. Menjalankan dataset validation.
6. Membuat dan memonitor training run.
7. Mengintegrasikan training dengan Unsloth Core.
8. Menyimpan model artifact dan metadata/lineage.
9. Menyimpan evaluation results.
10. Mendukung human-controlled promotion/rejection.
11. Menyediakan deployment state.
12. Memungkinkan frontend dan backend dikembangkan secara paralel melalui API contract.
13. Menyediakan environment development yang reproducible menggunakan Docker dan Docker Compose.

## Secondary Goals

* API mudah dites.
* Backend dapat dijalankan secara konsisten di local development.
* Struktur code modular dan mudah dikembangkan.
* Tidak mengunci aplikasi pada SQLite.
* Komponen training/job runner dapat diganti atau ditingkatkan ketika VM tersedia.
* Menyediakan foundation yang cukup untuk deployment ke VM.

---

# 3. Non-Goals

Untuk MVP **tidak membangun**:

* Kubernetes
* Kafka
* Airflow
* MLflow
* distributed training orchestration
* enterprise artifact registry
* cloud-native storage architecture
* complex authentication/authorization
* automatic model promotion
* automatic rollback engine
* multi-tenant architecture
* enterprise monitoring stack
* production-scale job queue

Infrastructure tersebut dapat dipertimbangkan setelah VM dan kebutuhan deployment/production sudah jelas.

---

# 4. Technology Stack

| Layer                | Technology                        |
| -------------------- | --------------------------------- |
| Language             | Python                            |
| API Framework        | FastAPI                           |
| Schema / Validation  | Pydantic                          |
| ORM                  | SQLAlchemy                        |
| Development Database | SQLite                            |
| Database Migration   | Alembic                           |
| ASGI Server          | Uvicorn                           |
| Job Execution        | Lightweight Python worker/process |
| Training             | Unsloth Core                      |
| Testing              | pytest                            |
| API Contract         | OpenAPI 3.1                       |
| Containerization     | Docker                            |
| Local Orchestration  | Docker Compose                    |

## Database Strategy

Development:

```text
FastAPI
   ↓
SQLAlchemy
   ↓
SQLite
```

Future VM/production:

```text
FastAPI
   ↓
SQLAlchemy
   ↓
PostgreSQL
```

Domain/service logic tidak boleh bergantung langsung pada SQLite-specific behavior.

---

# 5. Containerization

Development environment harus dapat dijalankan menggunakan Docker.

Minimal environment:

```text
docker-compose.yml
│
├── backend
│     └── FastAPI
│
├── worker
│     └── Lightweight Python Job Worker
│
└── volume
      └── SQLite database / application data
```

Catatan:

* Backend dan worker boleh menggunakan image yang sama tetapi menjalankan command/process berbeda.
* Jangan membuat container tambahan yang belum dibutuhkan.
* Tidak perlu Redis, Kafka, PostgreSQL, atau message broker untuk MVP.
* Struktur Compose harus memungkinkan komponen ML/training diganti atau diaktifkan ketika VM tersedia.
* Actual GPU/CUDA configuration tidak boleh diasumsikan sebelum environment VM diketahui.

## Docker Requirements

Minimal harus tersedia:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Jika diperlukan, konfigurasi environment dapat menggunakan:

```text
.env.example
```

Jangan commit secrets atau credential ke repository.

---

# 6. Core Entities

Minimal entities:

```text
Dataset
DatasetVersion
ValidationReport
TrainingRun
Model
ModelVersion
Evaluation
PromotionDecision
Deployment
Feedback
```

Relasi utama:

```text
Dataset
   │
   └── DatasetVersion
           │
           ▼
      ValidationReport
           │
           ▼
       TrainingRun
           │
           ▼
       ModelVersion
           │
           ├── Evaluation
           │      │
           │      ▼
           │  PromotionDecision
           │
           ▼
       Deployment
```

`training_run_id` harus menjadi bagian dari persisted model lineage.

---

# 7. Dataset Management

Backend harus mampu:

* mendaftarkan dataset
* membuat dataset version
* menyimpan source information
* menyimpan canonical schema metadata
* mengambil dataset version
* melakukan dataset intake/normalization
* memisahkan train/eval sesuai pipeline
* menyimpan manifest metadata

Canonical training format menggunakan ChatML-style message structure:

```json
{
  "id": "example-001",
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "metadata": {}
}
```

Dataset version menggunakan:

```text
dataset_version: INTEGER
```

Contoh:

```text
dataset_id = legal-qa
dataset_version = 1
```

---

# 8. Validation

Validation service harus mengikuti:

```text
docs/dataset/validation-rules.md
```

Status:

```text
VALID
INVALID
NEEDS_REVIEW
```

Warning bukan status tambahan.

Validation minimal mencakup aturan yang telah ditetapkan dalam validation rules, termasuk:

* schema validation
* required-field validation
* role/message validation
* malformed data
* duplicate detection
* train/eval leakage detection
* quality checks yang telah didefinisikan

Output berupa validation report.

Backend **tidak boleh membuat numeric threshold baru** yang belum ditentukan oleh project.

---

# 9. Training Runs

Training harus diperlakukan sebagai asynchronous job.

Request:

```http
POST /training-runs
```

tidak boleh menunggu proses training selesai.

Contoh lifecycle:

```text
QUEUED
   ↓
RUNNING
   ↓
COMPLETED
```

atau:

```text
QUEUED
   ↓
RUNNING
   ↓
FAILED
```

Training run harus menyimpan minimal:

* training_run_id
* dataset reference
* training configuration
* status
* timestamps
* execution metadata yang tersedia
* artifact reference ketika selesai
* error information ketika gagal

---

# 10. Lightweight Job Runner

MVP tidak menggunakan:

* Celery
* Redis
* Airflow
* Kafka
* distributed scheduler

Gunakan lightweight Python worker/process.

Model:

```text
FastAPI
   │
   ├── create TrainingRun
   │
   ▼
SQLite
status = QUEUED
   │
   ▼
Python Worker
   │
   ▼
Training Process
   │
   ▼
Unsloth Core
```

Worker bertugas:

1. mengambil queued job
2. mengubah status menjadi `RUNNING`
3. menjalankan training process
4. menangkap stdout/stderr/log metadata yang tersedia
5. menangkap exit status
6. menyimpan hasil training
7. menyimpan artifact location
8. mengubah status menjadi `COMPLETED` atau `FAILED`

Training process harus dipisahkan dari HTTP request lifecycle.

Worker harus dapat dijalankan secara independen dari FastAPI process.

---

# 11. Training Configuration

Baseline project berdasarkan research/project decision saat ini:

```text
Base model: Qwen/Qwen3.8-27B
PEFT method: DoRA
Precision: BF16
4-bit quantization: false
max_seq_length: 4096
```

Nilai configuration lainnya harus configurable.

Jangan meng-hard-code parameter training yang masih experimental atau belum ditetapkan.

Setiap training run harus menyimpan **snapshot training configuration** yang benar-benar digunakan pada run tersebut.

---

# 12. Model Registry

Model artifact utama:

```text
DoRA adapter
```

Merged model / GGUF:

```text
OPTIONAL
```

Minimal lineage:

```text
model_id
model_version
base_model
training_run_id
dataset_id
dataset_version
training_config
artifact_uri
artifact_type
evaluation_id
status
created_at
```

Model Registry dan Deployment harus tetap menjadi dua konsep berbeda.

### Model

Menjawab:

> "Artifact/model version apa yang kita miliki?"

### Deployment

Menjawab:

> "Model version mana yang sedang digunakan di environment tertentu?"

---

# 13. Evaluation

Backend menyimpan evaluation results.

Evidence yang telah ditetapkan:

1. eval loss trend
2. 20-question benchmark / majority-win
3. general-domain regression check

Backend tidak boleh membuat automatic promotion threshold yang belum ditentukan.

Evaluation result harus dapat dikaitkan dengan `model_version`.

---

# 14. Promotion Workflow

State machine:

```text
REGISTERED
     ↓
EVALUATED
     ↓
 ┌───────────┐
 ▼           ▼
PROMOTED   REJECTED
 ▼
DEPLOYED
 ▼
RETIRED
```

Promotion/rejection merupakan **human-controlled decision** untuk MVP.

Promotion decision menyimpan:

```text
decision_id
decided_by
decided_at
evidence_snapshot
rationale
```

`evidence_snapshot` merepresentasikan evidence pada saat keputusan dibuat.

Tidak boleh menggunakan numeric threshold yang belum ditetapkan sebagai automatic gate.

---

# 15. Deployment

Deployment state dipisahkan dari model registry.

Minimal:

```text
deployment_id
model_id
model_version
environment
status
deployed_at
```

Deployment stack konkret masih TBD berdasarkan environment VM dan keputusan serving stack.

Jangan mengasumsikan detail vLLM/llama.cpp sebelum environment tersebut dikonfirmasi.

---

# 16. API Contract

API harus mengikuti:

```text
docs/api/openapi.yaml
```

OpenAPI menjadi **formal API contract/source of truth** antara frontend dan backend.

Domain API minimal:

```text
/datasets
/validation
/training-runs
/models
/evaluations
/promotion
/deployments
/feedback
```

Frontend tidak boleh berkomunikasi langsung dengan:

* Unsloth
* worker
* database
* filesystem internal

Semua komunikasi frontend dilakukan melalui Backend API.

---

# 17. Frontend / Backend Parallel Development

Arsitektur development:

```text
                OpenAPI Contract
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Frontend           Backend
          + Mock API          Implementation
```

Frontend dapat menggunakan mock API berdasarkan OpenAPI sementara backend masih dikembangkan.

Backend menggunakan OpenAPI sebagai contract dan tidak boleh mengubah request/response schema secara ad-hoc.

---

# 18. Error Handling

Semua endpoint menggunakan standard error envelope.

Contoh:

```json
{
  "error": {
    "code": "DATASET_NOT_FOUND",
    "message": "Dataset version was not found",
    "details": {}
  }
}
```

HTTP status code dan schema harus mengikuti OpenAPI contract.

---

# 19. Testing

## Unit Tests

Minimal:

* dataset service
* validation service
* training state transitions
* model registry service
* evaluation service
* promotion state transitions

## API Tests

Minimal:

* request validation
* response schema
* error response
* lifecycle transitions

## Integration Tests

Minimal lifecycle:

```text
Dataset
   ↓
Validation
   ↓
Training Run
   ↓
Model
   ↓
Evaluation
   ↓
Promotion
```

Actual GPU training tidak perlu dijalankan pada setiap test.

Gunakan mock/stub training runner untuk local testing.

---

# 20. Development Without VM

Karena akses VM belum tersedia, seluruh backend harus tetap dapat dikembangkan dan dites secara lokal.

Local environment:

```text
Docker Compose
      │
      ├── FastAPI
      │
      ├── Worker
      │
      └── SQLite
```

Training dapat menggunakan mock runner:

```text
POST /training-runs
        ↓
Mock Worker
        ↓
COMPLETED
        ↓
Fake/Test Artifact
```

Tujuannya adalah memungkinkan seluruh lifecycle API diuji tanpa GPU.

Ketika VM tersedia:

```text
Mock Training Runner
        ↓
Actual Unsloth Runner
```

API contract tidak boleh perlu diubah hanya karena training runner diganti.

---

# 21. Project Structure

Agent harus memilih struktur final berdasarkan kebutuhan implementasi, tetapi minimal harus memisahkan:

```text
backend/
├── app/
│   ├── api/
│   ├── schemas/
│   ├── models/
│   ├── services/
│   ├── workers/
│   ├── db/
│   └── main.py
│
├── tests/
│
├── alembic/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

Struktur dapat disesuaikan jika agent menemukan alasan teknis yang lebih baik.

Jangan membuat abstraction layer yang tidak diperlukan hanya demi mengikuti pola arsitektur tertentu.

---

# 22. AI Agent Implementation Order

**AI agent WAJIB mengerjakan backend secara bertahap berdasarkan domain.**

Jangan mencoba mengimplementasikan seluruh backend sekaligus.

Urutan:

## Phase 0 — Foundation

Implementasikan:

* project structure
* Python environment
* FastAPI application
* configuration
* SQLAlchemy setup
* SQLite
* Alembic
* Dockerfile
* Docker Compose
* health endpoint
* testing foundation

**Output:**

Backend dapat dijalankan melalui Docker Compose dan database dapat dibuat/migrated.

---

## Phase 1 — Dataset Domain

Implementasikan:

* Dataset
* DatasetVersion
* dataset API
* persistence
* schemas
* service layer
* tests

Validation belum perlu diimplementasikan penuh pada phase ini.

**Output:**

Dataset lifecycle dapat dibuat dan dibaca melalui API.

---

## Phase 2 — Validation Domain

Implementasikan:

* ValidationReport
* validation service
* validation API
* validation rules
* duplicate detection
* leakage detection
* tests

Gunakan `docs/dataset/validation-rules.md` sebagai source of truth.

**Output:**

Dataset dapat divalidasi dan menghasilkan validation report.

---

## Phase 3 — Training Domain

Implementasikan:

* TrainingRun entity
* training API
* training state machine
* training configuration schema
* lightweight worker
* mock training runner
* tests

Jangan langsung bergantung pada VM atau GPU.

**Output:**

Training lifecycle dapat dijalankan secara end-to-end menggunakan mock runner.

---

## Phase 4 — Model Registry Domain

Implementasikan:

* Model
* ModelVersion
* artifact metadata
* lineage
* model registry API
* artifact storage abstraction
* tests

Gunakan:

```text
docs/registry/model-artifact-versioning-lineage.md
```

sebagai source of truth.

**Output:**

Training run dapat menghasilkan dan mendaftarkan model artifact beserta lineage.

---

## Phase 5 — Evaluation Domain

Implementasikan:

* Evaluation entity
* evaluation API
* evaluation result persistence
* relationship ke ModelVersion
* tests

Gunakan evidence yang sudah ditentukan project.

**Output:**

Model dapat memiliki evaluation record yang terdokumentasi.

---

## Phase 6 — Promotion Domain

Implementasikan:

* PromotionDecision
* promotion/rejection API
* state transition validation
* evidence snapshot
* rationale
* tests

Gunakan:

```text
docs/registry/model-promotion-approval-workflow.md
```

sebagai source of truth.

**Output:**

Model dapat dipromosikan atau ditolak secara human-controlled.

---

## Phase 7 — Deployment Domain

Implementasikan:

* Deployment entity
* deployment API
* deployment state
* model/deployment relationship
* deployment abstraction
* tests

Actual serving implementation dapat tetap mock/TBD sampai VM dan serving stack tersedia.

**Output:**

Model version dapat memiliki deployment state.

---

## Phase 8 — End-to-End Integration

Integrasikan seluruh domain:

```text
Dataset
   ↓
Validation
   ↓
Training Run
   ↓
Model Registry
   ↓
Evaluation
   ↓
Promotion
   ↓
Deployment
```

Pastikan:

* IDs konsisten
* lineage lengkap
* state transitions valid
* API contract konsisten
* database persistence bekerja
* worker bekerja
* Docker Compose dapat menjalankan environment
* integration tests tersedia

---

# 23. Agent Working Rules

AI agent harus:

1. Membaca dokumen WBS/architecture yang relevan sebelum mengimplementasikan domain.
2. Mengikuti `docs/api/openapi.yaml` sebagai API contract.
3. Tidak mengubah business semantics tanpa alasan.
4. Tidak mengarang threshold, metric, atau requirement yang belum ditentukan.
5. Jika menemukan konflik antar-dokumen, **berhenti pada domain tersebut dan laporkan conflict** sebelum membuat keputusan besar.
6. Mengimplementasikan satu phase/domain pada satu waktu.
7. Menjalankan tests setelah setiap phase.
8. Tidak melakukan premature abstraction.
9. Tidak menambahkan infrastructure yang belum dibutuhkan.
10. Menjaga backend tetap dapat dijalankan tanpa VM.
11. Menggunakan mock/stub untuk komponen yang membutuhkan GPU atau VM.
12. Memastikan Docker environment reproducible.
13. Memperbarui dokumentasi teknis jika perubahan implementation memang memerlukannya.
14. Tidak mengubah OpenAPI contract hanya untuk mempermudah implementasi tanpa mengevaluasi impact ke frontend.
15. Setelah setiap domain selesai, memberikan summary:

    * files changed
    * functionality implemented
    * tests executed
    * known limitations
    * next domain

---

# 24. Definition of Done

Backend MVP dianggap selesai apabila:

* [ ] FastAPI application berjalan.
* [ ] OpenAPI specification tervalidasi.
* [ ] SQLite persistence berjalan.
* [ ] SQLAlchemy models tersedia.
* [ ] Alembic migration tersedia.
* [ ] Dockerfile tersedia.
* [ ] Docker Compose environment berjalan.
* [ ] Dataset API berjalan.
* [ ] Validation API berjalan.
* [ ] Training Run API berjalan.
* [ ] Lightweight worker berjalan.
* [ ] Mock training runner berjalan.
* [ ] Model registry berjalan.
* [ ] Evaluation API berjalan.
* [ ] Promotion/rejection workflow berjalan.
* [ ] Deployment state dapat dikelola.
* [ ] Error handling konsisten.
* [ ] Unit tests tersedia.
* [ ] API tests tersedia.
* [ ] End-to-end integration test tersedia.
* [ ] Actual Unsloth training dapat diintegrasikan setelah VM tersedia.
* [ ] Tidak ada dependency enterprise infrastructure yang tidak diperlukan.

---

# 25. Success Criteria

MVP harus mampu mendemonstrasikan lifecycle berikut:

```text
Create Dataset
      ↓
Create Dataset Version
      ↓
Validate Dataset
      ↓
Create Training Run
      ↓
Queue Job
      ↓
Run Mock/Actual Training
      ↓
Register Model Artifact
      ↓
Attach Evaluation
      ↓
Human Promotion Decision
      ↓
Deploy Model
      ↓
Track Deployment
```

Success bukan diukur dari jumlah infrastructure yang dibangun.

**Tujuan utama adalah membuktikan closed-loop MLOps lifecycle secara functional, reproducible, dan dapat dikembangkan lebih lanjut ketika VM tersedia.**

---

# 26. Current Known Constraints / TBD

Hal-hal berikut **jangan diasumsikan oleh agent**:

* VM environment dan akses GPU.
* CUDA/driver version pada VM.
* Exact deployment/serving implementation.
* vLLM vs llama.cpp final decision.
* Numeric promotion/evaluation thresholds.
* Deployment authorization roles.
* Production PostgreSQL configuration.
* Production artifact storage.
* Production authentication/authorization.

Seluruh item tersebut dapat ditentukan setelah environment dan requirement project tersedia.

---

# 27. Guiding Principle

> **Build the smallest complete closed loop first.**

Prioritaskan:

```text
Correctness
   ↓
Traceability
   ↓
Reproducibility
   ↓
Testability
   ↓
Extensibility
```

Bukan:

```text
Complexity
   ↓
Infrastructure
   ↓
Premature Scaling
```

MVP harus cukup sederhana untuk selesai dalam timeline project, tetapi memiliki boundary yang jelas sehingga dapat dikembangkan menjadi platform MLOps yang lebih lengkap setelah kebutuhan dan infrastructure production tersedia.