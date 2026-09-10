# PRD — DEFNEX MLOps Multi-Server Architecture v2

**Status:** Draft / Architecture Baseline  
**Document Type:** Product Requirements Document (PRD)  
**Scope:** DEFNEX MLOps control plane, training orchestration, artifact/registry, staging/production inference, infrastructure management, and operational observability  
**Primary Audience:** DEFNEX engineering team, backend/frontend developers, DevOps/MLOps engineers, technical leads, project mentors  
**Version:** 2.0  
**Date:** 2026-09-07

---

## 1. Executive Summary

DEFNEX MLOps is designed as a domain-specific MLOps control plane for managing the lifecycle of fine-tuned defense-oriented language models. The system must support dataset management, training-job orchestration, model/adapter versioning, evaluation, staging deployment, production promotion, rollback, feedback-driven retraining, and infrastructure management.

The target architecture separates the **control plane** from **compute and serving environments**:

- **Backend VPS / Control Plane** manages the API, database, job queue, model registry, dataset metadata, deployment orchestration, credentials, and infrastructure registry.
- **Training Compute** may be a persistent GPU VPS or an ephemeral/on-demand Google Colab runner.
- **Inference Staging** is used to validate a model with DEFNEX and chatbot integrations before production.
- **Inference Production** serves the approved model to production consumers.
- **Object storage** holds immutable datasets/artifacts so model bytes are not tied to a particular compute machine.

The architecture deliberately adopts proven lifecycle concepts such as immutable versions, aliases/pointers, lineage, quality gates, and environment promotion, without requiring DEFNEX MLOps to become a general-purpose platform such as MLflow.

---

## 2. Problem Statement

The existing MLOps prototype demonstrates a working single-host closed loop:

```text
Dataset
  ↓
Training
  ↓
Evaluation
  ↓
Model/LoRA Artifact
  ↓
vLLM Deployment
  ↓
Inference
  ↓
Feedback
```

However, the current implementation is architecturally constrained by several assumptions:

1. Training and inference run on the same host.
2. There is only one training provider.
3. There is only one inference environment.
4. Artifacts are tied to a local/shared filesystem.
5. GPU locking is host-local.
6. There is no infrastructure/server registry.
7. Remote GPU/Colab execution is not implemented as a provider abstraction.
8. There is no formal staging → production promotion ladder.
9. Training jobs do not have robust stale-job recovery.
10. Automated CD is incomplete.
11. Engineering observability is limited to logs/health checks rather than a complete metrics/traces/logs pipeline.

The new PRD must preserve the working parts while removing these architectural constraints in a controlled, incremental way.

---

## 3. Goals

### 3.1 Primary Goals

- Provide one web application for the DEFNEX MLOps lifecycle.
- Separate control-plane responsibilities from GPU compute and inference serving.
- Support multiple training providers through a stable provider interface.
- Support both persistent GPU VPS training and ephemeral/on-demand Colab training.
- Make model and adapter versions immutable and traceable.
- Store artifact bytes independently from model registry metadata.
- Provide a staging environment for model validation before production.
- Support production promotion and rollback without retraining.
- Prevent simultaneous training/deployment GPU contention.
- Manage infrastructure and deployment targets from configuration rather than hard-coded server assumptions.
- Support feedback → dataset curation → retraining as a closed loop.
- Provide operational observability for API, jobs, training, deployments, and inference.
- Keep the architecture extensible without introducing unnecessary infrastructure complexity.

### 3.2 Secondary Goals

- Make the system usable by normal users for routine training workflows.
- Give administrators control over infrastructure, credentials, compute resources, and deployment settings.
- Enable future migration from MinIO to company-managed object storage with minimal application changes.
- Enable future migration from a Celery executor to another execution engine without rewriting the API/domain layer.

---

## 4. Non-Goals

The following are intentionally outside the first production-capable scope:

- Building a full replacement for MLflow.
- Supporting arbitrary distributed training frameworks.
- Treating Google Colab as an always-on worker pool.
- Automatically controlling arbitrary personal Google accounts.
- Storing Google account passwords inside DEFNEX MLOps.
- Building a Kubernetes platform before it is required.
- Adding Kafka/RabbitMQ solely for architectural completeness.
- Building a full multi-tenant enterprise IAM platform.
- Deploying a large observability stack before the required signals are known.
- Allowing production deployment before staging/quality gates pass.

---

## 5. Users and Roles

### 5.1 ADMIN

Administrative user with access to:

- infrastructure registry
- training providers
- compute resources
- inference environments
- credential references
- notebook/runner links
- system configuration
- audit logs
- model and dataset management
- deployment controls
- rollback

### 5.2 USER

Normal user with access to:

- datasets assigned to them
- training jobs
- training progress
- model versions they are authorized to see
- evaluation results
- staging status
- production model status
- inference/testing interfaces
- feedback submission

Users must not be able to read secret values or alter protected infrastructure settings unless explicitly authorized.

### 5.3 Role Hierarchy

```text
ADMIN
  └── includes all USER capabilities

USER
  └── operational/model lifecycle capabilities
      subject to RBAC/policy
```

---

## 6. Target Architecture

### 6.1 High-Level Topology

```text
                         ┌─────────────────┐
                         │    FRONTEND     │
                         │ Web Application │
                         └────────┬────────┘
                                  │ HTTPS
                                  ▼
                       ┌─────────────────────┐
                       │     BACKEND VPS     │
                       │     CONTROL PLANE   │
                       │                     │
                       │ FastAPI             │
                       │ PostgreSQL          │
                       │ Redis               │
                       │ Celery Worker       │
                       │ Model Registry      │
                       │ Dataset Metadata    │
                       │ Job Manager         │
                       │ Deployment Manager  │
                       │ Infrastructure Reg. │
                       │ Secret References   │
                       │ Observability        │
                       └──────────┬──────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
      ┌───────────┐        ┌─────────────┐       ┌─────────────┐
      │  MinIO    │        │ Training    │       │ Inference   │
      │ Artifacts │        │ Providers   │       │ Targets     │
      └───────────┘        └──────┬──────┘       └──────┬──────┘
                                  │                     │
                         ┌────────┴────────┐      ┌─────┴────────┐
                         │                 │      │              │
                         ▼                 ▼      ▼              ▼
                    ┌─────────┐      ┌─────────┐ STAGING      PRODUCTION
                    │ GPU VPS │      │  Colab  │   vLLM          vLLM
                    │ Worker  │      │ Runner  │
                    └─────────┘      └─────────┘
```

### 6.2 Reference Server Topology

The architecture must support the following logical server roles without hard-coding these names:

```text
server-1  backend/control plane
server-2  training compute
server-3  inference staging
server-4  inference production
```

For MVP deployment on limited infrastructure, roles may temporarily coexist on fewer physical machines. The application must still model them as separate logical resources.

### 6.3 Current Prototype Mapping

The current single GPU VM can be represented as a compute resource while the architecture is being developed. The application should not assume that the current VM is permanently both training and inference.

---

## 7. Core Product Flow

### 7.1 Standard Training-to-Production Flow

```text
Create/Select Dataset
        ↓
Create Training Configuration
        ↓
Create Training Job
        ↓
Queue
        ↓
Select Training Provider / Compute Resource
        ↓
Training
        ↓
Evaluation
        ↓
Artifact Packaging
        ↓
Register Immutable Model Version
        ↓
Deploy STAGING
        ↓
DEFNEX + Chatbot Integration Test
        ↓
Quality Gate / Approval
        ↓
Promote to PRODUCTION
        ↓
Monitor
        ↓
Feedback
        ↓
Dataset Curation
        ↓
Next Training Run
```

### 7.2 Training UI Flow

The frontend must allow the user to define:

- dataset version
- dataset format/schema
- base model
- model family
- training hyperparameters
- LoRA configuration
- maximum sequence length
- training compute/provider
- compute resource
- version/checkpoint name or version intent
- evaluation configuration

The backend creates a training job and never executes GPU-intensive training inside the HTTP request handler.

---

## 8. Frontend Product Requirements

## 8.1 Dashboard

Show high-level operational status:

- active training jobs
- queued jobs
- failed jobs
- current staging model
- current production model
- recent deployments
- inference health
- major errors
- resource availability

## 8.2 Datasets

Capabilities:

- upload dataset file(s)
- select/describe schema
- create immutable dataset version
- inspect metadata
- validate format
- track source and checksum
- associate dataset with training jobs and model lineage
- optionally curate feedback-derived records

Example:

```text
Dataset Family: defense-scenarios
Version: defense-scenarios-v1
Schema: scenario schema v1
Checksum: <sha256>
Created: 2026-09-07
```

## 8.3 Training

Capabilities:

- create training job
- select model/base model
- configure hyperparameters
- select provider
- select compute resource
- view live status
- view progress
- view loss/metrics
- view logs
- view heartbeat
- view artifact status
- cancel eligible jobs
- retry eligible failed/stale jobs

### 8.3.1 Progress Display

Recommended fields:

```text
Status: RUNNING
Epoch: 2 / 3
Step: 840 / 1200
Loss: 1.42
Elapsed: 01:47:32
Last heartbeat: 4 sec ago
Provider: Google Colab
Compute: colab-training-a
```

## 8.4 Model Registry

Show:

- model family
- version
- base model
- dataset version
- training job
- artifact ID
- evaluation score(s)
- quality-gate result
- aliases/environments
- status
- lineage
- creation timestamp
- checksum

Required actions:

- inspect version
- promote candidate
- deploy staging
- approve production promotion where authorized
- rollback production
- compare versions
- archive previous versions without deleting artifacts

## 8.5 Deployments / Environments

Display:

```text
STAGING
  Version: defnex-qwen2.5-0.5b-v4
  Status: healthy

PRODUCTION
  Version: defnex-qwen2.5-0.5b-v3
  Status: healthy
```

The frontend must clearly distinguish staging from production.

Deployment actions:

- deploy to staging
- run smoke test
- view deployment logs
- promote to production
- rollback production
- redeploy selected version

## 8.6 Infrastructure / Admin

Admin page must expose configuration of:

- servers
- training providers
- compute resources
- inference targets
- runner/notebook links
- workspace paths
- model paths
- vLLM settings
- credential references
- health status
- connectivity status

Users may see read-only infrastructure status as permitted.

## 8.7 External Tool Links

The web application may link to operational tools in a new tab:

- Grafana
- Prometheus
- Swagger/OpenAPI
- MinIO Console

These tools are not required to be embedded inside the application.

---

## 9. Training Provider Architecture

### 9.1 Provider Abstraction

The system must separate:

```text
TrainingProvider
      ↓
ComputeResource
      ↓
TrainingJob
```

A provider is the execution mechanism; a compute resource is the specific resource/account/server used by that provider.

### 9.2 Example Providers

```text
TrainingProvider
├── GPUVPSProvider
└── ColabProvider
```

### 9.3 Example Compute Resources

```text
Google Colab
├── colab-training-a
├── colab-training-b
└── colab-training-c

GPU VPS
├── server-2
└── server-5
```

### 9.4 Provider Contract

Conceptually:

```python
class TrainingProvider:
    def submit(self, job) -> str: ...
    def get_status(self, external_job_id) -> ...: ...
    def cancel(self, external_job_id) -> ...: ...
    def collect_result(self, external_job_id) -> ...: ...
```

The actual interface may differ, but the domain must not depend directly on a specific provider implementation.

### 9.5 Persistent GPU VPS Worker

A GPU VPS may run a persistent worker process/container. It can:

- claim queued jobs
- acquire the GPU lock
- execute training
- report heartbeat/progress
- upload artifacts to object storage
- finalize the job

### 9.6 Google Colab Runner

Colab must be treated as an **ephemeral/on-demand runner**, not a permanent worker.

The MVP flow is:

```text
Frontend creates job
        ↓
Backend status = QUEUED
        ↓
User clicks "Open Colab Runner"
        ↓
Notebook opens in new browser tab
        ↓
Operator authenticates to the appropriate Google account
        ↓
Operator runs the notebook
        ↓
Notebook claims/executes the job
        ↓
Artifact uploaded
        ↓
Backend marks completion
```

The frontend may store and expose a notebook URL configured by ADMIN.

### 9.7 Colab Access Model

DEFNEX MLOps must **not** store a Google account password.

The system should show which compute resource is intended, for example:

```text
Compute: Colab Training A
Account: configured training account
Notebook: [Open Colab Runner ↗]
```

The actual Google login is handled by Google/Colab outside DEFNEX MLOps.

If the operator is not authenticated to the intended account or lacks notebook/runtime access, the runner must fail clearly and instruct the operator to use the authorized account/process.

The architecture must not assume that sharing a notebook grants another user access to the subscription/compute resources of the account that owns it.

### 9.8 Colab Usage Policy

The system should not represent Colab as an always-on service. It should be explicitly labeled as on-demand compute because Colab runtime lifetime, availability, and usage limits depend on the selected Colab service/plan.

---

## 10. Job Management and State Machine

### 10.1 Job Manager

Use:

- FastAPI for HTTP/API
- PostgreSQL as system-of-record for job state
- Redis for broker/cache
- Celery worker for asynchronous execution

Training must not execute inside a synchronous HTTP handler.

A `JobManager` abstraction should shield the domain from Celery-specific calls.

### 10.2 Training Job States

Recommended state machine:

```text
QUEUED
  ↓
SUBMITTED
  ↓
STARTING
  ↓
RUNNING
  ↓
EVALUATING
  ↓
ARTIFACT_READY
  ↓
COMPLETED
```

Terminal/error states:

```text
FAILED
CANCELLED
TIMEOUT / STALE
```

### 10.3 State Rules

- `QUEUED`: waiting for compute.
- `SUBMITTED`: provider accepted execution request.
- `STARTING`: worker acquired and is initializing.
- `RUNNING`: training is active.
- `EVALUATING`: training complete and evaluation is running.
- `ARTIFACT_READY`: artifact exists and is verified.
- `COMPLETED`: registry and lineage records are finalized.
- `FAILED`: job ended with an error.
- `CANCELLED`: intentionally stopped.
- `STALE`: worker stopped reporting heartbeat beyond configured threshold.

### 10.4 Heartbeat

Running jobs must update a heartbeat timestamp independently from progress counters.

Heartbeat is required to detect:

- worker crash
- machine reboot
- lost process
- dead communication path
- orphaned job state

### 10.5 Retry

Retry must not silently mutate the original job history.

Preferred behavior:

```text
train-123 FAILED
        ↓ retry
train-124 RETRY_OF=train-123
```

The original record remains immutable as historical evidence.

---

## 11. Dataset Versioning

Dataset files must be immutable once a version is published.

Example:

```text
Dataset Family
  defense-scenarios

Versions
  defense-scenarios-v1
  defense-scenarios-v2
  defense-scenarios-v3
```

Dataset metadata should include:

- dataset ID
- version
- schema version
- file/object references
- checksum
- size
- created timestamp
- author/source
- validation status
- optional tags

Training jobs reference a specific dataset version rather than a mutable dataset name.

---

## 12. Model and Adapter Versioning

### 12.1 Model Family Naming

Version numbers are scoped to a model family.

Examples:

```text
id="qwen25-0.5b"
base_model="Qwen/Qwen2.5-0.5B-Instruct"
version="v1"

id="qwen38-27b"
base_model="Qwen/Qwen3.8-27B"
version="v1"
```

Human-readable names may be:

```text
 defnex-qwen2.5-0.5b-v1
 defnex-qwen3.8-27b-v1
```

Do not encode environment or lifecycle state in the version name.

Avoid names such as:

```text
model-v4-production
latest-final-fixed
```

Use metadata/aliases for state instead.

### 12.2 Immutability

Once a model version is registered:

- artifact bytes cannot be overwritten
- lineage cannot be silently rewritten
- version ID cannot change
- historical evaluation results remain attached to the run

New training results create a new version.

### 12.3 Artifact Contract

Every artifact should track:

- `artifact_id`
- model family
- model version
- base model
- dataset version
- training job ID
- training config/hash
- code commit
- artifact files
- checksum
- storage location
- created timestamp
- creator/provider

---

## 13. Artifact Storage

### 13.1 Object Store

Use **MinIO** as the MVP object store because it is S3-compatible and can run independently of the backend process.

The application must define an `ArtifactStore` abstraction.

```text
ArtifactStore
├── MinIO/S3 implementation
├── Local filesystem implementation (optional dev)
└── Future company object storage implementation
```

### 13.2 Storage Separation

Keep:

```text
PostgreSQL
  = metadata, registry, lineage, job state

MinIO/S3
  = dataset bytes, adapter/model artifacts, logs or large files where appropriate
```

The registry must never depend on a server-local model directory being the authoritative artifact store.

### 13.3 Direct Artifact Upload

For large artifacts, prefer:

```text
Compute
   ↓
MinIO/S3
   ↓
Backend notification/metadata finalization
```

rather than:

```text
Compute
   ↓
FastAPI
   ↓
MinIO
```

The backend should issue constrained/short-lived permissions or presigned operations where appropriate.

---

## 14. Model Registry

### 14.1 Responsibilities

The model registry is the authoritative catalog of:

- model families
- immutable versions
- evaluation results
- quality gate status
- lineage
- aliases
- deployment status
- lifecycle metadata

### 14.2 Version Listing

The API must support both:

```text
GET /models/{model_id}
GET /models/{model_id}/versions
```

The second endpoint is required for deployment selection, rollback UI, and history browsing.

### 14.3 Aliases / Environment Pointers

Use environment pointers such as:

```text
staging    → defnex-qwen2.5-0.5b-v4
production → defnex-qwen2.5-0.5b-v3
```

Aliases are mutable pointers; model versions remain immutable.

### 14.4 Rollback

Rollback is implemented by moving an environment pointer back to a previous immutable version:

```text
production → v4

v4 fails

production → v3
```

No retraining is required for rollback.

### 14.5 Lifecycle Status

Recommended states:

```text
TRAINED
EVALUATED
PROMOTED
STAGING
VALIDATED
PRODUCTION
ARCHIVED
REJECTED
```

`REJECTED` means the candidate failed and must not be promoted.

`ARCHIVED` means the version was previously valid/used but is no longer the active version.

---

## 15. Evaluation and Quality Gates

### 15.1 Evaluation Layers

#### Layer 1 — Structural Gate

Verify:

- valid output format
- required fields
- expected types
- valid JSON/schema where applicable
- no malformed structured responses

#### Layer 2 — Domain Quality / Groundedness

Evaluate whether:

- hypotheses are grounded in scenario indicators/evidence
- recommended actions are consistent with the scenario
- outputs do not introduce unsupported facts unnecessarily
- domain-specific requirements are met

#### Layer 3 — Golden / Regression Set

Maintain a fixed evaluation set for challenger-vs-champion comparison.

The set must be versioned.

Example:

```text
Golden Set v1
Golden Set v2
```

### 15.2 Quality Gate

The backend must decide pass/fail using server-side evaluation results where practical.

A client may submit scores for reporting, but the architecture must not treat arbitrary client-provided scores as automatically trustworthy production evidence.

### 15.3 Champion / Challenger

```text
Champion = current production model
Challenger = candidate model/version
```

Promotion should compare the candidate against configured quality thresholds and, where required, the current champion.

---

## 16. Staging and Production

### 16.1 Environment Model

The architecture must support at least:

```text
staging
production
```

Each environment maps to an `InferenceTarget`.

```text
InferenceTarget
├── inference-staging
└── inference-production
```

### 16.2 Promotion Ladder

```text
TRAINED
   ↓
EVALUATED
   ↓
STAGING
   ↓
VALIDATED
   ↓
PRODUCTION
```

### 16.3 Staging Purpose

Staging is not merely another vLLM endpoint. It is the integration validation environment for:

- DEFNEX integration
- chatbot integration
- request/response compatibility
- adapter loading
- response quality
- latency/smoke checks
- environment-specific issues

Production promotion must remain blocked until required staging checks pass.

---

## 17. vLLM Deployment

### 17.1 Serving Abstraction

Define an `InferenceTarget` / serving abstraction so the application does not hard-code one vLLM host.

Conceptually:

```python
class ServingTarget:
    def deploy(self, model_version, config): ...
    def smoke_test(self): ...
    def rollback(self, version): ...
    def status(self): ...
```

### 17.2 One Active Version per Target

The MVP requires one active model/adapter deployment per logical inference target for the chosen model family.

Example:

```text
STAGING      → v4
PRODUCTION   → v3
```

A previous version remains available in registry/storage but is not simultaneously active on that target unless explicitly supported later.

### 17.3 Adapter Deployment

For adapter-only changes:

1. load adapter
2. run smoke test
3. confirm expected model/version
4. move environment pointer
5. unload previous adapter where applicable

The implementation may use vLLM runtime LoRA loading/unloading when supported.

### 17.4 Base Model Changes

A base-model change should normally use a controlled recreate/redeploy process rather than assuming an adapter hot-swap is sufficient.

### 17.5 GPU Lock During Deployment

Deployment operations that mutate the GPU-serving process must participate in the GPU concurrency policy.

The goal is to prevent:

```text
Training uses GPU
        ×
Deployment restarts/loads GPU model
```

A single-host lock is acceptable for the current machine, but the abstraction must support cross-host locking later.

---

## 18. GPU Concurrency and Locking

### 18.1 Requirement

Training must acquire an exclusive compute lock before starting GPU-intensive work.

Deployment operations that materially affect the same GPU resource must use the same coordination mechanism.

### 18.2 Current Implementation

A host-local filesystem lock (`flock` or equivalent) is acceptable for a single-machine MVP.

### 18.3 Future Multi-Host Implementation

When resources span multiple servers, the locking abstraction may move to a centralized/distributed mechanism.

Potential implementations can include database advisory locks or another distributed lock mechanism depending on the deployment topology.

### 18.4 Safety Rules

- Do not kill unknown GPU processes.
- Do not reset shared GPUs.
- Lock acquisition timeout must produce a clear job state/error.
- Lock release must be guaranteed using structured cleanup/finally semantics.

---

## 19. Infrastructure Registry

### 19.1 Server Entity

A server/resource record should include:

- server ID
- name
- role
- environment
- host/IP
- SSH port
- credential reference
- workspace
- artifact path or transport config
- service endpoint
- provider type
- expected GPU information
- health status
- last health check

### 19.2 Server Roles

Examples:

```text
BACKEND
TRAINING
INFERENCE_STAGING
INFERENCE_PRODUCTION
```

### 19.3 Compute Resource

A compute resource is an addressable unit used by a provider.

Examples:

```text
server-2
colab-training-a
colab-training-b
```

### 19.4 Configuration Over Hard-Coding

The frontend must not need a redesign when:

- the staging server changes
- the production server changes
- a Colab account is replaced
- another GPU VPS is added
- a notebook URL changes

These must be configuration/data-model changes.

---

## 20. Remote Connectivity and Artifact Transport

### 20.1 SSH

SSH is used for control/transport when a remote GPU VPS requires it.

Potential operations:

- health check
- command execution
- deployment orchestration
- workspace preparation
- artifact synchronization
- service restart/recreate

### 20.2 SFTP / Object Storage

For large model artifacts, prefer object-storage-based transport whenever practical.

SSH/SFTP remains useful for operational control or environments where object storage access is not available.

The architecture must not assume that SSH means a shared filesystem.

### 20.3 Artifact Transfer Contract

A deployment pipeline should explicitly track:

```text
artifact_id
source storage
transfer target
checksum before transfer
checksum after transfer
transfer status
transfer timestamp
```

A deployment must not proceed when artifact verification fails.

---

## 21. Secret Management

### 21.1 Secret Types

Potential secrets include:

- SSH private keys
- API tokens
- database credentials
- object-store credentials
- external-provider credentials
- signing keys

### 21.2 SecretStore Abstraction

```text
SecretStore
├── EncryptedSecretStore (MVP)
├── VaultSecretStore (production option)
└── Company Secret Manager implementation
```

### 21.3 Vault Policy

Vault should **not** be a mandatory dependency in the core application Compose stack.

Production should ideally use an external Secret Manager/Vault service.

A local development Compose profile may include Vault for testing the integration, but taking DEFNEX MLOps down must not unnecessarily take the organization's shared secret infrastructure down.

### 21.4 Credential References

Database records should store:

```text
credential_ref = secret://infrastructure/server-2/ssh
```

rather than plaintext secret values.

### 21.5 Security Requirements

- TLS for frontend → backend
- no secrets in logs
- no secrets in frontend responses
- RBAC around credential operations
- encrypted secret storage
- audit access to sensitive operations
- rotation supported by the abstraction

---

## 22. Backend Services

The backend should be modular by domain rather than split into many independently deployed microservices.

Recommended logical modules:

```text
backend/
├── auth/
├── users/
├── datasets/
├── training/
├── jobs/
├── evaluation/
├── artifacts/
├── registry/
├── deployments/
├── infrastructure/
├── providers/
├── serving/
├── feedback/
├── secrets/
├── observability/
└── common/
```

A modular monolith is acceptable for the control plane initially.

---

## 23. Database

### 23.1 PostgreSQL

Use PostgreSQL as the authoritative relational database for production-oriented deployments.

SQLite may remain useful for very small local development, but the application must be built with a DB abstraction compatible with PostgreSQL from the beginning.

### 23.2 ORM / Migration

Recommended:

- SQLAlchemy 2.x
- Alembic migrations

### 23.3 Core Entities

At minimum:

```text
User
Role
Dataset
DatasetVersion
TrainingJob
TrainingProvider
ComputeResource
Model
ModelVersion
Artifact
Evaluation
QualityGate
Environment
InferenceTarget
Deployment
CredentialReference
Feedback
AuditLog
```

Additional entities may be introduced when implementation demonstrates the need.

---

## 24. Suggested Relational Model

```text
Dataset
  └── DatasetVersion
          └── TrainingJob

TrainingProvider
  └── ComputeResource
          └── TrainingJob

TrainingJob
  ├── Evaluation
  └── Artifact
          └── ModelVersion

Model
  └── ModelVersion
          ├── Evaluation
          ├── QualityGate
          ├── Deployment
          └── Alias/Environment Pointer

Environment
  └── InferenceTarget
          └── Deployment

User
  ├── TrainingJob
  ├── Feedback
  └── AuditLog
```

---

## 25. API Requirements

Representative API surface:

### Auth

```text
POST /auth/login
POST /auth/refresh
GET  /auth/me
```

### Datasets

```text
POST /datasets
POST /datasets/{id}/versions
GET  /datasets
GET  /datasets/{id}/versions
GET  /datasets/{id}/versions/{version}
```

### Training

```text
POST /training/jobs
GET  /training/jobs
GET  /training/jobs/{id}
POST /training/jobs/{id}/cancel
POST /training/jobs/{id}/retry
GET  /training/jobs/{id}/progress
GET  /training/jobs/{id}/logs
```

### Models / Registry

```text
GET  /models
GET  /models/{id}
GET  /models/{id}/versions
GET  /models/{id}/versions/{version}
POST /models/{id}/versions/{version}/promote
POST /models/{id}/versions/{version}/archive
POST /models/{id}/versions/{version}/reject
```

### Deployments

```text
GET  /deployments
POST /deployments/staging
POST /deployments/production
POST /deployments/{id}/rollback
GET  /deployments/{id}
```

### Infrastructure

```text
GET  /infrastructure/servers
POST /infrastructure/servers
PATCH /infrastructure/servers/{id}
GET  /infrastructure/servers/{id}/health

GET  /infrastructure/providers
POST /infrastructure/providers
GET  /infrastructure/compute-resources
POST /infrastructure/compute-resources
```

### Feedback

```text
POST /feedback
GET  /feedback
```

Exact routes may change during implementation; the domain capabilities must remain.

---

## 26. CI/CD Requirements

### 26.1 CI

CI must run on code changes and cover:

- backend linting
- type/static checks where used
- unit tests
- API tests
- security/dependency checks
- frontend build/tests
- migration validation

### 26.2 CD

CD must eventually support:

```text
merge/main
   ↓
CI green
   ↓
build image
   ↓
publish artifact/image
   ↓
deploy backend/control plane
   ↓
health check
```

Model promotion/deployment is a separate ML/CD workflow and must not be conflated with application-image deployment.

### 26.3 ML Deployment Flow

```text
Training
  ↓
Evaluation
  ↓
Quality Gate
  ↓
Registry
  ↓
Staging Deployment
  ↓
Integration Validation
  ↓
Production Promotion
```

---

## 27. Closed-Loop Feedback

Feedback is part of the product lifecycle, not an isolated form.

```text
Production / Staging Output
        ↓
Feedback
        ↓
Curation
        ↓
Dataset Candidate
        ↓
Validation
        ↓
New Dataset Version
        ↓
Retraining Job
        ↓
New Model Version
```

Feedback records should retain:

- source request/reference
- model version used
- environment
- feedback type
- reviewer/user
- optional corrected answer/annotation
- timestamp
- inclusion/curation status

No raw feedback record should automatically enter training without the configured curation/validation gate.

---

## 28. Observability

### 28.1 Two Layers

#### Engineering Observability

Use OpenTelemetry instrumentation for application signals and export/collect them into appropriate backends.

Core signals:

- logs
- metrics
- traces

Initial backend stack may use:

```text
OpenTelemetry
   ↓
Prometheus / metrics backend
   ↓
Grafana
```

Additional log/trace backends can be introduced when required.

#### Product/MLOps Observability

The DEFNEX frontend should expose product-level operational data such as:

- active jobs
- job duration
- job failures
- deployment success/failure
- staging health
- production health
- current model/version
- inference latency
- inference error rate
- resource availability

### 28.2 Trace Example

```text
User Request
   ↓
FastAPI
   ↓
Training Job Creation
   ↓
Job Queue
   ↓
Provider
   ↓
Compute
   ↓
Artifact Upload
   ↓
Registry
   ↓
Deployment
```

Critical lifecycle operations should carry correlated IDs such as job ID, model version, deployment ID, and trace ID.

---

## 29. Reliability and Failure Handling

The system must explicitly account for the following scenarios.

| Failure | Required behavior |
|---|---|
| Training process crash | Job becomes FAILED or STALE; cleanup executes |
| Worker/server reboot | Heartbeat timeout detects stale jobs |
| SSH disconnect | Remote operation returns failure; retry policy applies where safe |
| Colab disconnect | Job can become STALE; artifact is only accepted if verified |
| Backend restart | PostgreSQL remains source of truth; workers reconcile state |
| Artifact transfer failure | Deployment blocked; checksum failure recorded |
| Artifact corruption | Reject artifact; do not register/deploy |
| vLLM startup failure | Deployment marked FAILED; previous production version remains active |
| Smoke test failure | Deployment blocked or rolled back |
| Staging integration failure | Production promotion blocked |
| Production deployment failure | Keep prior production version active |
| Concurrent training/deployment | GPU lock prevents unsafe overlap |
| Duplicate deployment request | Operation must be idempotent or safely deduplicated |
| Stale RUNNING job | Transition to STALE according to timeout policy |
| Registry write failure after artifact upload | Artifact remains immutable/orphaned and is reconciled rather than overwritten |
| Backend down during training | Compute continues where possible; worker retries reporting/finalization |
| Redis unavailable | Job dispatch pauses; DB state remains authoritative |
| MinIO unavailable | Training completion may be blocked until artifact upload is confirmed |

### 29.1 Production Safety Principle

A failed candidate deployment must not destroy the currently healthy production version.

---

## 30. Deployment Safety and Idempotency

Deployment operations must be designed so repeated requests do not create inconsistent state.

Example:

```text
Requested: deploy v4 to staging

Request repeated
      ↓
Detect deployment for v4 already active/in-progress
      ↓
Return current operation/status
```

Promotion and rollback operations must be auditable.

---

## 31. Security Requirements

### Authentication

- secure authentication mechanism
- token/session expiration
- refresh strategy where applicable

### Authorization

RBAC must protect:

- deployment
- production promotion
- rollback
- infrastructure editing
- credential administration
- training provider configuration

### Transport

- HTTPS/TLS for web/API access
- SSH for remote server access

### Secrets

- never plaintext in frontend
- never logs
- never source-controlled
- stored via SecretStore

### Input Security

- request size limits
- dataset upload validation
- safe path handling
- artifact checksum validation
- command construction must avoid shell injection

### Auditability

Audit events should cover:

- login/admin actions
- server configuration changes
- provider configuration changes
- credential reference changes
- training execution
- model promotion
- production deployment
- rollback

---

## 32. Current Docker Compose Baseline

The control-plane Compose stack should be approximately:

```text
backend
worker
postgres
redis
minio
```

Optional development tooling may be included with profiles.

Vault should normally remain external in production.

The inference server may have a separate Compose deployment on staging/production hosts.

Example logical separation:

```text
Control Plane Compose
├── backend
├── worker
├── postgres
├── redis
└── minio

Inference Compose
└── vllm
```

This avoids coupling model serving lifecycle to the control plane.

---

## 33. Infrastructure Configuration UX

Admin should be able to configure a compute target using a form similar to:

```text
Name: GPU Server 2
Role: TRAINING
Environment: -
Provider: GPU VPS
Host: x.x.x.x
SSH Port: 22
Username: ubuntu
Credential: SSH Key #7
Workspace: /opt/defnex-mlops
Artifact Strategy: MinIO
GPU: H100
Status: Connected
```

For Colab:

```text
Name: Colab Training A
Role: TRAINING
Provider: Google Colab
Account Label: Colab Account A
Notebook URL: https://colab.research.google.com/...
Status: Configured
```

The account label is metadata, not a stored password.

---

## 34. Frontend Colab UX

For users who select a Colab compute resource:

```text
┌─────────────────────────────────────────────┐
│ Training Compute                            │
│                                             │
│ Provider: Google Colab                      │
│ Compute: Colab Training A                   │
│                                             │
│ Status: QUEUED                              │
│                                             │
│ [ Open Colab Runner ↗ ]                     │
│                                             │
│ Run the configured notebook in the         │
│ authorized Google/Colab account.           │
└─────────────────────────────────────────────┘
```

The link should open in a **new browser tab**.

After the runner starts, the DEFNEX page remains the primary status UI.

The frontend must not expose secret values or attempt to manage Google credentials.

---

## 35. Model Serving Configuration

Serving configuration should be versioned/configurable per inference target.

Possible fields:

- base model
- active adapter(s)
- max model length
- GPU memory budget
- dtype
- LoRA enablement
- max LoRAs
- port
- health endpoint
- smoke test prompt
- timeout

Example:

```text
Inference Target: staging
Base Model: Qwen/Qwen2.5-0.5B-Instruct
Adapter: defnex-qwen2.5-0.5b-v4
Max Model Len: 1024
GPU Budget: configured
```

Do not hard-code these values in frontend code.

---

## 36. Golden Test / Smoke Test Contract

Every deployment should execute a deterministic smoke test.

Minimum checks:

- endpoint reachable
- expected model ID reported
- request succeeds
- response non-empty
- response format valid where applicable
- optional task-specific validation

A deployment is not `HEALTHY` until required checks pass.

---

## 37. Data and Model Lineage

The system must answer:

> "Where did this production model come from?"

For each model version, it should be possible to reconstruct:

```text
Production Model v4
   ↓
Training Job #123
   ↓
Dataset defense-scenarios-v2
   ↓
Training Config Hash abc123
   ↓
Base Model Qwen/...
   ↓
Code Commit cb92718
   ↓
Artifact SHA256 ...
```

This lineage is one of the most important MLOps capabilities in the product.

---

## 38. Acceptance Criteria

### 38.1 Training

- [ ] User can create a training job from a dataset version and configuration.
- [ ] Training executes asynchronously.
- [ ] Job status is persisted in PostgreSQL.
- [ ] Progress and heartbeat are visible.
- [ ] Failed/stale jobs are distinguishable.
- [ ] Retry creates a traceable new execution.

### 38.2 Dataset

- [ ] User can upload a dataset.
- [ ] Dataset version is immutable.
- [ ] Dataset checksum is recorded.
- [ ] Training references an explicit dataset version.

### 38.3 Registry

- [ ] Model versions are immutable.
- [ ] Versions are listed through API/UI.
- [ ] Lineage is complete.
- [ ] Evaluation results are attached.
- [ ] Quality gate is recorded.
- [ ] staging/production pointers are represented.

### 38.4 Staging/Production

- [ ] Candidate can be deployed to staging.
- [ ] Staging smoke test runs automatically.
- [ ] Required integration checks can block promotion.
- [ ] Production promotion is controlled.
- [ ] Rollback returns to a previous immutable version.
- [ ] Failed production deploy does not destroy the current healthy version.

### 38.5 Infrastructure

- [ ] Admin can register a server/compute resource.
- [ ] Admin can assign role/environment.
- [ ] Admin can configure credential references.
- [ ] Server health is visible.
- [ ] Frontend is not hard-coded to one server.

### 38.6 Colab

- [ ] Admin can configure a notebook link.
- [ ] User can open the runner in a new tab.
- [ ] Notebook can claim the assigned job.
- [ ] Training progress returns to backend.
- [ ] Artifact is uploaded to MinIO.
- [ ] Google password is never stored in DEFNEX MLOps.
- [ ] Colab is modeled as on-demand compute, not a permanent worker.

### 38.7 Observability

- [ ] API logs are structured.
- [ ] Training/deployment logs are correlated with IDs.
- [ ] Core metrics are exposed.
- [ ] Critical lifecycle paths can be traced.
- [ ] Frontend presents useful operational status.

---

## 39. Migration Strategy from Current Prototype

Implementation should preserve working functionality and evolve incrementally.

### Phase 0 — Baseline / Evidence

Preserve the current proven capabilities:

- dataset input
- SFT training
- LoRA artifact
- evaluation
- vLLM deployment
- inference API
- SSE/progress where currently functional

### Phase 1 — Make the Existing System Safer

Priorities:

1. GPU deployment lock.
2. Job heartbeat.
3. Stale `RUNNING` detection.
4. Retry semantics.
5. Model-version listing.
6. Base-model alignment validation.

### Phase 2 — Storage and Registry Hardening

- PostgreSQL baseline.
- SQLAlchemy/Alembic.
- MinIO/S3 artifact store.
- immutable artifact metadata.
- dataset upload/versioning.
- server-side evaluation verification.

### Phase 3 — Multi-Environment Inference

- `staging` environment.
- `production` environment.
- inference target abstraction.
- promotion flow.
- production rollback.

### Phase 4 — Infrastructure and Provider Abstraction

- server registry.
- compute resource registry.
- SSH connector.
- artifact transfer.
- GPU VPS training provider.
- Colab runner provider.

### Phase 5 — Automated Delivery

- backend application CD.
- model promotion/deployment automation.
- environment gates.
- production deployment workflow.

### Phase 6 — Observability and Hardening

- OpenTelemetry.
- metrics dashboards.
- deployment/training traces.
- alerts.
- security hardening.
- failure/recovery drills.

---

## 40. Recommended Implementation Issue Order

### P0 — Safety / Correctness

1. Add GPU lock to deployment/hot-swap operations.
2. Add job heartbeat and stale detection.
3. Add reliable state transition validation.
4. Add artifact checksum verification.

### P1 — Registry and Versioning

5. Add model-version listing endpoint.
6. Add complete lineage metadata.
7. Add base-model/version compatibility checks.
8. Separate `REJECTED` and `ARCHIVED` semantics.

### P2 — Environments

9. Add staging/production entities.
10. Add inference target abstraction.
11. Add deployment promotion flow.
12. Add production rollback.

### P3 — Storage / Transport

13. Add MinIO/S3 artifact store.
14. Add artifact transfer contract.
15. Add remote server connectivity abstraction.

### P4 — Provider Infrastructure

16. Add `TrainingProvider` abstraction.
17. Add `ComputeResource` abstraction.
18. Add GPU VPS provider.
19. Add Colab runner provider.
20. Add admin infrastructure configuration.

### P5 — Automation / CD

21. Automate staging deployment after quality gate.
22. Automate production promotion according to approval policy.
23. Add deployment idempotency and recovery.

### P6 — Observability / Hardening

24. Add OpenTelemetry.
25. Add operational dashboards.
26. Add alerting.
27. Add failure/recovery test suite.
28. Perform security hardening.

---

## 41. Architectural Principles

### Principle 1 — Control Plane vs Data Plane

The backend orchestrates; compute systems perform the heavy work.

### Principle 2 — Immutable Versions

Artifacts and model versions are immutable. State changes happen through metadata/pointers.

### Principle 3 — Object Storage for Large Artifacts

Model and dataset bytes live in object storage, not inside PostgreSQL.

### Principle 4 — Staging Before Production

No candidate model goes straight from training to production without required gates.

### Principle 5 — Provider Agnostic Domain

Training logic must not be coupled to GPU VPS, Colab, or one future execution engine.

### Principle 6 — Configuration Over Hard-Coding

Server topology, notebook links, provider selection, and serving configuration are data/configuration.

### Principle 7 — Fail Safe

When a candidate fails, preserve the last known healthy production deployment.

### Principle 8 — Observable by Default

Jobs and deployments must emit enough information to diagnose failures without manually inspecting every server.

### Principle 9 — Avoid Premature Infrastructure Complexity

Use PostgreSQL + Redis + Celery + MinIO initially. Add Kafka, Kubernetes, distributed locking, Vault, or other infrastructure only when a concrete requirement justifies it.

---

## 42. Technology Baseline

### Control Plane

- FastAPI
- PostgreSQL
- SQLAlchemy 2.x
- Alembic
- Redis
- Celery

### Frontend

- Existing project frontend stack
- HTTPS
- SSE/WebSocket/polling as appropriate for live job status

### Training

- Unsloth
- PyTorch
- GPU VPS worker
- Google Colab notebook runner as on-demand provider

### Serving

- vLLM
- Docker / Docker Compose

### Artifact Storage

- MinIO / S3-compatible storage

### Secrets

- SecretStore abstraction
- encrypted store for MVP
- Vault/company secret manager as production integration

### Observability

- OpenTelemetry
- Prometheus
- Grafana

The stack must remain replaceable behind application interfaces where practical.

---

## 43. Operational Policies

### Shared GPU Policy

The system must assume GPUs can be shared with other workloads unless the infrastructure explicitly allocates exclusive capacity.

Never reset or terminate unknown workloads.

### Production Policy

- Production promotion requires authorization.
- Failed candidates remain non-production.
- Rollback must be possible using existing immutable artifacts.

### Artifact Retention

Artifacts required for rollback must not be deleted merely because a model is no longer active.

### Credential Policy

Credentials are references to secret storage, not application data.

### Colab Policy

Colab is an on-demand runner. The product should not imply guaranteed persistent compute availability from a manually launched notebook.

---

## 44. Open Decisions to Resolve During Implementation

These decisions should be finalized only when implementation requires them:

1. ~~Exact PostgreSQL deployment location and backup policy.~~ **Resolved**:
   single-node PostgreSQL, daily `pg_dump` with 7 daily + 4 weekly rolling
   retention. See `docs/governance/backup-dr-policy.md`.
2. ~~MinIO single-node vs redundant deployment.~~ **Resolved**: single-node
   MinIO, hourly application-level mirror to a second bucket/endpoint. See
   `docs/governance/backup-dr-policy.md`.
3. Exact distributed locking mechanism for multi-GPU servers.
4. Exact secret backend used in production.
5. Exact SSH library/connector implementation.
6. Whether production deployment requires a manual approval or can be fully automatic after staging.
7. Exact golden-set metrics for each DEFNEX task.
8. Exact vLLM deployment strategy per target: hot-swap, recreate, or both.
9. Whether logs belong in object storage, centralized log backend, or both.
10. Exact Colab authentication/operational policy approved by the organization.

These are implementation-level decisions and should not change the core domain model described in this PRD.

---

## 45. Reference Naming Convention

Use consistent identifiers across the platform:

```text
Dataset:
  defense-scenarios-v1

Model family:
  defnex-qwen2.5-0.5b

Model version:
  defnex-qwen2.5-0.5b-v4

Training job:
  train-20260907-001

Artifact:
  artifact-<immutable-id>

Deployment:
  deploy-20260907-001

Environment:
  staging
  production

Compute resource:
  server-2
  colab-training-a
```

Aliases/pointers:

```text
staging → defnex-qwen2.5-0.5b-v4
production → defnex-qwen2.5-0.5b-v3
```

---

## 46. Definition of Done for the Architecture MVP

The architecture MVP is considered complete when a user can perform the following end-to-end flow:

```text
1. Admin configures a training compute resource.
2. User selects a dataset version.
3. User selects base model + SFT configuration.
4. User creates a training job.
5. Job enters QUEUED state.
6. Training is executed by a configured provider.
7. Progress/heartbeat are visible.
8. Evaluation runs.
9. Immutable artifact is uploaded to MinIO.
10. Model version is registered with lineage.
11. Quality gate is evaluated.
12. Candidate deploys to STAGING.
13. Smoke/integration validation passes.
14. Authorized user promotes candidate to PRODUCTION.
15. Production model is visible in the registry.
16. A later version can be deployed.
17. A failure can trigger rollback to the previous immutable version.
18. Feedback can be collected and associated with the active version.
19. A future training job can consume a new curated dataset version.
20. All important lifecycle operations are observable and auditable.
```

For Colab specifically:

```text
1. Admin configures notebook URL.
2. User selects Colab compute.
3. Frontend shows "Open Colab Runner".
4. Link opens in a new tab.
5. Authorized operator logs into the intended Google account in Colab.
6. Operator runs the configured notebook.
7. Notebook claims the job.
8. Training completes or fails visibly.
9. Artifact is uploaded directly to object storage.
10. Backend finalizes registry/job state.
```

---

## 47. External Reference Notes

The architecture adopts concepts supported by current ecosystem practices:

- MLflow's current Model Registry documentation describes centralized model lifecycle management, versioning, lineage/metadata, aliases/tags, and environment-oriented promotion patterns. DEFNEX MLOps adopts these concepts selectively without adopting MLflow as the core platform.
- MLflow artifact-store architecture separates metadata from artifact bytes and supports object storage and other artifact backends.
- OWASP's Secrets Management guidance treats SSH keys, API keys, and database credentials as secrets and recommends centralized, controlled secret storage and lifecycle management.
- OpenTelemetry defines observability around signals such as traces, metrics, and logs and provides a vendor-neutral instrumentation/collection framework.
- Google Colab's current documentation describes dynamic resource availability and runtime limits; paid plans are usage/resource based and should not be treated as guaranteed always-on worker infrastructure.

These references are architectural inputs, not dependencies of the DEFNEX MLOps application.

Suggested references for implementation verification:

- MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry
- MLflow Model Registry Workflow: https://mlflow.org/docs/latest/ml/model-registry/workflow
- MLflow Artifact Store: https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/
- MLflow Backend Store: https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/
- OWASP Secrets Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
- OpenTelemetry Observability Primer: https://opentelemetry.io/docs/concepts/observability-primer/
- OpenTelemetry: https://opentelemetry.io/docs/what-is-opentelemetry/
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html

---

## 48. Final Architecture Summary

The intended DEFNEX MLOps architecture is:

```text
                         FRONTEND
                             │
                             ▼
                     ┌──────────────┐
                     │   FastAPI    │
                     │ Control Plane│
                     └──────┬───────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
      PostgreSQL          Redis             MinIO
      metadata            jobs             artifacts
          │                 │
          │                 ▼
          │              Celery
          │                 │
          │       ┌─────────┴─────────┐
          │       ▼                   ▼
          │   GPU VPS Worker      Colab Runner
          │       │                   │
          │       └─────────┬─────────┘
          │                 ▼
          │              MinIO
          │                 │
          └──────── Registry/Lineage
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 STAGING           PRODUCTION
                   vLLM               vLLM
                    │                  │
                 DEFNEX             DEFNEX
                 Chatbot             Chatbot
```

The system is intentionally designed as a **modular MLOps control plane**, not as a collection of unrelated services. The stable abstractions are:

```text
TrainingProvider
ArtifactStore
SecretStore
JobManager
ComputeResource
InferenceTarget
ModelRegistry
```

Those abstractions allow the platform to evolve from a single VM + free Colab prototype into a multi-server system without forcing a redesign of the frontend or domain model.
