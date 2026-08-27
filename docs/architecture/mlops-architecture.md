# WBS 1.1 — MLOps Closed Loop: Architecture & Technical Feasibility

Scope: the MLOps Closed Loop prototype only (dataset → validation → SFT → evaluation → model lifecycle →
deployment → feedback). This is a 1-month internal prototype, not the full DEFNEX platform (4-layer,
multi-domain intelligence fabric) — that broader system is existing context, not something this WBS builds.

Labels used throughout: **[PROJECT]** = confirmed from this vault's own notes, **[DOCS]** = confirmed from
official Unsloth documentation, **[REC]** = engineering recommendation, **[UNKNOWN]** = needs confirmation.

---

## 1. Architecture

### 1.1 Where this prototype sits relative to DEFNEX

**[PROJECT]** The vault already documents a 12-stage DEFNEX closed loop (`notebooklm/13`):
`Data Ingestion → Inference → Prediction Output → Evaluation → Feedback Capture → Data Curation & Validation
→ Training/SFT → Evaluation (post-training) → Model Versioning → Deployment → Production Run → Continuous
Feedback`. This WBS's assigned packages (1.1, 2.1, 2.2, 3.2, 3.3) map onto the middle six stages
(curation/validation through deployment). The DEFNEX notes also independently confirm a `.excalidraw`
diagram already exists titled "MLOps Closed Loop - SFT & Continuous Model Improvement" — worth opening
directly before finalizing any diagram here, since it may already encode decisions this doc would otherwise
duplicate. **[UNKNOWN]** whether this prototype must integrate with the full DEFNEX MCP/Kafka/Airflow/MLflow
stack (Layers 2–3) or is intentionally a standalone pipeline for now — nothing in the vault states this
explicitly for the *prototype*, only for the target production system. Treat the minimal architecture below
as the standalone case, per the "do not over-engineer" instruction.

### 1.2 Component boundaries and responsibilities

**Roles at a glance** (explicit separation — this is the mental model the rest of the document assumes):
- **Unsloth Core = the training engine.** It runs the actual SFT computation. It does not own state, does
  not talk to the frontend, and is not itself "the backend."
- **Backend = the orchestration/control layer.** It owns all state, decides when training runs, and is the
  only thing the frontend or the training runner talks to.
- **Frontend = the UI/client layer.** It never calls Unsloth or the training runner directly — only the
  backend API.
- **Model artifact/registry = the lifecycle & lineage layer.** It records what was produced, from what
  dataset/config, and its current status. See §1.6 for the boundary WBS 3.2 must implement.

| Component | Responsibility | Talks to |
|---|---|---|
| **Frontend** | Dataset upload UI, validation report view, training job trigger/status view, evaluation results view, promotion approve/reject UI, deployment status view | Backend/API only (never Unsloth directly) |
| **Backend/API** | Owns all state (dataset versions, validation results, training job records, model registry, promotion state, deployment status). Single integration point for the frontend. Runs validation. Triggers and monitors training. Records evaluation results. Enforces promotion state machine. | Frontend, Dataset storage, Training runner, Model artifact/registry |
| **Training runner** | Executes SFT using **Unsloth Core (the Python library)**, not Unsloth Studio's web service — see §1.4. Runs as a job on the GPU VM, invoked by the backend. | Backend (status/metrics push or poll), GPU/VM, dataset storage, model artifact storage |
| **Model artifact/registry** | Lifecycle & lineage layer — versioned model artifacts + metadata behind the boundary defined in §1.6 (internal schema/fields are WBS 3.2's job, not designed in this doc) | Backend, deployment step |
| **Deployment/serving** | Serves an approved model version for inference | Backend (status), inference clients |

**[REC]** Keep the backend as the single source of truth and the only thing the frontend talks to. This is
standard API-boundary hygiene, not a DEFNEX-specific decision, and it's what makes "frontend and backend can
develop in parallel against mocks" possible (§1.5).

### 1.3 Dataset → validation → SFT → evaluation → model lifecycle → deployment → feedback flow

```
[Frontend: upload]
      │
      ▼
[Backend API] ──stores──> [Dataset storage: raw/]
      │
      ▼
[Validation] (WBS 2.2 — designed separately)
      │  status: VALID / INVALID / NEEDS_REVIEW
      ▼
[Dataset storage: processed/ train/ eval/] (WBS 2.1 — designed separately)
      │
      ▼  (backend triggers training job on the GPU VM)
[Training runner: Unsloth Core SFT job] ──writes──> [Model artifact storage]
      │  status/metrics polled or pushed back to Backend API
      ▼
[Evaluation] (metrics + qualitative eval — WBS not yet assigned in detail; mentor's 3-signal
              promotion method already documented, see §3.5)
      │
      ▼
[Model registry] (WBS 3.2 — designed separately)
      │
      ▼
[Promotion / approval] (WBS 3.3 — designed separately)
      │  APPROVED
      ▼
[Deployment: serve model] ──> [Inference]
      │
      ▼
[Feedback capture] ──back to── [Dataset storage: raw/] (next iteration)
```

This is the same shape as the DEFNEX 12-stage loop, scoped down to what backend + Unsloth Core can actually
drive without the rest of the DEFNEX stack.

### 1.4 Frontend ↔ Backend/API ↔ Unsloth interaction — and why it is NOT "Backend ↔ Unsloth Studio"

**[DOCS]** Unsloth ships as three separate surfaces: **Unsloth Core** (the `pip install unsloth` Python
library — script/notebook-first, fully programmatic), **Unsloth Desktop**, and **Unsloth Studio** (a web
GUI with its own FastAPI backend). These have different automation stories:

- **Unsloth Studio's REST API** *does* exist and is documented at a surface level
  (`/api/train/start|stop|reset|status|metrics`, `/api/train/stream` for SSE progress,
  `/api/datasets/`, `/api/models/`, `/api/auth/*`). **[DOCS]** But: authentication is user-login JWT (not a
  service-account/API-key model suited to backend-to-backend calls), there is no published request/response
  schema for `/api/train/start`, and critically **there is no documented endpoint to retrieve the trained
  model artifact** — export is a UI-only or separate-CLI-only action (`cli.py export`). The docs consistently
  describe starting/stopping training as a UI click ("Click Start Training"), not as an automation
  surface. **[REC]** Do not architect the training orchestration around Unsloth Studio's REST API — it is the
  private backend for Studio's own web UI, not a supported third-party integration surface, and it has a
  hard artifact-retrieval gap.
- **Unsloth Core** is the officially-intended automation path: `FastLanguageModel.from_pretrained()` +
  `SFTTrainer` (HF TRL) + `save_pretrained_gguf()`/`save_pretrained_merged()`, callable directly from Python.
  **[PROJECT]** This is also exactly what all 9 documented SFT practice notebooks already do (Colab,
  T4 GPU, Qwen2.5-7B, 4-bit QLoRA) — so the automation path recommended here is a direct extension of
  practice already done, not a new unproven approach.
- **[DOCS]** Unsloth's separate inference API (`/v1/chat/completions`, `/v1/messages`,
  API-key auth `sk-unsloth-…`, backed by `llama-server`/GGUF) is well documented with real request/response
  examples and is a reasonable option for the *deployment* stage specifically, if GGUF export is the chosen
  artifact format — but it's a serving product, not a training-orchestration product, and should not be
  conflated with "Unsloth Studio automation."

**Recommended interaction shape:**

```
Frontend  <──REST/JSON──>  Backend API  <──spawns/monitors──>  Training runner process/container
                                                                (imports `unsloth`, runs SFTTrainer,
                                                                 same pattern as existing Colab notebooks,
                                                                 executed on the GPU VM instead of Colab)
```

The backend does not call "Unsloth Studio" as a service at all in this design — it runs Unsloth Core as a
library inside a job it controls (subprocess, container, or workflow-orchestrated script), and polls/receives
status the same way it would for any long-running job it owns. Unsloth Studio remains available as a
human-facing tool for practitioners who want a GUI (e.g. for ad-hoc Data Recipe dataset building or manual
experimentation), but it is not in the automated critical path.

### 1.5 API boundary for parallel frontend/backend development

**[REC]** This architecture is explicitly designed so frontend and backend can be built in parallel: the
backend API is the *only* contract between them (§1.2), so neither side needs the other's implementation to
exist, only the contract. Because the backend is the sole integration point, the frontend can be built
entirely against mocked responses for: dataset upload, dataset/validation status, training job creation &
status, evaluation result, model version list, promotion action, deployment status, feedback submission. The
full request/response contract for these is WBS-assigned separately (cross-cutting API contract) and
intentionally not designed in this document — but the architectural precondition for it (single backend
boundary, no frontend-to-Unsloth calls) is established here.

### 1.6 Model Artifact, Metadata & Lineage — architectural boundary for WBS 3.2

This document does not design the model registry's schema (that is WBS 3.2's job) — it defines the minimum
boundary this architecture assumes exists, so WBS 3.2 has a clear contract to implement against instead of an
open-ended one.

**Boundary (what the backend and training runner can rely on, nothing more):**
- **Register**: after a training job finishes, the registry accepts a call with (at minimum) the dataset
  version used, the training configuration, and the artifact's storage location, and returns a model
  identifier plus an initial status.
- **Read/list**: given a model identifier (or a filter), the registry returns that model's metadata and
  current lifecycle status.
- **Status update**: the registry accepts lifecycle-transition updates (evaluation recorded, promoted,
  deployed, rejected). The transition rules themselves are WBS 3.3's design, not this document's.
- **Lineage**: the registry is the single source of truth for the chain Dataset version → Training run →
  Model artifact → Evaluation result → Promotion decision → Deployment record. Every stage in §1.3's flow
  diagram that produces or consumes a model artifact does so *through* this boundary — never by passing raw
  file paths between components directly.

**Explicitly out of scope here** (left entirely to WBS 3.2): exact field names/types, immutable vs mutable
fields, UUID vs string IDs, JSON vs normalized storage, and the relational/data-model representation. This
section only guarantees the boundary exists and that the rest of the architecture is designed against it —
it does not redesign the registry.

---

## 2. Feasibility assessment

| Question | Answer |
|---|---|
| Can SFT training be automated end-to-end by our backend? | **Yes, via Unsloth Core as a library inside a backend-owned job**, not via Unsloth Studio's API. **[DOCS]+[PROJECT]** |
| Can we get real-time training progress? | Yes — either by having the training process itself report status/metrics to the backend (simplest, since we own the training script), or by shelling out through the Studio API's SSE stream if Studio is used manually. **[REC]**: prefer the former, it needs no dependency on Studio at all. |
| Can we retrieve the trained model artifact programmatically? | Yes, if training runs via Unsloth Core (we control the save/export call directly). **Not reliably** if training runs via Unsloth Studio (no documented export API). **[DOCS]** |
| Is the target base model available/confirmed? | **[UNKNOWN]** — see §4. |
| Is the GPU/VM available? | **[UNKNOWN]/planned, not yet provisioned** — see §4. |
| Is a minimal (non-DEFNEX-scale) architecture sufficient for a 1-month prototype? | Yes — see §1.3/§1.4; nothing here requires Kafka/Airflow/MLflow/K8s to exist first. |

**Overall: technically feasible for a minimal prototype**, provided the training runner is built against
Unsloth Core directly and the VM/base-model unknowns in §4 are resolved before SFT training work (not before
the rest of Phase A design/prep work, which does not need the VM).

---

## 3. Key decisions

**Decision 1 — Training orchestration goes through Unsloth Core, not Unsloth Studio's REST API.**
Why: Studio's training API has no documented artifact-retrieval endpoint and uses a user-login JWT model
unsuited to service-to-service calls; Core is scriptable, already proven in 9 practice iterations, and is
the path every official Unsloth tutorial uses. Evidence: [DOCS] `unsloth.ai/docs/new/studio/start.md`,
`unsloth.ai/docs/new/studio/export.md`; [PROJECT] `Praktik/Hasil Praktik SFT 1-9.md`. Alternatives
considered: scripting against Studio's REST API directly (rejected — undocumented request schema, no export
endpoint); shelling out to Studio's CLI (`cli.py train/export`) instead of importing the library (viable
fallback, but no documented flags, so Core's Python API is more predictable). Open question: none blocking —
this can be adopted now.

**Decision 2 (advisory only — WBS title unchanged) — work-package naming analysis.**
The existing WBS task title is kept **unchanged for now**; nothing here is an applied rename, only an
architectural observation for whoever owns the WBS document later. Why it's raised at all: "Integration
Foundation" reads as though orchestration is deferred to a later package, while §1.4 shows real
orchestration (job lifecycle, status, artifact handoff via Unsloth Core) is buildable now. If the title is
ever revisited, "Backend–Unsloth Integration & Training Orchestration" would describe the actual scope more
accurately than "...Foundation." Evidence: [DOCS] §1.4. Alternatives: keep either original name — both are
defensible, this is a naming/scoping call and does not affect the architecture in §1. Open question:
whether/when to revisit the title is left entirely to the mentor/team; **no rename has been applied here.**

**Decision 3 (candidate option, NOT locked) — deployment/serving: vLLM/llama.cpp is the leading candidate,
pending confirmation.**
This is explicitly **not** a final decision. It is pending two things that are both still open: (a) the
VM/environment (§4 — unresolved), and (b) which artifact format training actually exports (merged
safetensors vs GGUF — not yet decided). Why vLLM/llama.cpp is the leading candidate today: Unsloth's own
inference API only serves GGUF models via `llama-server`; the vault already has hands-on vLLM practice
(laptop iGPU, `Qwen3-0.6B`/`Qwen3-1.7B`, OpenAI-compatible server) and DEFNEX architecture notes independently
infer vLLM/Ollama for on-prem serving. Evidence: [PROJECT] `Praktik/praktikum-vllm/01,05`;
[PROJECT/INFERENCE] `notebooklm/03`/MERGED (explicitly marked inference, not confirmed); [DOCS]
`unsloth.ai/docs/basics/inference-and-deployment.md`. Alternatives: Unsloth's own `llama-server`-backed API
(viable if GGUF export is chosen and a second serving stack is undesirable — legitimate lighter-weight
option for a 1-month prototype, worth reconsidering if vLLM setup proves heavy). **Do not lock this in the
architecture** until the VM/environment and artifact-format questions are answered. Open question:
**[UNKNOWN]** which serving stack is actually intended for this prototype's deployment stage — not decided
anywhere in the vault; §5 flags this for confirmation.

**Decision 4 — Minimal prototype architecture excludes Kafka/Airflow/MLflow/Kubernetes.**
Why: those are DEFNEX's full-platform Layer 2/3 components for the multi-domain intelligence system, not
requirements for a 1-month SFT closed-loop prototype; nothing in the assigned WBS packages (1.1, 2.1, 2.2,
3.2, 3.3) requires them. Evidence: [PROJECT] `notebooklm/04`, `13`; explicit user instruction not to
over-engineer. Alternatives: use MLflow now for the model registry (WBS 3.2) — reasonable future upgrade,
but out of scope to require it in Phase A. Open question: none blocking.

---

## 4. Dependencies / blockers

| Item | Status | Detail |
|---|---|---|
| **VM/GPU access** | **VM-BLOCKED** | **[PROJECT]** A VM ("VM Len") with an H100 80GB GPU is planned but not yet provisioned (`Rencana Praktik VM...md` states "execution date not yet determined — waiting for VM"). All actual SFT practice so far ran on Google Colab T4 16GB. |
| **Exact base model name/version** | **[UNKNOWN] — needs confirmation** | The task brief says "Qwen3.8 27B." The vault itself is internally inconsistent: it uses "Qwen3.8-27B" in the VM plan and "Qwen 3.6/3.8 27B" elsewhere — never the exact plain string "Qwen3.8 27B." No publicly known Qwen model family matches this name as of current knowledge. This looks like a working placeholder in the author's own notes, not a confirmed upstream release name. **Do not start VM-side training until this is resolved** — it changes VRAM planning, chat template selection (WBS 2.1), and Unsloth compatibility. |
| **Network access on the VM** | **[UNKNOWN]** | Not documented anywhere in the vault; needed to confirm the training runner can reach a model hub (for base weights) and any artifact storage target. |
| **CUDA/driver version on the VM** | **[UNKNOWN]** | VM plan's own checklist already flags this as unconfirmed (needs `nvidia-smi` check once VM access exists), and flags confirming the H100 is the 80GB (not 48GB) variant. |
| **Storage for datasets/artifacts** | **[UNKNOWN]** | Not documented; needed before WBS 2.1/3.2 can assume a concrete artifact URI scheme. |
| **Container/runtime for the training job** | **[REC] not yet decided** | Nothing in the vault mandates a specific runtime; a plain Python venv/conda environment mirroring the existing Colab notebook dependencies is the lowest-effort option for a 1-month prototype; containerizing is a reasonable upgrade but not required to start. |
| **Credentials** | **[UNKNOWN]** | HF Hub token (for gated/base model download) and any artifact storage credentials are not documented as provisioned. |
| **Whether training data will be public or Len's production data** | **[UNKNOWN]** | Explicitly left as an open checklist item in the VM plan note itself; user's task brief already directs public datasets for Phase A regardless. |

Everything in §1–3 of this document (architecture, feasibility reasoning, naming decision, dataset/model
design in the other WBS packages) can proceed without the VM. Only actual SFT execution is VM-BLOCKED.

**Exact command/test to run once VM access exists** (to resolve the blockers above in one pass):
```bash
nvidia-smi                          # confirm GPU model + VRAM (H100 80GB vs 48GB)
nvcc --version                      # confirm CUDA version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ping -c3 huggingface.co             # confirm outbound network access for model hub
df -h                                # confirm available storage for datasets/checkpoints
```

---

## 5. Open questions

1. **[UNKNOWN]** Exact base model name/version — "Qwen3.8 27B" does not match any confirmed public model
   name; the vault's own notes are inconsistent about it too. Needs explicit confirmation before any VM-side
   training work or VRAM/chat-template planning.
2. **[UNKNOWN]** Is this MLOps Closed Loop prototype meant to eventually plug into DEFNEX's full MCP/Kafka/
   Airflow/MLflow stack, or is it intentionally a standalone pipeline for now? Affects how much future-proofing
   (if any) belongs in the model registry (WBS 3.2) and API contract.
3. **[UNKNOWN]** Which serving stack is intended for the deployment stage — vLLM (matches existing practice)
   vs Unsloth's own `llama-server`-backed inference API (matches GGUF export, one fewer moving part)?
4. **[UNKNOWN]** VM network/CUDA/storage/credentials — see §4 table; none of this is documented yet.
5. Work-package title: kept **unchanged** per explicit instruction (see Decision 2) — the naming analysis in
   §3 is advisory only and has not been applied. Not an open question unless the team later chooses to
   revisit it.
6. Already flagged by the note author, still open, and relevant to this closed loop specifically: how does
   the closed-loop feedback/data-lineage pipeline connect operator validation back into curation
   (`notebooklm/10`, priority-3 item) — directly overlaps with WBS 2.1/2.2/3.3, worth resolving there rather
   than here.

---

## 6. Recommended next steps

1. Confirm the exact base model name/version with the mentor/team (§5.1) — this is the single highest-leverage
   unknown, since it gates VRAM sizing, chat-template choice, and Unsloth compatibility checks.
2. Proceed with WBS 2.1 (dataset lifecycle/schema) and 2.2 (validation) now — fully VM-independent, and the
   canonical dataset schema decision there should reference whatever chat template the confirmed base model
   uses.
3. Proceed with WBS 3.2 (model registry) and 3.3 (promotion workflow) now — also VM-independent; 3.2's
   `artifact_uri` field should assume the Unsloth-Core-driven save/export path from Decision 1 (we always
   control the artifact ourselves, so no "wait for Studio's export API" dependency needed).
4. Get an explicit answer on §5.2 (standalone vs DEFNEX-integrated) before finalizing the API contract
   cross-cutting deliverable — it changes whether the contract needs to anticipate Kafka/MCP-facing fields.
5. Once VM access exists, run the verification commands in §4 first, then port the existing Colab notebook
   training script to the VM essentially unchanged (same Unsloth Core calls, larger model/GPU) before building
   any new orchestration layer around it — de-risks the VM transition by changing one variable (compute) at a
   time instead of compute + orchestration simultaneously.
