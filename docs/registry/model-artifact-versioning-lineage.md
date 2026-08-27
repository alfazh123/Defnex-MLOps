# WBS 3.2 — Model Artifact, Versioning & Lineage Management

Scope: the model registry boundary defined in `docs/architecture/mlops-architecture.md` §1.6 (Register /
Read-list / Status-update / Lineage). This document designs that boundary's schema and lifecycle. It does
**not** design promotion/approval decision logic or thresholds — that is WBS 3.3. It does **not** redesign
dataset schema/validation — see `docs/dataset/dataset-lifecycle-and-schema.md` and
`docs/dataset/validation-rules.md`, which this registry references by version/report path.

Labels: **[PROJECT]** = confirmed from this vault's own notes, **[REC]** = engineering recommendation,
**[UNKNOWN]** = needs confirmation.

---

## 1. Current practice — what this WBS is fixing

**[PROJECT]** Today, model artifacts have **no versioning at all** at the point they matter most: the final
adapter is always saved to the same fixed folder name `qwen_lora/`, overwritten on every run
(`Praktik/18 Aug 2026 - Praktik 1.md`, `Praktik/Latihan_SFT_Qwen2_5_(7B)...md`). The only versioning that
exists is the Colab *notebook* iteration number (`outputs_v2`, `outputs_v3`, `outputs_v4`, `outputs_v5`), which
tracks which practice session produced a checkpoint, not a deliberate model-artifact identity.

**[PROJECT]** Storage today is local-Colab-ephemeral only: the adapter folder is zipped and pushed through a
browser download popup so it survives session teardown
(`Praktik/Latihan_SFT_Qwen2_5_(7B)...md`: *"karena folder `qwen_lora/` akan hilang begitu sesi Colab berakhir
... cell ini membungkusnya jadi satu file .zip lalu memicu popup download browser"*). HF Hub push and merged/
GGUF export are documented as options but were never actually executed — the merge/export cells are
`if False:`-gated in every run. **[PROJECT]** No note anywhere records which dataset version + training config
produced a given artifact except by human memory; `Materi/11` names this directly as the motivating problem:
*"begitu ada beberapa iterasi model ... sulit menjawab pertanyaan sederhana seperti 'model v3 kemarin itu
pakai config LoRA yang mana? Dataset yang sudah dibersihkan versi keberapa?'"*

This WBS closes both gaps: give every trained artifact a stable version identity, and record its lineage at
register time instead of relying on memory.

**[PROJECT] Update since first draft**: the base model identity, flagged `[UNKNOWN]` throughout this document
and every prior WBS doc, is now confirmed: `Riset Metode Fine-Tuning Qwen3.8-27B di Single H100 80GB.md`
verified `Qwen/Qwen3.8-27B` is a real, official Alibaba/Qwen release (14 Aug 2026, Apache 2.0, ~28B dense
params, 55.6GB BF16 weights, 262k native context) — not a typo or placeholder as earlier docs cautiously
assumed. That same note also confirms the actual recommended fine-tuning method for the H100 80GB target:
**DoRA, bf16 base, no 4-bit quantization** (`use_dora=True` on top of the existing Unsloth LoRA flow) —
the opposite quantization call from the 7B/T4 practice this document was originally grounded in (QLoRA/4-bit
was mandatory there because VRAM was tight; here VRAM is loose enough that bf16 full-precision base +
adapter is the better default, full fine-tuning itself is confirmed infeasible/OOM even with 8-bit AdamW).
§3 and §6 below are updated to reflect this; sections not touched by this update still reflect the original
7B/T4-grounded reasoning where it still applies (e.g. adapter-vs-merged storage preference, §3).

---

## 2. Versioning scheme

**[REC]** Same philosophy as dataset versioning (`dataset-lifecycle-and-schema.md` §2): plain incrementing
integers, no meaning-encoded suffixes, no dedicated versioning infrastructure (no MLflow, no DVC-equivalent).

- `model_id` — stable identifier for one continuing fine-tuning line (e.g. one base model + one target
  capability). Does not change across re-training rounds.
- `version` — integer, incrementing per successful training run registered against that `model_id`. Starts at
  `1`.
- **[REC]** Do not encode meaning into the identifier itself (no `_final`, `_FIX`, `_v7`-in-a-filename style —
  the same anti-pattern already flagged for datasets in `Materi/11` and `dataset-lifecycle-and-schema.md`
  §2, which named `dataset_v7_final_FIX.csv` as the thing to avoid; no equivalent warning existed yet for
  model artifacts in the vault, but the current `qwen_lora/`-overwrite practice is the same failure mode by
  omission — no identity at all rather than a bad one).
- **[REC]** No MLflow/SQLite registry infrastructure now. `notebooklm/13` sketches a DEFNEX-scale
  MLflow Model Registry (tag format `v1.2.0-champion`), and `Materi/catatan dari chat gpt.md` explicitly
  tags a full "Model Registry" as *"overkill infra, untuk nanti"* twice. A flat metadata record (below) is
  sufficient for a 1-month prototype and is consistent with the same call already made for datasets.

---

## 3. Artifact types recorded per version

**[PROJECT]** The vault's own mentor guidance prefers keeping the LoRA adapter separate from the base model
rather than merging, specifically for multi-routing/serving cost reasons
(`Praktik/20 Aug 2026 - Bimbingan Mentor...md`: *"opsi 1 (adapter terpisah, base di-share) adalah yang paling
murah — base model 7B dimuat sekali di VRAM, tiap adapter cuma nambah beban kecil"*), and this matches actual
practice: only the adapter (~100MB, confirmed observed size) was ever reliably produced; merged model
(~15GB) and GGUF export were never executed.

- **Required**: `adapter` — the LoRA/DoRA adapter directory/archive. Always recorded when a training run
  completes successfully. **[PROJECT]** The confirmed target model's own research note reinforces this
  choice: the recommended method is DoRA — still an adapter (order of tens-to-hundreds of millions of
  trainable params, under 1% of the 28B total) rather than a full-parameter method, so "adapter is the
  required artifact" holds for the actual target model, not just the earlier 7B practice.
- **Optional**: `merged` and `gguf` — recorded only if/when actually exported. **[PROJECT] Size estimate
  revised**: the ~15GB merged-model figure was a 7B-practice number. For the confirmed target model
  (`Qwen/Qwen3.8-27B`, 55.6GB BF16 weights per `Riset Metode Fine-Tuning Qwen3.8-27B di Single H100 80GB.md`
  §2), a merged export would be on that order (~56GB), not ~15GB — relevant to open question §10.1's
  artifact-storage sizing. Whether these are ever produced still depends on the still-open deployment-stack
  decision (`mlops-architecture.md` Decision 3, not locked).

A single `version` can accumulate more than one artifact entry over time (adapter recorded first at training
completion, `merged`/`gguf` recorded later if exported for deployment) — this is why artifacts are a list on
the version record, not a single field.

---

## 4. Lifecycle status

**[REC]** Per the architecture boundary (§1.6): registry records status, WBS 3.3 designs the promotion
transition *rules*. This document only defines the status values themselves and what data must exist to
support that later decision.

`REGISTERED → EVALUATED → PROMOTED | REJECTED → DEPLOYED → RETIRED`

- **`REGISTERED`** — training run completed, adapter artifact recorded. No evaluation data yet.
- **`EVALUATED`** — the three evaluation signals (§5) have been recorded against this version. Still not a
  promotion decision.
- **`PROMOTED`** / **`REJECTED`** — WBS 3.3's decision, written back onto this record. Threshold logic lives
  in WBS 3.3, not here.
- **`DEPLOYED`** — currently serving.
- **`RETIRED`** — superseded by a later promoted version for the same `model_id`. Retiring a version is how
  rollback is expressed: point `current_deployed_version` (registry-level, not per-record) back at an earlier
  `PROMOTED` version and mark the failed one `RETIRED`, rather than deleting or overwriting anything.
  **[PROJECT]** No note documents an actual rollback-from-deployed event; the closest real precedent is
  choosing Epoch 1 over Epoch 2 pre-promotion in `Hasil Praktik SFT 6`/`7` (rejecting a newer checkpoint for a
  regression, before deployment) — the same append-only, never-overwrite principle is what this status model
  generalizes.

---

## 5. Evaluation signals stored (not decided) here

**[PROJECT]** The vault documents a specific 3-signal promotion method
(`Praktik/20 Aug 2026 - Bimbingan Mentor...md` §4, echoed in `Materi/16`): eval_loss trend, majority-win on a
fixed 20-question comparison table, and no general-domain regression. **No numeric threshold is defined
anywhere** for how much eval_loss must improve or how many of the 20 questions constitute "majority" — the
method is stated directionally only, and partial alignment is explicitly meant to be logged, not acted on
("*kalau cuma salah satu yang positif, catat sebagai temuan tapi jangan langsung jadikan checkpoint
default*").

This registry's job is to give WBS 3.3 somewhere to read that data from — not to compute the decision:

- `eval_loss_trend`: `{previous_version_eval_loss, this_version_eval_loss}` — direction, not a judgment.
- `qualitative_comparison`: `{question_table_version, wins, losses, ties, total}` (total is 20 per current
  practice, **[UNKNOWN]** whether that stays fixed).
- `general_domain_regression_check`: `{checked: bool, regressions_found: [...]}`.
- `dataset_validation_report_ref`: pointer to the WBS 2.2 report this training run's data was gated by
  (`validation/{dataset_id}/{version}/report.json`), so "was this run's data gate passed" is traceable from
  the model side — the exact link WBS 2.2 §10 already asked WBS 3.2 to confirm.

**[REC]** No threshold fields are added here (e.g. no `pass_threshold`) — inventing a number the vault never
used would misrepresent it as a confirmed practice. WBS 3.3 defines and applies thresholds against these raw
signal fields.

---

## 6. Lineage fields (register-time payload)

**[PROJECT]** Directly adopts the 7-field manual table `Materi/11` already proposed for exactly this purpose
(*"bentuk versioning yang sudah dipraktikkan tanpa disadari"* — base model, dataset & versi/seed, konfigurasi
LoRA, konfigurasi training, tanggal & pembuat, hasil evaluasi, checkpoint/artifact yang dihasilkan), extended
only with the status enum (§4) and artifact list (§3) needed to make it a queryable record instead of a
markdown table a human has to remember to fill in.

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
  "training_config": {
    "peft_method": "dora",
    "load_in_4bit": false,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": "UNKNOWN — fill from actual run config",
    "epochs": 2,
    "max_seq_length": 4096
  },
  "created_at": "2026-08-27T10:15:00Z",
  "created_by": "UNKNOWN — fill from actual run",
  "evaluation": {
    "eval_loss_trend": {"previous_version_eval_loss": null, "this_version_eval_loss": 0.842},
    "qualitative_comparison": {"question_table_version": 1, "wins": 13, "losses": 5, "ties": 2, "total": 20},
    "general_domain_regression_check": {"checked": true, "regressions_found": []}
  },
  "artifacts": [
    {"type": "adapter", "uri": "UNKNOWN — persistent storage location not yet decided", "size_bytes": 104857600}
  ],
  "promotion_decision_ref": null,
  "previous_model_id": null
}
```

`base_model` and `training_config.peft_method`/`load_in_4bit`/`max_seq_length` reflect the now-confirmed
target model and its recommended recipe (`Riset Metode Fine-Tuning Qwen3.8-27B di Single H100 80GB.md` §7):
DoRA, bf16 base (no 4-bit), `max_seq_length` raised from the 7B/T4 practice's 2048 as a starting point subject
to VRAM profiling on the actual VM. Remaining `"UNKNOWN — ..."` fields (learning rate, storage location,
created_by) are placeholders showing where a dependency still pending an actual training run plugs in — not
invented defaults.

**[UPDATE — persisted `training_run_id` added.]** `training_run_id` is now a required, persisted field on
this record, not merely an API-side convenience. The cross-cutting API contract
(`docs/api/mlops-api-contract.md` §1, §9.2) originally introduced `training_run_id` as an entity this schema
has no way to represent (WBS 3.2's lineage begins at `REGISTERED`, i.e. after a training run already
completed) and, lacking authorization to modify this document at the time, assembled it into API responses
without persisting it. That gap is closed here: the training runner/backend supplies `training_run_id` at
Register time (§7 below), so a registry record can always be traced back to the exact run that produced it
without depending on a separate, unpersisted join. See §9 Decision 5.

---

## 7. Registry interface (implements `mlops-architecture.md` §1.6)

| Operation | Input (minimum) | Output |
|---|---|---|
| **Register** | `training_run_id`, `dataset_id`, `dataset_version`, `training_config`, `artifacts[]` (at least the adapter) | `model_id`, `version`, initial `status: REGISTERED` |
| **Read/list** | `model_id` and/or `version`, or filter (e.g. `status=PROMOTED`) | matching record(s), full schema above |
| **Status update** | `model_id`, `version`, new `status`, optional `evaluation` payload (§5) or `promotion_decision_ref` (WBS 3.3) | updated record |
| **Lineage** | `model_id`, `version` | full chain: dataset version → training config → artifacts → evaluation → promotion decision → deployment record (deployment record itself is a later WBS, referenced by ID only here) |

No operation accepts or returns a raw file path outside the `artifacts[].uri` field — this matches the
architecture's explicit rule that stages talk to model artifacts only *through* the registry boundary, never
by passing paths directly between components.

---

## 8. Machine-readable schema (JSON Schema, draft-07)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ModelRegistryRecord",
  "type": "object",
  "required": ["model_id", "version", "status", "training_run_id", "dataset_id", "dataset_version", "created_at", "artifacts"],
  "properties": {
    "model_id": {"type": "string"},
    "version": {"type": "integer", "minimum": 1},
    "status": {"type": "string", "enum": ["REGISTERED", "EVALUATED", "PROMOTED", "REJECTED", "DEPLOYED", "RETIRED"]},
    "training_run_id": {"type": "string", "description": "Identifier of the training run that produced this version. See docs/api/mlops-api-contract.md §1/§3.4 for the training_run entity itself, which this document does not otherwise define."},
    "base_model": {"type": "string"},
    "dataset_id": {"type": "string"},
    "dataset_version": {"type": "integer"},
    "dataset_validation_report_ref": {"type": "string"},
    "training_config": {"type": "object"},
    "created_at": {"type": "string", "format": "date-time"},
    "created_by": {"type": "string"},
    "evaluation": {
      "type": "object",
      "properties": {
        "eval_loss_trend": {"type": "object"},
        "qualitative_comparison": {"type": "object"},
        "general_domain_regression_check": {"type": "object"}
      }
    },
    "artifacts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "uri"],
        "properties": {
          "type": {"type": "string", "enum": ["adapter", "merged", "gguf"]},
          "uri": {"type": "string"},
          "size_bytes": {"type": "integer"}
        }
      }
    },
    "promotion_decision_ref": {"type": ["string", "null"]},
    "previous_model_id": {"type": ["string", "null"]}
  },
  "additionalProperties": false
}
```

---

## 9. Key decisions

**Decision 1 — Flat metadata record, no MLflow/SQLite registry infrastructure.** Why: matches the same call
already made for dataset versioning (`dataset-lifecycle-and-schema.md` §2), and this project's own notes
(`Materi/catatan dari chat gpt.md`) twice mark a full Model Registry system as "overkill, untuk nanti."
Evidence: [PROJECT]. Alternatives: adopt `notebooklm/13`'s DEFNEX-scale MLflow Model Registry now — rejected,
that's target production infrastructure for the full platform, not this 1-month prototype. Open question:
whether this lightweight registry should later migrate into DEFNEX's MLflow registry — carried from WBS 1.1
§5.2 (standalone vs DEFNEX-integrated), not resolved here.

**Decision 2 — LoRA adapter is the required artifact; merged/GGUF are optional, recorded only if produced.**
Why: matches actual practice (adapter always produced, ~100MB observed; merge/GGUF cells never executed) and
matches the mentor's own stated storage-cost reasoning for keeping adapters separate for multi-routing.
Evidence: [PROJECT] `Praktik/20 Aug 2026 - Bimbingan Mentor...md`, `Praktik/18 Aug 2026 - Praktik 1.md`; further
reinforced by `Riset Metode Fine-Tuning Qwen3.8-27B di Single H100 80GB.md` §6, which confirms the actual
target model's recommended method (DoRA) is also adapter-based, not full-parameter — the confirmed base
model doesn't change this decision, it reinforces it. Alternatives: require a merged model at register time —
rejected, would force work that was never actually part of practice and blocks registration on a
deployment-stack decision that isn't made yet. Open question: none blocking.

**Decision 3 — The three evaluation signals are stored as raw fields; promotion thresholds are not computed
here.** Why: no numeric threshold for eval_loss improvement or majority-win count exists anywhere in the
vault — inventing one would misrepresent an unconfirmed number as project practice. Evidence: [PROJECT]
confirmed absent in `Praktik/20 Aug 2026 - Bimbingan Mentor...md` §4 (directional language only: "turun" vs
"naik", "mayoritas" with no count). Alternatives: define thresholds now as a `[REC]` placeholder — rejected,
same reasoning as WBS 2.2's PASS/FAIL threshold decision (§9.1 there): start by recording real numbers, let
WBS 3.3 set thresholds once real evaluation runs exist. Open question: carried to WBS 3.3.

**Decision 4 — Rollback is expressed as `RETIRED` status + re-pointing the deployed version, not a separate
mechanism.** Why: no note documents an actual rollback-from-deployed event to design against; the closest
precedent (SFT 6/7 epoch selection) is a pre-promotion comparison, not a rollback. An append-only status
history (never delete/overwrite a record) is the simplest mechanism that still supports "go back to the
previous promoted version" without inventing an untested rollback workflow. Evidence: [PROJECT] absence
confirmed by search; [REC] mechanism. Alternatives: a dedicated rollback API/versioned-pointer history table —
deferred as unnecessary complexity until a real rollback need is observed. Open question: none blocking.

**Decision 5 — `training_run_id` is a required, persisted field on the registry record, not merely an
API-assembled convenience.** Why: the cross-cutting API contract (`docs/api/mlops-api-contract.md` §9.2)
identified that no source document — including this one — schematizes "a training run in progress," and that
without a persisted field here, a registry record's link back to the run that produced it only exists inside
the backend's API layer, not in the registry's own data. That's a real lineage gap: this document's own stated
purpose (§1) is closing "sulit menjawab pertanyaan sederhana seperti 'model v3 kemarin itu pakai config LoRA
yang mana?'" (`Materi/11`) — the same question applies to "which training run produced this version," and it
deserves the same persisted answer as dataset/config lineage already gets. Evidence: [PROJECT]
`docs/api/mlops-api-contract.md` §1, §9.2 (gap identified during OpenAPI formalization); [REC] resolution.
Alternatives: leave the link as an API-only assembled field (the original design) — rejected now that the
gap has been identified and fixing it is a single additive field, not a redesign. Open question: none
blocking — `docs/api/mlops-api-contract.md` §9.2's `training_run` entity itself remains this document's
dependency, not the other way around; this document still does not define that entity's own schema.

---

## 10. Open questions

1. **[UNKNOWN]** Persistent artifact storage location/backend — today artifacts are local-Colab-ephemeral
   only (manually zipped and downloaded); this registry's `artifacts[].uri` needs *some* durable target
   (VM disk path, object storage, or HF Hub) before it can be more than a metadata shell. Directly tied to
   WBS 1.1 §4's open "storage for datasets/artifacts" blocker — not resolved here, carried forward.
2. **[UNKNOWN]** Promotion/rejection thresholds and rollback trigger conditions — explicitly WBS 3.3's design,
   not this document's. This registry only guarantees the data (§5) exists for that decision to be made
   against.
3. **[UNKNOWN]** Whether `merged`/`gguf` artifacts will ever actually be produced depends on the still-open
   deployment-stack decision (`mlops-architecture.md` Decision 3, not locked) — the schema supports recording
   them if/when that's resolved, without requiring it now. If produced, plan storage for ~56GB (revised
   estimate, §3), not the ~15GB figure from the earlier 7B-practice assumption.
4. **[RESOLVED — no longer open]** Base model identity: `Qwen/Qwen3.8-27B`, confirmed via
   `Riset Metode Fine-Tuning Qwen3.8-27B di Single H100 80GB.md` (verified against the official HF model
   card, released 14 Aug 2026, Apache 2.0). This was the single highest-leverage unknown carried from WBS 1.1
   §5.1 — resolving it here only updates this document's own placeholders (§6); WBS 1.1/2.1/2.2's open-questions
   sections still list it as unresolved and are unchanged by this update (out of this revision's scope).
5. **[UNKNOWN]** Exact adapter parameter count/size for `Qwen/Qwen3.8-27B` specifically — the research note
   itself flags this as unmeasured ("order puluhan-ratusan juta parameter ... bergantung hidden_dim & jumlah
   layer persis, yang belum dipublikasikan model card"). The ~100MB figure in this document's sample record
   (§6) is still the 7B-practice observed size, not a confirmed number for the actual target model.
6. Carried over unchanged from WBS 1.1: VM environment (network/CUDA/storage/credentials), DEFNEX integration
   boundary (whether this lightweight registry should later feed into a centralized DEFNEX MLflow registry per
   `notebooklm/13`'s INFERENCE-tagged centralization idea — flagged there as inference, not confirmed fact).

---

## 11. Recommended next steps

1. Decide a persistent artifact storage target (even a plain shared VM disk path is enough for Phase A) —
   this is the single blocker preventing `artifacts[].uri` from being anything but a placeholder, same root
   cause as the VM storage blocker already flagged in WBS 1.1. With the base model now confirmed
   (`Qwen/Qwen3.8-27B`), size the target for at least ~56GB headroom if merged/GGUF export is ever chosen
   (§3), not the smaller 7B-practice figure.
2. Retrofit register-time metadata capture into the *next* SFT training run — now unblocked on the base-model
   unknown specifically (confirmed `Qwen/Qwen3.8-27B`, DoRA/bf16 recipe already drafted in
   `Riset Metode Fine-Tuning Qwen3.8-27B di Single H100 80GB.md` §7), though still blocked on VM
   provisioning — rather than trying to reconstruct lineage for the 9 already-completed 7B/T4 practice
   iterations from memory. `Materi/11` already recommended this same forward-only approach for its proposed
   manual table.
3. When WBS 3.3 is designed, confirm it reads `evaluation` (§5) and writes `promotion_decision_ref` and
   `status` back onto this record — the two integration points this document assumes but does not itself
   build.
4. Confirm with WBS 2.2's owner (already requested there) that `dataset_validation_report_ref` is a stable,
   predictable path (`validation/{dataset_id}/{version}/report.json`) before wiring it into register-time
   payloads.
