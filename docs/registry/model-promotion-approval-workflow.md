# WBS 3.3 — Model Promotion / Approval Workflow

Scope: the promotion/approval decision logic explicitly deferred by `docs/registry/model-artifact-versioning-lineage.md`
(WBS 3.2, §4/§9 Decision 3) and by `docs/architecture/mlops-architecture.md` (WBS 1.1, §1.6: "the transition
rules themselves are WBS 3.3's design, not this document's"). This document designs **who/what decides a
promotion, on what evidence, and what gets recorded** — it does **not**:
- redesign the model registry schema (`model-artifact-versioning-lineage.md`, WBS 3.2) — this doc only adds
  the `promotion_decision` record that 3.2's `promotion_decision_ref` field already points to,
- redesign dataset schema/validation (`docs/dataset/dataset-lifecycle-and-schema.md`,
  `docs/dataset/validation-rules.md`, WBS 2.1/2.2),
- lock a deployment/serving stack (`mlops-architecture.md` §3 Decision 3 — still not locked, orthogonal to
  this document),
- invent numeric pass/fail thresholds. Per explicit instruction and per the vault's own evidence (§1, §6
  below), no numeric threshold for eval_loss improvement or majority-win count exists anywhere in the
  project's notes. Where a threshold would normally go, this document marks **UNKNOWN / TO BE DETERMINED**
  rather than proposing one.

Labels: **[PROJECT]** = confirmed from this vault's own notes, **[REC]** = engineering recommendation,
**[UNKNOWN]** = needs confirmation.

---

## 1. Evidence base — what promotion decisions actually look like today

**[PROJECT]** The only documented promotion method is the mentor's 3-signal method
(`Praktik/20 Aug 2026 - Bimbingan Mentor - Deep Dive Unsloth, Multi-Routing, Eval Checkpoint.md` §4):

1. **Kuantitatif** — `eval_loss` trend, epoch-over-epoch. Falling is a positive signal; rising is flagged as
   "sinyal overfitting, jangan otomatis pakai checkpoint terbaru" (a warning against automatic promotion, not
   a threshold).
2. **Kualitatif** — majority-win across a fixed 20-question comparison table, with no new failure pattern.
3. **No general-domain regression** — a fine-tuned checkpoint must not get *worse* at out-of-domain questions
   than the base model, relevant once multi-routing (base vs. adapter by domain) is in play.

**[PROJECT]** The method is explicitly **judgment-based, not automatable as written**: *"Kapan boleh naik ke
checkpoint yang 'lebih pintar': jangan berdasarkan satu sinyal saja... Kalau cuma salah satu yang positif,
catat sebagai temuan tapi jangan langsung jadikan checkpoint default."* No count, percentage, or loss-delta
number is ever given — "mayoritas" (majority) and "turun"/"naik" (down/up) are the only stated criteria, and
they are stated as inputs to a human decision, not as a formula.

**[PROJECT]** The eval table itself already tracks more than binary win/loss: the mentor's own note asks for a
**confidence level per comparison** (High/Medium/Low) in addition to the binary "Lebih baik?" column,
*"terutama untuk pertanyaan yang jawabannya sama-sama masuk akal tapi beda gaya"* (§4). This confidence field
is not yet represented in WBS 3.2's `qualitative_comparison` schema (`wins/losses/ties/total` only) — flagged
as an open question below (§11.1) rather than silently added to a document this WBS does not own.

**[PROJECT]** A real precedent for signal 3 (general-domain regression) already exists and shows the process
working as intended: `Praktik/Hasil Praktik SFT 7.md`'s own conclusion states the general-domain mitigation
attempted that iteration was insufficient — *"mixing dataset umum belum cukup dosis untuk menutup regresi
kemampuan umum, bahkan soal planet terdekat matahari regresinya lebih buruk dari sebelumnya"* (mixing in
general-domain data wasn't enough to close the regression gap; a general-knowledge question actually regressed
worse than before). This is a concrete, real instance of "regression found → do not promote as default," not a
hypothetical — the workflow below is designed to make this exact finding a recorded, first-class input to a
promotion decision instead of a note buried in a practice log.

**[PROJECT]** `Praktik/Hasil Praktik SFT 6.md` shows the human-judgment style concretely: the 20-question table
plus a follow-up action item to manually re-verify specific citations flagged as "the most relevant answers
used as 'better' in the table above" — i.e. even a recorded win is treated as provisional pending manual
spot-check, reinforcing that this is a human-supervised process, not an automated gate.

**[PROJECT]** No monitoring, alerting, or automated rollback-trigger system exists anywhere in the vault. Any
"something is wrong in production" signal today would be human-observed, not system-detected. This shapes §9
(rollback semantics) below.

---

## 2. Model lifecycle states and allowed transitions

**[PROJECT]** Reuses the enum already defined in `model-artifact-versioning-lineage.md` §4 verbatim — this
document does not add or rename states, only defines the transition *rules* that enum's own text explicitly
deferred:

```
REGISTERED → EVALUATED → PROMOTED | REJECTED → DEPLOYED → RETIRED
```

| From | To | Condition |
|---|---|---|
| *(none)* | `REGISTERED` | Training run completed, adapter artifact recorded (WBS 3.2 Register operation). No evaluation data required yet. |
| `REGISTERED` | `EVALUATED` | All three signal fields in WBS 3.2 §5 (`eval_loss_trend`, `qualitative_comparison`, `general_domain_regression_check`) are present on the record. Partial data does not qualify — a record with only `eval_loss_trend` filled stays `REGISTERED`. |
| `EVALUATED` | `PROMOTED` | A human promotion decision (§7) is recorded referencing this version's evaluation data. See §3. |
| `EVALUATED` | `REJECTED` | A human promotion decision recorded as a rejection. See §4. Also reachable directly if evaluation itself reveals a disqualifying regression (§4) — still a recorded human decision, not automatic. |
| `PROMOTED` | `DEPLOYED` | Deployment action taken against a `PROMOTED` version (§3). |
| `DEPLOYED` | `RETIRED` | Superseded by a later version being deployed for the same `model_id`, or an explicit rollback decision (§9). |
| `REJECTED` | *(terminal)* | **[REC]** No transition out of `REJECTED`. A rejected version is not "fixed in place" — per the append-only versioning philosophy already established in WBS 2.1/3.2, a fix produces a **new** `version` that re-enters the pipeline at `REGISTERED`. This keeps every version's history immutable instead of allowing silent re-decision on the same record. |

**[REC]** Note on the word "APPROVED" in the WBS 3.3 task brief: this document treats "APPROVED" and the
already-defined `PROMOTED` status (WBS 3.2 §4) as the same concept. Introducing a second, differently-named
status for the identical decision point would fragment the two documents' schemas for no benefit — the gate
described as "EVALUATED → APPROVED → DEPLOYED" in the task brief is designed here as `EVALUATED → PROMOTED →
DEPLOYED`, reusing WBS 3.2's existing enum unchanged.

---

## 3. Promotion gate: `EVALUATED → PROMOTED → DEPLOYED`

**[REC]** Two distinct gates, not one, even though a 1-month prototype may choose to execute them back-to-back:

1. **`EVALUATED → PROMOTED`** — the approval gate. Requires: complete evaluation data (§2 table) +
   a recorded human decision (§7, §8) with rationale. This is the gate this WBS is primarily about — it is
   where the 3-signal method (§1, §6) is actually consulted.
2. **`PROMOTED → DEPLOYED`** — the release gate. Requires: a `PROMOTED` version + a deployment target that
   exists. **[UNKNOWN]** This gate cannot be fully specified yet because the deployment/serving stack is not
   locked (`mlops-architecture.md` §3 Decision 3) — vLLM/llama.cpp is a leading candidate, not confirmed.

**[REC]** For Phase A specifically, it is reasonable for one human action to satisfy both gates at once (i.e.
"approve" in the frontend's promotion UI, per `mlops-architecture.md` §1.2's "promotion approve/reject UI",
immediately triggers deployment) rather than requiring two separate clicks — there is no documented need for
a separate release-manager role distinct from whoever makes the promotion call. Keeping the two gates
*conceptually* distinct in this document (rather than merging them into one status) means splitting them
later, if a separate deployment approval step is ever needed, does not require a schema change — only a
process change. This also keeps Model Registry state (`PROMOTED`) and deployment state (`DEPLOYED`,
`current_deployed_version`) conceptually separate per the task's explicit constraint, even when one human
action causes both to update.

**[UNKNOWN]** Whether Phase A actually needs the `PROMOTED`-but-not-yet-`DEPLOYED` gap to exist as an
observable waiting state (e.g. "approved, deploying later") or whether it's always instantaneous — no vault
note describes an actual deployment step ever having happened, so there is no practice to confirm this
against.

---

## 4. Rejection and `RETIRED` paths

**Rejection (`EVALUATED → REJECTED`):**
- **[REC]** A rejection is a human decision like a promotion, not a system-computed outcome — the same
  record structure (§8) is used for both, with `decision: REJECTED`.
- **[PROJECT]** Grounded directly in the vault's own guidance: *"kalau cuma salah satu yang positif, catat
  sebagai temuan tapi jangan langsung jadikan checkpoint default"* — i.e. partial/mixed signals are the
  **expected, common** rejection case, not an edge case. The SFT 7 general-domain-regression finding (§1) is
  exactly this: a real iteration where the fix was judged insufficient and the checkpoint should not become
  the new default.
- **[REC]** A `REJECTED` version is not deleted or hidden — it stays in the registry (per WBS 3.2's read/list
  operation) as a permanent record of what was tried and why it didn't qualify, consistent with WBS 2.1's
  append-only philosophy for datasets.

**`RETIRED` (from `DEPLOYED`):**
- **[PROJECT]** Reuses WBS 3.2 §4/§9 Decision 4 verbatim: retiring a version and re-pointing
  `current_deployed_version` (a registry-level pointer, not a per-record field) to an earlier `PROMOTED`
  version **is** the rollback mechanism — there is no separate rollback API. This document's job is only to
  define *who* can trigger that re-pointing (§7, §9) and *what minimum record* justifies it (§9), since WBS
  3.2 explicitly left the trigger/decision side to this WBS.
- Two paths into `RETIRED`, both ending in the same state:
  1. **Supersession** — a later version for the same `model_id` reaches `DEPLOYED`. The previously-deployed
     version is marked `RETIRED` automatically as part of that same transaction (system-triggered
     bookkeeping, not a separate decision — see §7).
  2. **Rollback** — a human decides the currently-deployed version should stop serving before a replacement
     exists. This is a decision (§9), not automatic, because no monitoring system exists to detect a
     production problem on its own (§1).

---

## 5. Required evidence for each promotion decision

**[REC]** A promotion or rejection decision (`EVALUATED → PROMOTED|REJECTED`) must reference, at minimum, the
full evaluation object WBS 3.2 §5 already defines as stored on the version record:

| Evidence field | Source | What it tells the decision-maker |
|---|---|---|
| `eval_loss_trend` | Training runner, recorded automatically at training completion (already captured by `eval_strategy="epoch"` per `Praktik/20 Aug 2026...md` §1/§4) | Direction only (down = positive signal, up = overfitting warning). No threshold — see §6. |
| `qualitative_comparison` (`wins/losses/ties/total`) | Human-run 20-question comparison, entered by whoever performs the eval pass | Majority direction only. No minimum win count is defined anywhere in the vault. |
| `general_domain_regression_check` (`checked`, `regressions_found[]`) | Human-run, only meaningful once multi-routing/out-of-domain testing exists | Presence of *any* entry in `regressions_found` is itself evidence against promotion — see the SFT 7 example (§1) — but this document does not define "how many regressions is too many," per the no-invented-thresholds constraint. |
| `dataset_validation_report_ref` | WBS 3.2 record, itself pointing to a WBS 2.2 validation report | Confirms the training run's input data passed the WBS 2.2 gate — a promotion decision on data that never passed validation should not happen; this is a precondition check, not a promotion signal itself. |

**[UNKNOWN]** Per-question confidence level (High/Medium/Low, requested by the mentor's own note, §1) is not
yet a field on WBS 3.2's `qualitative_comparison` object. Until that's added (WBS 3.2's call, not this
document's), the decision record's free-text `rationale` field (§8) is the only place this nuance can be
captured for Phase A.

**[REC]** The decision record (§8) must also capture a human-written **rationale**, not just a
PROMOTED/REJECTED flag — because the method itself has no formula, the "why" is the only artifact that makes a
later decision auditable or reviewable. This mirrors the mentor's own emphasis on recording *findings* even
when a promotion doesn't happen.

---

## 6. How the three evaluation signals inform (not decide) the outcome

**[PROJECT]** Restating the constraint directly from the source note (§1, §4): *"jangan berdasarkan satu
sinyal saja"* (don't decide on a single signal alone). This document therefore defines the **shape** of the
decision input, not a scoring function:

- **eval_loss trend down** + **qualitative majority-win** + **no new general-domain regression** → all three
  aligned positively is the case the vault calls out as the confident-promote case. Still a human confirms it
  — this document does not auto-promote even when all three align, because "confident" in the source note
  describes the evaluator's confidence, not a system state.
- **Any signal negative or absent** → per the vault's own instruction, record as a finding, default to **not**
  promoting. The SFT 7 case (§1) is exactly "general-domain regression found → do not make this the new
  default," recorded as a `REJECTED` decision with `rationale` citing the specific regression.
- **Mixed / partial signals** (e.g. loss down but qualitative table mixed) → **[UNKNOWN]** No project evidence
  says what to do here beyond "don't auto-promote, log the finding." This document leaves the call to the
  human decision-maker (§7) with mandatory rationale (§8) — it is not resolved by a rule here, because no rule
  exists in the source material without inventing one.

**[REC]** Numeric thresholds (minimum eval_loss delta, minimum win count, maximum tolerated regression count)
are explicitly marked **UNKNOWN / TO BE DETERMINED**. They should only be set once multiple real training
iterations on the actual target model (`Qwen/Qwen3.8-27B`, per
`Riset Metode Fine-Tuning Qwen3.8-27B di Single H100 80GB.md`) produce enough evaluation data to calibrate a
number against — inventing one now would misrepresent an unconfirmed guess as project practice, the same
reasoning WBS 2.2 and WBS 3.2 already applied to their own threshold decisions.

---

## 7. Who/what can trigger each transition

| Transition | Trigger | Actor | Automation level |
|---|---|---|---|
| `→ REGISTERED` | Training run completes | Training runner → Backend (WBS 3.2 Register op) | **System-triggered.** No human action needed. |
| `REGISTERED → EVALUATED` | All 3 evaluation fields present on the record | Whoever runs the eval pass (today: the practitioner manually, per `Praktik/20 Aug 2026...md`) submits the payload; Backend validates completeness and flips status | **[UNKNOWN]** Mixed — `eval_loss_trend` can be system-captured automatically at training time; `qualitative_comparison` and `general_domain_regression_check` require a human eval pass. No automated eval harness exists in the vault, so a human (or a future eval script) must be the one who calls WBS 3.2's Status-update operation with the evaluation payload. |
| `EVALUATED → PROMOTED` | Promotion decision | **Human only**, via the frontend's "promotion approve/reject UI" (`mlops-architecture.md` §1.2) | **Must not be automated.** No numeric threshold exists to automate against (§6) — automating this transition would require inventing the exact thresholds this document is instructed not to invent. |
| `EVALUATED → REJECTED` | Rejection decision | **Human only**, same UI/actor as above | Same reasoning as promotion — a rejection is also a judgment call, not a computed outcome. |
| `PROMOTED → DEPLOYED` | Deployment action | **[REC]** Same human action as promotion, or a separate explicit deploy action — undecided pending deployment-stack lock (§3) | Backend executes the deployment call; a human authorizes it. Not system-auto-triggered — there is no CI/CD-style "auto-deploy on promote" precedent in the vault, and none should be assumed for a 1-month prototype. |
| `DEPLOYED → RETIRED` (supersession) | A later version for the same `model_id` reaches `DEPLOYED` | **System-triggered bookkeeping**, as a side effect of the human-triggered `PROMOTED → DEPLOYED` transition for the new version | Not a separate decision — the backend marks the old version `RETIRED` in the same operation that deploys the new one. |
| `DEPLOYED → RETIRED` (rollback) | A human observes a production problem | **Human only** (§1: no monitoring/alerting system exists to detect this automatically) | See §9. |

**[REC]** No transition in this table is triggered by the training runner or Unsloth Core itself beyond the
initial `REGISTERED` write — this matches `mlops-architecture.md` §1.2's role boundary (Unsloth Core is the
training engine only; it does not own state or make lifecycle decisions).

---

## 8. Minimum promotion-decision record and its relationship to the model registry

**[REC]** A single record type covers promotion, rejection, and rollback decisions (§9) — one schema, not
three, since all three are "a human decision referencing evidence, with a rationale," differing only in the
`decision` value.

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

- **`evidence_snapshot`** is a **copy** of the evaluation object at decision time, not a live reference —
  **[REC]** because a version's evaluation data on the WBS 3.2 record is otherwise mutable (nothing in WBS 3.2
  marks it immutable once `EVALUATED`), a promotion decision must freeze the exact numbers that justified it,
  so a later re-run or correction of the eval data can't silently invalidate a past decision's audit trail.
- **`rationale`** is mandatory free text, not optional — per §5/§6, this is the only place nuance the fixed
  schema can't capture (confidence level, "one signal only" caveats, the SFT-7-style regression narrative)
  survives.
- **`rollback_of_version`** is `null` for ordinary promotion/rejection decisions; populated only for rollback
  decisions (§9).

**Relationship to the model registry (WBS 3.2):** this record is what WBS 3.2's already-reserved
`promotion_decision_ref` field (§6, §8 there) points to. The registry record itself only ever stores the
*pointer* and the resulting `status` — it does not duplicate `rationale` or `evidence_snapshot` inline. This
keeps the two documents' responsibilities separate exactly as WBS 3.2 already assumed: the registry is
lineage/state, this document is decision/evidence.

**Relationship to deployment state:** `current_deployed_version` (a registry-level pointer, per WBS 3.2 §4) is
updated as a side effect of a `PROMOTED → DEPLOYED` transition or a rollback decision — it is not stored on
this decision record and this record does not model deployment infrastructure (server, endpoint, etc.) at
all. That remains out of scope here, per the task's explicit instruction to keep registry and deployment
state conceptually separate and per `mlops-architecture.md` §3 Decision 3 (deployment stack not locked).

---

## 9. Rollback semantics (deliberately minimal)

**[REC]** Rollback is **not** a new mechanism — it reuses the same decision record (§8) with
`decision: "ROLLBACK"` and `rollback_of_version` set to the version being reverted *to*:

```json
{
  "decision_id": "rollback-qwen-sft-domain-x-v6-001",
  "model_id": "qwen-sft-domain-x",
  "version": 6,
  "decision": "ROLLBACK",
  "decided_by": "UNKNOWN — fill from actual decision-maker identity/role",
  "decided_at": "2026-08-27T14:30:00Z",
  "evidence_snapshot": null,
  "rationale": "UNKNOWN — must state the observed production problem; mandatory for this decision type since no automated trigger exists.",
  "rollback_of_version": 5
}
```

- `evidence_snapshot` is nullable here specifically — a rollback responds to an observed production problem,
  not necessarily to new offline evaluation data. `rationale` is where the observed problem must be described
  instead.
- Effect: `current_deployed_version` moves back to `rollback_of_version` (which must already be `PROMOTED` or
  previously `DEPLOYED`); the version being rolled back *from* transitions `DEPLOYED → RETIRED`.
- **[REC]** No separate "rollback API," no automatic rollback trigger, no health-check/monitoring integration
  is defined — none exists in the vault to design against (§1), and building one is explicitly the kind of
  infrastructure this Phase A prototype should not add ahead of an actual need. If an automated trigger is
  ever wanted, it plugs into this same decision record as its `decided_by` (e.g. `"system:health-check"`
  instead of a person) without changing the schema.
- **[PROJECT]** The closest real precedent remains the SFT 6/7 epoch selection (choosing Epoch 1 over Epoch 2
  for a regression, per WBS 3.2 §4) — but that was a **pre-promotion** comparison, not a rollback from an
  actually-deployed state. No note documents an actual rollback-from-deployed event; this section is
  therefore [REC] design, not [PROJECT] confirmed practice, same caveat WBS 3.2 §9 Decision 4 already carries.

---

## 10. Key decisions

**Decision 1 — Promotion and rejection transitions must be human-triggered, never automated, for Phase A.**
Why: the only documented promotion method has no numeric threshold (§1, §6); automating the gate would
require inventing one, which the task explicitly forbids and which would misrepresent a guess as confirmed
project practice. Evidence: [PROJECT] `Praktik/20 Aug 2026 - Bimbingan Mentor...md` §4 (method stated
directionally, never numerically). Alternatives: auto-promote when all three signals are directionally
positive — rejected, "confident" in the source note describes evaluator judgment, not a system rule, and even
a 3-for-3 case in the source material was never described as auto-actioned. Open question: none blocking —
this can be adopted now.

**Decision 2 — One decision-record schema covers promotion, rejection, and rollback.** Why: all three are
structurally the same event (a human decision, evidence, a rationale) differing only in outcome value;
maintaining three separate schemas would be unrequested complexity for a 1-month prototype. Evidence: [REC]
design choice, no direct precedent either way in the vault. Alternatives: a dedicated rollback API/table
separate from promotion decisions — rejected as over-engineering per explicit instruction, no real
rollback-from-deployed event exists to justify separate infrastructure (§9). Open question: none blocking.

**Decision 3 — The task brief's "APPROVED" state is implemented as WBS 3.2's existing `PROMOTED` status, not
a new state.** Why: introducing a second name for the same decision point across two documents in the same
WBS would fragment the schema without adding meaning. Evidence: [PROJECT] `model-artifact-versioning-lineage.md`
§4 already defines `PROMOTED` as exactly this gate. Alternatives: rename WBS 3.2's status to `APPROVED` —
rejected, this document is explicitly not permitted to modify WBS 3.2; keeping both point at the same concept
under one name is simpler than reconciling two names later. Open question: if a future revision wants
`APPROVED` as the literal string, that's a WBS 3.2 schema change, out of this document's scope.

**Decision 4 — Promotion decisions freeze a copy of the evaluation evidence (`evidence_snapshot`), not a live
reference.** Why: WBS 3.2 does not mark evaluation data immutable once a version is `EVALUATED`; without a
frozen copy, a later correction or re-run of evaluation data could silently invalidate why a past decision was
made, breaking auditability. Evidence: [REC], gap identified by cross-checking WBS 3.2's schema (no
immutability constraint present). Alternatives: reference evaluation data live from the registry record —
rejected for the auditability reason above; requiring WBS 3.2 to make evaluation fields immutable instead —
rejected as a change to a document this WBS may not modify, and a snapshot achieves the same guarantee more
simply. Open question: none blocking.

**Decision 5 — No numeric promotion thresholds are defined in this document.** Why: explicit task
instruction, and no threshold exists anywhere in the vault to ground one in real practice — the source
method is stated directionally only (§1, §6). Evidence: [PROJECT] confirmed absent by direct reading of
`Praktik/20 Aug 2026 - Bimbingan Mentor...md` §4. Alternatives: propose a starting `[REC]` threshold (e.g.
"promote if ≥11/20 wins") — rejected, this would misrepresent an invented number as grounded guidance, the
same reasoning already applied in WBS 2.2 and WBS 3.2's own threshold decisions. Open question: carried
forward to whoever calibrates thresholds once real evaluation data from actual training runs exists (§6).

---

## 11. Open questions

1. **[UNKNOWN]** Per-question confidence level (High/Medium/Low) requested by the mentor's own note (§1, §5)
   is not yet a field in WBS 3.2's `qualitative_comparison` schema. This document cannot add it without
   modifying WBS 3.2 (out of scope here) — flagged for whoever next revises WBS 3.2, or captured informally in
   this document's `rationale` field (§8) in the meantime.
2. **[UNKNOWN]** Whether `PROMOTED → DEPLOYED` should ever require a separate human action from
   `EVALUATED → PROMOTED`, or whether Phase A can safely collapse them into one click (§3). Depends on the
   still-unlocked deployment stack decision (`mlops-architecture.md` §3 Decision 3).
3. **[UNKNOWN]** Numeric promotion thresholds (eval_loss delta, win-count minimum, tolerated regression
   count) — deliberately left undefined per Decision 5. Needs real evaluation data across multiple training
   iterations on the confirmed target model to calibrate, not invented now.
4. **[UNKNOWN]** What "mixed/partial signal" decisions should default to beyond "log the finding, don't
   auto-promote" (§6) — no vault evidence resolves this beyond the human decision-maker's judgment call.
5. **[UNKNOWN]** Identity/role model for `decided_by` (§8, §9) — nothing in the vault specifies who
   (a named mentor, any team member, a specific "approver" role) is authorized to make promotion/rollback
   decisions. `mlops-architecture.md` §1.2 only establishes that the *frontend* exposes an approve/reject UI,
   not who is permitted to use it.
6. Carried over unchanged from WBS 3.2 §10: persistent artifact storage location, VM environment
   (network/CUDA/storage/credentials), and the DEFNEX-integration boundary — none of these block *this*
   document's design, all are training/deployment-execution blockers already tracked in WBS 1.1/3.2.

---

## 12. Recommended next steps

1. When an actual training run on `Qwen/Qwen3.8-27B` produces real `eval_loss`, 20-question comparison, and
   general-domain regression data, run at least 2-3 promotion decisions through this workflow manually
   (human fills the decision record by hand) before considering any tooling to semi-automate parts of it —
   mirrors the same "prove the shape works before building infrastructure around it" approach already used for
   dataset validation (WBS 2.2) and the registry (WBS 3.2).
2. Revisit open question §11.1 (confidence-level field) together with whoever next touches WBS 3.2's schema —
   this document's `rationale` field is a workable stopgap, not a long-term substitute for a structured field.
3. Once the deployment stack (`mlops-architecture.md` §3 Decision 3) is locked, resolve open question §11.2
   (single-click vs. two-step promote/deploy) concretely rather than leaving it a [REC] placeholder.
4. Confirm with the mentor/team who is actually authorized to make promotion and rollback decisions (§11.5)
   before this workflow is wired into a real frontend approve/reject UI — this document assumes "a human,"
   not a specific role, because no note names one.
