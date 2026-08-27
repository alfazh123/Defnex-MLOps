# WBS 2.2 — Dataset Validation & Quality Gate

Scope: validation rules and quality-gate flow applied to a `processed/{dataset_id}/{version}/` directory
(WBS 2.1) before it may be split into `train/`/`eval/`. This document does not redesign the canonical schema,
folder structure, or versioning convention — see `docs/dataset/dataset-lifecycle-and-schema.md`. It also does
not evaluate a trained model's behavior (hallucination rate, factual accuracy) — that is an evaluation-stage
concern for a later WBS, not a dataset-validation concern; this document only checks the *data*, not model
outputs.

Labels: **[PROJECT]** = confirmed from this vault's own SFT practice notes (with exact figures where
available), **[REC]** = engineering recommendation, **[UNKNOWN]** = needs confirmation.

---

## 1. Validation rule table

Rules are grouped by what happens to a record when they trigger. Every project-grounded rule cites the exact
figure/threshold already used in practice — these are not invented numbers.

### Hard errors (record is rejected — `INVALID`)

| # | Rule | Check | Evidence |
|---|---|---|---|
| H1 | Required fields present | `id`, `messages` (non-empty array), `metadata.source_dataset`, `metadata.source_id` all present and non-null | **[REC]** — schema baseline from WBS 2.1 §4 |
| H2 | Valid role values | Every `messages[].role` ∈ `{system, user, assistant}` | **[REC]** |
| H3 | Content is non-empty text | Every `messages[].content` is a non-null string, non-blank after `.strip()` | **[PROJECT]** — direct precedent: rows dropped if `Question`/`Answer` empty after strip (`18 Aug 2026 - Praktik 1` §4.3) |
| H4 | Minimum length | Assistant `content` word count > 20 | **[PROJECT]** — exact threshold used in every cleaning version (v1–v3): `len(answer.split()) > 20`, stated purpose "buang jawaban yang tidak informatif" |
| H5 | No residual empty-enumeration artifact | Content does not match the empty-link-artifact pattern (e.g. `"yaitu: , , dan ."` — leftover from bad link-stripping) | **[PROJECT]** — exact rule `ARTEFAK_LINK_KOSONG`, explicitly "dibuang seluruhnya, bukan diperbaiki, karena kalau tetap dipakai, model belajar meniru enumerasi kosong" |
| H6 | No residual boilerplate marker | Content does not match known source-boilerplate patterns (site footers, "baca/klik/simak ulasan" variants) | **[PROJECT]** — this project's own `MARKERS_REGEX`; presence at the *validation* gate means the upstream normalization step (WBS 2.1 §1 intake interface) already failed, so treat as a hard error here rather than re-cleaning silently |
| H7 | Exact-duplicate detection | No two records share identical normalized (`user.content`, `assistant.content`) pairs within the same `processed/` version | **[REC] — new, not previously done.** Explicitly confirmed: no dedup logic of any kind existed in any of the 9 practice iterations. This is a genuine gap being closed here, not a formalization of existing practice |
| H8 | Train/eval leakage | No record's normalized `user.content` appears in both `train/` and any `eval/{holdout,benchmark}/` split | **[PROJECT]** — reinstates a check that existed in early notebooks and was silently dropped by v3 ("cek kebocoran... tidak ada lagi di v3"); the vault confirms it was an **exact-match** check, not fuzzy/substring |
| H9 | Valid encoding | Content decodes as valid UTF-8, no control characters | **[REC]** — defensive; no encoding defect was ever found in this project's own data, but this has not been tested against a second data source either, so this is precautionary, not evidence of a known problem |

### Warnings (record kept as `VALID`, flagged informationally — not blocking)

| # | Rule | Check | Evidence |
|---|---|---|---|
| W1 | Rhetorical-question ending | Answer's last sentence ends in `?` or a question-word pattern | **[PROJECT]** — this was a real, measured problem (56%, 288/514 rows, `Hasil Praktik SFT 4`) but is now a warning, not an auto-reject: it's a style signal correlated with under-cleaned source text, not itself an invalid record |
| W2 | Approaching context length | Estimated token count of the full formatted example approaches `max_seq_length` (2048, per this project's own training config) | **[PROJECT]**-informed threshold — flags examples that risk truncation during training, worth a human glance rather than a silent truncation later |
| W3 | Length outlier | Assistant content length is a statistical outlier vs. the dataset's own distribution (e.g. beyond 3×IQR) | **[REC]** — no max length was ever enforced in practice (only descriptive stats: median 1,108 words / max 2,151 words pre-cleaning were reported, never capped); this closes that gap as a soft flag, not a hard cap, since a hard max was never shown to be necessary |
| W4 | Citation pattern present | Content matches `Pasal \d+`, `UU\.?\s*(No\.?\s*)?\d+[/.]\d{2,4}`, etc. | **[PROJECT]** — exact regex family used for the citation-density composition cap (see dataset statistics, §3); flagged per-record so the dataset-level composition check in §3 has something to aggregate |
| W5 | Mixed-script / non-target-language content | Content contains a large proportion of characters outside the record's declared `metadata.language` script | **[REC]** — precautionary; the only related project finding is a **model-generation** bug (base model switching into Mandarin mid-answer, attributed to a quantization artifact), which is explicitly NOT a data defect — this rule exists in case a future source dataset actually has wrong-language rows, not because this project's data has shown it |

### Quality-review candidates (`NEEDS_REVIEW` — held for human decision)

| # | Rule | Check | Evidence |
|---|---|---|---|
| Q1 | Borderline length | Assistant content between 20–40 words (just above the hard minimum) | **[REC]** — extension of H4; a record that barely clears the minimum-informativeness bar is a reasonable candidate for a human glance rather than automatic inclusion |
| Q2 | Near-duplicate (not exact) | Content pair is highly similar (e.g. fuzzy match) to another record without being byte-identical | **[REC]** — H7 only catches exact duplicates; near-duplicates were never checked for at all in practice, and are lower-confidence to auto-reject, so route to review instead of hard-rejecting |
| Q3 | Multiple warnings on one record | A record triggers 2+ rules from the Warnings table simultaneously | **[REC]** — compounding low-confidence signals is a reasonable review trigger even when no single signal is a hard error |
| Q4 | Answer near-identical to question | Assistant content is largely a restatement of the user content rather than a substantive response | **[REC]** — not previously encoded in this project, but a plausible low-value-pair signal worth human judgment rather than automatic rejection, since it's easy to get a naive check wrong |

---

## 2. Status model — critique of VALID / INVALID / NEEDS_REVIEW

**[REC]** The three-status set is kept, but with one clarification the brief didn't make explicit: **warnings
are not a fourth status.** A record with warnings is still `VALID` — warnings are attached as metadata
(`warnings: []`) on an otherwise-valid record, not a separate state. Only hard errors (§1, Hard errors table)
force `INVALID`, and only quality-review-candidate rules (§1, third table) force `NEEDS_REVIEW`. Collapsing
"has warnings" into its own status would make the state model harder to reason about (is a warned record
usable or not?) without adding real information — the `warnings` array already carries that detail.

- **`VALID`** — no hard errors, zero or more non-blocking warnings. Eligible for `train/`/`eval/` inclusion.
- **`INVALID`** — at least one hard error (§1 Hard errors table). Excluded from `train/`/`eval/`, logged with
  which rule(s) fired.
- **`NEEDS_REVIEW`** — no hard errors, but at least one quality-review-candidate rule fired. Excluded from
  automatic `train/`/`eval/` inclusion until a human resolves it (accept → `VALID`, reject → `INVALID`).

This is a per-record status. A separate, simpler **dataset-level gate decision** (§5) aggregates these counts
into a single PASS/FAIL for the whole `processed/` version — that is intentionally not a fourth status on the
same enum, since it answers a different question (can this version proceed to splitting?) than a per-record
status does.

---

## 3. Dataset statistics to compute

**[PROJECT]** These are grounded in what was actually measured by hand in practice (row counts at each
filter stage, composition ratios) — this section formalizes computing them automatically instead of manually
per notebook run:

- Row counts at each pipeline stage: raw → after empty-field filter → after length filter → after
  artifact/boilerplate filters → final `processed/` count. (Precedent: 8,078 raw → ~4,940 after cleaning →
  2,000 sampled, `Hasil Praktik SFT 6`.)
- Citation-pattern (or any declared composition-control tag) density as a % of the pool, compared against
  whatever composition target is configured for that dataset (precedent: 27.6% raw → capped to 10.0%,
  `Hasil Praktik SFT 7`). This is a **dataset-level** aggregate check, not a per-record rule — a single
  citation-bearing record is fine (W4); the dataset-level *proportion* is what practice found worth
  controlling.
- General/non-domain data mix ratio, if the dataset composition intentionally blends sources (precedent:
  5% → 15% experiments to address catastrophic forgetting, `Hasil Praktik SFT 7`/`8`).
- Length distribution (min/median/mean/max, word count) of assistant content.
- Duplicate count (H7) and near-duplicate count (Q2) — new metrics, never computed before.
- Language distribution (from `metadata.language`).
- Leakage check result (H8): count of overlapping records found between `train/` and each `eval/` subfolder,
  and which ones.

---

## 4. Sample validation result (per-record)

```json
{
  "record_id": "no_robots__v1__000123",
  "dataset_id": "no_robots",
  "dataset_version": "v1",
  "status": "NEEDS_REVIEW",
  "hard_errors": [],
  "warnings": ["W4_citation_pattern_present"],
  "review_flags": ["Q1_borderline_length"],
  "rule_set_version": "2.2.0",
  "checked_at": "2026-08-27T10:15:00Z"
}
```

## 5. Dataset-level validation report (sample)

```json
{
  "dataset_id": "no_robots",
  "dataset_version": "v1",
  "rule_set_version": "2.2.0",
  "run_at": "2026-08-27T10:15:00Z",
  "record_count": 500,
  "status_counts": {"VALID": 468, "INVALID": 21, "NEEDS_REVIEW": 11},
  "warnings_summary": {"W1_rhetorical_question": 4, "W4_citation_pattern_present": 32},
  "dataset_statistics": {
    "row_counts_by_stage": {"raw": 500, "after_empty_filter": 497, "after_length_filter": 481, "final": 468},
    "length_distribution_words": {"min": 8, "median": 61, "mean": 74.2, "max": 512},
    "duplicate_count": 3,
    "near_duplicate_count": 5,
    "leakage_check": {"checked_against": ["eval/holdout/v1", "eval/benchmark/v1"], "overlaps_found": 0}
  },
  "gate_decision": "PASS",
  "gate_reason": "No hard-error rate breach; no leakage found; NEEDS_REVIEW queue within threshold."
}
```

---

## 6. Recommended quality-gate flow

```
processed/{dataset_id}/{version}/ (WBS 2.1 output)
         │
         ▼
  Apply hard-error rules (H1-H9) per record ──> INVALID records set aside, logged, excluded
         │
         ▼
  Apply warning rules (W1-W5) per record ──> attached as metadata on VALID records
         │
         ▼
  Apply quality-review rules (Q1-Q4) per record ──> NEEDS_REVIEW records set aside, held for human decision
         │
         ▼
  Compute dataset-level statistics + leakage check (§3) ── dataset-wide, not per-record
         │
         ▼
  Write validation/{dataset_id}/{version}/report.json (§5)
         │
         ▼
  Gate decision:
    - Leakage found (H8)? ──> FAIL, block entirely regardless of other rates (systemic issue, not tolerable
      at any rate — a single leaked eval question invalidates that question for every future comparison)
    - Hard-error rate / NEEDS_REVIEW queue within configured thresholds? ──> PASS, VALID records proceed to
      train/eval split (WBS 2.1 §3)
    - Otherwise ──> FAIL, dataset version not promoted; cleaning step (WBS 2.1 intake/normalization) revisited
```

**[UNKNOWN]** The exact hard-error-rate and NEEDS_REVIEW-queue-size thresholds for PASS vs FAIL are not
defined anywhere in the vault and were never needed in practice (all cleaning was iterated on manually until
it "looked right"). **[REC]** Start permissive (e.g. flag but don't auto-fail below 100% leakage) and
tighten based on the first real run's numbers, rather than guessing thresholds now — this is one case where
"start simple and measure" is preferable to inventing a number without evidence.

---

## 7. Machine-readable validation result schema (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SFTValidationRecordResult",
  "type": "object",
  "required": ["record_id", "dataset_id", "dataset_version", "status", "rule_set_version", "checked_at"],
  "properties": {
    "record_id": {"type": "string"},
    "dataset_id": {"type": "string"},
    "dataset_version": {"type": "string"},
    "status": {"type": "string", "enum": ["VALID", "INVALID", "NEEDS_REVIEW"]},
    "hard_errors": {"type": "array", "items": {"type": "string"}},
    "warnings": {"type": "array", "items": {"type": "string"}},
    "review_flags": {"type": "array", "items": {"type": "string"}},
    "rule_set_version": {"type": "string"},
    "checked_at": {"type": "string", "format": "date-time"}
  },
  "additionalProperties": false
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SFTValidationDatasetReport",
  "type": "object",
  "required": ["dataset_id", "dataset_version", "rule_set_version", "run_at", "record_count",
               "status_counts", "gate_decision", "gate_reason"],
  "properties": {
    "dataset_id": {"type": "string"},
    "dataset_version": {"type": "string"},
    "rule_set_version": {"type": "string"},
    "run_at": {"type": "string", "format": "date-time"},
    "record_count": {"type": "integer", "minimum": 0},
    "status_counts": {
      "type": "object",
      "properties": {
        "VALID": {"type": "integer"}, "INVALID": {"type": "integer"}, "NEEDS_REVIEW": {"type": "integer"}
      },
      "required": ["VALID", "INVALID", "NEEDS_REVIEW"]
    },
    "warnings_summary": {"type": "object", "additionalProperties": {"type": "integer"}},
    "dataset_statistics": {"type": "object"},
    "gate_decision": {"type": "string", "enum": ["PASS", "FAIL"]},
    "gate_reason": {"type": "string"}
  },
  "additionalProperties": false
}
```

---

## 8. Key decisions (consolidated)

**Decision 1 — Warnings are metadata on `VALID` records, not a fourth status.** Why: keeps the status enum
simple and matches what the brief actually asked for (hard errors / warnings / quality-review as three
*kinds of findings*, not necessarily three *statuses*). Evidence: [REC], reasoned from the task's own
phrasing. Alternatives: a 4-value enum (`VALID`, `VALID_WITH_WARNINGS`, `NEEDS_REVIEW`, `INVALID`) — rejected
as unnecessary complexity; the `warnings` array already carries the same information. Open question: none
blocking.

**Decision 2 — Exact-duplicate and train/eval-leakage detection are new hard errors, not formalizations of
prior practice.** Why: duplicate detection never existed in any of the 9 practice iterations (confirmed
absent, not just undocumented); leakage checking existed once and was silently dropped — both are being
(re)introduced here deliberately because their absence is a documented, real gap, not a hypothetical one.
Evidence: [PROJECT] direct confirmation from practice notes. Alternatives: leave dedup as a future upgrade —
rejected, since this is exactly the kind of gap a "quality gate" WBS package exists to close. Open question:
none blocking.

**Decision 3 — Hallucination/factual-accuracy detection is explicitly out of scope for dataset validation.**
Why: those are properties of a *trained model's output*, not the dataset — this project's own practice
notes already draw this distinction (citation hallucination was measured via post-training generation, never
via a data-validation rule). Conflating them would make this WBS responsible for something it can't actually
check statically. Evidence: [PROJECT] `Hasil Praktik SFT 6-9` citation-hallucination findings were all
measured via the fixed 20-question eval set at inference time, never via a dataset-level rule. Alternatives:
none seriously considered — this is a scope boundary, not a design trade-off. Open question: none blocking.

**Decision 4 — No hard maximum length rule; length outliers are a warning (W3), not a rejection.** Why: no
maximum was ever enforced in practice despite descriptive stats showing wide variance (median 1,108 words,
max 2,151 pre-cleaning), and nothing in the vault suggests long answers were themselves a problem (only
*short*, boilerplate-contaminated ones were). Evidence: [PROJECT] `Hasil Praktik SFT 4` audit. Alternatives:
hard cap at some token count — rejected for lack of evidence it's needed; W2 (approaching `max_seq_length`)
already covers the actual practical risk (training-time truncation). Open question: none blocking.

---

## 9. Open questions

1. **[UNKNOWN]** PASS/FAIL thresholds for the dataset-level gate (§6) — never needed in practice since
   cleaning was iterated manually; recommend starting permissive and tightening from real numbers rather than
   guessing now.
2. Carried over, unchanged from WBS 1.1/2.1: base model identity, VM environment, DEFNEX integration
   boundary — none of them affect this document's rules, since validation operates on the model-agnostic
   canonical schema (WBS 2.1 §4).
3. **[UNKNOWN]** Whether near-duplicate detection (Q2) needs a specific similarity algorithm/threshold now,
   or can be deferred until the Phase A public datasets are actually run through this gate and a real
   near-duplicate rate is observed — recommend the latter, consistent with the decision-avoidance-without-
   evidence pattern already used elsewhere in this document (§6).

---

## 10. Recommended next steps

1. Run this rule set against the two Phase A datasets already selected in WBS 2.1 §5
   (`HuggingFaceH4/no_robots`, `yahma/alpaca-cleaned`) as the first real exercise — this requires no VM and
   will surface real threshold numbers to resolve open question §9.1.
2. Implement H8 (leakage check) first among the new rules — it directly closes the one gap this project has
   already been burned by once (the dropped overlap check).
3. When WBS 3.2 (model registry) is designed, confirm the registry can reference a specific
   `validation/{dataset_id}/{version}/report.json` as part of a training run's lineage record, so "was this
   dataset version's gate passed" is traceable from the model side too.
