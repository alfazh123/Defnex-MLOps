# WBS 2.1 — SFT Dataset Preparation & Data Intake

Scope: dataset directory structure, lifecycle, naming/versioning convention, canonical SFT schema, and public
dataset selection for Phase A. Validation *rules* are WBS 2.2 (separate document) — this doc only decides
where validation *artifacts* live. Model registry/lineage fields this feeds into are WBS 3.2's job; the
boundary it must satisfy is defined in `docs/architecture/mlops-architecture.md` §1.6.

Labels: **[PROJECT]** = confirmed from this vault's own notes, **[DOCS]** = confirmed from official Unsloth
documentation, **[HF]** = confirmed from a Hugging Face dataset's actual API/card, **[REC]** = engineering
recommendation, **[UNKNOWN]** = needs confirmation.

---

## 1. Directory structure & lifecycle — critique of the proposed layout

Proposed: `dataset/{raw/, processed/, validation/, train/, eval/}`.

**[PROJECT]** Context: no such directory-based structure has ever existed in this project — all 9 documented
SFT practice iterations cleaned/split data **in-memory** via HuggingFace `datasets` (`load_dataset` →
`.filter/.map/.shuffle`) inside a notebook, with no intermediate files saved to disk. The closest thing to
versioning was the *notebook filename* itself (`Latihan_SFT_Qwen2_5_(7B)_2.ipynb` → `_6.ipynb`) acting as a
de facto pipeline version. So this WBS is not correcting a bad existing structure — it's formalizing
something that never had a file-based structure before. **[REC]** the proposed 5-directory layout is
appropriate for that formalization step, with one clarification and one addition below.

| Directory | Purpose | Contents | Mutable? |
|---|---|---|---|
| `raw/` | Exact, unmodified snapshot of an ingested source dataset | Original HF dataset dump (e.g. Parquet/JSON as downloaded), one subfolder per `{dataset_id}/{version}/` | Immutable once written |
| `processed/` | Cleaned, deduplicated, canonical-schema records (§4) — still model-agnostic, no chat template applied yet | JSONL files in the canonical schema | Immutable once written (a new cleaning pass = new version, not an edit) |
| `validation/` | **Validation reports about a `processed/` version — not a copy of the data itself** | JSON validation-result artifacts (status, errors, warnings, stats) per WBS 2.2's schema | Append-only (one report per validation run) |
| `train/` | Training split, derived from a specific `processed/` version + split config | JSONL, canonical schema, one file per version | Immutable once written |
| `eval/` | Two distinct things, kept as two subfolders (see §3) | `eval/holdout/` (random statistical split) and `eval/benchmark/` (fixed hand-curated question set) | Immutable once written |

**Clarification (validation artifacts stored separately — yes):** the task explicitly asks whether
validation artifacts should be separate from data. **[REC] Yes** — `validation/` holds only the
machine-readable validation *report* (pass/fail status, per-record errors/warnings, dataset-level stats), not
a duplicate of the dataset. This keeps `processed/` as the single place data actually lives, and
`validation/` as an audit trail of what a quality gate said about it — which also matches how WBS 2.2's
VALID/INVALID/NEEDS_REVIEW status needs to be recorded and looked up later.

**Addition (§3 below): `eval/` needs two subfolders, not one.** The vault's own practice already discovered
this need the hard way — see §3.

### Lifecycle between directories

```
[Public HF dataset] ──download, unmodified──> raw/{dataset_id}/{version}/
                                                      │
                                              cleaning + canonicalization (§4 schema)
                                                      ▼
                                          processed/{dataset_id}/{version}/
                                                      │
                                    ┌─────────────────┼──────────────────────┐
                                    ▼                 ▼                      ▼
                        validation/{dataset_id}/{version}/   train/{...}/eval/holdout/{...}
                        (WBS 2.2 report, run against            (automatic random split,
                         the processed/ version)                 §3)
                                                                          │
                                                              eval/benchmark/{dataset_id}/{version}/
                                                              (fixed hand-curated questions, §3 —
                                                               NOT derived from processed/, sourced
                                                               separately, checked for overlap)
```

`raw/` and `processed/` are never edited in place — a new cleaning pass produces a new version directory.
This matches Materi/11's own recommendation (see §2) to avoid mutating named artifacts and instead track
lineage through versions.

### Dataset intake interface: source → intake → normalization → canonical ChatML → processed

**[REC]** This names the five stages already implied by the `raw/` → `processed/` step above, so intake logic
has an explicit interface to implement against rather than being folded silently into "cleaning":

1. **Source** — the external origin of the data (a public HF dataset for Phase A, or a future production
   source). Identified by `source_dataset` (e.g. an HF repo id) and `source_commit_or_snapshot_date`.
2. **Intake** — unmodified download, written as-is to `raw/{dataset_id}/{version}/`. No transformation happens
   here; this is the immutable capture step, and its manifest is what makes the source traceable later.
3. **Normalization** — format-specific conversion into the canonical shape: Alpaca `instruction/input/output`
   → `user`/`assistant` messages, ShareGPT `from`/`value` → `role`/`content`, etc. (§4). This is where
   per-source-format adapters live; it is not model-specific.
4. **Canonical ChatML** — the model-agnostic `messages` + `metadata` record (§4), the output of
   normalization, not yet deduplicated/validated or split.
5. **Processed/versioned dataset** — normalized records after cleaning (WBS 2.2 rules), written to
   `processed/{dataset_id}/{version}/` with its own manifest. This is what train/eval splitting (§3) and
   validation operate on.

**Minimum intake metadata for traceability** — every record must be traceable back through these five stages
without ambiguity. This is already mostly defined in §2's manifest and §4's per-record `metadata` block; the
one addition needed here is recording the *original* format before normalization:

- Per version (manifest, §2): `source_dataset`, `source_commit_or_snapshot_date`, and **`source_format`**
  (`alpaca` | `sharegpt` | `chatml` | `other`) — the format the source was in *before* normalization, so it's
  always clear which adapter produced a given `processed/` version.
- Per record (§4 `metadata` block): `source_dataset`, `source_id` (the row's identifier in the source),
  `license`, `cleaning_version`.

No new metadata scheme is introduced — this section only names the pipeline stages and adds `source_format`
to the manifest fields already defined in §2.

---

## 2. Naming & versioning convention

**[PROJECT]** `Materi/11 - Model & Dataset Versioning (Lineage).md` already contains a documented
recommendation, studied before this WBS existed: keep versioning **lightweight** (a metadata table, not
DVC/MLflow-level tooling — explicitly called "overkill, for later" at this project's scale), and explicitly
**avoid manual meaning-encoded filenames** like `dataset_v7_final_FIX.csv`, preferring seed/config-based
traceability instead. **[REC]** this document follows that guidance directly:

- **Dataset identifier**: `{dataset_id}` — a short slug for the source dataset (e.g. `no_robots`,
  `alpaca_cleaned`, `indonesian_legal_qa`), not a description of what changed in it.
- **Version**: a plain incrementing integer (`v1`, `v2`, ...) per `dataset_id` — no `_final`, `_FIX`, `_clean2`
  suffixes. What changed between versions belongs in the manifest, not the folder name.
- **Manifest** (`manifest.json`, one per version directory, in every `raw/`, `processed/`, `train/`, `eval/`
  version folder): `dataset_id`, `version`, `source_url_or_hf_id`, `source_commit_or_snapshot_date`,
  `source_format` (`alpaca` | `sharegpt` | `chatml` | `other` — the format before normalization, see the
  intake interface above), `seed`, `row_count`, `cleaning_steps_applied` (ordered list of rule names/versions
  — see WBS 2.2), `created_at` (ISO 8601), `created_by`. This is the same content shape as the 7-field manual
  metadata table
  `Materi/11` already recommends recording per practice session (base model, dataset+seed, config, date,
  author, eval results, artifact names) — narrowed to the dataset-relevant subset here.
- This `{dataset_id}/{version}` pair is exactly what `docs/architecture/mlops-architecture.md` §1.6 expects
  the model registry to record as `dataset_id`/`dataset_version` for lineage — so this convention is not just
  internal to this WBS, it's the value the model artifact boundary consumes downstream.

**[REC]** Do not version `validation/` reports independently — a validation report's identity is
`{dataset_id}/{version}` (which `processed/` version it validated) plus a `run_at` timestamp; no separate
version counter is needed for the report itself.

---

## 3. Train/eval split strategy — automatic split *and* a separate fixed benchmark set

**[PROJECT] — this is the single most important lesson to carry over from existing practice.** The vault's
9 SFT iterations actually used **two different, complementary mechanisms**, not one:

1. **Automatic statistical split** (introduced from iteration 6 onward): `train_test_split(test_size=0.05,
   seed=3407)`, i.e. a fixed-seed 95/5 random split, used to compute `eval_loss` during training. Before this
   was introduced, only train loss was tracked — with no way to detect overfitting.
2. **A fixed, hand-curated 20-question benchmark set**, deliberately written to be *outside* the training
   data, reused identically across all 9 iterations for qualitative before/after comparison — plus a second
   5-question general-knowledge set added later specifically to catch catastrophic forgetting. This is not a
   sample of the training data; it's purpose-built and lives independently.

**[PROJECT] — documented regression to explicitly avoid repeating:** early notebook versions had an
overlap-check cell verifying the benchmark questions never leaked into the training data — **this check was
silently dropped by the third notebook revision** and never reinstated. That's exactly the kind of gap a
quality gate should catch automatically rather than rely on a notebook cell someone remembers to keep.

**[REC]** Formalize both mechanisms as directory-level concepts, and make the leakage check a required,
automatic validation rule (feed this forward into WBS 2.2 rather than re-deciding it here):

- `eval/holdout/` — **automatic**, derived from `processed/` via a fixed-seed random split. **95/5 is a
  default candidate only** (it's what practice used, at a ~2,000-row pool size) — not a hard requirement.
  The actual ratio should be chosen per dataset size and experiment needs (e.g. a smaller pool may need more
  than 5% held out to get a stable eval signal). Used for quantitative metrics (eval loss) during training.
- `eval/benchmark/` — **manual**, curated independently of `processed/`, versioned the same way
  (`{dataset_id}/{version}/manifest.json`), used for qualitative before/after comparisons across training
  runs. Must never be auto-regenerated from `processed/`.
- **Train/eval leakage check between `train/` and both `eval/` subfolders is a required, automatic
  validation rule** (WBS 2.2), not an optional notebook cell — this directly closes the gap that actually
  regressed in practice.

Answering the task's direct question: **train/eval splitting should be automatic for the holdout set**
(reproducible, seed-based, no manual judgment needed) **and manual for the benchmark set** (it exists
specifically because it's hand-picked to probe known failure modes) — both are needed, and conflating them
into one `eval/` folder is what the original proposed structure was missing.

---

## 4. Canonical SFT dataset schema

**[DOCS]** Unsloth Core officially supports three dataset formats for SFT: **Alpaca** (flat
instruction/input/output columns), **ShareGPT** (`from`/`value`, human/gpt roles), and **ChatML/OpenAI
`messages`** (`role`/`content`, user/assistant) — no single format is declared mandatory, but ChatML/messages
is called out as *"probably the most used format"* and what Hugging Face itself defaults to.

**[DOCS]** Critically, the model-specific step is separable from the format-standardization step: converting
a dataset into ChatML `messages` (via `standardize_sharegpt`/manual mapping) is model-agnostic; only the
later `get_chat_template(tokenizer, chat_template="qwen-2.5")` call (or whichever template) is tied to the
specific base model. The official workflow applies both in one notebook, but nothing requires that — the
standardized format can be prepared and validated before the base model is finalized.

**[REC] — this directly resolves an open dependency from WBS 1.1:** since the exact base model
(`Qwen3.8-27B`-ish, per WBS 1.1 §4, still **[UNKNOWN]**) is not yet confirmed, the canonical **stored** schema
is deliberately **model-agnostic ChatML `messages`**, with the model-specific `get_chat_template()` call
deferred to train-time (inside the training runner, not baked into `processed/`/`train/`/`eval/` files).
This means dataset work in WBS 2.1/2.2 does not block on resolving the base-model unknown, and re-running
training against a different base model later does not require re-processing the dataset.

**[PROJECT] — this choice is also empirically supported, not just doc-driven:** the vault's own practice
independently arrived at the same conclusion the hard way — switching from Alpaca-string format to ChatML
`messages` (+ `train_on_responses_only()`, which requires knowing the model's instruction/response delimiter
strings — an argument *for* keeping storage template-agnostic and applying delimiters only at train time)
measurably fixed two real problems (a training-loss anomaly and a general-knowledge regression). This is
independent confirmation from two different sources ([DOCS] + [PROJECT]) converging on the same schema
choice.

### Canonical record schema

```json
{
  "id": "string — unique within dataset_id/version, stable across reprocessing of the same source row",
  "messages": [
    {"role": "system", "content": "string, optional"},
    {"role": "user", "content": "string, required"},
    {"role": "assistant", "content": "string, required"}
  ],
  "metadata": {
    "source_dataset": "string — e.g. HuggingFaceH4/no_robots",
    "source_id": "string — original row id/index in the source dataset, for traceability",
    "category": "string, optional — e.g. source dataset's own category/domain label",
    "language": "string — ISO 639-1, e.g. 'en' or 'id'",
    "license": "string — SPDX identifier of the source dataset's license",
    "cleaning_version": "string — which processed/ version's cleaning steps produced this record"
  }
}
```

- **Multi-turn is supported** (messages is an array, not a fixed 3-field record) — this matches ChatML's
  native shape and doesn't foreclose multi-turn data later, even though Phase A fixtures are single-turn.
- **Alpaca-format sources are converted at intake**, not stored natively: `instruction`+`input` →
  `user.content` (concatenated per Unsloth's own Alpaca-template convention),
  `output` → `assistant.content`. The flat Alpaca fields are a transient intake shape, never the resting
  schema — this is a deliberate correction of the example schema in the original brief
  (`input`/`output`/`instruction` as top-level fields): those are useful as an *ingestion* vocabulary for
  Alpaca-style sources, not as the canonical stored form.
- `id` and `metadata.source_id` together are what WBS 2.2's leakage/duplicate checks and WBS 3.2's lineage
  need to trace a training example back to its origin — kept deliberately simple (strings), no normalized
  relational design at this scale.

---

## 5. Public dataset selection for Phase A

**[HF]** Per the task's own instruction, selection is based on "exercises the real pipeline," not domain
relevance. Two datasets are recommended together, precisely because they exercise **different** intake code
paths:

| Dataset | Format | Rows | License | Why it's picked |
|---|---|---|---|---|
| **HuggingFaceH4/no_robots** | Native ChatML `messages` (role/content) | 10,000 (9,500 train / 500 test) | cc-by-nc-4.0 | Already in the canonical target format — exercises the "already-conversational, apply chat template directly" path with no conversion logic to validate |
| **yahma/alpaca-cleaned** | Alpaca flat columns (instruction/input/output) | 51,760 | cc-by-4.0 | Exercises the Alpaca→canonical-messages *conversion* path — this is exactly the code path that needs validation coverage, not just the already-clean case |

**[REC]** Slice a few hundred rows from each for Phase A pipeline smoke-testing (not real training quality) —
Unsloth's own docs cite 100 rows as a bare minimum for actual training quality, which is a reasonable floor
to sample down to for exercising validation/split/format logic quickly.

**[PROJECT] — noted for completeness, not a Phase A action item:** the vault's own past SFT practice already
used a public dataset too (`Azzindani/Indonesian_Legal_QA`, Apache-2.0, 8,078 rows) — so "public data first"
is already this project's own established practice, not a new constraint being imposed. That dataset remains
a reasonable next step *after* the pipeline is validated against the two picks above, since it's closer to a
real future domain and is not confidential — but it adds no new *format* coverage (it was consumed as flat
Q/A columns, structurally like Alpaca), so it isn't the right choice for the pipeline-exercising goal of
Phase A specifically.

**[REC] — explicitly rejected for Phase A, per the task's own instruction:** picking a dataset "about
defense/legal/regulation" just because it's topically closer to DEFNEX. Both picks above are deliberately
generic.

**[REC] — license note:** the licenses listed in the table above (cc-by-nc-4.0, cc-by-4.0) are sufficient for
**pipeline/engineering validation** in Phase A. They must be **reviewed again before any production or
commercial use** of a model trained on this data — cc-by-nc-4.0 in particular carries a non-commercial
restriction that Phase A's engineering-test use does not trigger, but a later production use might.

---

## 6. Key decisions (consolidated)

**Decision 1 — Canonical stored schema is model-agnostic ChatML `messages`, not Alpaca flat fields, with
chat-template application deferred to train time.** Why: decouples dataset work from the still-unconfirmed
base model (WBS 1.1 §4); matches Unsloth's own most-used format; matches what practice already discovered
empirically fixes real problems. Evidence: [DOCS] Unsloth Datasets Guide, chat-templates docs;
[PROJECT] `Hasil Praktik SFT 8`. Alternatives considered: store in Alpaca format (rejected — ties storage to
a template style already shown to cause issues); store already-templated per-model text (rejected — would
require re-processing on every base-model change). Open question: none blocking.

**Decision 2 — `eval/` is split into `holdout/` (automatic) and `benchmark/` (manual, fixed).** Why: this is
not a new idea, it's formalizing a pattern the project already used and then partially lost (the overlap
check was dropped after v3). Evidence: [PROJECT] `Praktik 1 §6.3`, `Latihan_SFT...§9`. Alternatives: a single
`eval/` folder relying on random sampling only (rejected — loses the deliberate, comparable-across-runs
benchmark question set that was the project's actual mechanism for detecting regressions like catastrophic
forgetting). Open question: none blocking.

**Decision 3 — Validation artifacts are reports in `validation/`, not a copy of the data.** Why: keeps a
single place data lives (`processed/`), avoids duplication, and matches what WBS 2.2's status/report design
needs to persist. Evidence: [REC], consistent with the task's explicit question about this. Alternatives:
store a validated-copy of data (rejected — duplicates storage for no benefit at this scale). Open question:
none blocking.

**Decision 4 — Naming avoids manual meaning-encoded suffixes; versions are plain integers backed by a
manifest.** Why: directly follows an existing, already-studied recommendation in this vault, not a new
external convention. Evidence: [PROJECT] `Materi/11`. Alternatives: content-hash-based version ids (viable
future upgrade, unnecessary complexity for Phase A). Open question: none blocking.

---

## 7. Open questions

1. **[UNKNOWN]** Exact base model name/version — unchanged from WBS 1.1 §5.1. This schema is designed to not
   need an answer yet, but the final `get_chat_template()` string used at train time still depends on it.
2. **[UNKNOWN]** VM environment for actual training — unchanged from WBS 1.1 §4/§5.4. Nothing in this
   document requires the VM; it only affects when real (non-Phase-A) datasets get processed at scale.
3. **[UNKNOWN]** Whether this dataset pipeline must integrate with DEFNEX's broader data stack (Kafka/Airflow)
   — unchanged from WBS 1.1 §5.2. This document assumes the standalone case, consistent with that doc.
4. **[UNKNOWN]** Final holdout split ratio for the actual prototype dataset(s) — 95/5 is documented here only
   as a **default candidate** (practice used it at a ~2,000-row pool size), not a fixed rule. It should be
   revisited per dataset size and experiment needs once real dataset sizes for this prototype are known — not
   blocking, just not to be treated as mandatory.
5. Whether `Azzindani/Indonesian_Legal_QA` (or a similar domain dataset) is ever intended to move from
   "informal past practice" into this formal pipeline, and if so, when — not needed for Phase A, but worth a
   team decision before WBS 2.2's validation rules are asked to handle domain-specific content (e.g. legal
   citation patterns) rather than generic English text.

---

## 8. Recommended next steps

1. Proceed with WBS 2.2 (validation rules) against the canonical schema in §4 — in particular, make the
   train/benchmark leakage check (§3) a hard validation rule, not an optional step, since that's the specific
   gap that already regressed once in this project's own history.
2. Stand up `raw/`, `processed/`, `validation/`, `train/`, `eval/{holdout,benchmark}/` for the two Phase A
   datasets (§5) as the first concrete exercise of this lifecycle — this can happen immediately, no VM needed.
3. When WBS 3.2 (model registry) is designed, confirm its `dataset_id`/`dataset_version` fields consume
   exactly the `{dataset_id}/{version}` pair defined in §2, so lineage tracing works end-to-end without a
   translation layer.
4. Defer any `Azzindani/Indonesian_Legal_QA`-based formalization (§7.5) until after the Phase A pipeline is
   proven against the two generic datasets.
