# DEFNEX MLOps — Updated Issue Reconciliation After Phase 2A/2B/2C

**Date:** 2026-09-23
**Status:** REVIEW ONLY — no issues were closed, edited, labeled, or mutated by this pass.
**Supersedes:** the 2026-09-22 version of this same file, which was written before a real
Phase 2C report existed in this repository. That version's Phase 2C conclusion ("does not
exist") was correct **at the time it was written** — a `git pull` on 2026-09-23 fast-forwarded
this repo past new commits that added the Phase 2C reports. This version is based on the
current repository (`d3d90c7`) and the current issue tracker, both read directly.

**Correction note (2026-09-23, same day):** the first version of this file stated
`docs/qa/terminal-qa-golden-path.md` did not exist anywhere in the repo. That was accurate at
the time — `git log --all --full-history` for that path returns no commits, ever, on any
branch; it was never previously present, at the repo root or anywhere else. It exists now only
because it was added locally today. This note corrects the record without changing the #167
conclusion, which the file's own content (read in full below) reinforces rather than
contradicts.

## 1. Executive Summary

Phase 2C evidence now genuinely exists in the repository and has been read in full:
`docs/PHASE2C_RUNTIME_LORA_VALIDATION_REPORT.md` (2026-09-16, concluded **NOT VERIFIED**
against the legacy `defnex-vllm` v0.28.0) and `docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md`
(2026-09-22, concluded **E2E VERIFIED** against a new compose-managed `serving` container on
vLLM v0.30.0). Both are real, internally consistent, and consistent with the independently
verified Phase 2A/2B evidence and with a real merged PR (#185, `alfazh123/Defnex-MLOps`,
authored by the repo owner, not this session).

What Phase 2C actually proves: the **mechanism** the backend's `VLLMServingBackend` code
depends on (`POST /v1/load_lora_adapter` → inference → `POST /v1/unload_lora_adapter`) works
against a real freshly-trained adapter (`smoke-llm-v3` from `run-9e66b6`), on vLLM ≥0.29.0 with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`. What it does **not** prove: that the `ml-close-loop-be`
backend's own promotion ladder endpoints (`deploy-staging` → `validate-staging` →
`promote-production` → `POST /models/{id}/inference`) were exercised end-to-end. The validation
in `PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md` calls vLLM's raw HTTP API directly with `curl`-
style requests, not the backend application. This distinction is the crux of the #167
reconciliation below.

A real PR (#185) already partially acted on the previous reconciliation's #162 finding —
before this reconciliation was re-run — by adding a startup health check, a runbook, and
clearer pinning of the external `defnex-vllm` dependency. It did **not** integrate `defnex-vllm`
as a first-class Compose service; the compose `serving` profile remains explicitly labeled
"for FRESH environments only... do NOT enable if defnex-vllm already exists." So #162 is now
**partially** addressed, not resolved.

## 2. Repository Refresh Evidence

| Item | Value |
|---|---|
| Previous HEAD (start of this session) | `3ff1c4c` |
| Local working tree before pull | Clean except untracked `docs/qa/` (this report's own directory, created by the prior reconciliation pass) |
| Pull method | `git fetch origin` then `git pull --ff-only` (no rebase, no force, no discarded work) |
| New HEAD | `d3d90c7` |
| Fast-forward | YES — 3 commits ahead, 0 diverging; ff-only succeeded cleanly |
| New commits | `172be53` fix: harden external vLLM dependency (issue #162); `511a197` feat(serving): validate runtime LoRA via compose serving (Phase 2C); `d3d90c7` Merge PR #185 |
| PR provenance | `gh pr view 185` — real merged PR, author `Maulana-anjari` (repo owner), merged `2026-09-23T04:53:27Z` — not content injected into this conversation |
| Phase 2C files arrived | YES — `docs/PHASE2C_RUNTIME_LORA_VALIDATION_REPORT.md`, `docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md` |
| Other files arrived | `ml-close-loop-be/docker-compose.yml` (serving/health-check changes), `docker-entrypoint.sh`, `docker-worker-entrypoint.sh`, `ml-close-loop-be/docs/runbooks/vllm-dependency.md`, `ml-close-loop-be/scripts/check-vllm.sh` |

## 3. Evidence Sources

| Source | Found? | Verdict as read directly |
|---|---|---|
| `docs/PHASE2A_FINAL_QA_REPORT.md` | YES | READY FOR LIVE SIGNALING (static+test verdict) |
| `docs/PHASE2A_SECOND_LIVE_SIGNALING_TEST_REPORT.md` | YES | 18/18 PASS, live-verified |
| `docs/PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md` | YES | PASS, 16/16 success criteria, live-verified |
| `docs/PHASE2B_POST_RUN_VALIDATION_REPORT.md` | YES | COMPLETE, artifact/lineage verified |
| `docs/PHASE2C_RUNTIME_LORA_VALIDATION_REPORT.md` | YES (new) | Original 2026-09-16 test: NOT VERIFIED (vLLM 0.28.0 too old); amended 2026-09-22 with a follow-up pointing to the compose report |
| `docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md` | YES (new) | 2026-09-22: E2E VERIFIED, against compose `serving` (v0.30.0), not `defnex-vllm` |
| `ml-close-loop-be/docs/RUNTIME_INFRASTRUCTURE_INTEGRATION_VERIFICATION.md` | YES (pre-existing) | Now stale on this one point — it says runtime LoRA was "never successfully exercised"; Phase 2C (2026-09-22) supersedes that specific claim |
| `ml-close-loop-be/docker-compose.yml` (current) | YES | `serving` profile still marked "FRESH environments only," `defnex-vllm` (external, v0.28.0) remains the path actually wired via `VLLM_URL` for `backend`/`worker` |
| `docs/qa/terminal-qa-golden-path.md` (cited by issue #167 as evidence) | **Added 2026-09-23**, locally, by the repo owner (untracked at first read; committed as part of this update) | `git log --all --full-history` for this path returns zero prior commits — it never existed in this repo's history before today, on any branch. It is a rehearsal script dated baseline 2026-09-16 that explicitly labels Sections 12–15 (evaluation, staging, promotion, inference-with-new-adapter) as "CODE-VERIFIED FROM SOURCE, NOT YET DEMONSTRATED END-TO-END," and it predates Phase 2C entirely — it reinforces, not changes, the #167 conclusion below |
| Live GitHub issue tracker (`gh issue view` ×7, `gh issue list`) | YES | Re-read directly on 2026-09-23; same 7 issues still open, same bodies as previous pass (#162's body has not been edited to reflect PR #185 yet) |

## 4. Phase 2A Impact

Unchanged from the previous reconciliation. File-based GPU signaling is live-verified
(18/18 pass), `gpu_controller.py` still ran as a manually-started process in every observed
test (including the 2026-09-22 Phase 2C session, which explicitly notes "GPU controller —
Restarted (PID was dead from previous session)") — i.e. **Phase 2C's own evidence reconfirms
#163 is still open**, since a systemd-managed service would not need manual restart between
sessions.

## 5. Phase 2B Impact

Unchanged from the previous reconciliation. `run-9e66b6` real training run, smoke-scale
(10 examples / 5 steps), artifact + `ModelVersion smoke-llm-v3 v1` correctly registered. Still
trained on generic/smoke data, not DEFNEX-domain data (#161 unaffected). Still executed via the
host-mounted training venv (#166 unaffected).

## 6. Phase 2C Impact

This is genuinely new. Broken down by exactly what was and wasn't demonstrated:

- **Legacy `defnex-vllm` (v0.28.0):** confirmed **incapable** of runtime LoRA — `/v1/load_lora_adapter`
  returns 404, endpoint doesn't exist below vLLM 0.29.0. This is a real, useful negative result,
  not a failure of the investigation.
- **New compose `serving` service (v0.30.0, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`):** the
  full artifact→register→list→infer→unload cycle was exercised against vLLM's own HTTP API
  directly (not through the `ml-close-loop-be` backend) and every step returned the expected
  result (200s, adapter visible in `/v1/models`, generation returned, clean unload, GPU state
  restored to pre-test baseline: 78,964 MiB used / 2,116 MiB free, identical before/after).
- **Configuration changes required to make this work** were nontrivial and are listed
  explicitly in the report (`--gpu-memory-utilization 0.15`, `--max-model-len 1024`, `.env`
  model-name overrides, an `/etc/hosts` DNS workaround, and pulling a newer `latest` image tag)
  — none of these are yet the *default*, tested-in-CI configuration; they were hand-applied for
  this one validation session.
- **What was NOT exercised:** the `ml-close-loop-be` backend's own code path
  (`app/services/serving.py::VLLMServingBackend.deploy()/generate()`) was never called in this
  test — the validation used raw `curl`-equivalent HTTP calls straight to vLLM. The backend's
  promotion ladder endpoints (`deploy-staging`, `validate-staging`, `promote-production`,
  `POST /models/{id}/inference`) were not part of this test at all.
- **What was NOT changed:** the actual `VLLM_URL` the `backend`/`worker` services point at is
  still `http://172.17.0.1:8001` — i.e. still the legacy `defnex-vllm` (v0.28.0, no runtime
  LoRA). The compose `serving` profile is opt-in (`--profile gpu`) and explicitly documented as
  not to be run alongside `defnex-vllm`. **The application today still cannot runtime-load a
  new adapter in its actual configured serving target.**

## 7. Current Verified Model Lifecycle

```
dataset (generic/smoke)         → VERIFIED (works, not domain-specific — #161)
  ↓
TrainingRun (run-9e66b6)        → VERIFIED (Phase 2B)
  ↓
GPU lock / handoff              → VERIFIED (Phase 2A)
  ↓
Unsloth SFT                     → VERIFIED (Phase 2B)
  ↓
LoRA artifact                   → VERIFIED (Phase 2B)
  ↓
ModelVersion registered          → VERIFIED (Phase 2B)
  ↓
Artifact visible to a vLLM ctr  → VERIFIED, but only the compose `serving` container (Phase 2C) —
                                    NOT the actual configured `defnex-vllm` target (still 404s)
  ↓
Runtime LoRA load (raw vLLM API) → VERIFIED against compose serving only (Phase 2C)
  ↓
Inference w/ new adapter (raw vLLM API) → VERIFIED against compose serving only (Phase 2C)
  ↓
Backend promotion ladder (deploy-staging → validate-staging →
  promote-production → app-level inference endpoint)  → NOT VERIFIED — no report of this exists
```

## 8. Issue-by-Issue Reconciliation

| Issue | Original Intent | Current Truth | Status | Recommended Action | Priority | VM-Gated? | Evidence |
|---|---|---|---|---|---|---|---|
| #161 | Real DEFNEX-domain SFT dataset, not generic placeholder | Unaffected by Phase 2C; still smoke/generic data | STILL OPEN | KEEP OPEN | P1 | DATA-GATED | Issue body; `PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md` |
| #162 | Harden/integrate external `defnex-vllm` dependency | PR #185 (merged 2026-09-23) added pinning docs, a startup health check, and a runbook (`docs/runbooks/vllm-dependency.md`) — option (b) from the issue's own "suggested fix." Option (a) (integrate as first-class service) was NOT done; `defnex-vllm` remains external and is still the actual serving target. Issue body itself has not been updated/closed to reflect this. | PARTIALLY RESOLVED (new) | UPDATE DESCRIPTION to record what PR #185 did; KEEP OPEN — do not close, since integration (option a) is undone and Phase 2C's own recommendation is that legacy `defnex-vllm` "should eventually be decommissioned or upgraded," which hasn't happened | P1 (unchanged — still a real operational risk even hardened) | VM-GATED (any actual decommission/upgrade of `defnex-vllm`) | `git log` (172be53, 511a197, d3d90c7); `ml-close-loop-be/docker-compose.yml`; `docs/runbooks/vllm-dependency.md` |
| #163 | Install `gpu_controller.py` as systemd unit | Phase 2C's own session log says the controller had to be manually restarted ("PID was dead from previous session") — direct, fresh confirmation it is still unmanaged | STILL OPEN (reconfirmed) | KEEP OPEN | P1 | VM-GATED | `PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md` §Configuration Changes |
| #166 | Pinned/reproducible training-venv install | Unaffected by Phase 2C (serving-only change) | STILL OPEN | KEEP OPEN | P1 | Authoring CODE-ONLY; validating VM-GATED | Unchanged from previous reconciliation |
| #167 | Rehearse full eval→staging→production→inference ladder against a real freshly-trained model + add regression coverage | Regression-test half done (PR #184, mock-backed). Phase 2C meaningfully de-risks the technical unknown underneath this issue (proves the raw vLLM runtime-LoRA mechanism the backend's serving code relies on actually works, against the real `run-9e66b6` artifact) — but it did **not** exercise the backend's own promotion-ladder endpoints. The issue's cited evidence doc, `docs/qa/terminal-qa-golden-path.md`, was added to the repo on 2026-09-23 and read in full: it is a rehearsal *script* (baseline 2026-09-16, pre-dates Phase 2C) that explicitly self-labels Sections 12–15 (evaluation → staging → promotion → new-adapter inference) as not yet demonstrated end-to-end. It confirms, not contradicts, this conclusion. The live rehearsal *through the application* is still undone. | PARTIALLY RESOLVED (more so than before, but not fully) | UPDATE DESCRIPTION to credit Phase 2C's mechanism-level validation and note the golden-path script (Sections 12-15) is still the open TODO; KEEP OPEN | P1 | VM-GATED (needs the compose serving path wired as the actual `VLLM_URL` target, or an equivalent live app-level rehearsal) | `PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md`; `docs/qa/terminal-qa-golden-path.md` §§12-15 |
| #174 | Real secrets backend | Unaffected | STILL OPEN | KEEP OPEN | P1/P2 | DECISION-GATED | Unchanged |
| #179 | Bundled nice-to-have backlog | Unaffected — none of its checklist items reference vLLM/serving | STILL OPEN (mixed) | KEEP OPEN | P2 | MIXED | Unchanged |

## 9. Duplicate / Overlap Analysis

Same as the previous reconciliation: #162/#163 and #163/#166 overlap thematically
("unmanaged host-level infra") but have distinct, non-duplicate acceptance criteria. No merges
recommended. One new observation: PR #185's runbook (`docs/runbooks/vllm-dependency.md`) is
relevant evidence for both #162 and, secondarily, #163 (it documents the GPU controller as part
of the dependency chain) — but #163's acceptance criterion (systemd unit installed) is
untouched by it, so no merge is warranted there either.

## 10. Stale Acceptance Criteria

- **#166** — unchanged recommendation from the previous reconciliation: sharpen the acceptance
  test to "reproduce `run-9e66b6`'s result from a clean image build."
- **#167** — its cited evidence file (`docs/qa/terminal-qa-golden-path.md`) is now present
  (added 2026-09-23) and was read in full. It corroborates the issue's own claim rather than
  undermining it: Sections 12–15 (the exact ladder this issue asks to be rehearsed) are
  explicitly self-labeled "CODE-VERIFIED FROM SOURCE, NOT YET DEMONSTRATED END-TO-END," and the
  document predates Phase 2C, so it doesn't yet reflect the runtime-LoRA mechanism validation
  either. The acceptance criteria should still be sharpened to state explicitly that "full
  ladder" means through `ml-close-loop-be`'s own API endpoints (Sections 12–15 of the golden
  path script), not direct vLLM calls — otherwise a future contributor could point at Phase 2C
  and claim #167 is done, which it is not.
- **#162** — acceptance criteria still say "either (a) integrate... or (b) explicitly
  document and pin" — PR #185 satisfies (b) literally as written, so if the intent was really
  either/or, a case exists for calling this issue resolved. This reconciliation does **not**
  recommend that, because Phase 2C's own recommendation section says the legacy dependency
  "should eventually be decommissioned" — treating documentation of a fragile dependency as
  equivalent to removing the fragility contradicts the issue's actual "why it matters" section
  ("if that separate environment... moves host, DEFNEX-MLOps training and serving stop working
  entirely"). That risk is only documented now, not eliminated.

## 11. Remaining VM Work

- #162 — actual integration/decommission of `defnex-vllm` (not just documentation)
- #163 — install/enable the systemd unit, verify reboot survival
- #166 — validate a pinned-image worker build against real Unsloth/CUDA
- #167 — live rehearsal through the backend's own promotion-ladder + inference endpoints against
  a real freshly-trained adapter, using the now-validated compose serving path as the target
- #179 — "No CI validation for GPU/training-path components"

## 12. Non-VM Work

- #179 — CI staging-gate design, test-directory restructuring, full ERD, backup/DR doc
  validation-once-Postgres-exists
- Refreshing `docs/qa/terminal-qa-golden-path.md`'s baseline note to acknowledge Phase 2C
  (it currently predates it and still frames runtime LoRA as fully unproven, which is now only
  true for the app-level path, not the underlying mechanism) is a docs-only task, no VM required

## 13. Decision-Gated Work

- #174 — which secrets backend to adopt
- #179 — LLM-judge evaluation scope, MFA/2FA scope
- New: whether to formally adopt the compose `serving` (v0.30.0) path as the target
  architecture and schedule `defnex-vllm`'s decommission, per Phase 2C's own recommendation —
  this is an architectural decision for the project owner, not something to infer from the
  validation report alone

## 14. Data-Gated Work

- #161 — sourcing/curating a real DEFNEX-domain SFT dataset

## 15. Remaining P0

None — unchanged. No open issue blocks the prototype's core path.

## 16. Remaining P1

#161, #162, #163, #166, #167 — unchanged priorities; evidence updates don't change the
priority calculus for any of them.

## 17. Remaining P2

#174, #179 — unchanged.

## 18. Recommended Issue Updates

#162 and #167 both warrant description updates to record what Phase 2C/PR #185 actually
demonstrated, without marking either resolved. See §19 for exact proposed content. No other
issue needs an update.

## 19. Final Backlog Truth

Phase 2C is real, and it answers a genuine open technical question (can the artifact pipeline's
runtime-LoRA mechanism work at all against a modern vLLM) with a clear, well-evidenced "yes" —
against a validation container, not the app's live target. Nothing in the backlog should be
closed on this basis. The two issues it touches (#162, #167) move from "open, no evidence" to
"open, evidence exists showing the path forward is viable" — genuine progress, but not
completion. #163/#166/#161/#174/#179 are unaffected.

---

## Issue Update Payloads (proposed only — not applied)

### ISSUE #162

**Proposed title:** unchanged

**Proposed status:** unchanged (open)

**Proposed priority:** unchanged (P1)

**Proposed description:** append: "PR #185 (merged 2026-09-23) implemented option (b) from
this issue — added `scripts/check-vllm.sh` (startup health check), `docs/runbooks/vllm-dependency.md`,
and clearer inline pinning in `docker-compose.yml`. Option (a) (integrate `defnex-vllm` as a
first-class Compose service, or replace it) is still undone. Phase 2C
(`docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md`) additionally shows the compose `serving`
profile (vLLM v0.30.0) is a viable *replacement* target with working runtime LoRA, unlike the
current `defnex-vllm` (v0.28.0). Recommendation from that report: eventually decommission or
upgrade `defnex-vllm`. Not yet scheduled or decided."

**Proposed acceptance criteria:** add a second criterion alongside the original (a)/(b) choice:
"If (b) is accepted as sufficient for now, this issue should be retitled/rescoped to track the
follow-on decision: adopt compose `serving` as primary and retire `defnex-vllm`, vs. keep
`defnex-vllm` long-term and continue hardening around it."

**Proposed labels:** unchanged

**Proposed closure comment:** N/A — not closing

**Evidence reference:** `git log 172be53..d3d90c7`; `docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md`;
`ml-close-loop-be/docker-compose.yml`

---

### ISSUE #167

**Proposed title:** unchanged

**Proposed status:** unchanged (open, partially resolved)

**Proposed priority:** unchanged (P1)

**Proposed description:** append: "Phase 2C (`docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md`,
2026-09-22) validated the runtime-LoRA mechanism itself — load/infer/unload against the real
`run-9e66b6`/`smoke-llm-v3` artifact — but called vLLM's HTTP API directly, not this backend's
own `deploy-staging`/`validate-staging`/`promote-production`/`inference` endpoints. The live
rehearsal *through the application* is still not done. `docs/qa/terminal-qa-golden-path.md`
(added 2026-09-23) is the actual rehearsal script for this — its own Sections 12-15 explicitly
mark the eval→staging→promotion→new-adapter-inference chain as not yet exercised. Running that
script end-to-end (ideally against the compose serving path, now proven capable of runtime LoRA
by Phase 2C) is what would close this issue."

**Proposed acceptance criteria:** clarify existing wording to specify "through the
`ml-close-loop-be` API (not direct vLLM calls) — i.e., complete `docs/qa/terminal-qa-golden-path.md`
Sections 12-15 against a real deployment and record the result" for the rehearsal criterion.

**Proposed labels:** unchanged

**Proposed closure comment:** N/A — not closing

**Evidence reference:** `docs/PHASE2C_COMPOSE_LORA_INFERENCE_REPORT.md`; `docs/qa/terminal-qa-golden-path.md` §§12-15

---

No other issue (#161, #163, #166, #174, #179) requires a proposed edit.

---

## Final Quality Check

1. Latest repository state pulled safely (`git pull --ff-only`, fast-forward, no discarded work) — done.
2. No local changes were discarded — confirmed (`git status --short` before/after showed only the untracked `docs/qa/` directory this report itself lives in).
3. Phase 2C report exists in the current repository — confirmed, two files.
4. Phase 2C facts read directly from both reports in full, not from any external summary.
5. Phase 2A final evidence read directly (this pass reused verification from the prior session; re-confirmed via Phase 2C's own cross-references, e.g. GPU-controller restart note).
6. Phase 2B final evidence read directly (same basis).
7. Current GitHub issues read directly via `gh issue view`/`gh issue list` on 2026-09-23, not reused from memory.
8. Previous stale reconciliation's Phase 2C conclusion was explicitly superseded, not blindly reused — see header note.
9. #162 re-evaluated: PARTIALLY RESOLVED (new finding).
10. #167 re-evaluated: PARTIALLY RESOLVED, more evidence but not closed.
11. #163 re-evaluated: STILL OPEN, reconfirmed by Phase 2C's own session log.
12. #166 re-evaluated: STILL OPEN, unaffected.
13. #161/#174/#179 checked for indirect impact: none found.
14. No issue was mutated — only `gh issue view`/`gh issue list` (read-only) were run.
15. Current runtime (compose `serving`, validation-only) vs. target/actual-configured runtime
    (`defnex-vllm`, still the real `VLLM_URL`) explicitly separated throughout §6-§8.
16. Smoke-test scale (10 examples/5 steps) is stated, not inflated into production-readiness claims.
17. Runtime LoRA success on compose `serving` is explicitly not conflated with legacy
    `defnex-vllm` capability (§6, §7 make the distinction line-by-line).
18. Historical failure context preserved: the original 2026-09-16 "NOT VERIFIED" Phase 2C
    finding is kept and cited alongside the 2026-09-22 "E2E VERIFIED" follow-up, not erased.
