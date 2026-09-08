# Ralph Agent Instructions — DEFNEX MLOps Backend

You are an autonomous coding agent working on the DEFNEX MLOps backend
(`ml-close-loop-be/`), a FastAPI service. This project runs inside a larger
monorepo (`Defnex-MLOps/`) — all your work happens under `ml-close-loop-be/`.

## Your Task

1. `cd ml-close-loop-be` (all commands below assume this as cwd).
2. Read the PRD at `scripts/ralph/prd.json`.
3. Read the progress log at `scripts/ralph/progress.txt` (check the Codebase
   Patterns section first).
4. Check you're on the branch from PRD `branchName`. If not, check it out or
   create it from `main`.
5. Pick the **highest priority** user story where `passes: false`.
6. Before implementing: read the source-of-truth doc(s) referenced in that
   story's acceptance criteria (under `../docs/`) — do not guess field names,
   types, or state machines.
7. Implement that single user story.
8. Run quality checks:
   ```bash
   source .venv/bin/activate  # create with: python3 -m venv .venv && pip install -e ".[dev]"
   alembic revision --autogenerate -m "<story id> <short message>"  # only if models changed
   alembic upgrade head
   pytest
   ```
9. Update `CLAUDE.md`/`AGENTS.md` files if you discover reusable patterns (see below).
10. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`.
11. Update `scripts/ralph/prd.json` to set `passes: true` for the completed story.
12. Append your progress to `scripts/ralph/progress.txt`.

## Project-Specific Rules (from docs/prd/mlops-backend-mvp.md §23)

- Do not invent numeric thresholds, evaluation metrics, or business rules not
  already present in the source docs.
- Do not change `docs/api/openapi.yaml` request/response schemas to make
  implementation easier — the frontend depends on this contract.
- If you find a genuine conflict between two source documents, **stop on that
  story**, leave `passes: false`, write the conflict into `progress.txt`
  under a `## Blocked` heading, and end your turn without guessing a
  resolution.
- No infrastructure not already named in `DEFNEX_MLOps_Multi_Server_Architecture_v2_PRD.md`
  (repo root). The PRD names PostgreSQL, Redis, Celery, MinIO, vLLM, Unsloth, Docker; it does
  NOT name Kubernetes, Kafka, Airflow, MLflow, or Vault — do not introduce those.
- Work on ONE story per iteration. Do not start a second story even if time
  remains.
- Keep changes scoped to `ml-close-loop-be/` unless a story explicitly says
  otherwise (e.g. it does not touch `docs/` or `ml-close-loop-fe/`).

## Progress Report Format

APPEND to `scripts/ralph/progress.txt` (never replace, always append):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---
```

## Consolidate Patterns

If you discover a reusable pattern, add it to a `## Codebase Patterns`
section at the TOP of `progress.txt` (create it if missing). Only add
patterns that are general and reusable, not story-specific details.

## Quality Requirements

- ALL commits must pass `pytest` (and `alembic upgrade head` if migrations
  changed).
- Do NOT commit broken code.
- Keep changes focused and minimal — no premature abstraction, no
  speculative flexibility.
- Follow existing code patterns already established in Phase 0
  (`app/config.py`, `app/db/session.py`, `app/api/health.py`).

## Stop Condition

After completing a user story, check if ALL stories in `prd.json` have
`passes: true`.

If ALL stories are complete and passing, reply with:
<promise>COMPLETE</promise>

If there are still stories with `passes: false` (including ones you left
blocked), end your response normally — another iteration will pick up the
next story.

## Important

- Work on ONE story per iteration.
- Commit frequently.
- Keep tests green.
- Read the Codebase Patterns section in `progress.txt` before starting.
