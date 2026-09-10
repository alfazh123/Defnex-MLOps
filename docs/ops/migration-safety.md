# Migration safety: expand/contract for production Postgres

## Why this exists

`CLAUDE.md` requires "Setiap perubahan skema wajib lewat migrasi baru" but gives no
guidance on making that migration *safe* to run against a live database. In this
codebase "live" specifically means: `app/workers/training_worker.py` runs an
infinite polling loop (`run_forever` at `app/workers/training_worker.py:206`, which
repeatedly calls `process_next_job` at `app/workers/training_worker.py:98`) that
does

```python
select(TrainingRun)
    .where(TrainingRun.status.in_(["PENDING", "STALE"]))
    .order_by(TrainingRun.created_at)
```

(`app/workers/training_worker.py:131-133`) against `training_runs`, independent of
and concurrently with the FastAPI process. A migration that locks that table for
more than an instant, or that a mid-rollout API pod can't read/write against, stalls
or crashes the worker loop mid-poll. Per PRD §23.1, PostgreSQL is production's
single source of truth and the worker keeps running against it while the app is
redeployed — there is no maintenance window built into this architecture, so
migrations must assume readers/writers are active throughout.

This document is process/checklist only. It adds no code and changes no schema.

## The expand/contract pattern

Never combine "add a new required thing" and "remove the old thing" in one
migration + one deploy. Split every breaking schema change into four steps, each
its own Alembic revision (`alembic revision --autogenerate -m "..."` per
`CLAUDE.md`'s Commands section) and, where code changes are involved, its own
deploy:

1. **Expand** — add the new column/table *nullable* (or with a server-side
   default), additive only. Old code and the worker keep working unmodified
   because nothing they already read or write changed shape.
2. **Backfill + dual-write** — deploy application code that writes the new
   column/table alongside the old one (or backfills it), while still reading the
   old one so in-flight rows aren't broken. Backfill existing rows in batches, not
   a single `UPDATE ... SET` over the whole table (that takes a table-wide lock
   under Postgres MVCC on a large `training_runs`/`model_versions` table).
3. **Cut over reads** — once backfill is confirmed complete, deploy code that
   reads/writes only the new column/table.
4. **Contract** — a separate migration that drops the old column/table or adds
   the `NOT NULL` constraint, only after step 3 has been running long enough that
   no code path still depends on the old shape.

Each step ships independently. If step 2's deploy has a bug, steps 3-4 simply
haven't happened yet and the old code path (which still fully works, since step 1
only added things) keeps serving traffic.

## Concrete example from this schema

Suppose a future change needs `training_runs.priority` (an integer used to order
the worker's poll query) instead of the current strict `created_at` FIFO ordering
at `app/workers/training_worker.py:133`. `TrainingRun` is defined at
`app/models/training.py:9-56`; `training_run_id` is a `String` primary key
(`app/models/training.py:14`) and `status`/`created_at` are exactly the two
columns the worker's `WHERE`/`ORDER BY` depend on
(`app/models/training.py:21,23`).

- **Expand** (migration A): `op.add_column("training_runs", sa.Column("priority",
  sa.Integer(), nullable=True))`. The worker's existing query at
  `app/workers/training_worker.py:131-133` is untouched — it doesn't select
  `priority` yet, so this migration is a pure additive `ALTER TABLE ... ADD
  COLUMN` (fast on Postgres for a nullable column with no default, no table
  rewrite).
- **Backfill + dual-write** (code deploy, no schema change): update
  `training_service` (per `CLAUDE.md`'s "all business rules... live in
  app/services") so run creation sets `priority` going forward, and run a
  batched backfill job that sets `priority` for existing rows in chunks (e.g. by
  `training_run_id` range), never a single unbounded `UPDATE`. The worker query
  still orders by `created_at` — nothing reads `priority` yet.
- **Cut over reads** (code deploy): change
  `app/workers/training_worker.py:131-133`'s `.order_by(TrainingRun.created_at)`
  to `.order_by(TrainingRun.priority, TrainingRun.created_at)`, deployed only
  after the backfill in the previous step is verified complete for every row the
  worker could still pick up (i.e. every non-terminal `PENDING`/`STALE` row).
- **Contract** (migration B, later, its own revision per the "never edit old
  migrations" rule in `CLAUDE.md`): `alembic.op.alter_column("training_runs",
  "priority", nullable=False)` once nothing depends on `priority` being
  possibly-null. Only issue this after step 3 has been live long enough to be
  confident every row has a `priority`.

The same shape applies to `model_versions` (`app/models/model.py:21-88`, e.g. its
`status` transition set in `model_service`) and `deployments`
(`app/models/deployment.py:9-34`, e.g. its `status`/`environment` columns) — any
column change there that the worker's registration path
(`model_service.register_model_version`, the worker's only caller per
`CLAUDE.md`'s "closed loop" section) or `deployment_service.deploy` reads must go
through the same four steps rather than a single migration that both adds and
enforces a constraint at once.

## What NOT to do in one migration

- Don't add a `NOT NULL` column without a default to a table the worker reads —
  existing INSERTs from in-flight code (old app version still deployed on another
  pod, or the worker itself constructing a row) will start failing the moment the
  migration lands, before the new code that populates the column is live.
- Don't rename a column the worker's `select()`/`.where()`/`.order_by()` touches
  in the same migration that changes the code — `sqlalchemy` maps columns by
  name; renaming and redeploying are two different moments in time and Postgres
  migrations run independently of Docker Compose's `backend`/`worker` container
  restart order (`docker compose up --build` starts both, not atomically).
- Don't drop a column/table in the same PR that stops using it. Ship the "stop
  using it" deploy first, verify, then drop it in a later migration.
- Don't run a full-table `UPDATE` or add a `NOT NULL`/type-changing
  `ALTER COLUMN` as part of the same migration that also creates the column —
  each of those can take a lock Postgres holds for the statement's duration,
  and if `training_runs`/`model_versions`/`deployments` grow large this stalls
  the worker's poll query for the same duration.

## PR review checklist for migrations touching `training_runs`, `model_versions`, or `deployments`

Use this whenever a migration under `ml-close-loop-be/alembic/versions/` touches
one of the three tables the worker reads or writes
(`app/workers/training_worker.py`, `app/services/training_service.py`,
`app/services/model_service.py`, `app/services/deployment_service.py`):

- [ ] Is this migration **additive only** (new nullable column, new table, new
      index created `CONCURRENTLY`-equivalent where the target DB supports it)?
      If it also drops/renames/tightens a constraint, split it into a separate
      later revision.
- [ ] If a column is added `NOT NULL`, is there a default, or is this actually
      step 4 (contract) of an already-completed expand/contract sequence?
- [ ] Does the corresponding service change (`app/services/*.py`) keep reading
      the pre-migration shape until the backfill is verified — i.e. no PR ships
      "add column" and "worker requires column" in the same commit?
- [ ] Is any backfill batched (bounded per statement), not a single unbounded
      `UPDATE`/scan over the whole table?
- [ ] Does the PR identify which step (expand / backfill+dual-write / cut-over /
      contract) it is, and confirm the prior step is already deployed and
      verified in the target environment?
- [ ] Was this migration generated with `alembic revision --autogenerate -m
      "..."` against a current `down_revision` (not hand-edited to insert
      itself earlier in the chain), and does `alembic upgrade head` succeed
      locally before the PR is opened?
- [ ] If multiple in-flight branches added migrations in parallel (this project
      routinely has several issue branches touching `alembic/versions/`
      concurrently), has `alembic heads` been checked for more than one head
      before merging, and is a merge revision (`alembic merge heads`) added if
      needed rather than silently picking one head?
- [ ] Does the migration avoid editing any previously merged file under
      `alembic/versions/` (per `CLAUDE.md`: "jangan edit migrasi lama yang sudah
      ada")?
