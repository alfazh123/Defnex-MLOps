# Restore runbook: PostgreSQL and MinIO

Companion to `docs/governance/backup-dr-policy.md`. This is a step-by-step
procedure, not a script — a full-loss recovery needs a human to verify each step
(wrong target host, wrong dump file, etc. are exactly the mistakes a blind script
would make worse). None of the commands below have been run against a real
database or bucket as part of writing this document (LOCAL DEVELOPMENT ONLY /
no production infrastructure was touched); verify each command against your own
target host names, credentials, and current dump filenames before running it.

## Before you start

- Confirm you are restoring onto a *new or intentionally emptied* target — never
  restore over a database/bucket that might still hold newer, un-backed-up data
  without first snapshotting it aside.
- Confirm which incident you're recovering from: total loss of the primary host
  (restore fresh), or a bad migration/bad write (restore to a scratch instance
  first, diff, then decide what to bring back) — the steps differ below.

## Restoring PostgreSQL from a `pg_dump`

1. **Identify the dump to restore.** Dumps live under `BACKUP_DIR` from
   `ml-close-loop-be/scripts/backup/pg_backup.sh`, named
   `<PGDATABASE>-<UTC timestamp>.dump` in `daily/` or `weekly/`. Pick the most
   recent one that predates the incident (for a bad-migration scenario, this may
   mean deliberately picking the dump from *before* the bad migration ran, not the
   latest one).
   ```bash
   ls -la "$BACKUP_DIR"/daily "$BACKUP_DIR"/weekly
   ```
2. **Provision the target PostgreSQL instance** (fresh container/VM/managed
   instance) if the primary host itself was lost. Confirm you can connect:
   ```bash
   psql -h "$PGHOST" -U "$PGUSER" -d postgres -c '\conninfo'
   ```
3. **Create an empty target database** (skip if restoring into a database that
   already exists and is confirmed empty/scratch):
   ```bash
   createdb -h "$PGHOST" -U "$PGUSER" "$PGDATABASE"
   ```
4. **Restore the dump** (custom format, matches `pg_dump -F c` used by the backup
   script):
   ```bash
   pg_restore -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" --no-owner --clean \
     "$BACKUP_DIR/daily/<chosen-file>.dump"
   ```
   `--clean` drops conflicting objects first if the target isn't fully empty;
   omit it for a truly empty database. `--no-owner` avoids role-mismatch errors
   when restoring onto a different host/user than the one that produced the dump.
5. **Point the application at the restored database.** Update `DATABASE_URL` in
   the running environment's config (see `ml-close-loop-be/.env.example`) and
   restart the `backend` and `worker` processes. Do **not** start the `worker`
   before verifying step 6 — a worker polling a partially-restored database can
   pick up and act on inconsistent `training_runs` rows
   (`app/workers/training_worker.py:131-133` polls `status IN (PENDING, STALE)`
   immediately on the next iteration).
6. **Verify before resuming traffic**:
   - Row counts on the tables the closed loop depends on (`training_runs`,
     `model_versions`, `deployments` — see `app/models/training.py`,
     `app/models/model.py`, `app/models/deployment.py`) look sane, not empty,
     not obviously truncated mid-table.
   - `alembic current` (from `ml-close-loop-be/`) matches the schema version the
     application code expects; run `alembic upgrade head` if the dump predates a
     migration that has since shipped (this restore path can put you behind
     `head`, unlike a live database).
   - No `training_runs` row is `RUNNING` with a stale `heartbeat_at` — those are
     from before the incident and should already be reclaimed as `STALE` by
     `training_service.mark_stale_runs` on the worker's next poll, but confirm
     rather than assume mid-incident.
7. **Resume the worker and backend**, then monitor the first few poll cycles.

## Restoring MinIO artifacts from the secondary bucket

Two scenarios: (a) the primary MinIO is gone and needs to be rebuilt from the
secondary copy, or (b) you only need specific objects back (e.g. one deleted
artifact) without a full rebuild.

### (a) Full rebuild of a lost primary

1. **Provision a new primary MinIO instance** (or reuse the existing
   `docker-compose.yml` `minio` service definition against a fresh volume).
2. **Register both endpoints with `mc`** if not already configured:
   ```bash
   mc alias set defnex-primary   http://<new-primary-host>:9000   "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
   mc alias set defnex-secondary http://<secondary-host>:9000     "$BACKUP_ACCESS_KEY" "$BACKUP_SECRET_KEY"
   ```
3. **Create the target bucket** on the new primary if it doesn't exist:
   ```bash
   mc mb defnex-primary/artifacts
   ```
4. **Mirror the secondary back onto the primary** (reverse direction from the
   normal backup sync in `ml-close-loop-be/scripts/backup/minio_backup_sync.sh`):
   ```bash
   mc mirror --overwrite defnex-secondary/artifacts-backup defnex-primary/artifacts
   ```
5. **Verify object counts/checksums** match between the two buckets before
   pointing the application back at the primary:
   ```bash
   mc du defnex-primary/artifacts
   mc du defnex-secondary/artifacts-backup
   ```
6. **Verify at least one restored artifact's checksum** against its recorded
   value: the app's own checksum algorithm is
   `compute_checksum_from_bytes`/`_compute_checksum` in
   `ml-close-loop-be/app/services/artifact_storage.py`, and
   `MinioArtifactStorage.verify_checksum` (same file) is the code path that
   would reject a corrupted artifact at deploy time — spot-check with it (or its
   equivalent) rather than assuming the mirror was byte-perfect.
7. **Point `MINIO_ENDPOINT`/`MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY`** (see
   `ml-close-loop-be/.env.example`) at the rebuilt primary and restart `backend`
   and `worker`.

### (b) Recovering a single object without a full rebuild

```bash
mc cp defnex-secondary/artifacts-backup/<key> defnex-primary/artifacts/<key>
```
Then verify that object's checksum the same way as step 6 above before treating
it as usable for a rollback.

## Post-restore checklist

- [ ] `alembic current` matches expectations; `alembic upgrade head` run if needed.
- [ ] `training_runs`/`model_versions`/`deployments` row counts sanity-checked.
- [ ] No orphaned `RUNNING` training run left inconsistent by the restore.
- [ ] MinIO object count/checksums spot-checked against the pre-incident state.
- [ ] `worker` process (training_worker) only restarted after the checks above.
- [ ] Incident + RPO actually incurred (data lost between last backup and
      incident time) recorded somewhere durable for post-mortem, since this
      policy's RPO is a target, not a guarantee.
