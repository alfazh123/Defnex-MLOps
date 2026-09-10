# Backup & disaster-recovery policy: PostgreSQL and MinIO

Resolves the two open decisions in `DEFNEX_MLOps_Multi_Server_Architecture_v2_PRD.md`
§44 ("Open Decisions to Resolve During Implementation"):

> 1. Exact PostgreSQL deployment location and backup policy.
> 2. MinIO single-node vs redundant deployment.

## Scope and constraints this policy operates under

- Single shared H100 GPU, small team, no Kubernetes (`CLAUDE.md` "Constraints proyek"
  and PRD §4 Non-Goals) — this rules out managed multi-region replicas, streaming
  WAL standbys, or a MinIO distributed/erasure-coded cluster as MVP requirements.
  Those remain valid future upgrades, not blockers to shipping a policy now.
- PostgreSQL is the authoritative production datastore (PRD §23.1, `CLAUDE.md`
  "DB: SQLAlchemy 2.0 ORM + Alembic... PostgreSQL menjadi basis otoritatif"); the
  current prototype still runs on SQLite for local dev (`ml-close-loop-be/.env.example`
  `DATABASE_URL=sqlite:///./data/app.db`), so this policy targets the Postgres
  deployment that PRD v2 migrates to, not the current SQLite dev default.
  SQLite dev databases are not covered by this policy — they are disposable and
  reseedable via `ml-close-loop-be/seed.py --reset`.
- Model/adapter artifacts in MinIO are immutable once written and are the only way
  to roll back a deployment (PRD §41 P2/P3, PRD §43, `CLAUDE.md` "Artifact
  model/version bersifat immutabel... Artifact yang dipakai untuk rollback jangan
  dihapus"). Losing the MinIO bucket without a second copy means losing rollback
  capability entirely, not just losing history — this is why MinIO backup, not just
  PostgreSQL backup, is in scope.

## Decision

### PostgreSQL

- **Deployment location**: single-node PostgreSQL instance (no replica/standby at
  MVP scale), consistent with the "no Kubernetes" / small-team constraint. This can
  be upgraded to a streaming-replication standby later without changing this backup
  policy.
- **Backup method**: logical backup via `pg_dump` (custom format, `-F c`), run on a
  schedule via cron (matches the project's existing broker-free operational style —
  no Celery/Airflow dependency is introduced for this). Implemented at
  `ml-close-loop-be/scripts/backup/pg_backup.sh`.
- **Schedule**: daily.
- **Retention**: rolling 7 daily dumps + 4 weekly dumps (script prunes anything
  older automatically).
- **Storage location**: dumps are written to `BACKUP_DIR`, which MUST be a
  filesystem separate from the database host (network volume, secondary disk, or
  synced off-host) — a backup that lives on the same disk as the database it backs
  up does not survive the failure it exists for.
- **RPO (Recovery Point Objective): up to 24 hours.** Any writes committed after the
  most recent nightly dump are lost in a full-loss disaster scenario. This is
  accepted at MVP scale; WAL archiving/point-in-time recovery is an explicit
  non-goal for now (the issue's own recommendation: "tidak perlu WAL
  streaming/replica dulu"), and is a natural upgrade path once RPO needs tighten.
- **RTO (Recovery Time Objective): ~1-2 hours**, dominated by (a) provisioning a
  fresh PostgreSQL instance/container and (b) `pg_restore` time, which scales with
  database size. This is a rough MVP-scale estimate (small dataset — dataset
  metadata, training run records, model registry rows — not raw model bytes, which
  live in MinIO), not a measured SLA.

### MinIO

- **Deployment topology**: single-node MinIO (already the case per
  `ml-close-loop-be/docker-compose.yml` — the `minio` service has no cluster/erasure
  configuration), consistent with the "no Kubernetes" constraint.
- **Redundancy strategy**: not via MinIO server-side clustering, but via scheduled
  **application-level mirroring** of the artifact bucket to a second bucket/endpoint
  (a second MinIO instance, or any S3-compatible target). Implemented at
  `ml-close-loop-be/scripts/backup/minio_backup_sync.sh` using `mc mirror`.
- **Schedule**: hourly. Artifacts are immutable/write-once (PRD §41 P2/P3), so
  frequent mirroring is cheap (new objects only, no churn from in-place edits) and
  meaningfully lowers RPO compared to the once-daily PostgreSQL dump.
- **Deletion safety**: the sync is one-directional and never deletes from the
  target (`mc mirror --overwrite`, no `--remove`) — an accidental delete on the
  primary must never propagate to the backup copy, since the backup exists
  specifically to survive primary-side accidents.
- **RPO: up to 1 hour** for artifact bytes (bounded by the hourly mirror interval).
- **RTO: ~15-60 minutes** to point `MINIO_ENDPOINT`/`ARTIFACT_BACKEND` (see
  `ml-close-loop-be/.env.example`) at the secondary bucket, or to re-populate a
  freshly provisioned primary from the secondary via a reverse `mc mirror` — see
  the restore runbook for the exact steps.

## What this policy deliberately does not cover (future work, not blockers)

- Point-in-time recovery (WAL archiving) for PostgreSQL.
- Automated failover / standby promotion.
- MinIO server-side erasure coding or multi-node clustering.
- Encryption-at-rest for backup files (should be added before backups are ever
  written to a location outside the team's own infrastructure).

These are consciously deferred, matching PRD §44's framing that these are
"implementation-level decisions" to revisit "only when implementation requires
them" — not omissions.

## Related documents

- Restore procedure: `docs/governance/restore-runbook.md`.
- Backup scripts: `ml-close-loop-be/scripts/backup/pg_backup.sh`,
  `ml-close-loop-be/scripts/backup/minio_backup_sync.sh`.
