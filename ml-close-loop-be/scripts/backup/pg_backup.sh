#!/usr/bin/env bash
# Scheduled PostgreSQL logical backup with rolling retention (issue #133, PRD §44
# open decision "exact PostgreSQL deployment location and backup policy").
#
# Policy (see docs/governance/backup-dr-policy.md): daily `pg_dump`, retained as
# 7 daily + 4 weekly copies, written to a location OTHER than the database host
# (BACKUP_DIR below is expected to be a mounted network volume / secondary disk,
# not local ephemeral storage on the DB machine).
#
# This script does NOT run against any real database on its own — it only runs
# when invoked (by cron or manually) with PG* connection variables pointing at a
# real instance. It is safe to `bash -n` / read without touching any database.
#
# Usage:
#   PGHOST=... PGPORT=5432 PGUSER=... PGPASSWORD=... PGDATABASE=... \
#     BACKUP_DIR=/mnt/backups/postgres ./pg_backup.sh
#
# Cron example (02:00 daily, on a host separate from the DB server):
#   0 2 * * * BACKUP_DIR=/mnt/backups/postgres PGHOST=db.internal PGUSER=defnex \
#     PGDATABASE=defnex /opt/defnex/ml-close-loop-be/scripts/backup/pg_backup.sh \
#     >> /var/log/defnex/pg_backup.log 2>&1
#
# Requires: the `postgresql-client` package (`pg_dump`) on the machine running
# this script. Does not require the app's Python venv.

set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./data/backups/postgres}"
PGDATABASE="${PGDATABASE:?PGDATABASE is required (database name to back up)}"
# Retention: keep the last 7 daily dumps + the last 4 Sunday (weekly) dumps.
DAILY_RETENTION_DAYS="${DAILY_RETENTION_DAYS:-7}"
WEEKLY_RETENTION_WEEKS="${WEEKLY_RETENTION_WEEKS:-4}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
day_of_week="$(date -u +%u)"  # 1=Monday .. 7=Sunday
dump_file="${BACKUP_DIR}/daily/${PGDATABASE}-${timestamp}.dump"

mkdir -p "${BACKUP_DIR}/daily" "${BACKUP_DIR}/weekly"

echo "[pg_backup] dumping database '${PGDATABASE}' to ${dump_file}"
# -F c: custom format (compressed, supports parallel/selective restore via pg_restore).
pg_dump -F c -f "${dump_file}" "${PGDATABASE}"

# Weekly copy: on Sundays, promote the daily dump into the weekly retention set too.
if [ "${day_of_week}" = "7" ]; then
    weekly_file="${BACKUP_DIR}/weekly/${PGDATABASE}-${timestamp}.dump"
    cp "${dump_file}" "${weekly_file}"
    echo "[pg_backup] copied weekly snapshot to ${weekly_file}"
fi

echo "[pg_backup] pruning daily dumps older than ${DAILY_RETENTION_DAYS} days"
find "${BACKUP_DIR}/daily" -name "${PGDATABASE}-*.dump" -mtime "+${DAILY_RETENTION_DAYS}" -print -delete

# 4 weekly copies ~= 28 days; prune anything older than that window.
weekly_retention_days=$((WEEKLY_RETENTION_WEEKS * 7))
echo "[pg_backup] pruning weekly dumps older than ${weekly_retention_days} days (${WEEKLY_RETENTION_WEEKS} weeks)"
find "${BACKUP_DIR}/weekly" -name "${PGDATABASE}-*.dump" -mtime "+${weekly_retention_days}" -print -delete

echo "[pg_backup] done"
