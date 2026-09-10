#!/usr/bin/env bash
# Scheduled sync of the MinIO artifact bucket to a second storage target (issue #133,
# PRD §44 open decision "MinIO single-node vs redundant deployment").
#
# Policy (see docs/governance/backup-dr-policy.md): primary stays single-node MinIO
# (per the project's "no Kubernetes" constraint), but every object in the artifact
# bucket is mirrored to a second bucket/endpoint on a schedule. Artifacts are
# immutable once written (PRD §41 P2/P3, CLAUDE.md "Artifact model/version bersifat
# immutabel") and used for rollback (PRD §43) — losing MinIO without a copy means
# losing the ability to roll back at all, not just losing history.
#
# Uses the MinIO Client (`mc`), the standard tool for MinIO-to-MinIO/S3 mirroring —
# not a new app dependency, just an ops tool on the machine running this script.
# Install: https://min.io/docs/minio/linux/reference/minio-mc.html
#
# This script does NOT run against any real MinIO instance on its own — it only
# copies data when invoked with SOURCE_ALIAS/TARGET_ALIAS already configured via
# `mc alias set`. It is safe to `bash -n` / read without touching any storage.
#
# One-time alias setup (not part of this script, done once per host):
#   mc alias set defnex-primary   http://<primary-minio-host>:9000   "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
#   mc alias set defnex-secondary http://<secondary-host-or-s3>:9000 "$BACKUP_ACCESS_KEY" "$BACKUP_SECRET_KEY"
#
# Usage:
#   SOURCE_ALIAS=defnex-primary SOURCE_BUCKET=artifacts \
#   TARGET_ALIAS=defnex-secondary TARGET_BUCKET=artifacts-backup \
#     ./minio_backup_sync.sh
#
# Cron example (hourly — artifacts are write-once, so frequent sync is cheap and
# keeps RPO low; see docs/governance/backup-dr-policy.md):
#   0 * * * * SOURCE_ALIAS=defnex-primary SOURCE_BUCKET=artifacts \
#     TARGET_ALIAS=defnex-secondary TARGET_BUCKET=artifacts-backup \
#     /opt/defnex/ml-close-loop-be/scripts/backup/minio_backup_sync.sh \
#     >> /var/log/defnex/minio_backup.log 2>&1

set -euo pipefail

SOURCE_ALIAS="${SOURCE_ALIAS:?SOURCE_ALIAS is required (mc alias for the primary MinIO)}"
SOURCE_BUCKET="${SOURCE_BUCKET:-${MINIO_BUCKET:-artifacts}}"
TARGET_ALIAS="${TARGET_ALIAS:?TARGET_ALIAS is required (mc alias for the secondary storage)}"
TARGET_BUCKET="${TARGET_BUCKET:-${SOURCE_BUCKET}-backup}"

echo "[minio_backup_sync] mirroring ${SOURCE_ALIAS}/${SOURCE_BUCKET} -> ${TARGET_ALIAS}/${TARGET_BUCKET}"

# --overwrite: re-upload if a target object's checksum differs from source.
# Artifacts are write-once (immutable), so in steady state this only copies new
# objects; --overwrite exists to self-heal a previously interrupted/partial sync.
# No --remove: a delete on the primary must never propagate to the backup copy —
# the backup is the last line of defense against exactly that kind of accident.
mc mirror --overwrite "${SOURCE_ALIAS}/${SOURCE_BUCKET}" "${TARGET_ALIAS}/${TARGET_BUCKET}"

echo "[minio_backup_sync] done"
