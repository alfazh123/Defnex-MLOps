"""Centralized role -> permission policy (issue #123).

Replaces the hardcoded `role != "admin"` binary check that used to live in
`require_admin` (app/api/deps.py) for the endpoints where "any admin can do
anything" is no longer accurate. Deliberately NOT a policy engine
(Casbin/OPA): out of proportion for a single-service FastAPI monolith and
against the project's no-Vault/Kafka/K8s constraint (CLAUDE.md). Just a
declarative dict a router dependency can look up.

`User.role` (app/models/user.py) stays a free-text string column - no schema
change needed, the five values below are simply the ones this module knows
about. `require_admin` is intentionally left in app/api/deps.py: endpoints
outside the critical set identified by issue #123 (dataset intake, eval
sets, user management, feedback moderation, artifact transfers) are out of
scope and keep the binary admin check unchanged.
"""

from __future__ import annotations


class Role:
    """The five roles the backend recognizes. Values match `User.role` strings."""

    ADMIN = "admin"
    ML_ENGINEER = "ml_engineer"
    DATA_ENGINEER = "data_engineer"
    REVIEWER = "reviewer"
    USER = "user"

    ALL = frozenset({ADMIN, ML_ENGINEER, DATA_ENGINEER, REVIEWER, USER})


# Permission strings for the critical endpoints named in issue #123 (promote,
# deploy, rollback, infra credential write). Only permissions actually
# enforced somewhere in app/api are defined here - no speculative ones.
PROMOTE = "model:promote"
VALIDATE_STAGING = "model:validate"
DEPLOY = "deployment:deploy"
ROLLBACK = "deployment:rollback"
INFRA_CREDENTIAL_WRITE = "infra:credential:write"

ROLE_PERMISSIONS: dict[str, set[str]] = {
    Role.ADMIN: {PROMOTE, VALIDATE_STAGING, DEPLOY, ROLLBACK, INFRA_CREDENTIAL_WRITE},
    Role.ML_ENGINEER: {PROMOTE, DEPLOY, ROLLBACK},
    Role.DATA_ENGINEER: set(),
    Role.REVIEWER: {VALIDATE_STAGING},
    Role.USER: set(),
}


def has_permission(role: str, permission: str) -> bool:
    """True if `role` (a `User.role` string) holds `permission`. Unknown roles hold
    nothing - fail closed rather than raising, since role is untrusted string data."""
    return permission in ROLE_PERMISSIONS.get(role, set())
