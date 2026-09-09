"""ComputeResource model (issue #75, PRD §9.3, §19.3).

An addressable unit used by a training provider — e.g. "server-2", "colab-training-a".
Adding a new resource is a data/config change, not a code change (PRD §19.4).
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ComputeResource(Base):
    """A specific server/account that a TrainingProvider executes on (PRD §9.3).

    Fields map to the admin infrastructure config form (PRD §33): name, role,
    environment, provider_type, host, GPU info, SSH connection details, and a
    credential reference (PRD §21 — `secret://...`, never plaintext).
    """

    __tablename__ = "compute_resources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    # "training", "inference", "both"
    role: Mapped[str] = mapped_column(String, default="training")
    # "staging", "production"
    environment: Mapped[str] = mapped_column(String, default="staging")
    # "local", "gpu_vps", "colab"
    provider_type: Mapped[str] = mapped_column(String, default="local")
    host: Mapped[str | None] = mapped_column(String, nullable=True)
    # GPU details as JSON, e.g. {"model": "H100", "memory_gb": 80}
    gpu_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ssh_host: Mapped[str | None] = mapped_column(String, nullable=True)
    ssh_port: Mapped[int | None] = mapped_column(nullable=True)
    ssh_username: Mapped[str | None] = mapped_column(String, nullable=True)
    # PRD §21/§43: credential reference, never plaintext secrets
    credential_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    is_healthy: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
