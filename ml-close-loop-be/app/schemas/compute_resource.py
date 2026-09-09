"""Schemas for compute resource API (issue #78, PRD §8.6, §33)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ComputeResourceCreateRequest(BaseModel):
    """Request to register a new compute resource (PRD §33)."""

    name: str
    role: Literal["training", "inference", "both"] = "training"
    environment: Literal["staging", "production"] = "staging"
    provider_type: Literal["local", "gpu_vps", "colab"] = "local"
    host: str | None = None
    gpu_info: dict | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_username: str | None = None
    credential_ref: str | None = None


class ComputeResourceUpdateRequest(BaseModel):
    """Partial update for a compute resource (PATCH). All fields optional."""

    name: str | None = None
    role: Literal["training", "inference", "both"] | None = None
    environment: Literal["staging", "production"] | None = None
    provider_type: Literal["local", "gpu_vps", "colab"] | None = None
    host: str | None = None
    gpu_info: dict | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_username: str | None = None
    credential_ref: str | None = None
    is_healthy: bool | None = None


class ComputeResource(BaseModel):
    """A compute resource record (PRD §19.3)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    role: str
    environment: str
    provider_type: str
    host: str | None = None
    gpu_info: dict | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_username: str | None = None
    credential_ref: str | None = None
    is_healthy: bool = True
    created_at: datetime
    updated_at: datetime


class HealthStatus(BaseModel):
    """Read-only health/connectivity status for a compute resource (PRD §8.6)."""

    resource_id: int
    name: str
    is_healthy: bool
    provider_type: str
    message: str
