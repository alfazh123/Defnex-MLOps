from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DeployResult(BaseModel):
    """Response for POST .../deploy (openapi.yaml DeployResult)."""

    model_id: str
    current_deployed_version: int
    previous_deployed_version: int | None = None


class DeploymentStatus(BaseModel):
    """The model-level deployment pointer (openapi.yaml DeploymentStatus), a separate resource
    from any single version's registry record. `status` has no full enum yet - only "DEPLOYED"
    is confirmed by any source document - and is null when nothing has ever been deployed."""

    model_id: str
    current_deployed_version: int | None = None
    deployed_at: datetime | None = None
    status: Literal["DEPLOYED"] | None = None
