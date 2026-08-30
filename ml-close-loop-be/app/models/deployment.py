from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Deployment(Base):
    """The deployment pointer state (PRD §15), kept conceptually separate from the
    model registry (`Model`/`ModelVersion`) - no foreign key to either, only the
    plain `model_id`/`model_version` values, per this story's own acceptance
    criteria and PRD §12's "Model Registry dan Deployment harus tetap menjadi dua
    konsep berbeda."
    """

    __tablename__ = "deployments"

    deployment_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String)
    model_version: Mapped[int] = mapped_column()
    environment: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    deployed_at: Mapped[datetime] = mapped_column()
