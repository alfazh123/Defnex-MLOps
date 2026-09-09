from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Environment(Base):
    """A deployment target environment (PRD §16.1).

    The migration that creates this table seeds the `default` (backward-compat), `staging` and
    `production` rows. `Deployment.environment` records which environment a deploy targeted;
    keep this standalone relative to the register/deploy split so an existing deployment row is
    never invalidated by an environment ladder change (PRD §12 keeps registry and deployment
    separate concepts).
    """

    __tablename__ = "environments"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
