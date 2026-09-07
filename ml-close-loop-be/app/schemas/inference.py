"""Inference request/response schemas (issue #41, openapi.yaml Inference*)."""

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Body for POST /api/v1/models/{model_id}/inference.

    `target` is either a deployment alias (``prod``, resolved via the single
    `deployment_service.resolve_alias` function) or an explicit version number (the model's
    DEPLOYED version). `prompt` is the text to generate from.
    """

    target: str = Field(
        description="Deployment alias (e.g. 'prod') or an explicit deployed version number"
    )
    prompt: str = Field(min_length=1, description="Prompt to generate from")


class InferenceResponse(BaseModel):
    """A generation result. `model_id` / `version` identify the concrete version whose adapter
    actually served the request (issue #41) - not an alias, so callers know exactly which
    model produced the output."""

    model_id: str
    version: int
    generation: str
