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
    # Issue #179: previously the only sampling knob was the server-side
    # settings.inference_max_tokens default - no per-request control existed at all.
    # Both optional/None-default so an existing caller that omits them is unaffected.
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Override the server default max_tokens for this request",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Sampling temperature for this request (backend-default if omitted)",
    )


class InferenceResponse(BaseModel):
    """A generation result. `model_id` / `version` identify the concrete version whose adapter
    actually served the request (issue #41) - not an alias, so callers know exactly which
    model produced the output."""

    model_id: str
    version: int
    generation: str
