"""OpenAI Chat Completions schemas (issue #226/#227/#228, openapi.yaml OpenAI-compatible surface).

This is the standard, not a DEFNEX-shaped one. The point of moving to the OpenAI wire
format is that an off-the-shelf client — the official SDK, LangChain, OpenWebUI, a plain
`curl` — can point at this backend unchanged, which is what "sesuaikan API dengan standard
OpenAI API untuk komunikasi dengan model, test inference, dll" asks for. A bespoke envelope
would have satisfied the words and defeated the purpose.

Consequences that are deliberate, not accidents:

- The model reference lives in the body field `model`, not a path parameter, because that
  is where a client puts it. This is why the old `POST /api/v1/models/{model_id}/inference`
  is gone rather than kept alongside (issue #226 is explicitly breaking).
- `developer` is accepted alongside `system` for the system role: the OpenAI spec's current
  examples use `developer`, `system` remains supported, and a client using either must work.
- `usage` is always present on a non-streaming response. The spec's `*_details` sub-objects
  are optional and this layer has no source for them, so they are omitted rather than faked.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

# OpenAI's current role vocabulary. `system` predates `developer` and is still accepted;
# rejecting it would break older clients for no benefit.
ChatRole = Literal["system", "developer", "user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    # Accepted and echoed back per the spec's message shape, but this backend has no tool
    # registry and never emits them (see openapi.yaml: tools/tool_calls are out of scope).
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Body for POST /v1/chat/completions.

    `model` addresses a version two ways, matching how an OpenAI client already thinks
    about model names: `"<model_id>"` serves that model's currently DEPLOYED version, and
    `"<model_id>:<version>"` pins one explicitly.
    """

    model: str = Field(
        description=(
            'Model reference. "<model_id>" serves the DEPLOYED version; '
            '"<model_id>:<version>" pins an explicit version.'
        ),
        examples=["defnex-domain-sft", "defnex-domain-sft:3"],
    )
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    # Issue #230: sampling parameters. Declared here and propagated to the serving backend;
    # `n > 1` and multiple `stop` values still yield a single choice from this backend (see
    # the endpoint docstring) rather than silently pretending to honour them.
    top_p: float | None = Field(default=None, gt=0, le=1)
    n: int | None = Field(default=None, ge=1, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    # Streaming is declared for contract completeness but not implemented; the endpoint
    # rejects `stream: true` with a clear error instead of returning a non-streaming body
    # to a client that asked for SSE (issue #231 owns the implementation).
    stream: bool = False
    user: str | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str = Field(
        description="The concrete model reference that served this request"
    )
    choices: list[ChatCompletionChoice]
    usage: Usage

    @staticmethod
    def new_id() -> str:
        """OpenAI-shaped identifier: `chatcmpl-` plus random hex.

        Prefixed rather than a bare UUID so it is recognisable in a log as a chat completion
        and cannot be confused with the `cmpl-` shape older completions responses use.
        """

        return f"chatcmpl-{uuid.uuid4().hex}"


class ModelCard(BaseModel):
    """One entry of GET /v1/models (issue #229)."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = Field(default="defnex")


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


def now_epoch() -> int:
    return int(time.time())
