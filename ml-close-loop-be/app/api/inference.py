"""OpenAI-compatible inference surface (issues #226, #227, #228, #229).

`POST /v1/chat/completions` and `GET /v1/models` follow the OpenAI wire format so an
off-the-shelf client works against this backend unchanged. The previous
`POST /api/v1/models/{model_id}/inference` is removed: the model reference has to live in
the body field `model` for a standard client to send it at all, so keeping the old shape
would mean two ways to ask the same question and one of them wrong (issue #226 — this is
the explicitly breaking change).

Authentication is still this service's JWT, not an OpenAI `sk-...` API key. The meeting note
did not ask for API-key auth, and changing the security model is a separate decision
(openapi.yaml records this as out of scope).
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.errors import APIError
from app.db.session import get_db
from app.models.model import Model, ModelVersion
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelList,
    Usage,
    now_epoch,
)
from app.services import deployment_service, inference_service
from app.services.serving import InferenceError, get_serving_backend

router = APIRouter(tags=["OpenAI Compatibility"])


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    # `name` is an optional part of the OpenAI message shape; emitting it as `null` is
    # technically valid but is not what the spec's own examples look like, and a strict
    # client may reject an unexpected null. Drop unset optionals instead.
    response_model_exclude_none=True,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
def create_chat_completion(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ChatCompletionResponse:
    """Generate a chat completion (issue #226).

    Model addressing: `"<model_id>"` serves that model's DEPLOYED version, and
    `"<model_id>:<version>"` pins one explicitly. Unknown model or version is a 404; a
    version that exists but is not DEPLOYED is a 409; an upstream serving failure is a 502
    — never a 500 traceback (issue #41).
    """
    if request.stream:
        # Issue #231 has not landed. Saying so beats answering a `stream: true` request with
        # a non-streaming body, which an SDK would try to parse as SSE and fail on.
        raise APIError(
            501,
            "STREAMING_NOT_SUPPORTED",
            "stream=true is not implemented yet; retry with stream=false "
            "(streaming is issue #231).",
        )

    model_id, _ = inference_service.parse_model_ref(request.model)
    if db.get(Model, model_id) is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" not found')

    try:
        model_version = inference_service.resolve_model_ref(db, request.model)
    except ValueError as exc:
        _model_id, version = inference_service.parse_model_ref(request.model)
        if version is not None:
            # Disambiguate by whether the version exists at all, rather than by parsing the
            # exception message -- the same approach the old router took.
            existing: ModelVersion | None = db.scalar(
                select(ModelVersion).where(
                    ModelVersion.model_id == model_id,
                    ModelVersion.version == version,
                )
            )
            if existing is None:
                raise APIError(404, "MODEL_NOT_FOUND", str(exc)) from exc
            raise APIError(409, "INFERENCE_NOT_ALLOWED", str(exc)) from exc
        raise APIError(404, "DEPLOYMENT_NOT_FOUND", str(exc)) from exc

    system_prompt, prompt = inference_service.flatten_messages(request.messages)
    if not prompt.strip():
        raise APIError(
            400,
            "EMPTY_MESSAGES",
            "messages[] contains no usable content: every message was empty.",
        )

    try:
        generation = inference_service.generate(
            get_serving_backend(),
            model_version,
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except InferenceError as exc:
        raise APIError(502, "INFERENCE_FAILED", str(exc)) from exc

    return ChatCompletionResponse(
        id=ChatCompletionResponse.new_id(),
        created=now_epoch(),
        # Echo the concrete reference, not the alias the client sent: "which version
        # answered this" is the first thing anyone debugging a completion needs.
        model=f"{model_version.model_id}:{model_version.version}",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generation.text),
                finish_reason="stop",
            )
        ],
        usage=Usage(**generation.to_usage()),
    )


@router.get("/v1/models", response_model=ModelList)
def list_models(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ModelList:
    """List the models a client can address (issue #229).

    One entry per *served* version, not per registered model: a client pointing at
    `/v1/models` is deciding what to send in the `model` field, and listing versions that
    are not DEPLOYED would invite requests that are guaranteed to 409. The `id` is the
    explicit `model_id:version` form, which is directly usable.
    """

    versions = db.scalars(
        select(ModelVersion)
        .where(ModelVersion.status == "DEPLOYED")
        .order_by(ModelVersion.model_id, ModelVersion.version)
    ).all()
    return ModelList(
        data=[
            ModelCard(
                id=f"{v.model_id}:{v.version}",
                created=int(v.created_at.timestamp()),
                owned_by="defnex",
            )
            for v in versions
        ]
    )


# Re-exported so the alias vocabulary stays discoverable from this module, which is where a
# reader of the OpenAI surface would look for it.
SUPPORTED_ALIASES = deployment_service.SUPPORTED_ALIASES
