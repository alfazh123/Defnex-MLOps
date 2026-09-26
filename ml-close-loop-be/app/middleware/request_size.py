from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Slack added on top of the per-file cap for a multipart body: the part headers, the
# boundary delimiters and the form fields all count toward Content-Length but are not part
# of the uploaded file. Without it, a file exactly at the limit would be rejected by the
# body guard even though it is within the documented file limit.
MULTIPART_ENVELOPE_OVERHEAD_BYTES = 1024 * 1024


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies over `max_body_size` with a 413 + `REQUEST_TOO_LARGE`.

    `path_overrides` maps a path *prefix* to a different body limit for routes that
    legitimately carry more than a JSON body — currently only the dataset intake upload
    routes, whose 100 MB dataset file was previously unreachable because this middleware
    capped every route at 1 MB (issue #242, audit finding T3). A prefix-scoped override is
    used rather than a single raised global limit so a large upload allowance does not
    loosen the rest of the API.

    Known limitation: the body is buffered into memory here so it can be replayed to the
    downstream handler, which then reads it a second time (double buffering). A 100 MB
    upload therefore costs ~200 MB of RSS. Streaming the upload straight to
    `DatasetStorage` instead of buffering is the real fix and is deliberately left out of
    this issue's scope — see the note in `docs/BACKLOG.md` §5.3.
    """

    def __init__(
        self, app, max_body_size: int, path_overrides: dict[str, int] | None = None
    ):
        super().__init__(app)
        self.max_body_size = max_body_size
        self.path_overrides = path_overrides or {}

    def _limit_for(self, path: str) -> int:
        for prefix, limit in self.path_overrides.items():
            if path.startswith(prefix):
                return limit
        return self.max_body_size

    async def dispatch(self, request: Request, call_next):
        limit = self._limit_for(request.url.path)

        def _too_large() -> JSONResponse:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": f"Request body too large (max {limit} bytes)",
                    }
                },
            )

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    return _too_large()
            except ValueError:
                pass

        # Chunks are collected and joined once at the end. `body += chunk` on a 100 MB
        # upload is O(n^2) in the number of chunks (bytes are copied on every append),
        # which is measurably slow at the new intake limit.
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                return _too_large()
            chunks.append(chunk)

        request._body = b"".join(chunks)
        return await call_next(request)
