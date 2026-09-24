# Phase 2C Application-Layer HTTP Response Fix Report

**Issue:** #167
**Date:** 2026-09-23
**Status:** Application-layer blocker fixed; live deployment ladder NOT yet verified

---

## 1. Problem

During the Phase 2C live rehearsal, `POST /api/v1/models/smoke-llm-v3/versions/1/deploy-staging`
returned HTTP 409 `STAGING_DEPLOY_NOT_ALLOWED` despite the underlying vLLM adapter load succeeding.

## 2. Root Cause

`request_sync_with_retry()` in `app/services/http_retry.py:132` unconditionally called
`resp.json()` on all HTTP 2xx responses. vLLM 0.30.0's `POST /v1/load_lora_adapter` returns:

```
HTTP 200
Content-Type: text/plain
Body: Success: LoRA adapter 'smoke-llm-v3-v1' added successfully.
```

The JSON decode failure raised `json.JSONDecodeError` (a `ValueError`), which was **not** caught by
the retry loop's `except httpx.HTTPStatusError` or `except httpx.RequestError` clauses. The
exception propagated through `VLLMServingBackend.deploy()` → `_deploy_locked()` →
`deployment_service.deploy()` → `stage_deploy()` → `deploy_to_staging()`, and was surfaced by the
API layer as `STAGING_DEPLOY_NOT_ALLOWED`.

The adapter was actually loaded successfully on vLLM; the bug was purely in the response parsing.

## 3. Code Change

### `app/services/http_retry.py`

Added `parse_json: bool = True` parameter to both `request_with_retry()` (async) and
`request_sync_with_retry()` (sync).

- `parse_json=True` (default): preserves existing behavior — calls `resp.json()` and returns
  `dict[str, Any]`. All existing callers are unaffected.
- `parse_json=False`: returns the raw `httpx.Response` object without attempting JSON parsing.

Return type widened from `dict[str, Any]` to `dict[str, Any] | httpx.Response`.

### `app/services/serving.py`

- `VLLMServingBackend.deploy()` (line 230): added `parse_json=False` to the `request_sync_with_retry`
  call for `POST /v1/load_lora_adapter`.
- `VLLMServingBackend.unload()` (line 258): added `parse_json=False` to the `request_sync_with_retry`
  call for `POST /v1/unload_lora_adapter`.
- `VLLMServingBackend.generate()`: **unchanged** — `/v1/completions` returns JSON and continues
  to use the default `parse_json=True`.

## 4. Test Coverage

### `tests/test_retry.py` (7 new tests)

| Test | Validates |
|------|-----------|
| `test_sync_request_parse_json_true_returns_dict` | Default behavior preserved |
| `test_sync_request_parse_json_false_returns_response_object` | Returns raw Response, no `.json()` call |
| `test_sync_request_parse_json_false_plain_text_no_error` | **Phase 2C regression**: plain-text 200 doesn't raise |
| `test_sync_request_parse_json_true_still_raises_on_non_json` | parse_json=True preserves JSONDecodeError |
| `test_async_request_parse_json_true_returns_dict` | Async default behavior preserved |
| `test_async_request_parse_json_false_returns_response_object` | Async returns raw Response |
| `test_async_request_parse_json_false_plain_text_no_error` | Async plain-text no error |

### `tests/test_vllm_serving.py` (4 new tests)

| Test | Validates |
|------|-----------|
| `test_load_plain_text_response_succeeds` | Deploy succeeds with plain-text vLLM response |
| `test_unload_plain_text_response_succeeds` | Unload succeeds with plain-text vLLM response |
| `test_unload_404_json_still_handled` | Existing 404 JSON handling preserved |
| `test_load_plain_text_not_converted_to_serving_error` | Plain-text 200 is NOT a serving error |

## 5. Validation Results

### Tests

| Suite | Before | After |
|-------|--------|-------|
| `test_retry.py` | 7 passed | 14 passed (+7 new) |
| `test_vllm_serving.py` | 8 passed, 8 failed | 12 passed, 8 failed (+4 new pass, 8 pre-existing fail) |

All 8 failures in `test_vllm_serving.py` are **pre-existing** `BaseModelMismatchError` from
`SERVED_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` in `.env` — unrelated to this fix.

### Lint

```
ruff check .                    → All checks passed!
ruff format --check .           → 231 files already formatted
```

## 6. Remaining Blocker for Issue #167

This fix resolves the **application-layer parsing bug**. It does **not** prove the full live
promotion ladder. The following steps still require a separate live rehearsal:

1. **deploy-staging** — load adapter into vLLM via `/v1/load_lora_adapter` (plain-text response)
2. **validate-staging** — human approval of staged candidate
3. **promote-production** — deploy to production via `/v1/load_lora_adapter`
4. **Backend inference** — generate text via `/v1/completions` (JSON response, unchanged)
5. **New adapter identity verification** — confirm adapter name matches `{model_id}-v{version}`

### Distinction

- **Raw vLLM inference verification**: direct HTTP call to vLLM `/v1/completions`
- **Application-layer inference verification**: call through the application's inference endpoint

Both require a live GPU with vLLM 0.30.0 running.

## 7. Files Changed

| File | Change |
|------|--------|
| `app/services/http_retry.py` | Added `parse_json` parameter to both sync and async functions |
| `app/services/serving.py` | Added `parse_json=False` to deploy() and unload() calls |
| `tests/test_retry.py` | Added 7 regression tests for parse_json parameter |
| `tests/test_vllm_serving.py` | Added 4 regression tests for plain-text vLLM responses |
| `docs/qa/PHASE2C_HTTP_RESPONSE_FIX_REPORT.md` | This report |
