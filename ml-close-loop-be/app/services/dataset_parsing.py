"""Shared dataset-file parsing for the intake pipeline (issue #245, audit finding T6).

Before this module, ``app/api/intake.py`` (``_detect_format`` + ``_count_records``) and
``app/services/dataset_storage.py`` (``read_records``) each had their own copy of the
"bytes -> records" logic. Two copies meant two answers: a workbook could be counted one
way at ``inspect`` and read another way at ``validate`` (``docs/BACKLOG.md`` §5.3
"parser ganda"). It also meant every parser exception (``json.JSONDecodeError``,
``UnicodeDecodeError``, ``ValueError``, ``FileNotFoundError`` for a missing ``openpyxl``)
propagated out of the endpoint as an unhandled 500 instead of a coded client error.

This module is the single implementation. Every failure mode is a
:class:`DatasetParseError` carrying the wire ``code`` the API layer should surface, so
callers never have to guess which exception type to catch.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

# The formats this module can actually parse. NOTE: this is deliberately narrower than
# `app.schemas.dataset.SourceFormat`, which also permits the *normalization* targets
# "alpaca"/"sharegpt"/"chatml"/"other". Those are inputs a normalizer (issue #243 / A11)
# consumes; they are not file encodings, and silently handing one to a parser is what
# produced the 500s described above. `assert_supported_format` turns the difference into an
# explicit, coded rejection.
SUPPORTED_FORMATS: tuple[str, ...] = ("jsonl", "json", "csv", "xlsx")

# Extensions the intake endpoint accepts, mapped to the format they imply.
EXTENSION_FORMATS: dict[str, str] = {
    ".jsonl": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".xlsx": "xlsx",
}


class DatasetParseError(Exception):
    """A dataset file could not be turned into records.

    `code` is the wire error code for `openapi.yaml`'s Error envelope (e.g. the
    `ErrorResponse.error.code` returned by the intake endpoints) and `http_status` is the
    status the API layer should answer with. Carrying both here is what keeps the
    exception mapping in exactly one place instead of at every `except` site.
    """

    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(message)


def assert_supported_format(source_format: str) -> None:
    """Reject a format this module cannot parse, with a code the client can act on."""

    if source_format not in SUPPORTED_FORMATS:
        raise DatasetParseError(
            "UNSUPPORTED_FORMAT",
            f"Cannot parse source_format {source_format!r}. "
            f"Readable formats: {', '.join(SUPPORTED_FORMATS)}.",
        )


def format_from_filename(filename: str) -> str | None:
    """The format implied by a file's extension, or None if the extension is not one we read.

    Extension-only, so it is safe to call on a path that may not exist (the validate step
    cross-checks the declared `source_format` against the staged file's name before it ever
    reads the bytes). For `.json` this only reports the *claimed* format -- use `detect_format`
    when the content is available and the claim needs confirming.
    """

    return EXTENSION_FORMATS.get(Path(filename).suffix.lower())


def detect_format(filename: str, content: bytes) -> str:
    """Detect the file format from its extension, confirming `.json` really holds an array.

    Returns "unknown" for anything the intake endpoint should reject. `.json` is sniffed
    rather than trusted because a `.json` file holding a single object (not an array) is
    not a record file and `parse_bytes` rejects it.
    """

    claimed = format_from_filename(filename)
    if claimed == "json":
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "unknown"
        return "json" if isinstance(data, list) else "unknown"
    return claimed or "unknown"


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetParseError(
            "UNPARSEABLE_FILE",
            f"File is not valid UTF-8 text (byte offset {exc.start}): {exc.reason}. "
            "Re-save the file as UTF-8 and upload it again.",
        ) from exc


def _parse_jsonl(content: bytes) -> list:
    text = _decode_text(content)
    records: list = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise DatasetParseError(
                "UNPARSEABLE_FILE",
                f"Line {lineno} is not valid JSON: {exc.msg} (column {exc.colno}). "
                "A .jsonl file must hold exactly one JSON value per non-blank line.",
            ) from exc
    return records


def _parse_json(content: bytes) -> list:
    text = _decode_text(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetParseError(
            "UNPARSEABLE_FILE",
            f"File is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).",
        ) from exc
    if not isinstance(data, list):
        raise DatasetParseError(
            "UNPARSEABLE_FILE",
            f"A .json dataset must be a top-level array of records, got "
            f"{type(data).__name__}.",
        )
    return list(data)


def _parse_csv(content: bytes) -> list[dict]:
    text = _decode_text(content)
    return list(csv.DictReader(io.StringIO(text)))


def _parse_xlsx(source: str | io.BytesIO) -> list[dict]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise DatasetParseError(
            "MISSING_DEP",
            "Reading .xlsx uploads requires the optional 'intake' extra (openpyxl). "
            "Install it, or upload the sheet as CSV.",
        ) from exc

    try:
        workbook = openpyxl.load_workbook(source, read_only=True)
    except Exception as exc:
        raise DatasetParseError(
            "UNPARSEABLE_FILE", f"Could not read the .xlsx workbook: {exc}"
        ) from exc
    try:
        sheet = workbook.active
        rows = list(sheet.iter_rows(values_only=True))
    finally:
        workbook.close()

    if not rows:
        return []
    headers = [str(header) for header in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:]]


def parse_bytes(content: bytes, source_format: str) -> list:
    """Parse an in-memory dataset file into a list of records.

    A record is whatever the file held, not necessarily a dict: a `.jsonl` line of
    `[1, 2]` parses fine but is not a usable record, and the H0 shape rules in
    `validation_service` report it per record (issue #245) rather than this function
    either crashing or silently dropping it.
    """

    assert_supported_format(source_format)
    if source_format == "jsonl":
        return _parse_jsonl(content)
    if source_format == "json":
        return _parse_json(content)
    if source_format == "csv":
        return _parse_csv(content)
    if source_format == "xlsx":
        return _parse_xlsx(io.BytesIO(content))
    # Unreachable: `assert_supported_format` above already raised for anything else.
    raise DatasetParseError(
        "UNSUPPORTED_FORMAT", f"Unsupported format: {source_format}"
    )


def parse_file(path: Path, source_format: str) -> list:
    """Parse a staged dataset file on disk into a list of records."""

    assert_supported_format(source_format)
    if source_format == "xlsx":
        # openpyxl streams from the file handle; reading a 100 MB workbook into memory
        # just to hand BytesIO back to it would be pure waste.
        return _parse_xlsx(str(path))
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DatasetParseError(
            "UNPARSEABLE_FILE", f"Could not read the staged file {path.name}: {exc}"
        ) from exc
    return parse_bytes(content, source_format)
