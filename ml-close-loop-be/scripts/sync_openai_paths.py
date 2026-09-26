"""Regenerate the two OpenAI paths + schemas in openapi.yaml for issue #226.

The static spec is the frozen contract (CLAUDE.md: "change the implementation to match the
spec, not the reverse"), so the edit has to land in the file by hand -- but hand-writing a
FastAPI-generated path block is exactly how drift starts. This script renders the blocks from
the live app's own `app.openapi()` and splices them in, so the shape is generated rather than
transcribed.

Run:  python scripts/sync_openai_paths.py     (from ml-close-loop-be/)
"""

from __future__ import annotations

import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402

SPEC = pathlib.Path(__file__).resolve().parent.parent / "openapi.yaml"
NEW_PATHS = (
    "/v1/chat/completions",
    "/v1/models",
    # Added in the same batch; kept here so one script owns every path this branch
    # introduced, and a future change to any of them cannot be forgotten.
    "/api/v1/datasets/intake/import-hf",
    "/api/v1/models/{model_id}/versions/{version}/lineage",
)
OLD_PATH = "  /api/v1/models/{model_id}/inference:"
OLD_SCHEMAS = ("InferenceRequest", "InferenceResponse")
NEW_SCHEMAS = (
    "ChatCompletionRequest",
    "ChatMessage",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "Usage",
    "ModelCard",
    "ModelList",
    "HfImportRequest",
    "HfImportResponse",
)


def dump(obj) -> str:
    return yaml.safe_dump(
        obj, sort_keys=False, default_flow_style=False, width=88
    ).rstrip("\n")


def main() -> None:
    generated = app.openapi()
    text = SPEC.read_text()
    lines = text.split("\n")

    # 1. Drop the old inference path (if still present), then splice each generated path in
    #    after the last existing path block.
    #
    #    Not alphabetical insertion: the static spec is not sorted (`intake/inspect` follows
    #    `intake/validate`), and `/v1/...` sorts after every `/api/v1/...` path, so an
    #    "insert at first greater" search finds nothing and silently appends to the end of
    #    the file -- inside `components`, where the path is then invisible to any reader.
    #    Anchoring on the `components:` key is unambiguous.
    #
    #    Idempotent on purpose: re-running must refresh the generated blocks, not require a
    #    pristine spec.
    if OLD_PATH in lines:
        del lines[slice(*_path_block_bounds(lines, OLD_PATH))]
    for path in NEW_PATHS:
        # Replace an existing block for this path rather than adding a second one.
        if f"  {path}:" in lines:
            del lines[slice(*_path_block_bounds(lines, f"  {path}:"))]
    components_at = lines.index("components:")
    for path in NEW_PATHS:
        block = f"  {path}:\n" + _indent(dump(generated["paths"][path]), 4)
        lines[components_at:components_at] = block.split("\n") + [""]
        components_at += len(block.split("\n")) + 1
    text = "\n".join(lines)

    # 2. Swap the schemas: drop the old pair, insert the new ones alphabetically-ish next to
    #    where they were, so the diff stays reviewable.
    spec = yaml.safe_load(text)
    components = spec["components"]["schemas"]
    for name in OLD_SCHEMAS:
        components.pop(name, None)
    for name in NEW_SCHEMAS:
        if name in generated["components"]["schemas"]:
            components[name] = generated["components"]["schemas"][name]
    spec["components"]["schemas"] = dict(sorted(components.items()))

    SPEC.write_text(
        yaml.safe_dump(spec, sort_keys=False, default_flow_style=False, width=88)
    )
    print(
        f"synced {len(NEW_PATHS)} paths and {len(NEW_SCHEMAS)} schemas into {SPEC.name}"
    )


def _path_block_bounds(lines: list[str], header: str) -> tuple[int, int]:
    """`[start, end)` of the `header:` path block, up to the next path or `components:`.

    The `components:` fallback matters: the LAST path in the file has no successor, and a
    plain "next line starting with `  /`" search raises StopIteration there.
    """

    start = lines.index(header)
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if (lines[i].startswith("  /") and lines[i].rstrip().endswith(":"))
            or i == (lines.index("components:"))
        ),
        len(lines),
    )
    return start, end


def _indent(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in block.split("\n"))


if __name__ == "__main__":
    main()
