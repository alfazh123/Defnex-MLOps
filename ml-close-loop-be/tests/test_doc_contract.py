"""Docs <-> code contract checks (#37).

- Every route registered on the FastAPI app must have a matching path key in
  openapi.yaml, and vice versa (fails if a path exists on only one side).
- Every field on `app.config.Settings` must have an entry in `.env.example`
  (fails when a new setting is added without updating the example file).
- The endpoint count claimed by both READMEs (23) must match openapi.yaml.
"""

import re
from pathlib import Path

from app.config import Settings
from app.main import app

REPO_BE = Path(__file__).resolve().parents[1]
OPENAPI_YAML = REPO_BE / "openapi.yaml"
ENV_EXAMPLE = REPO_BE / ".env.example"


def _static_openapi_paths() -> set[str]:
    """Path keys under `paths:` in openapi.yaml.

    Path keys are indented by two spaces and start with `/` (e.g.
    `/api/v1/datasets/{dataset_id}/versions`); no other key in the file does.
    """
    text = OPENAPI_YAML.read_text()
    return set(re.findall(r"^\s{2}(/[^\s:]+):\s*$", text, re.M))


def _env_example_keys() -> set[str]:
    keys = set()
    for line in ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys


def test_fastapi_routes_and_openapi_paths_match_both_directions():
    app_paths = set(app.openapi()["paths"])
    spec_paths = _static_openapi_paths()

    assert app_paths, "no routes registered on the app"
    assert spec_paths, "no paths parsed from openapi.yaml"
    assert app_paths == spec_paths


def test_documented_endpoint_count_matches_openapi():
    actual = len(_static_openapi_paths())
    claims = {
        REPO_BE.parent / "README.md": r"(\d+) routes",
        REPO_BE / "README.md": r"(\d+) REST endpoints",
    }
    for path, pattern in claims.items():
        match = re.search(pattern, path.read_text())
        assert match, f"endpoint-count claim not found in {path}"
        assert int(match.group(1)) == actual, (
            f"{path.name} claims {match.group(1)} paths but openapi.yaml has {actual}"
        )


def test_every_settings_field_has_env_example_entry():
    env_keys = _env_example_keys()
    assert env_keys, "no KEY=VALUE entries parsed from .env.example"
    missing = {field.upper() for field in Settings.model_fields} - env_keys
    assert not missing, f"Settings fields missing from .env.example: {sorted(missing)}"
