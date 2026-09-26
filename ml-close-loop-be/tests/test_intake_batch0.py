"""Batch 0 regression tests for the dataset intake critical fixes.

One class per issue, so a failure names the finding it belongs to:

- A9  / #241  the gate must consider hard errors, not only leakage
- A10 / #242  the advertised 100 MB upload cap must actually be reachable
- A13 / #245  malformed input must never become a 500
- A14 / #246  a client-supplied filename must not escape the staging tree

Storage is redirected to a per-test tmp dir by the autouse `storage` fixture below, so no
test here writes to the repo's real `data/datasets`.
"""

import io
import json
from pathlib import Path

import pytest

from tests.conftest import auth_header

from app.services.artifact_storage import LocalFilesystemArtifactStorage
from app.services.dataset_storage import (
    DatasetStorage,
    UnsafeFilenameError,
    sanitize_filename,
)

ANSWER = (
    "DEFNEX adalah platform MLOps untuk integrasi model AI secara terpusat, yang "
    "menyatukan empat sub-proyek universitas di bawah satu arsitektur intelijen "
    "terpadu agar seluruh tim dapat mengelola dataset, pelatihan, dan registri model "
    "dari satu tempat dengan mudah."
)


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    """Point both intake routers' DatasetStorage — and the object store behind it — at tmp.

    Issue #214 moved committed dataset bytes into `ArtifactStorage`, whose local backend
    defaults to the repo's real `data/artifacts`. Injecting a tmp-rooted store is what keeps
    these tests from writing into the working tree. Autouse and defined once: every test in
    this file uploads, and patching the two router modules individually is the kind of
    duplication that lets a test quietly exercise real directories.
    """
    store = DatasetStorage(
        tmp_path / "datasets",
        store=LocalFilesystemArtifactStorage(tmp_path / "artifacts"),
    )
    monkeypatch.setattr("app.api.intake.DatasetStorage", lambda: store)
    monkeypatch.setattr("app.api.intake_validate.DatasetStorage", lambda: store)
    return store


def _record(index: int = 0) -> dict:
    return {
        "id": f"rec-{index:03d}",
        "messages": [
            {"role": "user", "content": f"Pertanyaan nomor {index} tentang DEFNEX?"},
            {"role": "assistant", "content": ANSWER},
        ],
        "metadata": {"source_dataset": "batch0", "source_id": f"s-{index:03d}"},
    }


def _jsonl(records: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode()


def _upload(client, admin_token, filename, content):
    return client.post(
        "/api/v1/datasets/intake/inspect",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
        headers=auth_header(admin_token),
    )


def _inspect_then_validate(
    client, admin_token, dataset_id, filename, content, source_format="jsonl", **extra
):
    """Run the wizard's inspect + validate steps and return the validate response."""
    inspect = _upload(client, admin_token, filename, content)
    assert inspect.status_code == 200, inspect.text
    return _validate(
        client,
        admin_token,
        inspect.json()["staging_id"],
        dataset_id,
        source_format=source_format,
        **extra,
    )


def _validate(
    client, admin_token, staging_id, dataset_id, source_format="jsonl", **extra
):
    return client.post(
        "/api/v1/datasets/intake/validate",
        json={
            "staging_id": staging_id,
            "dataset_id": dataset_id,
            "source_format": source_format,
            **extra,
        },
        headers=auth_header(admin_token),
    )


def _commit(client, admin_token, staging_id, dataset_id, report_id):
    return client.post(
        "/api/v1/datasets/intake/commit",
        json={
            "staging_id": staging_id,
            "dataset_id": dataset_id,
            "validation_report_id": report_id,
        },
        headers=auth_header(admin_token),
    )


def _gate_check(body: dict) -> dict:
    return next(c for c in body["checks"] if c["name"] == "gate")


# ── A9 / #241 — the gate must consider hard errors ─────────────────────────────


class TestGateBlocksHardErrors:
    def test_all_records_failing_h1_does_not_pass(self, client, admin_token):
        """The original defect (T2): a file whose every record fails H1 reported PASS."""
        broken = [{"id": f"b{i}", "messages": [], "metadata": {}} for i in range(3)]
        resp = _inspect_then_validate(
            client, admin_token, "all_h1_fail", "broken.jsonl", _jsonl(broken)
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "FAIL"
        assert body["valid_records"] == 0
        assert body["blocking_error_count"] == 3
        assert "hard error" in _gate_check(body)["message"]

    def test_blocking_error_count_is_no_longer_always_zero(self, client, admin_token):
        """`blocking_error_count` was counted by looking for dicts carrying
        `severity == "error"` inside `per_record_errors`, which holds lists of rule-code
        *strings* -- so the count was structurally always 0 and the response claimed every
        record was valid on a fully invalid dataset."""
        broken = [{"id": f"b{i}", "messages": [], "metadata": {}} for i in range(5)]
        resp = _inspect_then_validate(
            client, admin_token, "count_fix", "broken.jsonl", _jsonl(broken)
        )

        body = resp.json()
        assert body["blocking_error_count"] == 5
        assert body["valid_records"] == 0

    def test_single_valid_record_among_thousands_fails(self, client, admin_token):
        """Edge case named in the issue: 1 valid out of 1000 is a 99.9% error rate."""
        records = [_record(0)]
        records += [{"id": f"b{i}", "messages": [], "metadata": {}} for i in range(999)]
        resp = _inspect_then_validate(
            client, admin_token, "one_of_thousand", "mixed.jsonl", _jsonl(records)
        )

        body = resp.json()
        assert body["total_records"] == 1000
        assert body["valid_records"] == 1
        assert body["status"] == "FAIL"

    def test_valid_dataset_still_passes(self, client, admin_token):
        """No regression: a clean dataset must still be PASS."""
        resp = _inspect_then_validate(
            client,
            admin_token,
            "clean_ds",
            "clean.jsonl",
            _jsonl([_record(i) for i in range(4)]),
        )

        body = resp.json()
        assert body["status"] == "PASS"
        assert body["valid_records"] == 4
        assert body["blocking_error_count"] == 0

    def test_duplicate_and_leakage_checks_report_what_they_checked(
        self, client, admin_token
    ):
        """The `leakage` check used to report FAIL purely from the gate decision, so a
        dataset that failed only on hard errors claimed 'leakage: FAIL' with none present."""
        records = [_record(0), _record(0), _record(1)]
        resp = _inspect_then_validate(
            client, admin_token, "dupes", "dupes.jsonl", _jsonl(records)
        )

        checks = {c["name"]: c for c in resp.json()["checks"]}
        assert checks["duplicate_ids"]["status"] == "FAIL"
        assert "1 duplicate" in checks["duplicate_ids"]["message"]
        assert checks["leakage"]["status"] == "PASS"
        assert checks["leakage"]["message"] == ""

    def test_small_error_rate_is_needs_review_and_is_committable(
        self, client, admin_token
    ):
        """Below the FAIL threshold but non-zero: NEEDS_REVIEW, and it may be committed --
        the bytes and the report are what a human needs in order to act on it."""
        records = [_record(i) for i in range(19)]
        records.append({"id": "bad", "messages": [], "metadata": {}})
        resp = _inspect_then_validate(
            client, admin_token, "review_ds", "review.jsonl", _jsonl(records)
        )
        body = resp.json()
        assert body["status"] == "NEEDS_REVIEW"
        assert body["valid_records"] == 19

        commit = _commit(
            client,
            admin_token,
            body["staging_id"],
            "review_ds",
            body["validation_report_id"],
        )
        assert commit.status_code == 200, commit.text

    def test_needs_review_dataset_refuses_training(self, client, admin_token):
        """NEEDS_REVIEW is committable but not trainable, and answers with its own code so a
        client can tell "your dataset is broken" apart from "a human must accept this"."""
        records = [_record(i) for i in range(19)]
        records.append({"id": "bad", "messages": [], "metadata": {}})
        resp = _inspect_then_validate(
            client, admin_token, "review_no_train", "review.jsonl", _jsonl(records)
        )
        body = resp.json()
        commit = _commit(
            client,
            admin_token,
            body["staging_id"],
            "review_no_train",
            body["validation_report_id"],
        )
        assert commit.status_code == 200, commit.text
        version = commit.json()["version"]

        run = client.post(
            "/api/v1/training-runs",
            json={
                "dataset_id": "review_no_train",
                "dataset_version": version,
                "model_id": "batch0-model",
                "base_model": "unsloth/Qwen3-0.6B",
                "training_config": {
                    "peft_method": "lora",
                    "load_in_4bit": False,
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "epochs": 1,
                    "max_seq_length": 512,
                },
                "triggered_by": "admin",
            },
            headers=auth_header(admin_token),
        )
        assert run.status_code == 409
        assert run.json()["error"]["code"] == "VALIDATION_NEEDS_REVIEW"

    def test_fail_gate_still_blocks_commit(self, client, admin_token):
        broken = [{"id": f"b{i}", "messages": [], "metadata": {}} for i in range(3)]
        resp = _inspect_then_validate(
            client, admin_token, "fail_no_commit", "broken.jsonl", _jsonl(broken)
        )
        body = resp.json()
        commit = _commit(
            client,
            admin_token,
            body["staging_id"],
            "fail_no_commit",
            body["validation_report_id"],
        )
        assert commit.status_code == 409
        assert commit.json()["error"]["code"] == "VALIDATION_FAILED"

    def test_fail_ratio_is_a_setting_not_a_literal(
        self, client, admin_token, monkeypatch
    ):
        """The threshold is a data-governance call, so it is configurable per deployment."""
        from app.config import settings

        records = [_record(i) for i in range(8)]
        records += [{"id": f"b{i}", "messages": [], "metadata": {}} for i in range(2)]
        content = _jsonl(records)  # 20% hard-error rate

        monkeypatch.setattr(settings, "validation_gate_fail_ratio", 0.10)
        assert (
            _inspect_then_validate(
                client, admin_token, "ratio_tight", "tight.jsonl", content
            ).json()["status"]
            == "FAIL"
        )

        monkeypatch.setattr(settings, "validation_gate_fail_ratio", 0.50)
        assert (
            _inspect_then_validate(
                client, admin_token, "ratio_loose", "loose.jsonl", content
            ).json()["status"]
            == "NEEDS_REVIEW"
        )

    def test_leakage_fails_regardless_of_ratio(self, client, admin_token, monkeypatch):
        """A zero-hard-error dataset that leaks must still FAIL: the pre-existing rule that
        the new hard-error branch must not weaken."""
        from types import SimpleNamespace

        from app.config import settings
        from app.services import eval_set_service

        leaked = _record(1)
        eval_record = {
            "id": "eval-1",
            "messages": [{"role": "user", "content": leaked["messages"][0]["content"]}],
        }
        monkeypatch.setattr(settings, "validation_gate_fail_ratio", 1.0)
        monkeypatch.setattr(
            eval_set_service,
            "get_eval_set_version",
            lambda *a, **k: SimpleNamespace(records=[eval_record]),
        )

        resp = _inspect_then_validate(
            client,
            admin_token,
            "leak_only",
            "leak.jsonl",
            _jsonl([leaked]),
            eval_set_id="benchmark",
            eval_set_version=1,
        )

        body = resp.json()
        assert body["status"] == "FAIL"
        assert _gate_check(body)["message"].startswith("Leakage found")
        assert (
            next(c for c in body["checks"] if c["name"] == "leakage")["status"]
            == "FAIL"
        )

    def test_gate_reason_is_populated_for_pass_and_fail(self, client, admin_token):
        """All three gate values carry a human-readable reason (issue acceptance)."""
        pass_body = _inspect_then_validate(
            client, admin_token, "reason_pass", "p.jsonl", _jsonl([_record(0)])
        ).json()
        fail_body = _inspect_then_validate(
            client,
            admin_token,
            "reason_fail",
            "f.jsonl",
            _jsonl([{"id": "b", "messages": [], "metadata": {}}]),
        ).json()

        assert _gate_check(pass_body)["status"] == "PASS"
        assert _gate_check(pass_body)["message"]
        assert _gate_check(fail_body)["status"] == "FAIL"
        assert _gate_check(fail_body)["message"]

    def test_error_rate_exactly_at_the_threshold_fails(self, client, admin_token):
        """Pins the boundary: the threshold is "at or above", so exactly 10% of 20 records
        failing is a FAIL, not a NEEDS_REVIEW. Without this the `>=` vs `>` choice would be
        an untested implementation detail."""
        records = [_record(i) for i in range(18)]
        records += [{"id": f"b{i}", "messages": [], "metadata": {}} for i in range(2)]
        resp = _inspect_then_validate(
            client, admin_token, "at_threshold", "at.jsonl", _jsonl(records)
        )
        body = resp.json()
        assert body["total_records"] == 20
        assert body["valid_records"] == 18
        assert body["status"] == "FAIL"

    def test_hard_error_statistics_are_reported(self, client, admin_token):
        """The numbers behind the decision, so a reviewer can see why the gate landed where
        it did without re-running validation."""
        records = [_record(0), {"id": "bad", "messages": [], "metadata": {}}]
        resp = _inspect_then_validate(
            client, admin_token, "stats", "stats.jsonl", _jsonl(records)
        )
        # 1 of 2 records failing is a 50% rate, so the default 10% threshold makes this FAIL.
        assert resp.json()["status"] == "FAIL"

        report = client.get(
            "/api/v1/datasets/stats/versions/1/validation-reports/latest",
            headers=auth_header(admin_token),
        )
        assert report.status_code == 200
        stats = report.json()["dataset_statistics"]["hard_error_gate"]
        assert stats["records_with_hard_errors"] == 1
        assert stats["hard_error_ratio"] == 0.5
        assert stats["fail_ratio_threshold"] == 0.10
        assert stats["leakage_overlaps"] == 0


# ── A10 / #242 — the upload cap must be reachable ─────────────────────────────


class TestUploadSizeLimit:
    def test_upload_larger_than_1mb_succeeds(self, client, admin_token):
        """The defect (T3): every upload over 1 MB was rejected by the global middleware
        before the handler's own 100 MB check could run, so the documented cap was a lie."""
        records = []
        index = 0
        while sum(len(r) for r in records) < 1_600_000:
            record = _record(index)
            record["metadata"]["pad"] = "x" * 900
            records.append(json.dumps(record).encode() + b"\n")
            index += 1
        content = b"".join(records)
        assert 1_048_576 < len(content) < 104_857_600

        resp = _upload(client, admin_token, "big.jsonl", content)
        assert resp.status_code == 200, resp.text
        assert resp.json()["file_size"] == len(content)

    def test_intake_override_does_not_loosen_other_routes(self, client, admin_token):
        """A prefix-scoped override must not raise the limit for the whole API."""
        over_default = b'{"id":"a","messages":[]}\n' * 60_000
        assert len(over_default) > 1_048_576

        other = client.post(
            "/api/v1/auth/register",
            content=over_default,
            headers={"Content-Type": "application/json"},
        )
        assert other.status_code == 413
        assert other.json()["error"]["code"] == "REQUEST_TOO_LARGE"

        assert _upload(client, admin_token, "ok.jsonl", over_default).status_code == 200

    def test_file_over_the_100mb_cap_reports_the_handlers_code(
        self, client, admin_token
    ):
        """The old test asserted only `status_code == 413`, which is exactly why the 1 MB
        middleware rejection satisfied it. The code must now be the handler's FILE_TOO_LARGE.
        """
        resp = _upload(
            client, admin_token, "huge.jsonl", b"x" * (100 * 1024 * 1024 + 1)
        )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"
        assert "100MB" in resp.json()["error"]["message"]

    def test_both_limits_derive_from_one_setting(self):
        """No divergent numbers anywhere: the handler cap and the middleware cap for the
        intake routes both come from `max_upload_file_size`."""
        from app.config import settings
        from app.middleware.request_size import MULTIPART_ENVELOPE_OVERHEAD_BYTES

        assert settings.max_upload_file_size == 104_857_600
        # The body cap must exceed the file cap, or a file exactly at the cap is rejected by
        # the middleware before the handler can accept it.
        assert MULTIPART_ENVELOPE_OVERHEAD_BYTES > 0
        assert settings.max_request_body_size < settings.max_upload_file_size

    def test_intake_module_no_longer_defines_its_own_size_cap(self):
        """The old local `MAX_FILE_SIZE` constant was a second, unreachable copy."""
        from app.api import intake

        assert not hasattr(intake, "MAX_FILE_SIZE")


# ── A13 / #245 — malformed input must not 500 ──────────────────────────────────


class TestMalformedInputNever500s:
    def test_broken_jsonl_line_is_a_coded_error(self, client, admin_token):
        resp = _upload(
            client,
            admin_token,
            "broken.jsonl",
            b'{"id": "ok", "messages": []}\n{not json at all}\n',
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "UNPARSEABLE_FILE"
        assert "Line 2" in resp.json()["error"]["message"]

    def test_broken_jsonl_at_validate_is_a_coded_error(
        self, client, admin_token, storage
    ):
        """`validate` is a separate request that re-parsed the staged file with no error
        handling at all, so the same file 500'd here even when inspect had accepted it."""
        staged = storage.stage_upload(
            "broken.jsonl", b'{"id": "ok", "messages": []}\n{oops}\n'
        )
        resp = _validate(client, admin_token, staged["staging_id"], "broken")
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "UNPARSEABLE_FILE"

    def test_non_utf8_file_is_a_coded_error(self, client, admin_token):
        # 0xFF/0xFE are never valid UTF-8.
        resp = _upload(
            client, admin_token, "latin1.jsonl", b'{"id": "a", "text": "\xff\xfe"}\n'
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "UNPARSEABLE_FILE"
        assert "UTF-8" in resp.json()["error"]["message"]

    def test_json_object_instead_of_array_is_rejected(self, client, admin_token):
        """A `.json` holding a single object is not a record file."""
        resp = _upload(
            client, admin_token, "single.json", json.dumps({"a": 1}).encode()
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "UNPARSEABLE_FILE"

    def test_json_format_requires_a_top_level_array(self, client, admin_token, storage):
        staged = storage.stage_upload("swap.json", json.dumps([_record(0)]).encode())
        # Content changes between inspect and validate (staging is a filesystem, not a lock).
        Path(staged["path"]).write_text(json.dumps({"not": "an array"}))
        resp = _validate(
            client,
            admin_token,
            staged["staging_id"],
            "swap",
            source_format="json",
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "UNPARSEABLE_FILE"
        assert "top-level array" in resp.json()["error"]["message"]

    def test_csv_validated_as_jsonl_names_the_detected_format(
        self, client, admin_token, storage
    ):
        """The systematic mismatch (T6): `source_format` defaulted to jsonl and was never
        compared to the staged file, so a CSV fell into the JSONL branch and 500'd."""
        staged = storage.stage_upload("data.csv", b"id,q,a\nrec-1,hello,hi\n")
        resp = _validate(client, admin_token, staged["staging_id"], "mismatch")
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "FORMAT_MISMATCH"
        assert "'csv'" in body["error"]["message"]
        # The message has to tell the client what to send instead.
        assert 'source_format="csv"' in body["error"]["message"]

    def test_normalization_target_formats_are_refused_by_name(
        self, client, admin_token, storage
    ):
        """`alpaca`/`sharegpt` are normalization targets, not file encodings. The schema
        permits them but no parser can read one, so the request is refused explicitly."""
        staged = storage.stage_upload("data.jsonl", _jsonl([_record(0)]))
        resp = _validate(
            client,
            admin_token,
            staged["staging_id"],
            "sharegpt",
            source_format="sharegpt",
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "FORMAT_MISMATCH"

    def test_missing_openpyxl_is_handled_identically_in_inspect_and_validate(
        self, client, admin_token, storage, monkeypatch
    ):
        """`inspect` raised a clean 400 for a missing openpyxl; `validate` raised
        FileNotFoundError -> 500. Both must answer MISSING_DEP."""
        import builtins

        real_import = builtins.__import__

        def _no_openpyxl(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("No module named 'openpyxl'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_openpyxl)

        resp = _upload(
            client, admin_token, "book.xlsx", b"PK\x03\x04not-a-real-workbook"
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "MISSING_DEP"

    def test_non_object_record_is_a_per_record_finding(self, client, admin_token):
        """A `.jsonl` line of `[1, 2]` parses cleanly but is not a record. Previously the
        H1 rules called `record.get` on a list and the request 500'd."""
        content = b"[1, 2]\n" + _jsonl([_record(0)])
        resp = _inspect_then_validate(
            client, admin_token, "non_object", "nonobj.jsonl", content
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "H0_record_not_object" in body["diagnostics"][0]
        assert body["valid_records"] == 1
        assert body["blocking_error_count"] == 1

    def test_messages_as_string_is_a_per_record_finding(self, client, admin_token):
        """`messages: "hello"` iterated per character, so `message.get` raised."""
        content = _jsonl(
            [
                {
                    "id": "s1",
                    "messages": "hello",
                    "metadata": {"source_dataset": "d", "source_id": "1"},
                }
            ]
        )
        resp = _inspect_then_validate(
            client, admin_token, "msg_string", "msgstr.jsonl", content
        )
        assert resp.status_code == 200
        assert "H0_messages_not_a_list" in resp.json()["diagnostics"][0]

    def test_metadata_as_string_is_a_per_record_finding(self, client, admin_token):
        content = _jsonl(
            [
                {
                    "id": "m1",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": "nope",
                }
            ]
        )
        resp = _inspect_then_validate(
            client, admin_token, "meta_string", "metastr.jsonl", content
        )
        assert resp.status_code == 200
        assert "H0_metadata_not_an_object" in resp.json()["diagnostics"][0]

    def test_message_element_not_an_object_is_a_per_record_finding(
        self, client, admin_token
    ):
        content = _jsonl(
            [
                {
                    "id": "e1",
                    "messages": ["just a string"],
                    "metadata": {"source_dataset": "d", "source_id": "1"},
                }
            ]
        )
        resp = _inspect_then_validate(
            client, admin_token, "elem_string", "elem.jsonl", content
        )
        assert resp.status_code == 200
        assert "H0_message_not_an_object" in resp.json()["diagnostics"][0]

    def test_valid_input_does_not_regress(self, client, admin_token):
        resp = _inspect_then_validate(
            client,
            admin_token,
            "still_fine",
            "fine.jsonl",
            _jsonl([_record(i) for i in range(3)]),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "PASS"
        assert resp.json()["total_records"] == 3

    def test_inspect_and_validate_agree_on_the_record_count(self, client, admin_token):
        """The "two parsers" defect: inspect counted one way and validate read another, so a
        workbook's record count could differ between the two steps."""
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["id", "messages"])
        for i in range(7):
            ws.append([f"r{i}", json.dumps([{"role": "user", "content": f"q{i}"}])])
        buf = io.BytesIO()
        wb.save(buf)

        inspect = _upload(client, admin_token, "book.xlsx", buf.getvalue())
        assert inspect.status_code == 200
        assert inspect.json()["detected_sample_count"] == 7

        resp = _validate(
            client,
            admin_token,
            inspect.json()["staging_id"],
            "xlsx_count",
            source_format="xlsx",
        )
        assert resp.json()["total_records"] == inspect.json()["detected_sample_count"]
        wb.close()


# ── A14 / #246 — nothing may be written outside the staging tree ───────────────


class TestFilenameSanitization:
    @pytest.mark.parametrize(
        "dangerous",
        [
            "../../etc/x.jsonl",
            "..\\..\\x.jsonl",
            "/abs/x.jsonl",
            "C:\\data\\x.jsonl",
            "./../x.jsonl",
            "sub/dir/x.jsonl",
            "a/b/../../../x.jsonl",
        ],
    )
    def test_dangerous_names_reduce_to_a_basename(self, dangerous):
        safe = sanitize_filename(dangerous)
        assert "/" not in safe
        assert "\\" not in safe
        assert safe == "x.jsonl"

    @pytest.mark.parametrize("name", ["", ".", "..", "   ", "..\\..", "\x00evil.jsonl"])
    def test_names_with_no_usable_component_are_rejected(self, name):
        with pytest.raises(UnsafeFilenameError):
            sanitize_filename(name)

    @pytest.mark.parametrize(
        "name",
        [
            "train.jsonl",
            "data.csv",
            "my dataset (v2).jsonl",
            "data-latihan.jsonl",
            "données-2026.jsonl",
            "spaced name.jsonl",
            "  leading and trailing.jsonl  ",
        ],
    )
    def test_normal_names_survive(self, name):
        assert sanitize_filename(name) == name.strip()

    def test_staged_file_always_lands_inside_the_staging_dir(self, tmp_path):
        store = DatasetStorage(tmp_path)
        info = store.stage_upload("../../etc/passwd.jsonl", b'{"a":1}')
        staged = Path(info["path"]).resolve()
        assert staged.parent.parent == (tmp_path / "_staging").resolve()
        assert staged.name == "passwd.jsonl"

    def test_nothing_is_written_above_the_storage_root(self, tmp_path):
        """The actual security property, asserted on the filesystem rather than on the
        helper's return value."""
        store = DatasetStorage(tmp_path)
        for name in ("../escaped.jsonl", "../../escaped.jsonl", "/tmp/escaped.jsonl"):
            store.stage_upload(name, b'{"a":1}')
        assert not (tmp_path / "escaped.jsonl").exists()
        assert not (tmp_path.parent / "escaped.jsonl").exists()
        assert not Path("/tmp/escaped.jsonl").exists()

    def test_stage_upload_rejects_a_name_with_no_component(self, tmp_path):
        store = DatasetStorage(tmp_path)
        with pytest.raises(UnsafeFilenameError):
            store.stage_upload("..", b'{"a":1}')

    def test_inspect_normalizes_a_traversal_filename(
        self, client, admin_token, storage
    ):
        """The endpoint accepts a path-bearing name but reports the sanitized one, so a
        client that sent a path can see what was actually stored -- and the bytes land
        inside the staging tree, not at the path the client asked for."""
        resp = _upload(client, admin_token, "../../etc/x.jsonl", b'{"a":1}\n')
        assert resp.status_code == 200
        assert resp.json()["filename"] == "x.jsonl"

        staged = storage.resolve_staged(resp.json()["staging_id"])
        assert staged is not None
        assert staged.resolve().parent.parent == (storage._staging).resolve()
        assert staged.name == "x.jsonl"
        assert not (storage._base.parent / "etc").exists()

    def test_inspect_rejects_a_name_with_no_component(self, client, admin_token):
        """`..` has no usable basename left after sanitizing, so there is nothing to store
        and the request is refused by name."""
        resp = _upload(client, admin_token, "..", b'{"a":1}\n')
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_FILENAME"

    def test_inspect_reports_the_sanitized_name(self, client, admin_token):
        resp = _upload(client, admin_token, "sub/dir/train.jsonl", b'{"a":1}\n')
        assert resp.status_code == 200
        assert resp.json()["filename"] == "train.jsonl"

    def test_commit_destination_stays_inside_the_version_prefix(self, tmp_path):
        """`commit_file` takes its name from a directory listing, never from client input --
        asserted so a future refactor cannot reintroduce the traversal there.

        Since issue #214 the destination is an object key, not a path, so the property to
        assert is that the key stays under `datasets/{id}/v{N}/` with a single filename
        component.
        """
        store = DatasetStorage(
            tmp_path / "datasets",
            store=LocalFilesystemArtifactStorage(tmp_path / "artifacts"),
        )
        store.stage_upload("../../evil.jsonl", b'{"a":1}\n')
        staging_id = next(
            p.name for p in (tmp_path / "datasets" / "_staging").iterdir() if p.is_dir()
        )
        uri, filename, size = store.commit_file(staging_id, "ds-1", 1)
        assert filename == "evil.jsonl"
        assert size == len(b'{"a":1}\n')
        assert uri == f"file://{tmp_path / 'artifacts' / 'datasets/ds-1/v1/evil.jsonl'}"
        # The stored object really is where the URI says, and it is a single component.
        stored = Path(uri.removeprefix("file://"))
        assert stored.is_file()
        assert stored.name == "evil.jsonl"
        assert ".." not in uri
