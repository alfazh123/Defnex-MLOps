# Validation gate policy

**Status:** aktif (issue #241, audit finding T2) · **Rule set:** `2.2.0` · **Owner:** data governance

Dokumen ini adalah sumber kebenaran untuk keputusan `gate_decision` pada
`ValidationReport` (`POST /api/v1/datasets/intake/validate` dan
`POST /api/v1/datasets/{dataset_id}/versions/{version}/validate`).

## Masalah yang diperbaiki

Sebelum issue #241, logika gate adalah:

```python
leakage_found = leakage_overlaps > 0
gate_decision = "FAIL" if leakage_found else "PASS"
```

`per_record_errors` **sudah dihitung** oleh rule set H1–H9, dan endpoint intake bahkan
mengirimkannya ke klien sebagai `blocking_error_count` — tapi tidak pernah dipakai untuk
menentukan gate. Akibatnya file yang **setiap record-nya** gagal H1 tetap melaporkan
`PASS`, bisa di-commit, dan bisa ditraining.

Dua bug terkait ditemukan bersama-sama:

| # | Bug | Akibat |
|---|---|---|
| 1 | Gate hanya melihat leakage | Dataset 100% invalid lolos sebagai `PASS` |
| 2 | `blocking_error_count` menghitung `isinstance(e, dict) and e.get("severity") == "error"`, padahal `per_record_errors` berisi list of **string** kode rule | `blocking_error_count` **selalu 0** dan `valid_records == total_records` untuk dataset yang seluruh record-nya invalid |

Bug 2 menyembunyikan bug 1 dari test: fixture bernama `VALID_RECORD` di
`tests/test_intake_validate.py` berisi jawaban 10 kata (H4: min 20 kata), jadi "record
valid" di test itu sebenarnya invalid — dan assertion `PASS` tetap hijau karena gate-nya
bocor. Kedua fixture dan assertion sekarang sudah dikoreksi.

## Keputusan

Gate bersifat tiga nilai, berbasis **rasio hard-error**:

```
invalid_ratio = record_dengan_hard_error / total_record
fail_ratio    = VALIDATION_GATE_FAIL_RATIO   (default 0.10)
```

| Keputusan | Kondisi | Efek |
|---|---|---|
| **FAIL** | Leakage (H8) ditemukan, **apapun** rasio hard-error-nya | Commit ditolak `409 VALIDATION_FAILED`; training ditolak |
| **FAIL** | `invalid_ratio >= fail_ratio` | Idem |
| **NEEDS_REVIEW** | `0 < invalid_ratio < fail_ratio` | **Boleh** commit (butuh supaya byte + laporan tersedia untuk diperbaiki manusia); **ditolak** training dengan `409 VALIDATION_NEEDS_REVIEW` |
| **PASS** | `invalid_ratio == 0` | Commit dan training diizinkan |

`>=` bersifat inklusif: tepat 10% dari 20 record = 2 record invalid → **FAIL**. Ambang ini
dipatok oleh test (`test_error_rate_exactly_at_the_threshold_fails`).

### Mengapa leakage tidak tunduk pada ambang

Leakage train/eval membatalkan nilai seluruh dataset, bukan hanya beberapa record. Satu
record yang bocor sudah cukup untuk membuat metrik eval tidak dipercaya, jadi tidak ada
"cukup sedikit" yang bisa diterima di sini.

### Mengapa NEEDS_REVIEW boleh commit tetapi tidak boleh training

`DatasetVersion.status` (`PENDING`/`PROCESSING`/`PROCESSED`/`FAILED`) menjelaskan **keadaan
byte**, sedangkan `gate_decision` menjelaskan **mutu**. Keduanya sengaja dipisah.
Dataset NEEDS_REVIEW yang sudah di-commit adalah bahan kerja: tanpa versi yang tersimpan,
mustahil membandingkan, men-diff, atau menghitung ulang record yang diperbaiki. Yang
dilarang adalah pemakaiannya sebagai bahan training, dan itu ditegakkan di
`app/api/training.py`.

Kalau bisnis memutuskan dataset NEEDS_REVIEW tidak boleh sama sekali masuk registry,
ubah satu kondisi di `app/api/intake_validate.py` (`commit_intake`) — bukan dengan
menulis ulang status versi, karena itu akan mengaburkan makna `PROCESSED`.

### Apa yang **tidak** menentukan gate ini

- **Warning (PII, W1–W5).** `PII_*` sengaja hanya warning dan tidak pernah memengaruhi
  gate — lihat komentar `_PII_PATTERNS` di `validation_service.py`. Menyiszipkan PII ke
  dataset pertahanan adalah keputusan review manusia, bukan alasan untuk menolak file.
- **Per-record `status_counts.NEEDS_REVIEW`.** Selalu `0`. Status per-record butuh
  threshold kelas warning (W2/W5) yang memang tidak didefinisikan di `validation-rules.md`
  §9 — persis yang dokumen itu rekomendasikan untuk ditunda. `NEEDS_REVIEW` di
  `gate_decision` adalah verdict tingkat-dataset yang berbeda dan **reachable**. Keduanya
  tidak boleh dicampur.

## Konsekuensi yang harus diketahui

Memperbaiki gate membuat upload CSV/XLSX **gagal** divalidate sebagai dataset yang bisa
ditraining, dan itu memang benar:

Baris CSV datar `{id, q, a}` tidak punya array `messages`, jadi setiap record kena H1 →
100% → FAIL. Mengubahnya menjadi `{"messages": [...]}` adalah tugas **normalizer**, yang
belum ada (**issue #243 / A11**). Sampai normalizer ada, jalur CSV/XLSX berguna hanya untuk
`inspect` (statistik + checksum), dan test
`test_validate_csv_records`/`test_validate_xlsx_records` sekarang mengassert outcome itu
dengan jujur.

## Konfigurasi

| Setting | Default | Arti |
|---|---|---|
| `VALIDATION_GATE_FAIL_RATIO` | `0.10` | Rasio hard-error pada atau di atas mana gate FAIL |

Default 0.10 (satu dari sepuluh) dipilih sebagai titik awal yang bisa dioverride per
deployment, **bukan** sebagai angka final — ini keputusan tata kelola data, dan angka yang
paling tepat adalah properti dari dataset yang masuk, bukan dari kodenya. Naikkan
untuk dataset yang noisy-by-design, turunkan untuk dataset yang wajib bersih.

## Angka di balik keputusan

Setiap laporan menyimpan 근거 keputusannya di `dataset_statistics.hard_error_gate`:

```json
{
  "records_with_hard_errors": 2,
  "hard_error_ratio": 0.1,
  "fail_ratio_threshold": 0.1,
  "leakage_overlaps": 0
}
```

## Rujukan

- Issue #241 · audit `docs/BACKLOG.md` §5.2 T2, §4.1 D-gap7
- `app/services/validation_service.py` — `validate_dataset_version`, `gate_decision`
- `app/api/intake_validate.py` — `checks[].name == "gate"`, blokir commit
- `app/api/training.py` — blokir `VALIDATION_NEEDS_REVIEW`
- `tests/test_intake_batch0.py::TestGateBlocksHardErrors`
- `validation-rules.md` §2 (status per-record), §9 (ambang yang belum didefinisikan)
