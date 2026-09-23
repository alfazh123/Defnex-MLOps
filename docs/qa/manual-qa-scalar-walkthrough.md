# Panduan QA Manual: Dataset → Production via Scalar

**Tujuan:** klik-klik manual di Scalar (`http://localhost:8000/scalar`) untuk membuktikan closed loop DEFNEX MLOps benar-benar jalan, dari upload dataset sampai model live di production — cocok untuk didemokan ke mentor.

**Setiap klaim di panduan ini sudah diverifikasi langsung ke kode** (`file:baris`), bukan asumsi — kutipan ada di tiap langkah supaya kamu bisa cross-check sendiri kalau perilakunya berubah nanti.

---

## 0. Sebelum mulai

### 0.1 Backend harus jalan

```bash
cd ml-close-loop-be
docker compose up -d backend
```

Cek sehat: buka `http://localhost:8000/api/v1/health` → `{"status":"ok","checks":{"db":"ok"}}`.

### 0.2 PENTING — dua worker terpisah harus dinyalakan manual

Ini yang paling sering bikin orang stuck: **training dan evaluasi tidak jalan otomatis** begitu kamu klik "Send" di Scalar. Keduanya cuma pindah status ke `PENDING`/menunggu, lalu diam — ada proses worker terpisah yang harus polling dan benar-benar mengeksekusinya.

- **Training worker (service `worker`) — JANGAN dipakai untuk demo lokal.** `docker compose up -d worker` menjalankan `python -m app.workers.training_worker`, yang secara hardcode selalu memakai `LocalSubprocessProvider` → `UnslothTrainingRunner` (`app/workers/training_worker.py:274-280`) — ini training **Unsloth asli**, bukan simulasi. Runner ini men-spawn `python3 app/training/run_training.py`, yang meng-`import unsloth, torch, transformers, datasets, trl` (`app/training/run_training.py:50-54`) — paket-paket ini **sengaja tidak di-install** di container `backend`/`worker` (training venv memang harus terpisah dari serving venv, dan training asli cuma boleh jalan di GPU H100 yang sebenarnya, bukan di container lokal — `CLAUDE.md`). Kalau kamu tetap jalankan `docker compose up worker` di lokal, setiap training run akan **selalu** `FAILED` dengan pesan `No module named 'datasets'`, apa pun dataset/config yang kamu pakai.

  **Untuk demo lokal, majukan training run secara manual** lewat `docker compose exec` — ini memanggil fungsi worker yang sama persis (`process_next_job`) tapi dengan `MockTrainingRunner` (dipakai test suite proyek ini juga) alih-alih Unsloth asli, jadi selalu berhasil tanpa GPU:
  ```bash
  docker compose exec backend python -c "
  from app.db.session import SessionLocal
  from app.workers.training_worker import process_next_job
  from app.workers.mock_runner import MockTrainingRunner

  db = SessionLocal()
  processed = process_next_job(db, MockTrainingRunner())
  db.commit()
  print('status:', processed.status if processed else 'tidak ada run PENDING')
  "
  ```
  Jalankan ini setiap kali kamu selesai bikin training run baru dan mau majukan statusnya (gantikan langkah "tunggu worker" di bagian 3). Poll interval training worker asli (5 detik, `app/workers/training_worker.py:240`) jadi tidak relevan untuk demo lokal.

- **Evaluation worker** — **TIDAK ADA di `docker-compose.yml` sama sekali** (dicek: `grep -rn evaluation_worker` di semua file yml = nihil). Kamu harus jalankan manual di terminal terpisah:
  ```bash
  docker compose exec backend python -m app.workers.evaluation_worker
  ```
  Biarkan terminal ini terbuka selama demo (dia polling tiap 5 detik juga, `app/workers/evaluation_worker.py:89`). Kalau lupa jalankan ini, model akan macet permanen di status `REGISTERED` — evaluasi tidak akan pernah selesai.

  **Beda dengan training worker — ini TIDAK butuh bypass sama sekali, perintah di atas sudah benar apa adanya.** Training gagal di lokal karena `UnslothTrainingRunner` di-hardcode tanpa syarat (`app/workers/training_worker.py:274-280`), sedangkan evaluation memanggil `get_serving_backend()` yang membaca `settings.serving_backend` — defaultnya `"mock"` (`app/config.py:27`, tidak di-override di `.env`), jadi otomatis pakai `MockServingBackend` tanpa perlu GPU/model asli sama sekali (`app/services/serving.py:326-331`). Sudah diverifikasi langsung: perintah di atas dijalankan apa adanya terhadap `demo-support-model` versi 1, hasilnya `status: "EVALUATED"` dengan sinyal evaluasi asli terisi (`eval_loss_trend`, `qualitative_comparison`, `general_domain_regression_check`) — bukan cuma status berubah tanpa data.

### 0.3 WAJIB — reset database dulu kalau sudah pernah dipakai testing

Langkah 1 di bawah mengandalkan aturan "akun pertama yang pernah register di database otomatis jadi admin" (`app/api/auth.py:122-127`). Kalau database ini sudah pernah dipakai testing sebelumnya (ada user lain di dalamnya), akun baru yang kamu register **TIDAK** akan otomatis admin — dia akan jadi `user` biasa, dan setiap langkah yang butuh admin (dataset intake, eval-set) akan gagal `403 FORBIDDEN`.

Cek dulu (kalau kamu punya akses shell):
```bash
sqlite3 data/app.db "SELECT username, role FROM users;"
```
Kalau hasilnya tidak kosong, **reset dulu** supaya demo bersih dan bisa diulang kapan pun tanpa nebak-nebak:
```bash
docker compose stop backend worker
rm data/app.db
docker compose exec backend rm -rf /app/data/artifacts   # WAJIB, lihat penjelasan di bawah
docker compose up -d backend   # migrasi jalan otomatis, bikin DB kosong baru
```

**Kenapa `data/artifacts/` juga wajib dibersihkan, bukan cuma `data/app.db`:** artifact model bersifat immutable — sekali sebuah folder `{model_id}-{base_model_slug}-v{N}` dibuat, sistem menolak menimpanya selamanya (`app/services/artifact_storage.py:113-116`, `ArtifactExistsError`). Kalau kamu cuma hapus `data/app.db` tapi biarkan `data/artifacts/`, database akan mengira `model_id` yang sama belum pernah punya versi 1 (mau alokasi ulang versi 1), padahal folder artifact versi 1 lama masih ada di disk dari sesi sebelum reset — training run baru akan gagal dengan `ArtifactExistsError: Refusing to overwrite...` walau semua langkah sebelumnya benar. Kedua hal ini (`data/app.db` dan `data/artifacts/`) harus direset BARENGAN, tidak boleh cuma salah satu.

### 0.4 Siapkan file dataset contoh

Buat file `demo-dataset.jsonl` di komputer kamu, isi persis seperti ini (satu JSON per baris, format `jsonl`):

```jsonl
{"id": "r1", "messages": [{"role": "user", "content": "Bagaimana cara reset password akun saya?"}, {"role": "assistant", "content": "Silakan buka halaman login, klik tautan lupa password, lalu ikuti instruksi yang dikirim ke email terdaftar Anda untuk membuat password baru dengan aman."}], "metadata": {"source_dataset": "demo-support-dataset", "source_id": "demo-1"}}
{"id": "r2", "messages": [{"role": "user", "content": "Berapa lama waktu pengiriman standar?"}, {"role": "assistant", "content": "Waktu pengiriman standar biasanya membutuhkan tiga sampai lima hari kerja tergantung lokasi tujuan dan ketersediaan kurir di wilayah Anda pada saat ini."}], "metadata": {"source_dataset": "demo-support-dataset", "source_id": "demo-2"}}
{"id": "r3", "messages": [{"role": "user", "content": "Apakah bisa membatalkan pesanan yang sudah dibayar?"}, {"role": "assistant", "content": "Pembatalan pesanan yang sudah dibayar dapat dilakukan selama status masih diproses, silakan hubungi layanan pelanggan kami secepatnya sebelum barang dikirim ke alamat tujuan."}], "metadata": {"source_dataset": "demo-support-dataset", "source_id": "demo-3"}}
```

**Dua aturan validasi yang harus dipenuhi persis** (sudah diuji, contoh di atas lolos bersih tanpa warning):
- **H1** — `metadata` WAJIB punya **dua** field: `source_dataset` DAN `source_id`, bukan cuma salah satu (`app/services/validation_service.py:77-84`). Kalau cuma isi `source_id` saja, semua baris kena `H1_missing_required_field` (masih lolos gate tapi kotor untuk demo).
- **H4** — jawaban `assistant` harus **lebih dari** 20 kata, pas 20 kata pun tetap ditolak (`app/services/validation_service.py:99-101`, `<= 20` yang ditolak). Contoh di atas sudah dihitung manual supaya semuanya 21+ kata.

---

## 1. Login — dapatkan token, pasang sekali di Scalar

### 1.1 Register (akun pertama otomatis jadi admin)

Endpoint: `POST /api/v1/auth/register`

```json
{"username": "demo_admin", "password": "Demo1234", "role": "admin"}
```

**Catatan penting:** field `role` di body ini **diabaikan sepenuhnya oleh server** — role sebenarnya dihitung otomatis: akun pertama yang pernah register di database jadi `admin`, semua akun berikutnya jadi `user`, apa pun yang kamu isi di `role` (`app/api/auth.py:117-127`). Jadi: **cukup register SATU akun untuk seluruh demo ini** — akun itu otomatis admin dan punya semua izin yang dibutuhkan di setiap langkah.

→ Response `201`, `{id, username, role: "admin", created_at}`.

### 1.2 Login

Endpoint: `POST /api/v1/auth/login`

```json
{"username": "demo_admin", "password": "Demo1234"}
```

→ Response `200`: `{"access_token": "...", "refresh_token": "...", "token_type": "bearer", "user": {...}}` (`app/schemas/auth.py:12-16`).

### 1.3 Pasang token sekali di Scalar/Swagger

Klik tombol **Authorize** (🔒) di pojok atas Scalar/Swagger, tempel isi `access_token` (tanpa perlu ketik `Bearer` — Scalar otomatis menambahkannya karena endpoint ini sekarang benar-benar terdaftar sebagai `HTTPBearer` security scheme). Sekali diisi, token ini otomatis terpasang ke SEMUA request berikutnya di sesi Scalar ini — tidak perlu isi ulang tiap endpoint.

---

## 2. Dataset intake (upload file asli)

Ada dua jalur di kode: endpoint lama `POST /datasets/{id}/versions` (cuma catat metadata, tidak pernah menyentuh isi file) dan wizard upload baru `intake/inspect → intake/validate → intake/commit` (`app/api/intake.py`, `app/api/intake_validate.py`). **Pakai yang wizard** — itu satu-satunya jalur yang benar-benar membaca file yang kamu upload.

### 2.1 Inspect

Endpoint: `POST /api/v1/datasets/intake/inspect`

Di Scalar, field body-nya `multipart/form-data` dengan satu field `file` — klik "Choose File", pilih `demo-dataset.jsonl` (`app/api/intake.py:91`).

→ Response `200`: catat nilai `staging_id` dari response, dipakai di langkah berikutnya.

```json
{"staging_id": "...", "filename": "demo-dataset.jsonl", "detected_format": "jsonl", "detected_sample_count": 3, "checksum_sha256": "...", "parse_status": "ok"}
```

### 2.2 Validate

Endpoint: `POST /api/v1/datasets/intake/validate`

```json
{"staging_id": "<dari langkah 2.1>", "dataset_id": "demo-support-dataset"}
```

→ Response `200`, cek `status` **bukan** `"FAIL"`, dan catat `validation_report_id`.

### 2.3 Commit

Endpoint: `POST /api/v1/datasets/intake/commit`

```json
{"staging_id": "<dari 2.1>", "dataset_id": "demo-support-dataset", "validation_report_id": "<dari 2.2>", "display_name": "Demo Support Dataset"}
```

→ Response `200`, **catat angka `version` yang benar-benar dikembalikan** (bukan asumsi `1` — kalau kamu sudah pernah coba dataset_id yang sama sebelumnya, dan gagal, nomor versi ikut naik walau commit-nya gagal/dibatalkan). Pakai angka aslinya di semua langkah berikutnya. Dataset version sekarang berstatus `PROCESSED` — sudah siap dipakai training, **tidak perlu** panggil endpoint validate lama lagi.

---

## 3. Training run

Endpoint: `POST /api/v1/training-runs` (bisa dipanggil user mana pun, tidak butuh admin — `app/api/training.py:36`)

```json
{
  "dataset_id": "demo-support-dataset",
  "dataset_version": 1,
  "model_id": "demo-support-model",
  "base_model": "unsloth/Qwen3-0.6B",
  "training_config": {
    "peft_method": "lora",
    "load_in_4bit": false,
    "lora_r": 8,
    "lora_alpha": 16,
    "epochs": 1,
    "max_seq_length": 512
  },
  "triggered_by": "demo_admin"
}
```

→ Response `201`, `status: "PENDING"`, `model_version: null`.

**Majukan run ini secara manual** (lihat langkah 0.2 — JANGAN `docker compose up worker`, itu akan selalu `FAILED` di lokal):
```bash
docker compose exec backend python -c "
from app.db.session import SessionLocal
from app.workers.training_worker import process_next_job
from app.workers.mock_runner import MockTrainingRunner

db = SessionLocal()
processed = process_next_job(db, MockTrainingRunner())
db.commit()
print('status:', processed.status if processed else 'tidak ada run PENDING')
"
```

Lalu cek statusnya:

Endpoint: `GET /api/v1/training-runs/{training_run_id}` → `status: "COMPLETED"` dan `model_version` terisi angka (mis. `1`). Kalau perintah di atas mencetak `tidak ada run PENDING`, pastikan `training_run_id` di request sebelumnya benar dan run-nya memang masih `PENDING` (belum pernah dicoba lewat worker asli sebelumnya sampai `FAILED`).

---

## 4. Model teregistrasi

Endpoint: `GET /api/v1/models/demo-support-model/versions/1`

→ `status: "REGISTERED"`, `training_run_id` sesuai run di atas, `artifacts[0].uri` terisi.

---

## 5. Evaluation (server-side, bukan angka karangan)

### 5.1 Buat golden/eval set

Endpoint: `POST /api/v1/eval-sets/demo-eval-set/versions`

```json
{"records": [{"messages": [{"role": "user", "content": "Bagaimana cara mengembalikan barang yang cacat?"}]}]}
```

→ Response `201`, catat `version` (biasanya `1`).

### 5.2 Trigger evaluasi

Endpoint: `POST /api/v1/models/demo-support-model/versions/1/evaluation`

```json
{"eval_set_id": "demo-eval-set", "eval_set_version": 1}
```

→ Response `200`. Status model **belum berubah** di sini — evaluasi berjalan async (`app/api/models.py:105-109`), mirip training.

### 5.3 Tunggu evaluation worker (langkah 0.2!)

Kalau kamu sudah jalankan `docker compose exec backend python -m app.workers.evaluation_worker` dan biarkan terbuka, dalam ~5-10 detik dia akan memproses ini. Kalau belum — jalankan sekarang, ini **wajib**, tidak akan pernah selesai sendiri. **Tidak seperti training worker, ini tidak butuh bypass/`MockTrainingRunner` apa pun** — jalankan perintahnya apa adanya, sudah diverifikasi langsung berhasil.

Cek: `GET /api/v1/models/demo-support-model/versions/1` → ulangi sampai `status: "EVALUATED"`.

Dengan `SERVING_BACKEND=mock` (default lokal), evaluasi akan selalu "menang" 100% tanpa regresi (`app/services/serving.py:119-122`) — jadi gate promosi di langkah berikutnya akan lolos dengan mudah, ini memang perilaku mock, bukan bug.

---

## 6. Naik tangga promosi: Staging → Validated → Production

Ini jalur **PRD-compliant** yang sebenarnya (PRD §16.2 & Principle 4: "no candidate model goes straight from training to production"). Ada jalur pintas lama (`/decisions` + `/deploy` langsung) yang masih ada di kode untuk kompatibilitas — jangan pakai itu untuk demo ke mentor, lihat Lampiran A.

Semua tiga endpoint di bawah ini butuh body yang sama:
```json
{"decided_by": "demo_admin", "rationale": "<alasan kamu>"}
```

### 6.1 Deploy ke staging

Endpoint: `POST /api/v1/models/demo-support-model/versions/1/deploy-staging`

```json
{"decided_by": "demo_admin", "rationale": "Eval loss turun, tidak ada regresi domain umum."}
```

→ Response `201`, `decision: "STAGING"`. Cek model: `GET .../versions/1` → `status: "STAGING"`.

### 6.2 Validasi di staging

Endpoint: `POST /api/v1/models/demo-support-model/versions/1/validate-staging`

```json
{"decided_by": "demo_admin", "rationale": "Sudah dicoba manual di staging, hasilnya sesuai ekspektasi."}
```

→ Response `201`, `decision: "VALIDATED"`. Cek model: `status: "VALIDATED"`.

### 6.3 Promosi ke production

Endpoint: `POST /api/v1/models/demo-support-model/versions/1/promote-production`

```json
{"decided_by": "demo_admin", "rationale": "Lolos validasi staging, siap production."}
```

→ Response `201`, `decision: "PRODUCTION"`. Ini benar-benar memindahkan pointer serving (real deploy call + smoke test kalau `SERVING_BACKEND=vllm`), bukan cuma ubah label.

**Catatan istilah:** status registry model tetap tertulis `"DEPLOYED"` setelah ini (bukan `"PRODUCTION"`) — itu memang sengaja, cuma `decision` di response yang bilang `"PRODUCTION"` (`app/api/promotion.py:163`). Kalau mentor tanya kenapa beda, itu bukan bug.

### 6.4 Konfirmasi live di production

- `GET /api/v1/models/demo-support-model/versions/1` → `status: "DEPLOYED"`.
- `GET /api/v1/models/demo-support-model/deployment` → `{"model_id", "current_deployed_version": 1, "status": "DEPLOYED", "deployed_at": "..."}`.
- `GET /api/v1/models/demo-support-model/deployment/prod` → alias yang sama, cara lain memastikan production benar-benar menunjuk ke versi ini.

**Ini titik paling bagus untuk stop dan foto/screenshot buat demo ke mentor** — modelnya sudah lewat seluruh tangga (training → evaluated → staging → validated → production), semua tercatat dengan `decided_by`/`rationale`/`evidence_snapshot` di tiap langkah.

---

## 7. (Opsional) Rollback — butuh DUA versi model dulu

Rollback **tidak bisa** didemokan dengan cuma 1 versi model. Aturannya: versi yang jadi *target* rollback harus berstatus `PROMOTED`, `RETIRED`, atau ada di tangga staging (`STAGING`/`VALIDATED`/`PRODUCTION`) — **bukan** `DEPLOYED` (`app/services/promotion_service.py:426-435`). Versi 1 yang baru saja kamu promosikan ke production sekarang berstatus `DEPLOYED` — dia justru versi yang *sedang aktif*, bukan target buat di-rollback-ke.

Supaya rollback masuk akal, kamu perlu versi ke-2 yang menggantikan versi 1 dulu (versi 1 otomatis jadi `RETIRED` begitu versi 2 di-deploy — itu baru bisa jadi target rollback). Cara paling cepat: ulangi langkah 3-6 dengan `model_id` yang **sama** (`demo-support-model`) tapi `dataset_version` baru, biar dapat versi model ke-2. Setelah versi 2 production dan versi 1 berstatus `RETIRED`:

Endpoint: `POST /api/v1/models/demo-support-model/rollback`

```json
{"rollback_of_version": 1, "decided_by": "demo_admin", "rationale": "Simulasi: ditemukan regresi di versi 2, kembali ke versi 1."}
```

→ Response `201`, `decision: "ROLLBACK"` — pointer production pindah balik ke versi 1.

---

## Lampiran A — Jalur lama (jangan dipakai untuk demo utama)

Kode ini masih ada untuk kompatibilitas mundur, TAPI melompati staging sama sekali (langsung EVALUATED → PROMOTED → DEPLOYED) — bertentangan dengan PRD Principle 4:

- `POST /models/{id}/versions/{v}/decisions` — `{"decision": "PROMOTED", "decided_by": "...", "rationale": "..."}` → status `PROMOTED`. Ini juga yang memicu **eval gate** (cek eval-set reference, majority win, no regression, eval loss membaik) — kalau gagal, 409 `PROMOTION_GATE_BLOCKED`.
- `POST /models/{id}/versions/{v}/deploy` (tanpa body, atau `{"environment": "staging"|"production"}`) → langsung ubah status deployment tanpa audit trail selengkap jalur ladder.

Sebutkan ke mentor kalau ini jalur lama yang sengaja dipertahankan untuk kompatibilitas, bukan cara yang direkomendasikan sekarang.

---

## Lampiran B — Siapa boleh apa (RBAC)

Karena kamu cuma register **satu** akun (otomatis admin), kamu tidak akan pernah kena blokir izin selama demo — admin punya semua izin di bawah ini (`app/rbac.py:42-48`):

| Aksi | Role yang diizinkan |
|---|---|
| Dataset intake, eval-set create | `admin` saja (`require_admin`, cek role String `"admin"`) |
| Buat training run, trigger evaluasi | siapa pun yang login |
| `/decisions`, `/promote-production` | `admin`, `ml_engineer` |
| `/deploy-staging`, `/deploy`, rollback | `admin`, `ml_engineer` |
| `/validate-staging` | `admin`, `reviewer` |

**Catatan:** role `ml_engineer`/`data_engineer`/`reviewer` ada di kode (`app/rbac.py`) tapi **tidak bisa dipilih lewat API** — `/auth/register` cuma menerima `admin` (akun pertama) atau `user` (akun berikutnya), dan tidak ada endpoint untuk ubah role user lain. Kalau mentor mau lihat perbedaan role selain admin/user, itu butuh perubahan kode dulu (endpoint role-assignment belum ada).

---

## Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `{"error":{"code":"MISSING_TOKEN",...}}` | Token belum dipasang, atau salah paste | Klik Authorize (🔒) lagi, pastikan cuma tempel token murni tanpa kutip/spasi ekstra |
| Training run macet di `PENDING` selamanya | Belum dimajukan manual | Jalankan perintah `docker compose exec backend python -c "..."` di langkah 0.2/bagian 3 |
| Training run jadi `FAILED` dengan pesan `No module named 'datasets'` (atau `unsloth`/`torch`/`transformers`) | **Ini yang terjadi kalau kamu jalankan `docker compose up -d worker` di lokal** — worker asli butuh venv Unsloth+GPU yang memang tidak ada di container lokal (`app/training/run_training.py:50-54`), bukan bug config kamu | Jangan pakai `docker compose up worker` untuk demo lokal — pakai perintah manual `process_next_job(MockTrainingRunner())` di langkah 0.2. Kalau ada run lama yang sudah `FAILED` gara-gara ini, buat training run baru (langkah 3) lalu majukan pakai perintah manual itu |
| Model macet di `REGISTERED` walau sudah trigger evaluation | Evaluation worker tidak jalan — **ini bukan bagian dari `docker compose up worker`** | Jalankan manual: `docker compose exec backend python -m app.workers.evaluation_worker` |
| `409 STAGING_DEPLOY_NOT_ALLOWED` di deploy-staging | Model belum `EVALUATED` | Selesaikan langkah 5 dulu |
| `409 GATE_NOT_MET` di promote-production | Belum lewat `validate-staging` | Jalankan langkah 6.2 dulu, jangan lompat dari staging langsung ke promote |
| `409 VALIDATION_FAILED` di dataset intake | Ada baris JSONL yang gagal validasi (mis. jawaban assistant <20 kata) | Perbaiki file, ulangi dari langkah 2.1 (staging_id baru) |
| Sudah pernah jalankan `docker compose up worker` sebelumnya, sekarang ada training run lama berstatus `FAILED` mengganggu | Sisa dari worker asli yang gagal (lihat baris di atas) | Bikin `dataset_id`/`model_id` baru untuk training run berikutnya, atau abaikan saja run lama yang `FAILED` — tidak mengganggu run baru |
| `ArtifactExistsError: Refusing to overwrite existing immutable artifact at data/artifacts/{model_id}/...` | `data/app.db` pernah direset TANPA ikut membersihkan `data/artifacts/` — folder artifact lama untuk `model_id`+`base_model`+versi yang sama masih ada di disk dari sebelum reset, sementara database (yang baru) mengira versi itu belum pernah dibuat | Ikuti langkah 0.3 dengan benar (`docker compose exec backend rm -rf /app/data/artifacts` **bersamaan** dengan hapus `data/app.db`), atau kalau cuma satu `model_id` yang bentrok: `docker compose exec backend rm -rf /app/data/artifacts/{model_id}` lalu ulangi training run-nya |
| Muncul warning `compute_resource_not_found compute_resource_id=...` saat training diproses manual | **Aman, boleh diabaikan** — training run yang dibuat lewat panduan ini tidak set `compute_resource_id`, jadi ini fallback normal ke `MockTrainingRunner` (`app/workers/training_worker.py:85-92`), bukan kegagalan | Tidak perlu tindakan, training tetap lanjut normal setelah warning ini |
