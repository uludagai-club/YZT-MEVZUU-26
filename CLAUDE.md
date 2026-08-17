# CLAUDE.md — Proje Haritası

TEKNOFEST hava aracı tespit/tanıma sistemi. Pipeline: **YOLO → VRAG → VLM → LLM**.
Kod/yorumlar Türkçe. Bu dosya her oturumda otomatik yüklenir — burada yazılı olanı yeniden keşfetme.

## Çalıştırma

- **Başlat:** `Sistemi_Baslat.bat` (çift tıkla). Web arayüzü: <http://127.0.0.1:8000/goruntule/>
- **Ön koşul:** Ollama (11434) açık, `llama3.2:1b` yüklü (yalnız LLM için). VLM artık vLLM'de (8002) — `Sistemi_Baslat.sh` başlatır. Bkz. `VLLM_KULLANIM.md`.
- **venv:** `LLM/.venv/Scripts/python.exe` — **uv ile yönetiliyor, `pip` modülü YOK.** Paket için `uv pip`.
- **Python çağrısı (repo kökünden):** `LLM/.venv/Scripts/python.exe -m ...` (PYTHONPATH kök; LLM servisi için `--app-dir src`).

## Servisler / portlar

| Port | Servis | Nasıl başlar |
|------|--------|--------------|
| 8000 | Backend + web (FastAPI, `entegrasyon/backend/main.py` → `main:app`) | Sistemi_Baslat.bat |
| 8001 | LLM operasyonel karar (`operational_decision.api.main:app`, app-dir `LLM/src`) | **Backend'in lifespan'i `subprocess.Popen` ile başlatır** (main.py) |
| 8002 | vLLM — VLM servisi, OpenAI-uyumlu (`Qwen3-VL-2B-Instruct-AWQ-4bit`) | `Sistemi_Baslat.sh` (backend'den ÖNCE; `VLLM_Baslat.sh`) |
| 11434 | Ollama — **yalnız LLM** (`llama3.2:1b`, CPU) | Harici, elle |

Backend'i öldürünce 8001 subprocess'i **öksüz kalabilir** — temiz başlatma için ikisini de kapat.

## Veri akışı (bir kare)

`src/core/pipeline.py` orkestratör. Her hedef (track) için:
1. **YOLO** (`src/detection/slicer.py`) — SAHI dilimli tespit. GPU, fp16 (`quantize=16`).
2. **VRAG** (`src/vrag/engine.py`) — SigLIP2 embedding + Qdrant retrieval → model kimliği. **Best-moment** tetikleme (kalite kapısı, her karede değil).
3. **VLM** (`src/vlm/engine.py`) — vLLM'de `Qwen3-VL-2B-Instruct-AWQ-4bit`, VRAG bağlamıyla görsel analiz → JSON. Kolaj 384px'e küçültülür. Protokol URL'den seçilir: `VLM_API_URL` içinde `/v1/` varsa OpenAI formatı, yoksa Ollama `/api/generate`.
4. **LLM** (8001'e POST, `_async_llm_task`) — izin/uçuş planı/NOTAM/envanter → stratejik karar.

Sonuç `entegrasyon/backend/pipeline_adapter.py` üzerinden web arayüzüne (`entegrasyon/web/index.html`) `vlm`/`llm`/VRAG alanları olarak akar.

## Modeller & cihaz

| Bileşen | Model | Cihaz | Not |
|---------|-------|-------|-----|
| YOLO | `models/best.pt` | GPU | `YOLO_QUANTIZE=16` (Ultralytics 8.4+ `half` kaldırdı → `quantize`) |
| VRAG | `google/siglip2-so400m-patch14-384` | `cuda:0` fp16 | CPU'da worker-thread BLAS deadlock'u vardı → GPU'ya alındı |
| VLM | `cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit` | **vLLM 8002**, GPU ~3.8GB | 4-bit AWQ. Kare başına ~3s (Ollama'da ~50s'ti). `VLLM_GPU_UTIL=0.41` ölçülmüş değer — 0.36'da KV cache açılmıyor. vLLM CPU'ya **taşamaz**, sığmazsa OOM. **"Instruct" varyant ŞART** — Thinking varyantı num_predict'i düşünmeye harcayıp JSON'ı boş döndürür. |
| LLM | `llama3.2:1b` | **CPU** (`num_gpu:0`) | `LLM/.../ollama_client.py` |

**VRAM (RTX 4070 Laptop 8GB) sıkı.** Ölçülen bütçe: vLLM ~3.8GB + backend (YOLO+SigLIP2) ~2.7GB = **~6.5/8.1GB**. Yeni model/ayar eklerken bunu düşün — vLLM taşarsa CPU'ya düşmez, çöker.

## VRAG / Qdrant

- Embedded/local Qdrant: `output/qdrant_db`, koleksiyon `uav_knowledge`, ~24839 nokta, dim **1152**, Cosine.
- **Thread-safe DEĞİL** → `engine.py`'de `_search_lock` tüm embed+search'ü serileştirir.
- **Oy-tabanlı birleşik skor (ince-ayrım):** SigLIP2 benzer uçaklara (F-16↔F-35, HÜRKUŞ↔HÜRJET) neredeyse aynı skoru verdiği için, tek en-yüksek hit'e bakmak şanslı bir referansa kanıp yanlış #1 seçiyordu. Artık her model `mean(top-3 hit) + VRAG_VOTE_WEIGHT × (ilk-K oy oranı)` ile sıralanır → tutarlı en-yakın model kazanır. Ayar: `VRAG_VOTE_TOPK=20`, `VRAG_VOTE_WEIGHT=0.10`. Eğitim/reindex YOK.
- Margin güven kapısı: `margin = top1 − top2` (birleşik skor), eşik `VRAG_MARGIN_ESIGI=0.015`, düşük margin → `dusuk_guven`. Taban: modelin en iyi ham kosinüsü ≥ `VRAG_MIN_SCORE`.
- **Yeniden indeksleme:** `VRAG_Yeniden_Indeksle.bat` (artımlı, yeni görsel), `VRAG_Temiz_Yeniden_Insa.bat` (sıfırdan, "EVET" onayı). Ingest: `python -m src.vrag.ingest --img_dir veriler`.
- Veri: `veriler/` — 8 kategori, ~2306 görsel, ~87 model. Her görselin ~10 augmentasyon varyasyonu indekste.

## Kritik sözleşmeler & tuzaklar

- **VLM→LLM `arac_sinifi`:** LLM adapter'ı (`LLM/config/visual_adapter.yaml` + `upstream_vlm_adapter.py`) **yalnızca `sabit_kanat`/`doner_kanat`** kabul eder. VLM serbest metin üretir (`hava_araci`, `askeri_ucak`…) → uyumsuzsa **HTTP 500**. `pipeline._async_llm_task` LLM'e gitmeden önce ikili kanat tipine **normalize eder** (UI'daki zengin değere dokunmadan).
- **LLM bağlam yönlendirme:** `src/core/context_router.py` model→`video_id` çözer (`raw_vlm_context_routes.json`). Eski bug: sabit `video_id="live_video"` → tüm platformlar yanlış bağlamda (PLT_KAAN).
- **Gözlem zamanı:** duvar saati değil, **video ofseti** (`frame/fps`). İzin/uçuş planı geçerlilik penceresi buna bağlı.
- **Menşei vs Sahiplik:** Menşei = üretici ülke (metadata `ulke`). Sahiplik = işletici, Türkiye envanterinden runtime çözülür (`entegrasyon/backend/sahiplik.py`).
- **Video akışı (`/video` MJPEG):** oturum aktifken ilk kareyi ~90s bekler (soğuk model yükleme ~44s). Boştayken ~5s'de kapanır.

## Ayarlar

`src/config.py` merkezi. Öne çıkanlar: `VRAG_DEVICE`, `VRAG_MARGIN_ESIGI`, `YOLO_QUANTIZE`, `VLM_NUM_CTX=4096`, `VLM_NUM_PREDICT=256`, `VLM_IMG_MAX_SIZE=384`, best-moment (`VRAG_MIN_POOL`, `VRAG_RECALL_IQA_GAIN`, `VRAG_MIN/MAX_INTERVAL_S`).

**Hız ayarları** (hepsi tek config değeri, geri alınabilir): `PROC_MAX_WIDTH=1920` (4K→1080p indir, SAHI patch azalt), `VLM_MIN_CROP_POOL_SIZE=6`/`VLM_MIN_HITS_FOR_COLLAGE=5` (VLM erken tetiklenir), `VLM_MIN_RECALL_INTERVAL_S=3.0`/`VLM_RECALL_IQA_GAIN=1.30` (VLM az tekrar → YOLO ile GPU çekişmesi az). **FPS darboğazı: 8GB'da YOLO+SigLIP+qwen3-vl GPU paylaşımı** — 5-10fps arası, videoya/yüke göre değişken.

## Dizin özet

- `src/core/` — pipeline, context_router, visualizer
- `src/detection/slicer.py` — YOLO+SAHI
- `src/tracking/` — tracker, camera_motion, track_state
- `src/vrag/` — engine, embedder, db, ingest, augmentation
- `src/vlm/` — engine, prompts
- `entegrasyon/backend/` — main (FastAPI), pipeline_adapter, sahiplik, ayarlar
- `entegrasyon/web/index.html` — tek web arayüzü
- `LLM/` — operasyonel karar servisi (ayrı venv `LLM/.venv`, kendi `config/`, tests)
- `vllm_servis/` — VLM sunucusunun AYRI venv'i (vllm kendi torch'unu dayatır, ana venv'i kırar). Kod yok, sadece ortam.
- `VLLM_KULLANIM.md` — vLLM kurulum/çalıştırma/VRAM ayarı/sorun giderme
