Aşağıdaki liste, bu oturumda yaptığımız her şeyi kapsıyor. Arkadaşına verirken şunu da ilet: **bir kısmı evrensel bug düzeltmesi, bir kısmı bu makineye (8 GB VRAM, RTX 4070 Laptop) özel ayar.** İkisini ayırdım.

---

## A. Ubuntu ortam kurulumu (kod değil)

| İş | Komut / not |
|---|---|
| uv kur | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| venv | `cd LLM && uv venv --python 3.11` → `LLM/.venv/bin/python` (Windows'taki `Scripts/python.exe` değil). Ubuntu 22.04 deposundaki `python3.11` paketi `3.11.0~rc1`, **kullanma** — uv'nin indirdiği 3.11.15 |
| torch | `uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130` |
| kalan paketler | `uv pip install -r requirements-linux.txt` |
| NVIDIA | 550 ve 580 sürücüleri bir aradaydı → `sudo apt-mark manual nvidia-driver-580 nvidia-dkms-580` **(önce bu, yoksa autoremove 580'i siler)**, sonra 16 adet `-550` paketini purge, sonra reboot |
| Ollama | 0.3.14 `qwen3-vl`'i tanımıyor, pull sessizce başarısız oluyordu → `curl -fsSL https://ollama.com/install.sh \| sh` (0.32.9) |
| SigLIP2 | HF cache taşınmamıştı, ~4.3 GB yeniden indi. Xet CDN hata verirse `HF_HUB_DISABLE_XET=1` |
| `requirements-linux.txt` | BOM + CRLF + içindeki `Using Python 3.11.9 environment at: LLM\.venv` kirli satırı temizlendi (bu satır `uv pip install -r`'ı patlatıyor) |
| `veriler/` | `unzip` Türkçe adları bozmuştu; Python `zipfile` ile yeniden açıldı. **Not:** ZIP cp437 kodlu olduğu için `İ ı ğ ş` zaten kayıptı, geri getirilemedi |
| scriptler | `chmod +x *.sh` |

---

## B. Gerçek bug düzeltmeleri — her ortamda geçerli

**1. `src/vrag/embedder.py` — SigLIP2 fp32 yükleniyordu**

```python
# ESKİ (yükleme tepesi ~4.4 GB)
self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device)
if self._use_half: self.model = self.model.half()

# YENİ (tepe yarıya iner)
if self._use_half:
    self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, dtype=torch.float16).to(self.device)
else:
    self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device)
```
Ölçülen kazanç: backend süreci **3892 → 2640 MiB**. Bu düzeltme olmadan VRAG "CUDA out of memory" alıp **sessizce devre dışı** kalıyordu (sistem çalışır görünüp model tanıma yapmıyor).

**2. `src/core/pipeline.py` — LLM hiç tetiklenmiyordu**

LLM kapısı `process_frame`'in track döngüsündeydi, yani sadece o an *yaşayan* track'ler için bakılıyordu. VLM çağrısı ~50 sn sürdüğü için sonuç döndüğünde track çoktan ölmüş oluyordu (ölçüm: Track 0 02:02:50'de gönderildi, 02:03:41'de döndü, o sırada aktif track 53+). Düzeltme: `_async_vlm_task` içinde `track.vlm_result = json_result` satırından hemen sonra LLM'i **doğrudan tetikle** (`last_llm_vlm_hash` zaten çift çağrıyı engelliyor). Bu aynı zamanda "video bitse bile çıktı gelsin" isteğini de çözer.

**3. `src/tracking/track_state.py` — birleşmede sonuç kayboluyordu**

`merge_from` içinde `vlm_done` taşınıyor ama `vlm_result` taşınmıyordu → LLM kapısı (`vlm_done and vlm_result`) kalıcı kapalı, VLM kapısı (`not vlm_done`) da kapalı → o track için bir daha hiç sonuç üretilemiyordu. `vlm_result`, `llm_result`, `vrag_matches`, `last_llm_vlm_hash`, `is_llm_querying` de devredilecek şekilde eklendi.

**4. `entegrasyon/backend/main.py` — INFO logları kapalıydı**

Backend `src/main.py`'yi hiç import etmediği için kök logger WARNING'de kalıyordu; `[VLM BEKLEMEDE]`, `[LLM] Karar alındı`, `[VRAG] Embedder hazır` gibi teşhis satırları hiç görünmüyordu. `import logging` + kök logger'ı INFO'ya çeken blok eklendi (uvicorn kendi handler'ını eklemiş olsa bile seviye zorlanıyor).

**5. `src/core/pipeline.py` — LLM analyze timeout 30 → 120 sn**

`llama3.2:1b` bilerek CPU'da; VLM de CPU'ya taşınınca çekişmeden karar 30 sn'yi aşıp `Read timed out` veriyordu.

---

## C. vLLM geçişi (VLM Ollama'dan taşındı)

- **Yeni venv:** `vllm_servis/.venv` (Python 3.12, `uv pip install vllm==0.27.1`) — ayrı tutuldu ki mevcut torch'a dokunmasın
- **Yeni dosya `VLLM_Baslat.sh`** — port 8002, OpenAI uyumlu. Zorunlu ayarlar ve sebepleri:
  - `VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer'in JIT sampling çekirdeği çalışma anında `nvcc` arıyor, sistemde CUDA toolkit yok (`Could not find nvcc`). Örnekleme zaten greedy
  - `--mm-processor-kwargs '{"max_pixels": 200704}'` — vLLM bellek profilini **en kötü görsel boyutuna** göre çıkarıyor; sınırsızken KV cache'e yer kalmıyor (`No available memory for the cache blocks`)
  - `--no-enable-log-requests` (0.27'de `--disable-log-requests` kaldırıldı)
- **`src/vlm/engine.py`** — istek gövdesi artık URL'den seçiliyor: `"/v1/" in api_url` → OpenAI formatı (`content` dizisi + `image_url` data-URI + `max_tokens` + `response_format`), değilse eski Ollama formatı. Cevap ayrıştırma zaten `choices`'ı destekliyordu
- **`src/core/pipeline.py`** — VLM ısınma kodu da aynı kuralla iki formatlı yapıldı
- **`Sistemi_Baslat.sh`** — tek komuta indirildi: önce vLLM'i başlatır, `/v1/models` 200 dönene kadar bekler, açılamazsa backend'i **hiç başlatmaz**. Üç tuzak vardı:
  1. `pkill -f "vllm.entrypoints"` öksüz motoru ıskalıyor — süreç kendini **`VLLM::EngineCore`** diye yeniden adlandırıyor
  2. Desen başa sabitlenmeli (`'^VLLM::'`), yoksa script'i çalıştıran kabuğu da öldürüyor
  3. **bash, ön planda çocuk süreç varken sinyal trap'ini erteler** → uvicorn arka plana alınıp `wait` ile beklenmeli, yoksa Ctrl+C'de temizlik hiç çalışmaz ve vLLM 3.8 GB VRAM'i rehin alır
  - Kapanış temizliği **port tabanlı** (PID'e güvenmiyor), `VLLM_ATLA=1` ile Ollama'ya dönülebiliyor

Ölçülen sonuç: **VLM gecikmesi ~50 sn → 3 sn**, VLM %87 CPU'dan %100 GPU'ya.

---

## D. Bu donanıma özel ayarlar — arkadaşının kartına göre değişir

| Ayar | Değer | Sebep |
|---|---|---|
| `VLM_MODEL_NAME` | `cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit` | 4B AWQ **4.13 GB**, 8 GB'a sığmıyor. vLLM Ollama gibi CPU'ya taşıyamaz, sığmazsa OOM verip ölür |
| `VLM_API_URL` | `http://127.0.0.1:8002/v1/chat/completions` | |
| `VLLM_GPU_UTIL` | `0.41` | 0.36'da KV cache **−0.18 GiB** çıkıyor. Ölçülmüş çalışan değer |
| `VLLM_MAX_LEN` | `1536` | |
| `BATCH_SIZE` (SAHI) | 8 → **4** | YOLO aktivasyon tepesi SigLIP2'ye yer bırakmıyordu |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Parçalanma; ~400 MiB kazandırdı |

Son VRAM tablosu: vLLM 3796 + backend 2640 = **6444 / 8188 MiB**. Daha büyük kartta 4B modele çıkılabilir ve `BATCH_SIZE` geri 8 yapılabilir.

---

## E. VRAG doğruluğu (kısmen tamamlandı — durumu net söyle)

**Teşhis (ölçülmüş):** indeks sağlam (self-retrieval **20/20**), ama gerçek video kırpıntılarında top-1 doğruluğu **2/40**. Baskın yanlış cevap hep `Bayraktar KIZILELMA`. Kök sebep **domain gap**: F-35 referansları yakın çekim/yüksek detay, KIZILELMA referansları uzaktan uçuşta siluet — sorgu kırpıntıları da öyle. SigLIP2 şekli değil çekim tarzını eşleştiriyor.

**Denenip elenen fikir:** referanslara "detay kaybı" augmentasyonu. Ölçtüm: F-35'i +0.038 yukarı, KIZILELMA'yı −0.037 aşağı çekiyor **ama orijinaller indekste kaldığı için yetmiyor**. Reindekse girmeden ölçmek iyi oldu — arkadaşın da bu yola sapmasın.

**Uygulanan düzeltmeler:**
- `src/vrag/engine.py` — margin artık **ham kosinüs** üzerinden (eskiden oy terimiyle şişmiş birleşik skordandı: birleşik marj 0.060 vs ham 0.020, eşik 0.015 → kapı hiç açılmıyordu); küçük kırpıntı → `dusuk_guven`; çıktıya gerçek `benzerlik` alanı eklendi
- `src/config.py` — `VRAG_MARGIN_ESIGI` 0.015 → **0.030**, yeni `VRAG_MIN_CROP_PX = 64`
- `src/vlm/engine.py` — prompt'ta artık ham benzerlik gösteriliyor (eskiden ham 0.93 iken VLM'e **"%99"** gidiyordu, VLM de yanlışı onaylıyordu) + `dusuk_guven` durumunda VLM'e açık "VRAG kararsız, kendi gözlemine güven" uyarısı
- `entegrasyon/backend/pipeline_adapter.py` — arayüz de `benzerlik` gösteriyor

**Doğrulama durumu:** çevrimdışı ölçtüm — 40 kırpıntının **33'ü (%82)** artık `dusuk_guven` işaretleniyor (öncesi pratikte sıfır). **Canlı uçtan uca doğrulaması tamamlanmadı.** Bu düzeltmeler "kendinden emin yanlış"ı "belirsiz"e çeviriyor; **doğruluğu artırmıyor**. Gerçek çözüm veri seti işi: karışan sınıflara (F-35, F-16, KAAN, KIZILELMA) uzaktan çekilmiş uçuş fotoğrafları ekleyip reindekslemek.

---

## F. Veri / indeks

`VRAG_Yeniden_Indeksle.sh` çalıştırıldı: 787 silindi, 802 eklendi. **88 → 80 model**, 24839 → 18624 nokta. 8 model düştü çünkü diskte hiç görselleri yok: `KAAN TF-X, TUSAŞ ŞİMŞEK, STM ALPAGU-B, MQ-1 Predator, Su-27 Flanker, Su-35S, T-38 Talon, UH-1 Huey`. Model adları da ASCII'ye döndü (ZIP karakter kaybı). Yedek: `output/qdrant_db_YEDEK_20260813_0209`.

⚠️ Arkadaşın Windows'tan `veriler/`'i **7-Zip veya doğrudan kopya** ile taşısın, ZIP ile değil — yoksa aynı `İ ı ğ ş` kaybını yaşar.

---

## G. Değişen dosyalar (yedekler scratchpad'de `*.bak`)

```
src/vrag/embedder.py            src/vrag/engine.py
src/core/pipeline.py            src/vlm/engine.py
src/config.py                   src/tracking/track_state.py
entegrasyon/backend/main.py     entegrasyon/backend/pipeline_adapter.py
Sistemi_Baslat.sh (yeniden yazıldı)
YENİ: VLLM_Baslat.sh, vllm_servis/
```

Arkadaşın Windows'ta kalacaksa **B bölümündeki 5 düzeltme doğrudan geçerli** (özellikle SigLIP2 fp16, LLM tetikleme ve track birleştirme — bunlar platformdan bağımsız gerçek hatalar). C ve D bölümleri ise Linux + 8 GB VRAM'e özel.