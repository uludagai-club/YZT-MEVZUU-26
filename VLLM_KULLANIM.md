# vLLM (VLM Servisi) — Kurulum ve Kullanım

Görsel dil modeli (VLM) artık **Ollama'da değil, vLLM'de** çalışıyor.
Port **8002**, OpenAI-uyumlu API.

> `README.md` ve `KURULUM_UBUNTU.md`'nin VLM'i Ollama'da anlatan bölümleri
> bu geçişten önce yazılmıştı. VLM için geçerli olan bu dosyadır.
> **Ollama hâlâ gerekli** — ama sadece LLM (`llama3.2:1b`, CPU) için.

---

## Neden vLLM?

| | Ollama (eski) | vLLM (şimdi) |
|---|---|---|
| Model | `qwen3-vl:4b-instruct` | `cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit` |
| Ağırlık | 4B, ~3.5 GB | 2B, 4-bit AWQ |
| Kare başına VLM | ~50 sn | ~3 sn |
| VRAM taşarsa | CPU'ya böler (yavaşlar) | **OOM verir, çöker** |

Son satır kritik: Ollama sığmayınca sessizce CPU-split'e düşüp yavaşlıyordu.
vLLM bunu yapamaz — sığmazsa doğrudan patlar. Bu yüzden VRAM bütçesi elle
ayarlanır (aşağıya bak).

---

## Kurulum

vLLM **kendi venv'inde** durur (`vllm_servis/.venv`), ana `LLM/.venv`'den ayrı.
Sebep: vLLM kendi torch sürümünü dayatıyor ve ana ortamdaki
torch/ultralytics/transformers üçlüsünü kırıyor.

```bash
cd vllm_servis
uv venv --python 3.12
uv pip install vllm
cd ..
```

İlk çalıştırmada model Hugging Face'ten iner (~2 GB, bir kerelik).
Sonrası tamamen offline.

---

## Çalıştırma

### Normal yol — hiçbir şey yapma

`./Sistemi_Baslat.sh` vLLM'i **kendisi başlatır** (sıra: vLLM 8002 → backend 8000
→ LLM 8001). Backend açılırken VLM'i ısıtmaya çalıştığı için vLLM önce gelmek
zorunda. Log: `output/vllm.log`.

İlk açılış **~90 saniye** sürer (ağırlık yükleme + CUDA graph derleme).
Script `/v1/models` uçuna sorarak hazır olmasını bekler, en fazla 300 sn.

### Tek başına başlatmak

```bash
./VLLM_Baslat.sh
```

Ortam değişkenleriyle ayarlanır:

| Değişken | Varsayılan | Ne işe yarar |
|---|---|---|
| `VLLM_MODEL` | `cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit` | Model kimliği |
| `VLLM_PORT` | `8002` | Dinlenecek port |
| `VLLM_GPU_UTIL` | `0.41` | Kartın **toplam** belleğinin oranı |
| `VLLM_MAX_LEN` | `1536` | Azami bağlam uzunluğu |

```bash
VLLM_GPU_UTIL=0.45 ./VLLM_Baslat.sh
```

### Ollama'ya geri dönmek

```bash
VLLM_ATLA=1 ./Sistemi_Baslat.sh
```
Bu tek başına **yetmez** — `src/config.py`'de şunlar da eski değerlerine
çevrilmeli:
```python
VLM_API_URL    = "http://localhost:11434/api/generate"
VLM_MODEL_NAME = "qwen3-vl:4b-instruct"
```

---

## Kod tarafı bağlantısı

`src/config.py`:
```python
VLM_API_URL    = "http://127.0.0.1:8002/v1/chat/completions"
VLM_MODEL_NAME = "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit"
```

`src/vlm/engine.py` hangi protokolü konuşacağını **URL'den anlıyor**:

```python
openai_uyumlu = "/v1/" in self.api_url
```

Yani adreste `/v1/` geçiyorsa OpenAI sohbet formatı, geçmiyorsa Ollama
`/api/generate` formatı kullanılır. Ayrı bir "backend seç" ayarı yok —
**adresi değiştirmek protokolü de değiştirir.** URL'den `/v1/`'i düşürürsen
vLLM'e Ollama formatı gönderilir ve istek sessizce başarısız olur.

---

## VRAM bütçesi (8 GB'da sıkı)

Aynı kartı üç şey paylaşıyor:

| Bileşen | Yaklaşık |
|---|---|
| vLLM (VLM) | ~3.8 GB |
| Backend (YOLO + SigLIP2) | ~2.7 GB |
| **Toplam** | **~6.5 / 8.1 GB** |

`VLLM_GPU_UTIL=0.41` rastgele değil, **ölçülmüş** bir değer:
0.36'da KV cache'e yer kalmıyordu (`-0.18 GiB` açık veriyordu).
Yükseltirsen backend'e yer kalmaz, düşürürsen vLLM KV cache açamaz.

Bir de `--mm-processor-kwargs '{"max_pixels": 200704}'` var (448×448). vLLM
bellek profilini **en kötü senaryodaki görüntü boyutuna** göre çıkarıyor;
sınır koymazsan profil tepesi tüm bütçeyi yiyip KV cache'e yer bırakmıyor
("No available memory for the cache blocks"). Biz zaten 384px kolaj
gönderiyoruz, 448 fazlasıyla yeterli.

Sıralama önemli: önce backend'i başlatıp VRAM'i ölç, sonra `VLLM_GPU_UTIL`'i
ayarla.

---

## Sorun giderme

| Belirti | Sebep / çözüm |
|---|---|
| `No available memory for the cache blocks` | `VLLM_GPU_UTIL`'i kademeli artır (0.41 → 0.45). Backend kapalıyken dene. |
| `CUDA out of memory` | Toplam bütçe aşıldı. Backend'i kapat, `nvidia-smi` ile boş VRAM'e bak. |
| `Could not find nvcc` | flashinfer JIT ile CUDA derlemeye çalışıyor, sistemde toolkit yok (sadece sürücü). `VLLM_Baslat.sh` bunu `VLLM_USE_FLASHINFER_SAMPLER=0` ile kapatıyor — örnekleme zaten greedy, flashinfer gerekmiyor. |
| Kapattım ama VRAM dolu | **Öksüz `VLLM::EngineCore`.** vLLM kendini `exec` ile python'a çeviriyor, EngineCore ayrı süreç olup adını değiştiriyor; tek PID'i öldürmek yetmiyor. Temizlik: `lsof -ti tcp:8002 \| xargs -r kill -9` ve `pkill -f '^VLLM::'` |
| VLM'den hiç cevap yok | `VLM_API_URL`'de `/v1/` var mı? Yoksa Ollama formatı gönderilir. |
| 90 sn geçti hâlâ hazır değil | `tail -50 output/vllm.log` |

---

## Doğrulama

Ayakta mı:
```bash
curl -s http://127.0.0.1:8002/v1/models | head -c 300
```

Gerçekten cevap veriyor mu:
```bash
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit",
       "messages":[{"role":"user","content":"1+1 kac?"}],
       "max_tokens":16}'
```

Üç servisin hepsi:
```bash
curl -s -o /dev/null -w "vLLM    8002: %{http_code}\n" http://127.0.0.1:8002/v1/models
curl -s -o /dev/null -w "backend 8000: %{http_code}\n" http://127.0.0.1:8000/goruntule/
curl -s -o /dev/null -w "LLM     8001: %{http_code}\n" http://127.0.0.1:8001/docs
```
Üçü de `200` dönmeli. Not: `http://127.0.0.1:8000/` **404 döner** — bu normal,
kök adres tanımlı değil, sağlık kontrolü `/goruntule/` ile yapılır.
