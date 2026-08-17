# Ubuntu'da Kurulum ve Çalıştırma

TEKNOFEST hava aracı tespit/tanıma sistemi (YOLO → VRAG → VLM → LLM) için Ubuntu kurulum rehberi.
Windows'taki çalışan ortamdan **dondurulmuş kesin sürümlerle** kurulum yapılır — `requirements.txt`'lerdeki eski pinlere güvenilmez.

> **Referans çalışan ortam (Windows):** Python 3.11.9 · torch 2.13.0 + CUDA 13.0 · transformers 5.14.1 · numpy 2.4.6 · ultralytics 8.4.115 · qdrant-client 1.19.0

---

## 0. ZIP'i aç

```bash
cd ~/Downloads
sudo apt update && sudo apt install -y unzip
unzip Mevzuu-26.zip
cd Mevzuu-26
# Türkçe klasör adları bozuksa (T├╝rkiye gibi) unzip yerine bsdtar:
#   sudo apt install -y libarchive-tools && bsdtar -xf ../Mevzuu-26.zip
```
Doğrula: `ls "veriler/1. Türkiye öncelikli İHA - SİHA - TİHA - UCAV/"` düzgün görünmeli.

> **Not:** ZIP içinde `.git` ve `LLM/.venv` **bilerek yok**. venv'i aşağıda sıfırdan kuracağız.

---

## 1. Sistem paketleri

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv build-essential lsof curl git
# opencv'nin çalışması için (headless değil normal opencv-python kullanıyoruz):
sudo apt install -y libgl1 libglib2.0-0
```

## 2. NVIDIA sürücü + CUDA doğrula (GPU şart)

Sistem, YOLO + SigLIP2 + VLM'i **GPU'da** çalıştırır (8GB VRAM sıkı). NVIDIA sürücüsü kurulu olmalı:

```bash
nvidia-smi          # sürücü + CUDA sürümünü göster
```
- Çıktı gelmiyorsa sürücü kur: `sudo ubuntu-drivers autoinstall` → yeniden başlat.
- Sağ üstteki **CUDA Version** değerini not et (torch'u buna göre kuracağız).
- Dizüstü + hibrit grafik (Optimus) ise NVIDIA GPU'nun aktif olduğundan emin ol.

## 3. uv (paket yöneticisi) kur

Proje `pip` değil **uv** kullanıyor.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc        # veya yeni terminal aç
uv --version
```

## 4. venv oluştur + bağımlılıkları kur

Tek venv, `LLM/.venv` konumunda, Python 3.11 ile:

```bash
cd ~/Downloads/Mevzuu-26         # proje kökü (kendi yolunla değiştir)
cd LLM && uv venv --python 3.11 && cd ..
source LLM/.venv/bin/activate
```

**Önce torch/torchvision (CUDA sürümüne göre — ayrı index):**
```bash
# nvidia-smi'deki CUDA 13.x ise cu130; farklıysa cu128/cu126 vb. seç (pytorch.org).
uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```
> Tam sürüm (2.13.0+cu130) senin CUDA'nda yoksa, `nvidia-smi`'ye uyan en yakın torch'u kur —
> ufak sürüm farkı sorun olmaz, yeter ki `import torch; torch.cuda.is_available()` **True** dönsün.

**Sonra kalan her şey (kilitli liste, torch'a dokunmaz):**
```bash
uv pip install -r requirements-linux.txt
```

**Doğrula:**
```bash
python -c "import torch, ultralytics, transformers, qdrant_client; print('torch cuda:', torch.cuda.is_available())"
```
`torch cuda: True` görmelisin.

## 5. Ollama kur + LLM modelini çek

Ollama artık **yalnızca LLM** için gerekli (port 11434). VLM vLLM'e taşındı — bkz. Adım 5b.

```bash
curl -fsSL https://ollama.com/install.sh | sh   # systemd servisi olarak kurulur, otomatik başlar
ollama pull llama3.2:1b                          # LLM (kodda CPU'ya sabitli, num_gpu:0)
ollama list
```

## 5b. vLLM kur (VLM servisi, port 8002)

VLM kendi venv'inde durur — vLLM kendi torch sürümünü dayattığı için ana
`LLM/.venv`'i kırar, o yüzden ayrı:

```bash
cd vllm_servis
uv venv --python 3.12
uv pip install vllm
cd ..
```

Model (`cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit`, ~2 GB) ilk çalıştırmada
Hugging Face'ten iner. `./Sistemi_Baslat.sh` vLLM'i kendisi başlatır.

📄 Ayrıntı, VRAM ayarı ve sorun giderme: **[VLLM_KULLANIM.md](VLLM_KULLANIM.md)**

## 6. Çalıştır

Script'lere çalıştırma izni ver (bir kez):
```bash
chmod +x Sistemi_Baslat.sh VRAG_Yeniden_Indeksle.sh
```

Sistemi başlat:
```bash
./Sistemi_Baslat.sh
```
- Eski 8000/8001/8002 süreçlerini kapatır; sırayla vLLM'i (8002), backend'i (8000) başlatır. Backend kendi içinde LLM'i (8001) başlatır.
- vLLM'in ilk açılışı **~90 sn** sürer (ağırlık + CUDA graph). Log: `output/vllm.log`.
- Hazır olunca tarayıcıda aç: **http://127.0.0.1:8000/goruntule/**
- Durdurmak: **Ctrl+C** (backend'i kapatır; 8001 alt-süreci de birlikte kapanır).

## 7. (Opsiyonel) VRAG yeniden indeksleme

`veriler/` klasörüne görsel ekleyip/çıkardıysan (bkz. `DENGE_PLANI.md`):
```bash
# ÖNCE backend'i durdur (Ctrl+C) — ikisi aynı Qdrant dosyasını kilitler.
./VRAG_Yeniden_Indeksle.sh
```
Artımlı çalışır: yeni görselleri ekler, silinenleri çıkarır. Full rebuild gerekmez.

---

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| `./Sistemi_Baslat.sh: bad interpreter: ^M` | CRLF satır sonu. `sudo apt install dos2unix && dos2unix *.sh` (veya `sed -i 's/\r$//' *.sh`) |
| `Permission denied` (script) | `chmod +x *.sh` |
| `torch.cuda.is_available()` **False** | NVIDIA sürücü yok/uyumsuz. `nvidia-smi` kontrol; torch'u doğru cuXXX index'ten kur |
| `lsof: command not found` | `sudo apt install -y lsof` |
| Türkçe klasör adları bozuk | ZIP'i `bsdtar -xf` ile aç (Adım 0); `locale` çıktısı UTF-8 olmalı |
| Port 8000/8001 takılı kaldı | `lsof -ti:8000 -ti:8001 \| xargs -r kill -9` |
| `libGL.so.1` hatası (opencv) | `sudo apt install -y libgl1 libglib2.0-0` |
| Ollama modeli bulunamadı | `ollama pull llama3.2:1b` |
| Port 8002 takılı / VRAM dolu kaldı | Öksüz `VLLM::EngineCore`. `lsof -ti tcp:8002 \| xargs -r kill -9` ve `pkill -f '^VLLM::'` |
| `No available memory for the cache blocks` | `VLLM_GPU_UTIL`'i artır (0.41 → 0.45). Bkz. VLLM_KULLANIM.md |
| `Could not find nvcc` | flashinfer JIT; script `VLLM_USE_FLASHINFER_SAMPLER=0` ile kapatıyor, zararsız |
| VLM'den hiç cevap yok | `VLM_API_URL` içinde `/v1/` var mı? Yoksa Ollama formatı gönderilir |

## VRAM notu

8 GB'da ölçülen bütçe: vLLM ~3.8 GB + backend (YOLO+SigLIP2) ~2.7 GB = **~6.5 / 8.1 GB**.
Ubuntu'da masaüstü ortamı da biraz VRAM yer — takılırsa `nvidia-smi` ile bak, gerekirse hafif
bir masaüstü oturumu (veya headless) kullan.

⚠️ vLLM, Ollama'dan farklı olarak VRAM yetmezse CPU'ya **taşımaz** — doğrudan OOM verip çöker.
Bu yüzden `VLLM_GPU_UTIL` elle ayarlanır; 0.41 ölçülmüş çalışan değerdir (0.36'da KV cache
açılmıyordu). Ayrıntı: VLLM_KULLANIM.md
