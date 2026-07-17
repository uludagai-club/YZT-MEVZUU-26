# VRAG — Hava Aracı Tanıma (Visual RAG)

Teknofest hava aracı tanıma sistemi için **görsel retrieval hattı**. Havadan
çekilmiş, kırpılmış (crop) bir uçak görüntüsünün **hangi model** olduğunu
(F-16, Bayraktar TB2, A320 vb.) referans veritabanıyla karşılaştırarak bulur.

## Mimari — 3 Katmanlı Boru Hattı

```
   ┌──────────────┐      ┌───────────────────────┐      ┌────────────────────┐
   │  1. TESPİT   │      │   2. RETRIEVAL         │      │   3. DOĞRULAMA     │
   │    (YOLO)    │─crop▶│   (Visual RAG)         │─aday▶│      (VLM)         │
   │              │      │   DINOv2 → Qdrant      │      │  adaylar arasından │
   │ uçağı bulur  │      │   top-k benzer referans│      │  seçer + gerekçe   │
   └──────────────┘      └───────────────────────┘      └────────────────────┘
      (bu repo dışı)          ◀── BU REPO ──▶                (Qwen2.5-VL, yerel)
```

1. **Tespit (YOLO):** Havadan görüntüde uçağı bulur ve kırpar. *Bu repo dışında*
   geliştiriliyor; buradaki hat crop'u hazır girdi kabul eder.
2. **Retrieval (Visual RAG):** Crop, **DINOv2** ile embedding'e çevrilir;
   **Qdrant**'taki referans görseller arasından en benzer top-k aday bulunur.
3. **Doğrulama (VLM):** Yerel/offline **Qwen2.5-VL** adaylar arasından
   karşılaştırmalı seçim yapar (sıfırdan tahmin etmez → halüsinasyon azalır);
   Türkçe gerekçe döndürür.

## Kurulum

Python 3.12+ gerekir.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

**GPU (CUDA):** `torch`'u CUDA yapılı kurun (yoksa otomatik CPU'ya düşer):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

DINOv2-base ilk çalıştırmada HuggingFace'ten iner (~350 MB).

## Veri Hazırlığı

Referans fotoğraflarını `data/reference/<model>/` altına koyun ve her klasöre
bir `metadata.json` ekleyin. Ayrıntı: [data/reference/README.md](data/reference/README.md).

```
data/reference/
├── F_16_Block50/
│   ├── metadata.json      → {"model": "...", "kategori": "...", "ayirt_edici_ozellikler": "..."}
│   ├── f16_ust_01.jpg
│   └── ...
└── Bayraktar_TB2/ ...
```

## Kullanım

**İndeksleme** (referansları Qdrant'a yazar; her çalıştırmada sıfırdan kurar):

```bash
python -m vrag ingest
```

**Arama** (tek crop → top-k aday):

```bash
python -m vrag search yol\to\crop.jpg
python -m vrag search crop.jpg --topk 3 --kategori "Savaş Uçağı"
python -m vrag search crop.jpg --dogrula      # Qwen2.5-VL ile nihai kararı da verir
```

Örnek çıktı:

```
Sorgu: crop.jpg

#  Model          Skor    Kategori     Açı      Referans
-  -------------  ------  -----------  -------  ----------------------------------
1  F-16 Block 50  0.9123  Savaş Uçağı  üstten   data\reference\F_16_Block50\...jpg
2  Bayraktar TB2  0.7440  SİHA         üstten   data\reference\Bayraktar_TB2\...jpg
```

> Sonuçlar **model bazında tekilleştirilir**: aynı modelin farklı varyasyonları
> top-k'yı doldurmaz; her model için en iyi skor gösterilir.

## Proje Yapısı

| Dosya | Sorumluluk |
|-------|-----------|
| `vrag/config.py` | Tüm sabitler ve yollar (model, Qdrant, augmentation, arama). |
| `vrag/gomleme.py` | Embedding: soyut arayüz + DINOv2 (batch, GPU/CPU, önbellekli). |
| `vrag/artirma.py` | Augmentation: döndürme / ölçekleme / blur varyasyonları. |
| `vrag/vektor_deposu.py` | Qdrant sarmalayıcı (embedded mod, kategori filtreli arama). |
| `vrag/indeksleme.py` | Ingest: tara → augment → embed → Qdrant (idempotent). |
| `vrag/arama.py` | Arama + model bazında tekilleştirme. |
| `vrag/dogrulama.py` | VLM doğrulama (Qwen2.5-VL) — soyut backend + Qwen; adaylar arasından karşılaştırmalı seçim. |
| `vrag/__main__.py` | `ingest` / `search` komutları için CLI. |

## Yapılandırma

Önemli ayarlar `vrag/config.py` içinde:

- `GOMLEME_MODELI` — embedding modeli (varsayılan `facebook/dinov2-base`).
- `VARSAYILAN_TOPK`, `HAM_ARAMA_LIMITI` — arama davranışı.
- `DONME_ACILARI`, `OLCEK_FAKTORLERI`, `BULANIKLIK_YARICAPI` — augmentation.
- `VLM_MODELI`, `VLM_ADAY_SAYISI`, `VLM_4BIT`, `VLM_MAX_PIKSEL` — VLM doğrulama.
  Varsayılan 4-bit (8 GB VRAM için, bitsandbytes); >12 GB GPU'da `VLM_4BIT=False` (bf16).

Embedding modeli değiştirilebilir: `gomleme.py` içindeki `GomlemeModeli` arayüzünü
uygulayan yeni bir sınıf yazıp `gomleme_modeli_al()` fabrikasında döndürmek yeterli
(ileride CLIP/SigLIP hibrit için).

## Sonraki Fazlar

- **VLM ince ayarı** — istem/çıktı kalibrasyonu; gerekirse 7B modele veya API
  backend'e geçiş (`dogrulama.py` içindeki `VLMDogrulayici` arayüzü değiştirilebilir).
- **YOLO bağlanması** — canlı crop'ların doğrudan aramaya beslenmesi.
- **Gerçek zamanlılık** — video akışı için optimizasyon (şimdilik kapsam dışı).
