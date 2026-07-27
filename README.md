# VRAG — Hava Aracı Tanıma (Visual RAG)

Teknofest için **görsel retrieval** sistemi. Havadan çekilmiş, kırpılmış bir uçak
görüntüsünün **hangi model** olduğunu (F-16, Kaan, B-2, ANKA...) bulur. Modeli
yeniden eğitmeden, referans veritabanından **en benzer** kaydı bulur → yeni uçak
eklemek = birkaç fotoğraf + yeniden indeksleme.

Bu, projenin **sade/optimal** sürümüdür: tek encoder (**SigLIP2-so400m**),
retrieval-only, perspektif augmentation + güven kapısı açık.

## Mimari — 2 katmanlı boru hattı

```
   ┌──────────────┐     ┌────────────────────────────────┐
   │  1. TESPİT   │     │   2. RETRIEVAL                 │
   │    (YOLO)    │crop▶│   SigLIP2 → Qdrant             │
   │ uçağı kırpar │     │  en benzer top-k model         │
   └──────────────┘     └────────────────────────────────┘
     (repo dışı)             ◀──────── BU REPO ────────▶
```

Görüntü **SigLIP2-so400m** ile vektöre çevrilir, **Qdrant**'ta en benzer
referanslar bulunur, model bazında tekilleştirilir. **Nihai tahmin = top-1**
(en yüksek benzerlik); geri kalan adaylar sıralı gösterilir.

**Neden VLM yok?** Önceki sürümde adayları bir görsel-dil modeline (Qwen2.5-VL)
seçtiren 3. katman denendi; ölçümde güçlü retrieval'ı **bozdu** (top-1 %88.9→%64.4)
ve ~100× yavaştı. Retrieval-only hem daha isabetli hem çok daha hızlı.

## Doğruluk

Leave-one-out (67 model, 586 sorgu): **top-1 %87.9 · top-3 %97.6** (perspektif
augmentation açık). SigLIP2-so400m, ölçtüğümüz 11 encoder içinde en iyisi —
kontrastif modeller self-supervised DINOv2'yi açık ara geçti.

**Güven kapısı:** ilk iki aday çok yakınsa (margin < `MARGIN_ESIGI`=0.015) tahmin
"düşük güven" işaretlenir; ölçümde yanlış tahminlerin ~%86'sını yakalar.

## Kurulum

Python 3.12+ ve CUDA'lı bir GPU (8 GB rahat yeter) gerekir.

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
CUDA'lı torch: `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
Encoder ilk kullanımda HuggingFace'ten iner (~1.6 GB).

## Kullanım

```bash
python -m vrag ingest                 # veriler/ → Qdrant indeksi (ilk kez şart)
python app.py                         # web arayüzü → http://127.0.0.1:5000
python -m vrag search crop.jpg        # komut satırı (top-k + güven)
```
Web arayüzü: fotoğraf yükle → **Tanı**. *Benchmark Çalıştır* doğruluğu ölçer;
*Yeniden İndeksle* veriyi değiştirince indeksi kurar.

## Veri

`veriler/<kategori>/<model>/` altına fotoğraflar + her model klasörüne bir
`metadata.json`:

```json
{
  "model": "F-16 Fighting Falcon",
  "kategori": "Savaş Uçağı",
  "ayirt_edici_ozellikler": "Tek motor, tek dikey kuyruk, karın hava alığı",
  "ulke": "ABD",
  "uretici": "Lockheed Martin",
  "rol": "Çok Rollü Avcı",
  "alternatif_adlar": ["Viper"],
  "motor_sayisi": 1,
  "silahli": true
}
```

Şu an **67 model, 586 görsel** → augmentation ile **3516 vektör**. `ulke`/`rol`
sonuç kartında gösterilir ve **arayüzden filtrelenebilir**; `alternatif_adlar`
kopya modelleri tespit etmeye yarar (ör. Kaan = TF-X = MMU).

> **Üstten referans (asıl isabet kazancı):** sorgu havadan, referanslar çoğu
> yandan. Bir modele **üstten fotoğraf** eklemek = klasörüne koy + `python -m vrag
> ingest`. Havadan sorguda isabeti en çok bu artırır.

## Proje yapısı

| Dosya | İş |
|-------|-----|
| `vrag/config.py` | Tüm ayarlar (encoder, Qdrant, augmentation, güven kapısı) |
| `vrag/gomleme.py` | SigLIP2 encoder (görüntü → vektör) |
| `vrag/artirma.py` | Augmentation (döndürme / ölçek / blur / perspektif) |
| `vrag/vektor_deposu.py` | Qdrant sarmalayıcı |
| `vrag/indeksleme.py` | Ingest: tara → augment → embed → Qdrant |
| `vrag/arama.py` | Retrieval + tekilleştirme + güven kapısı |
| `vrag/degerlendirme.py` | Leave-one-out benchmark |
| `vrag/__main__.py` | `ingest` / `search` CLI |
| `app.py` + `templates/` + `static/` | Flask web arayüzü |

## Sonraki adımlar
- **YOLO bağlanması** — canlı crop'ların doğrudan hatta beslenmesi.
- **Üstten referans görselleri** — havadan sorgu/yandan referans uçurumunu kapatır.
- **Gerçek OOD** — DB'de hiç olmayan uçağı reddetme (güven kapısının ötesinde).
