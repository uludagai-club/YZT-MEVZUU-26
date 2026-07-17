# Referans Görsel Klasörü

Her hava aracı modeli için bir alt klasör açın ve içine o modelin referans
fotoğraflarını (`.jpg`, `.jpeg`, `.png`) koyun. Model başına ~8 fotoğraf idealdir.
İndeksleme sırasında her fotoğraftan augmentation ile ek varyasyonlar üretilir.

## Klasör düzeni

```
data/reference/
├── F_16_Block50/
│   ├── metadata.json
│   ├── f16_ust_01.jpg
│   ├── f16_yan_02.jpg
│   └── ...
├── Bayraktar_TB2/
│   ├── metadata.json
│   └── ...
└── ...
```

## metadata.json

Her model klasörüne bir `metadata.json` koyun (`_ornek_metadata.json`'u kopyalayıp
doldurabilirsiniz):

```json
{
  "model": "F-16 Block 50",
  "kategori": "Savaş Uçağı",
  "ayirt_edici_ozellikler": "Tek motor, tek dikey kuyruk"
}
```

- **model** — insan-okur model adı (arama sonucunda görünür).
- **kategori** — şunlardan biri: `Savaş Uçağı`, `İHA`, `SİHA`, `Yolcu Uçağı`,
  `Drone`. Kategori filtreli arama (`--kategori`) bu alanı kullanır.
- **ayirt_edici_ozellikler** — VLM doğrulama katmanında kullanılacak kısa not.

`metadata.json` yoksa: klasör adı model adı kabul edilir, kategori `bilinmiyor` olur.

## Çekim açısı

Dosya adında `ust` / `yan` geçerse açı otomatik etiketlenir
(ör. `tb2_ust_01.jpg` → `üstten`). Havadan tanımada üstten görseller önemlidir.

## İndeksleme

Fotoğrafları yerleştirdikten sonra proje kökünde:

```
python -m vrag ingest
```
