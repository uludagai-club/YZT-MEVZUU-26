# VRAG — Hava Aracı Tanıma Sistemi: Ne Yaptık, Nasıl Çalışır?

Bu belge projenin **amacını**, **mimarisini**, **veri akışını** ve **teknik
kararlarını** açıklar. Hızlı kurulum/kullanım için [README.md](README.md)'ye bak;
burada işin *nasıl* çalıştığını anlatıyoruz.

---

## 1. Amaç

Teknofest için bir **hava aracı model tanıma** sistemi. Havadan/uzaktan çekilmiş,
kırpılmış bir uçak görüntüsünün **hangi model** olduğunu (F-16, Kaan, Bayraktar
TB2, ANKA...) bulur. Sadece "uçak var" demez; **markasını/modelini** söyler.

Klasik bir sınıflandırıcı (ör. tek bir CNN) yerine **Visual RAG** (Retrieval-
Augmented Generation) yaklaşımı kullanıyoruz: model doğrudan tahmin etmek yerine,
önce bir **referans veritabanından** en benzer adayları bulur, sonra bir görsel-
dil modeli bu **adaylar arasından** seçer. Bu yaklaşımın avantajı:

- **Genişletilebilir:** Yeni bir uçak modeli eklemek = birkaç referans fotoğraf
  eklemek. Modeli yeniden eğitmek gerekmez.
- **Halüsinasyon az:** Son karar hep adaylar arasından çıkar; sistem "olmayan bir
  uçak" uyduramaz.
- **Açıklanabilir:** Hangi referansa benzediğini ve neden seçildiğini gösterir.

---

## 2. Mimari — 3 Katmanlı Boru Hattı

```
   ┌──────────────┐      ┌───────────────────────┐      ┌────────────────────┐
   │  1. TESPİT   │      │   2. RETRIEVAL         │      │   3. DOĞRULAMA     │
   │    (YOLO)    │─crop▶│   (Visual RAG)         │─aday▶│      (VLM)         │
   │ uçağı bul+kır│      │   DINOv2 → Qdrant      │      │  Qwen2.5-VL seçer  │
   └──────────────┘      └───────────────────────┘      └────────────────────┘
     (bu repo dışı)           ◀──── BU REPO ────▶              (yerel/offline)
```

| Katman | Görev | Teknoloji | Durum |
|--------|-------|-----------|-------|
| 1. Tespit | Görüntüde uçağı bulur, kırpar (crop) | YOLO | Repo dışı; crop hazır girdi kabul edilir |
| 2. Retrieval | Crop'a en benzer referansları getirir | DINOv2 + Qdrant | ✅ Çalışıyor |
| 3. Doğrulama | Adaylar arasından nihai kararı verir | Qwen2.5-VL (yerel) | ✅ Çalışıyor |

Bu proje **2. ve 3. katmanı** içerir. YOLO ayrı geliştiriliyor; buraya kırpılmış
görüntü hazır gelir.

---

## 3. Katman Katman: Nasıl Çalışır?

### 3.1 Retrieval (Visual RAG) — "En benzerini bul"

**Adım 1 — Embedding (görüntü → vektör).**
Kırpılan görüntü **DINOv2** (`facebook/dinov2-base`) modelinden geçirilip 768
boyutlu bir sayı vektörüne (embedding) dönüştürülür. Bu vektör, görüntünün
"parmak izi" gibidir: benzer görünen uçaklar benzer vektörler üretir.

> **Neden DINOv2?** Self-supervised (etiketsiz) eğitilmiş bir görüntü modelidir;
> nesnelerin **şekil/silüet** benzerliğini yakalamada çok güçlüdür. Havadan uçak
> tanımada asıl ayırt edici sinyal silüettir (kanat, kuyruk, gövde şekli), o yüzden
> ideal. Vektörler L2-normalize edilip **kosinüs benzerliği** ile karşılaştırılır.

**Adım 2 — Vektör veritabanı (Qdrant).**
Tüm referans görsellerin vektörleri **Qdrant**'a yazılır. Qdrant, "bu vektöre en
yakın N vektör hangileri?" sorusunu çok hızlı cevaplar. Docker'sız, **yerel/embedded
modda** çalışır (`qdrant_db/` klasörü). Her vektörle birlikte **payload** tutulur:

- `model` — uçak modelinin adı
- `kategori` — Savaş Uçağı / Bombardıman Uçağı / İHA / SİHA / Yolcu Uçağı / Drone
- `dosya_yolu` — referans görselin yolu
- `aci` — çekim açısı (üstten/yandan, dosya adından çıkarılır)

> **Neden Qdrant?** Veritabanı küçük (~400 vektör), performans için değil;
> **kategori filtreli arama** ("sadece savaş uçakları arasında ara") ve kolay
> **genişletilebilirlik** için. Yeni model eklemek = yeni vektör + payload.

**Adım 3 — Augmentation (indeksleme sırasında).**
Her referans fotoğraftan tek vektör değil, **birkaç varyasyon** üretilir:
döndürme (±12°), ölçekleme (0.8×) ve hafif bulanıklık (drone kamerası netsizliği).
Böylece sorgu görüntüsü biraz döndürülmüş/bulanık olsa bile eşleşme tutar.
Sonuç: **34 model**, 292 fotoğraf → **1460 vektör** (fotoğraf başına ~5 varyasyon).

**Adım 4 — Model bazında tekilleştirme (aggregation).**
Bir modelin 5 varyasyonu top-5'i doldurmasın diye, arama sonuçları **model
bazında** tekilleştirilir: her model için yalnızca **en iyi skor** alınır. Böylece
top-5 = 5 farklı model olur.

**Sonuç:** Sorgu görseli → model adı + benzerlik skoru + referans görsel içeren
sıralı aday listesi.

### 3.2 Doğrulama (VLM) — "Adaylar arasından seç ve açıkla"

Retrieval genelde doğru modeli üst sıraya koyar ama benzer uçakları (ör. F-16 vs
F-5E) karıştırabilir. Burada bir **görsel-dil modeli (VLM)** devreye girer:
**Qwen2.5-VL** (yerel, internetsiz çalışır).

VLM'e şunlar verilir:
1. Sorgu görseli (kırpılmış uçak)
2. En iyi 3 adayın referans görselleri
3. "Bu görsel hangi adaya ait? Silüete, kanat/kuyruk yapısına, motor sayısına
   bak. **Sadece adaylar arasından** seç." talimatı

VLM adaylardan birini seçer ve **Türkçe bir gerekçe** üretir. Kritik nokta: VLM
sıfırdan tahmin etmez; sadece verilen adaylar arasından seçer → **halüsinasyon
azalır**.

> **Neden yerel/offline model (API değil)?** Teknofest'te canlı çalışırken çoğu
> senaryo internet gerektirmez; sistemin tamamı (DINOv2 + Qdrant + VLM) yerel
> çalışsın istiyoruz. Model **4-bit** (bitsandbytes) yüklenir → 8 GB ekran kartına
> sığar.

---

## 4. Bir Fotoğraf Gönderince Ne Oluyor? (Uçtan Uca Akış)

```
Kullanıcı fotoğrafı yükler
        │
        ▼
[1] DINOv2 ile embedding çıkar  ────────────►  768 boyutlu vektör
        │
        ▼
[2] Qdrant'ta en yakın ~64 vektörü ara
        │
        ▼
[3] Model bazında tekilleştir → top-5 aday   ─►  (model, skor, referans)
        │
        ▼
[4] En iyi 3 adayı + sorgu görselini VLM'e ver
        │
        ▼
[5] Qwen2.5-VL adaylar arasından seçer + gerekçe
        │
        ▼
Sonuç: TAHMİN + güven + açıklama + aday tablosu
```

**Web arayüzünde** bu akış tek tıkla çalışır; **CLI'da** ise
`python -m vrag search <görsel> --dogrula` komutuyla.

---

## 5. Bileşenler (Dosya Haritası)

```
VRAG 0/
├── vrag/                     # Ana Python paketi
│   ├── config.py             # TÜM ayarlar tek yerde (model adları, yollar, eşikler)
│   ├── gomleme.py            # Embedding: soyut arayüz + DINOv2 (batch, GPU/CPU)
│   ├── artirma.py            # Augmentation (döndür / ölçekle / bulanıklaştır)
│   ├── vektor_deposu.py      # Qdrant sarmalayıcı (kur / ekle / kategori filtreli ara)
│   ├── indeksleme.py         # Ingest: referansları tara → embed → Qdrant'a yaz
│   ├── arama.py              # Arama + model bazında tekilleştirme
│   ├── dogrulama.py          # VLM doğrulama (soyut backend + Qwen2.5-VL)
│   └── __main__.py           # CLI (ingest / search)
├── app.py                    # Flask web sunucusu
├── templates/index.html      # Web arayüzü (HTML)
├── static/style.css, script.js  # Arayüz stil + etkileşim
├── veriler/<model>/          # Referans fotoğraflar + metadata.json
├── qdrant_db/                # Qdrant veritabanı (ingest üretir)
└── config / requirements / README
```

### Tasarım vurgusu: değiştirilebilir backend'ler

İki ağır bileşen **soyutlanmıştır**, yani kolayca değiştirilebilir:

- **`GomlemeModeli`** (gomleme.py) — embedding modeli arayüzü. Bugün DINOv2;
  yarın CLIP/SigLIP eklemek bu arayüzü uygulamaktan ibaret.
- **`VLMDogrulayici`** (dogrulama.py) — VLM arayüzü. Bugün yerel Qwen2.5-VL;
  istenirse API tabanlı (Claude/Gemini) bir backend eklenebilir.

Model değiştirmek çoğu zaman `config.py`'de tek satır (`GOMLEME_MODELI` /
`VLM_MODELI`) değiştirmekle olur.

---

## 6. Nasıl Kullanılır?

### Veri hazırlama
Her model için `veriler/<model_adı>/` klasörüne fotoğraflar + bir `metadata.json`
(model, kategori, ayırt edici özellikler). Sonra indeksle:

```bash
python -m vrag ingest --dizin veriler
```

### Komut satırından tanıma
```bash
python -m vrag search "yol\to\crop.jpg" --topk 5 --dogrula
```

### Web arayüzü
```bash
python app.py         # → http://127.0.0.1:5000
```
Fotoğrafı sürükle-bırak → **Tanı** → sonuç.

---

## 7. Önemli Teknik Kararlar (ve Neden)

| Karar | Neden |
|-------|-------|
| **DINOv2** (ColPali değil) | ColPali doküman retrieval modeli; doğal görüntü/silüet benzerliği için yanlış araç. DINOv2 silüette güçlü. |
| **Qdrant embedded** (Docker'sız) | Kurulum kolaylığı; küçük veri; kategori filtresi + genişletilebilirlik. |
| **Augmentation** | Tek referanstan çok varyasyon → döndürme/bulanıklığa dayanıklı eşleşme. |
| **Model bazında tekilleştirme** | Aynı modelin varyasyonları top-k'yı doldurmasın. |
| **Yerel VLM + 4-bit** | Offline çalışsın; 8 GB VRAM'e sığsın. |
| **Modeller sunucuda "sıcak"** | Web'de her istekte yeniden yükleme yok; ilk açılış ~35 sn, sonra istek ~saniyeler. |
| **Türkçe kod + çıktı** | Mevcut proje stiliyle tutarlılık; gerekçeler Türkçe. |

---

## 8. VLM Modeli: 3B → 7B Yolculuğu (Dürüst Notlar)

Doğrulama katmanında önce **Qwen2.5-VL-3B** (küçük, hızlı) denendi. Bulgular:

- **Seçim** işini makul yapıyor (benzer adaylar arasından doğru modeli seçebiliyor).
- Ama **serbest-metin gerekçe** üretemiyor: 5 farklı istem denendi; küçük model
  ya kendine verilen örneği/etiketi **kopyalıyor** ya da tekrara/çöpe düşüyordu.
  Bu bir *kapasite* sınırı, format meselesi değil.

Bu yüzden gerçek **görsele-dayalı gerekçe** için **Qwen2.5-VL-7B**'ye geçildi
(daha iyi talimat izleme + betimleme). 7B, 8 GB karta **4-bit** olarak sığar ve
web sunucusunda DINOv2 ile **birlikte** bellekte kalabilir (~7 GB).

İlk denemede 7B çok yavaştı: **~146 sn/sorgu**. Sebep, 4 görselin yüksek
çözünürlükte işlenmesiydi (görsel token sayısı patlıyor). **Çözünürlük
düşürülünce** (`VLM_MAX_PIKSEL`) çıkarım **~8 sn**'ye indi — *ve doğruluk arttı*:
düşük çözünürlük gereksiz detayı eleyip **silüete** odaklandığı için birbirine
çok benzeyen modelleri (TB2/TB3, Wing Loong I/II) daha iyi ayırdı.

> **Dersler:** (1) Küçük modelde gerekçe kalitesi prompt ile düzelmez — model
> büyütmek gerekir. (2) VLM hızının asıl kaldıracı **görsel çözünürlüğüdür**,
> model boyutu değil. (3) Silüet işi için yüksek çözünürlük gerekmez, hatta
> zarar verebilir.

---

## 9. Bilinen Sınırlar & Sonraki Adımlar

- **Sadece veritabanındaki modeller:** Sistem yeni uçak "üretmez"; yalnızca
  eklenmiş modeller arasından seçer. Yeni model = yeni referans + `ingest`.
- **Çekim açısı önemli:** Referanslar çoğunlukla yandan/çeşitli açılardansa, tam
  tepeden bir sorguda benzerlik düşebilir. İyileştirme: üstten referanslar eklemek.
- **Güven skoru:** Şu an "güven" = seçilen adayın görsel benzerlik skorudur (VLM'in
  kesinlik ölçüsü değil).
- **Sıradaki:** (a) YOLO'yu bağlayıp canlı crop akışı; (b) gerçek zamanlılık
  optimizasyonu; (c) referans setini üstten görsellerle zenginleştirme.

---

*Bu belge projenin mevcut durumunu özetler. Ayarların tümü `config.py`'dedir;
mimari değişiklikler için ilgili modülün başındaki açıklamalara bakılabilir.*
