# Veri Seti Denge Planı (`veriler/`)

**Karar:** Taban **20** (her model ≥ 20 görsel). Çıkarma = **sadece kalite** (büyük modelleri tavana indirmek yok; oy sapması gerekirse kodla damperlenir).
**Durum (2026-08-12):** 82 model, 2137 görsel, min 6 – max 59, medyan 30.
**Hedefe eklenecek:** 31 model, ~**315 görsel**.

> **Neden?** Aç model iki kere zarar verir: (1) az açı çeşitliliği → zayıf retrieval, (2) oy-tabanlı skorda az oy → büyük modeller lehine sapma. Balance, geçen eklediğimiz vote skorunu adil yapar.
>
> **Not:** F-16/F-35, HÜRKUŞ/HÜRJET karışması bir **veri sorunu değil** (zaten 39-59 görselleri var) — SigLIP2 ince-ayrım limiti. Bunlara görsel eklemek karışmayı çözmez, boşuna emek.

---

## Ekleme kalite kuralları (retrieval için kritik)

- **Çeşitli açı ekle**, near-duplicate değil: ön / yan / arka / alt / manevra / farklı irtifa. (Augmentation zaten küçük döndürme-parlaklık oynamalarını üretiyor — sen **gerçek bakış açısı** çeşitliliği kat.)
- Doğru **varyant**: örn. F-16 Block'ları, Phantom vs Terminator karıştırma. Yanlış model karede olmasın.
- Makul çözünürlük (minik/bulanık ekleme), temiz + **çeşitli arka plan**.
- Filigran/logo ağırlıklı, kolaj, çizim/render yerine gerçek fotoğraf tercih.

## Çıkarma (sadece kalite)

Sil: near-duplicate, yanlış etiketli / karede yanlış uçak, minik-bulanık, ağır filigranlı, çizim/oyun render'ı.
**Büyük modeli sırf sayı için kesme.** (Oy sapması kalırsa `engine.py`'de oyu sınıf boyutuna normalize ederiz — iyi veri atmaktan iyi.)

---

## Alışveriş listesi (kutu işaretle)

### >>> ÖNCELİK 1 — Türk İHA/SİHA (Kat-1, TEKNOFEST önceliği)
- [ ] STM KARGU — 7 → +13
- [ ] STM TOGAN — 7 → +13
- [ ] Bayraktar DİHA — 8 → +12
- [ ] Tusaş ANKA — 8 → +12
- [ ] Bayraktar TB3 — 9 → +11
- [ ] STM ALPAGU-B — 9 → +11
- [ ] Vestel KARAYEL — 9 → +11
- [ ] TUSAŞ ŞİMŞEK — 10 → +10

### >>> ÖNCELİK 2 — Yabancı İHA (Kat-8, komple aç)
- [ ] Orion UAV — 8 → +12
- [ ] Neuron UCAV — 9 → +11
- [ ] Taranis UCAV — 9 → +11
- [ ] Wing Loong I — 9 → +11
- [ ] Forpost-R — 10 → +10
- [ ] Wing Loong II — 10 → +10
- [ ] X-47B — 10 → +10
- [ ] CH-4 — 12 → +8
- [ ] Hermes 450 — 12 → +8
- [ ] Hermes 900 — 13 → +7
- [ ] MQ-9 Reaper — 17 → +3
- [ ] WZ-7 Soaring Dragon — 19 → +1

### >>> ÖNCELİK 3 — Helikopter (Kat-6)
- [ ] AS532 Cougar — 6 → +14
- [ ] AH-1 Super Cobra — 7 → +13
- [ ] T625 GÖKBEY — 10 → +10

### >>> ÖNCELİK 4 — Nakliye/Özel görev (Kat-4)
- [ ] Airbus C-295W — 7 → +13
- [ ] Airbus A330-243MRTT — 8 → +12
- [ ] P-8A Poseidon — 8 → +12
- [ ] casa cn-235 — 10 → +10

### >>> Jet boşlukları (Kat-2 / Kat-3)
- [ ] NF-5 Türk Yıldızları — 8 → +12
- [ ] F-5E Tiger 2 Swiss Air Force — 9 → +11
- [ ] MiG-35 — 10 → +10
- [ ] F-4E Phantom II Terminator — 17 → +3

---

## İş akışı (sırayı BOZMA)

1. **Önce çıkar** (kalite): kötü görselleri sil. → Bu sayıları düşürür, o yüzden önce.
2. **Sonra ekle**: yeni görselleri ilgili model klasörüne at. Ad önemsiz — sıradaki numara ver, en son toparlanır.
3. **Numaraları toparla**: `renumber.py --apply` (her klasör tekrar 1..N olur).
4. **Backend'i KAPAT** (8000). Qdrant dosya kilidi çakışmasın.
5. **`VRAG_Yeniden_Indeksle.bat`'i BİR KEZ çalıştır.** Artımlı: yeni görselleri embed eder, silinenleri indeksten çıkarır, numaralandırma desync'ini de düzeltir. (Full rebuild GEREKMİYOR.)
6. **Backend'i başlat** (`Sistemi_Baslat.bat`), birkaç modeli spot-test et.
7. **Denge teyidi**: sayım scriptini tekrar çalıştır → min ≥ 20 mi?

## Sonra (opsiyonel)

- Denge sonrası hâlâ büyük-model sapması varsa: `src/vrag/engine.py`'de oy katkısını sınıf boyutuna normalize et (kod değişikliği, veri atmadan).
