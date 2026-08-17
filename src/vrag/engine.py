import json
import logging
import threading
from pathlib import Path

import numpy as np
from src.vrag.embedder import VisualEmbedder
from src.vrag.db import QdrantManager
from src.config import VRAG_MIN_SCORE, VRAG_MARGIN_ESIGI, VRAG_MIN_CROP_PX, VRAG_VOTE_TOPK, VRAG_VOTE_WEIGHT

log = logging.getLogger(__name__)

VERI_DIZINI = Path(__file__).resolve().parent.parent.parent / "veriler"


def _metadata_onbellegi_olustur() -> dict[str, dict]:
    """model adı → {ulke, uretici, rol} eşlemesi (veriler/**/metadata.json'dan).

    BUG-FIX: ingest.py, indeksleme sırasında Qdrant payload'ına yalnızca
    model/class/source_file/variation yazıyordu — ulke/uretici/rol hiç
    depolanmıyordu (kaynak metadata.json'da var olsa bile). Bu, arama
    sonuçlarının hep "Bilinmiyor" dönmesine yol açıyordu. Saatler süren
    yeniden indeksleme yerine, arama anında model adına göre metadata'yı
    diskten okuyup zenginleştiriyoruz.
    """
    onbellek: dict[str, dict] = {}
    if not VERI_DIZINI.is_dir():
        return onbellek
    for meta_path in VERI_DIZINI.rglob("metadata.json"):
        try:
            veri = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        model = veri.get("model")
        if model:
            onbellek[model] = {
                "ulke": veri.get("ulke", "Bilinmiyor"),
                "uretici": veri.get("uretici", "Bilinmiyor"),
                "rol": veri.get("rol", "Bilinmiyor"),
            }
    return onbellek


class VRAGEngine:
    def __init__(self):
        log.info("[VRAG] VRAG Engine başlatılıyor...")
        self.embedder = VisualEmbedder()
        self.db = QdrantManager(vector_size=self.embedder.vector_size)
        # Qdrant local/embedded client thread-safe DEĞİL. VRAG'ın iki çağıranı
        # (pipeline'ın bağımsız VRAG task'ı + VLM-içi bağlam araması) farklı
        # thread'lerden aynı client'a erişince embedded Qdrant kilitlenebiliyor.
        # Bu kilit tüm arama+embedding erişimini kaynağın yanında serileştirir —
        # böylece VRAG artık yavaş VLM/Ollama kilidine (pipeline._ai_gate) bağımlı
        # değil, sadece kendisiyle (hızlı, CPU) serileşir.
        self._search_lock = threading.Lock()
        self._metadata_onbellegi = _metadata_onbellegi_olustur()
        log.info(f"[VRAG] {len(self._metadata_onbellegi)} model için metadata önbelleğe alındı.")

    def search_similar_vehicle(self, image: np.ndarray, top_k: int = 2) -> list:
        """
        Hedef kameradan gelen kırpılmış (crop) görüntüyü VRAG DB'de arar
        ve eşleşmeleri döndürür.

        Güven kapısı: top-1 ile top-2'nin (FARKLI modellerin) skor farkı
        (margin) VRAG_MARGIN_ESIGI'den küçükse en iyi eşleşmeye
        "margin"/"dusuk_guven" alanları eklenir. Margin, model bazında
        tekilleştirilmiş sonuçlar üzerinden hesaplanır — aynı modelin 8
        augmentation varyasyonu top-2'yi doldurup yanlışlıkla "kararlı"
        (büyük margin) izlenimi vermesin diye.
        """
        try:
            # Qdrant embedded thread-safe olmadığı için tüm embedding+arama
            # erişimi tek kilit altında serileşir (bkz. __init__ açıklaması).
            with self._search_lock:
                emb = self.embedder.embed_image(image)
                # Oy sayımı + ortalama için bol ham sonuç çek (her modelin ~24
                # referansı ve augmentasyonları var — tutarlılığı ölçmek için
                # yeterince derin bakmalı).
                ham_limit = max(VRAG_VOTE_TOPK * 3, top_k * 16, 64)
                results = self.db.search(query_vector=emb, limit=ham_limit)

            if not results:
                return []

            # İlk-K ham hit içindeki model dağılımı (oy sayımı).
            topk_modeller = [
                h.payload.get("model", "Bilinmiyor") for h in results[:VRAG_VOTE_TOPK]
            ]

            # Model bazında TÜM skorları ve sınıfı topla.
            grup: dict[str, dict] = {}
            for hit in results:
                model = hit.payload.get("model", "Bilinmiyor")
                g = grup.get(model)
                if g is None:
                    grup[model] = {
                        "scores": [float(hit.score)],
                        "class": hit.payload.get("class", "Bilinmiyor"),
                    }
                else:
                    g["scores"].append(float(hit.score))

            # Birleşik skor: (en iyi 3 hit'in ortalaması) + oy oranı katkısı.
            # Tek şanslı yüksek-hit'e kanmaz; sorgunun TUTARLI olarak en yakın
            # olduğu modeli öne çıkarır (benzer uçaklarda kritik).
            adaylar = []
            for model, g in grup.items():
                sk = sorted(g["scores"], reverse=True)
                max_skor = sk[0]
                mean3 = float(np.mean(sk[:3]))
                oy = topk_modeller.count(model)
                birlesik = mean3 + VRAG_VOTE_WEIGHT * (oy / VRAG_VOTE_TOPK)
                meta = self._metadata_onbellegi.get(model, {})
                adaylar.append({
                    "model": model,
                    "class": g["class"],
                    "score": round(birlesik, 4),
                    "ulke": meta.get("ulke", "Bilinmiyor"),
                    "uretici": meta.get("uretici", "Bilinmiyor"),
                    "rol": meta.get("rol", "Bilinmiyor"),
                    "_max_skor": max_skor,   # taban filtresi için ham kosinüs
                })

            siralanmis = sorted(adaylar, key=lambda m: m["score"], reverse=True)

            # Güven kapısı (margin) — HAM KOSİNÜSLER üzerinden.
            # BUG-FIX (kapı hiç açılmıyordu): margin eskiden BİRLEŞİK skorlar
            # üzerinden hesaplanıyordu. Birleşik skor oy terimini (+0.10*oy/K)
            # içerdiği için, çok oy alan aday ile ikincisi arasındaki fark
            # yapay olarak büyüyor ve her şey "yüksek güven" görünüyordu.
            # Ölçüm (40 gerçek kırpıntı, F-35 videosu): birleşik marj ~0.060,
            # aynı kareye ait HAM marj ~0.020 — eşik 0.015 olduğu için kapı
            # pratikte hiç tetiklenmiyor, sistem %95 yanılırken kendinden
            # emin görünüyordu. Sıralama yine birleşik skora göre (ince ayrım
            # için kasıtlı), ama GÜVEN ölçüsü ham benzerlikten geliyor.
            margin = 0.0
            dusuk_guven = False
            if siralanmis:
                margin = siralanmis[0]["_max_skor"] if len(siralanmis) < 2 else (
                    siralanmis[0]["_max_skor"] - siralanmis[1]["_max_skor"]
                )
                dusuk_guven = (
                    VRAG_MARGIN_ESIGI > 0
                    and len(siralanmis) >= 2
                    and margin < VRAG_MARGIN_ESIGI
                )

            # Kırpıntı çok küçükse kimlik iddiası güvenilmez: SigLIP2 girişi
            # 384px, 44px'lik bir hedef ~9 kat büyütülüyor ve ayırt edici
            # detay (kokpit, kanard, kuyruk açısı) kalmıyor. Ölçüm: doğru
            # çıkan tek örnek 183x189px kırpıntıdan geldi, yanlışlar 34x59'a
            # kadar iniyordu. Sonucu bastırmıyoruz — düşük güven işaretliyoruz.
            try:
                kh, kw = image.shape[:2]
                if min(kh, kw) < VRAG_MIN_CROP_PX:
                    dusuk_guven = True
            except Exception:
                pass

            matches = []
            for i, aday in enumerate(siralanmis[:top_k]):
                # Taban: modelin EN İYİ ham kosinüsü VRAG_MIN_SCORE üstünde olmalı
                # (en az bir güçlü görsel eşleşme şart — birleşik skor değil).
                if aday["_max_skor"] >= VRAG_MIN_SCORE:
                    aday = dict(aday)
                    # "benzerlik" = GERÇEK ham kosinüs. VLM prompt'u ve arayüz
                    # bunu göstermeli; "score" oy terimiyle şişmiş sıralama
                    # metriği ve %99 gibi yanıltıcı değerler üretiyordu
                    # (ham 0.931 iken birleşik 0.991 -> VLM yanlışı onaylıyordu).
                    aday["benzerlik"] = round(float(aday["_max_skor"]), 4)
                    if i == 0:
                        aday["margin"] = round(float(margin), 4)
                        aday["dusuk_guven"] = dusuk_guven
                    aday.pop("_max_skor", None)
                    matches.append(aday)
            return matches
        except Exception as e:
            log.error(f"[VRAG] Arama sırasında hata oluştu: {e}")
            return []
