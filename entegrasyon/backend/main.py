# -*- coding: utf-8 -*-
"""VRAG backend (FastAPI) â€” YOLO tespit+takip + VRAG model tanÄ±ma.

UÃ§lar:
  POST /oturum/baslat {video_yolu}  â†’ video iÅŸlemeyi baÅŸlat (arka plan thread)
  POST /oturum/durdur               â†’ durdur
  GET  /durum                       â†’ oturum durumu
  GET  /video                       â†’ MJPEG (kutulu kare akÄ±ÅŸÄ±)
  WS   /hedefler                    â†’ canlÄ± hedef JSON'u (C# panelleri)
  POST /tani  (dosya)               â†’ tek gÃ¶rselde VRAG (model + adaylar)
  GET  /meta                        â†’ model/Ã¼lke/rol listeleri
  GET  /gecmis?adet=N               â†’ tespit geÃ§miÅŸi (log)
"""
import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import ayarlar
from pipeline_adapter import PipelineAdapter

# ─── Logging (src/main.py'deki merkezi yapılandırmanın backend karşılığı) ─────
# Backend, src/main.py'yi hiç import etmediği için kök logger WARNING'de kalıyordu:
# pipeline'ın "[VLM BEKLEMEDE] ... hangi kapı tıkalı" ve "[LLM] Karar alındı"
# gibi INFO teşhis satırları hiç görünmüyordu. Aynı kalıp burada tekrarlanıyor.
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)          # uvicorn handler eklemis olsa bile seviye INFO olsun
if not any(getattr(h, "_uav_backend", False) for h in _root_logger.handlers):
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    _sh._uav_backend = True                   # cift eklemeyi engelle (reload/import)
    _root_logger.addHandler(_sh)
for _gurultulu in ("httpx", "urllib3", "huggingface_hub", "transformers", "PIL"):
    logging.getLogger(_gurultulu).setLevel(logging.WARNING)


class Durum:
    def __init__(self):
        self.fuzyon: PipelineAdapter | None = None
        self.calisiyor = False
        self.son_kare: bytes | None = None
        self.son_hedefler: list = []
        self.frame_no = 0
        self.kaynak = ""
        self.kilit = threading.Lock()
        # BUG-FIX (çifte pipeline / VRAG kilit çakışması): "Başlat"a art arda
        # basılırsa (ör. çift tıklama, hızlı yeniden deneme) iki ayrı
        # _video_dongu thread'i aynı anda "durum.fuzyon is None mı?" kontrolünü
        # geçip İKİ AYRI PipelineAdapter (ve dolayısıyla iki VRAGEngine/Qdrant
        # bağlantısı) oluşturmaya çalışabiliyordu. İkincisi, Qdrant'ın yerel
        # veritabanı dosyasını kilitli bulup VRAG'ı hiç başlatamıyordu — sonuç:
        # VRAG'dan hiç cevap gelmiyor, VLM de desteksiz kalıp "Bilinmiyor"
        # döndürüyordu. Bu kilit, pipeline oluşturma/sıfırlama bloğunun aynı
        # anda yalnızca bir thread tarafından çalıştırılmasını garanti eder.
        self.pipeline_kilit = threading.Lock()
        self.thread: threading.Thread | None = None
        self.meta_cache: dict = {}


durum = Durum()


import queue
import subprocess
import sys

llm_process = None

def _meta_tara() -> dict:
    """VRAG-final/veriler metadata'sından model/ülke/rol listelerini çıkarır."""
    kok = ayarlar.VRAG_DIZINI / "veriler"
    modeller, ulkeler, roller = set(), set(), set()
    for mp in kok.rglob("metadata.json"):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("model"):
            modeller.add(m["model"])
        if m.get("ulke"):
            ulkeler.add(str(m["ulke"]))
        if m.get("rol"):
            roller.add(str(m["rol"]))
    return {"model_sayisi": len(modeller), "ulkeler": sorted(ulkeler),
            "roller": sorted(roller)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_process
    print("Modeller video başlatıldığında kendi iş parçacığında (thread) yüklenecek...", flush=True)
    durum.meta_cache = _meta_tara()
    
    # LLM Modülünü Başlat
    llm_dir = str((Path(__file__).resolve().parent.parent.parent / "LLM"))
    print(f"[BACKEND] LLM API başlatılıyor... ({llm_dir})", flush=True)
    try:
        # LLM'i ana ortamın Python'u ile başlat (Kullanıcının .venv'si)
        # Kopyalanmış bozuk .venv sorununu aşmak için sys.executable zorunlu kılındı.
        llm_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "operational_decision.api.main:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", "8001"],
            cwd=llm_dir
        )
    except Exception as e:
        print(f"[BACKEND] LLM başlatılırken hata: {e}")
        
    print(f"Hazır. {durum.meta_cache['model_sayisi']} model indeksli.", flush=True)
    yield
    if durum.fuzyon:
        durum.fuzyon.kapat()
    if llm_process:
        llm_process.terminate()

app = FastAPI(title="VRAG Backend", lifespan=lifespan)
app.mount("/goruntule", StaticFiles(directory=str(Path(__file__).parent.parent / "web"), html=True), name="goruntule")

def _kare_okuyucu(cap, q):
    while durum.calisiyor:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            q.put(frame, timeout=1.0)
        except queue.Full:
            continue
    q.put(None)

def _video_dongu(video_yolu: str):
    durum.calisiyor = True

    # BUG-FIX: bu blok (pipeline oluşturma/sıfırlama) art arda gelen iki
    # /oturum/baslat isteğinde aynı anda çalışırsa, "durum.fuzyon is None mı?"
    # kontrolünü ikisi de geçip iki ayrı PipelineAdapter (iki ayrı VRAG/Qdrant
    # bağlantısı) oluşturmaya çalışabiliyordu — bkz. Durum.pipeline_kilit
    # tanımındaki not. Kilitle bu bloğu tek seferde tek thread'e indiriyoruz.
    with durum.pipeline_kilit:
        # CUDA Thread kilitlemesini önlemek için modeli bu thread içinde yükle
        if durum.fuzyon is None:
            print("[BACKEND] Modeller belleğe alınıyor, lütfen bekleyin...", flush=True)
            from pipeline_adapter import PipelineAdapter
            durum.fuzyon = PipelineAdapter(source_fps=25.0)
        else:
            # Otomatik tam sıfırlama: farklı çözünürlükte bir video ile yeniden
            # başlatılsa bile (tracker, kamera hareketi kompanzasyonu, sahne-diff
            # önbelleği) her oturum temiz baştan başlasın — model ağırlıklarını
            # yeniden yüklemeden (~10-15sn kazanç), yalnızca çözünürlüğe bağlı
            # önbellekleri sıfırlar. Böylece video değiştirmek için backend'i
            # kapatıp açmaya gerek kalmaz.
            print("[BACKEND] Yeni oturum: tracker + kamera hareketi + VLM önbelleği sıfırlanıyor...", flush=True)
            durum.fuzyon.pipeline.tracker.reset()
            durum.fuzyon.pipeline._prev_gray = None
            # BUG-FIX: tracker.reset() ByteTrack'in ID sayacını da sıfırlıyor, yani
            # yeni videodaki ilk hedef eski videodaki aynı track_id'yi alabiliyordu.
            # VLM'in kendi önbelleği (self.vlm) ayrı bir nesne olduğu için tracker
            # sıfırlanınca otomatik temizlenmiyordu — eski videodan kalan cevap
            # (ör. "Kaan", tutarlılık 2/2) hiç yeni analiz yapılmadan gösteriliyordu.
            durum.fuzyon.pipeline.vlm.reset_all()
            # BUG-FIX: pipeline_adapter.py, henüz kendi tazesi olmayan hedefler için
            # (ör. yeni track ilk birkaç karede) en son bilinen VLM/LLM sonucunu
            # (last_vlm_payload/last_llm_payload) "hayalet" olarak gösteriyordu —
            # bunlar PipelineAdapter nesnesinde saklandığı için (oturumlar arası
            # yaşıyor) yeni video başlasa bile eski sonuç ekranda kalmaya devam
            # ediyordu. Yeni oturumda bunlar da temizlenmeli.
            durum.fuzyon.last_vlm_payload = None
            durum.fuzyon.last_llm_payload = None

    cap = cv2.VideoCapture(video_yolu)
    if not cap.isOpened():
        durum.calisiyor = False
        print(f"Kaynak açılamadı: {video_yolu}", flush=True)
        return
        
    frame_queue = queue.Queue(maxsize=30)
    reader_thread = threading.Thread(target=_kare_okuyucu, args=(cap, frame_queue), daemon=True)
    reader_thread.start()
    
    son_kare_bgr = None            # geç gelen VLM/LLM için son işlenen ham kare
    video_dogal_bitti = False
    while durum.calisiyor:
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if frame is None:
            video_dogal_bitti = True   # kullanıcı durdurmadı, video bitti
            break

        son_kare_bgr = frame
        annotated, hedefler = durum.fuzyon.isle(frame)

        # MJPEG için kareyi sıkıştır
        ok, jpg = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, ayarlar.JPEG_KALITE])
        if ok:
            with durum.kilit:
                durum.son_kare = jpg.tobytes()
                durum.son_hedefler = hedefler
                durum.frame_no += 1

    cap.release()

    # ─────────────────────────────────────────────────────────────
    # VİDEO BİTTİ — HEMEN SONLANDIRMA.
    # process_frame her karede track.vlm_result/llm_result'ı SNAPSHOT'lar
    # (bkz. pipeline.py). VLM/LLM async görevleri son kareden SONRA
    # tamamlanırsa (LLM tipik olarak geç gelir) sonuç hiç snapshot'lanmıyor,
    # "video bitti" diye kayboluyordu. Çözüm: son kareyi tutup bir süre
    # periyodik yeniden serialize et (geç sonuçları yakalar), sonra
    # calisiyor=True tutup son karede beklet — sistem sonlanmasın, kullanıcı
    # durdurana veya yeni video başlatana kadar son sonuçlar ekranda kalsın.
    if video_dogal_bitti and son_kare_bgr is not None:
        print("[BACKEND] Video bitti — geç gelen VLM/LLM sonuçları bekleniyor, son kare tutuluyor...", flush=True)
        yakalama_baslangic = time.time()
        while durum.calisiyor:
            # İlk ~18 sn: uçuşta olan VLM (3-7sn) + LLM tamamlanıp panele düşsün.
            if (time.time() - yakalama_baslangic) < 18.0:
                try:
                    annotated, hedefler = durum.fuzyon.isle(son_kare_bgr)
                    ok, jpg = cv2.imencode(".jpg", annotated,
                                           [cv2.IMWRITE_JPEG_QUALITY, ayarlar.JPEG_KALITE])
                    if ok:
                        with durum.kilit:
                            durum.son_kare = jpg.tobytes()
                            durum.son_hedefler = hedefler
                except Exception as e:
                    print(f"[BACKEND] Son kare yeniden işleme hatası (önemsiz): {e}", flush=True)
                time.sleep(1.0)
            else:
                # Sonuçlar alındı: GPU harcamadan son kareyi tut (kullanıcı durdurana kadar).
                time.sleep(0.5)

    durum.calisiyor = False
    print("Oturum sonlandı.", flush=True)


@app.post("/oturum/baslat")
def oturum_baslat(govde: dict):
    yol = govde.get("video_yolu", "")
    if not Path(yol).exists():
        return JSONResponse({"hata": f"Video bulunamadÄ±: {yol}"}, status_code=400)
    # Ã‡alÄ±ÅŸan oturum varsa Ã¶nce DURDUR (yeni videoya sorunsuz geÃ§iÅŸ).
    if durum.calisiyor:
        durum.calisiyor = False
        eski = durum.thread
        if eski and eski.is_alive():
            eski.join(timeout=4)
    with durum.kilit:                 # eski videonun son karesini/hedeflerini temizle
        durum.son_kare = None
        durum.son_hedefler = []
        durum.frame_no = 0
    durum.calisiyor = True
    durum.kaynak = yol
    durum.thread = threading.Thread(target=_video_dongu, args=(yol,), daemon=True)
    durum.thread.start()
    return {"ok": True, "kaynak": yol}


@app.post("/oturum/durdur")
def oturum_durdur():
    durum.calisiyor = False
    return {"ok": True}


@app.get("/durum")
def durum_al():
    return {"calisiyor": durum.calisiyor, "kaynak": durum.kaynak,
            "frame_no": durum.frame_no, "hedef_sayisi": len(durum.son_hedefler),
            "model_sayisi": durum.meta_cache.get("model_sayisi", 0)}


@app.get("/video")
def video():
    def uret():
        bos = 0
        while True:
            with durum.kilit:
                kare = durum.son_kare
            if kare:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + kare + b"\r\n")
                bos = 0
            else:
                # BUG-FIX ("video oynamıyor"): eski sabit 300 (~10sn) timeout, ilk
                # oturumda modeller (YOLO+SigLIP) yüklenirken (~20-30sn) devreye girip
                # akışı İLK KARE gelmeden kapatıyordu → tarayıcıdaki <img> bağlantısı
                # ölüyor ve video kalıcı boş kalıyordu. Artık: oturum AKTİFKEN
                # (calisiyor=True, modeller yükleniyor olabilir) ilk kareyi uzun süre
                # bekleriz; oturum yoksa kısa sürede kapatırız (boşta akışı tutmayalım).
                bos += 1
                limit = 2700 if durum.calisiyor else 150   # ~90sn yükleme / ~5sn boşta
                if bos > limit:
                    break
            time.sleep(1 / 30)
    return StreamingResponse(uret(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/hedefler")
async def hedefler_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with durum.kilit:
                paket = {"frame": durum.frame_no, "hedefler": durum.son_hedefler}
            await ws.send_json(paket)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.post("/tani")
async def tani(dosya: UploadFile = File(...)):
    # Tek görselde VRAG tanıma henüz canlı hatta bağlı değil (eski Kokpit
    # vrag_adapter'a bağımlıydı, o kaldırıldı). Görsel çözülüyor ama tanıma
    # yapılmıyor — ileride pipeline'ın vrag_engine'ine bağlanabilir.
    veri = await dosya.read()
    arr = np.frombuffer(veri, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"hata": "Görsel çözülemedi."}, status_code=400)
    return {"model": None, "adaylar": []}


@app.get("/meta")
def meta():
    return durum.meta_cache


_referans_yollari: dict[str, str] = {}


def _referans_bul(model: str):
    if model in _referans_yollari:
        return _referans_yollari[model]
        
    target_lower = model.lower().strip()
    
    for mp in (ayarlar.VRAG_DIZINI / "veriler").rglob("metadata.json"):
        try:
            m_data = json.loads(mp.read_text(encoding="utf-8"))
            meta_model = m_data.get("model", "")
            folder_name = mp.parent.name
            
            if meta_model.lower().strip() == target_lower or folder_name.lower().strip() == target_lower:
                for p in sorted(mp.parent.iterdir()):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        _referans_yollari[model] = str(p)
                        return str(p)
        except (OSError, ValueError):
            continue
    return None


@app.get("/referans")
def referans(model: str):
    """Bir modelin dataset'teki temsili fotoÄŸrafÄ±nÄ± dÃ¶ndÃ¼rÃ¼r ('iÅŸte bu X' kanÄ±tÄ±)."""
    yol = _referans_bul(model)
    if yol and Path(yol).exists():
        return FileResponse(yol)
    return JSONResponse({"hata": "referans bulunamadÄ±"}, status_code=404)


@app.get("/gecmis")
def gecmis(adet: int = 100):
    # Tespit geçmişi kaydı henüz canlı hatta bağlı değil (eski Kokpit kayit
    # modülüne bağımlıydı, o kaldırıldı). Şimdilik boş liste döner.
    return {"kayitlar": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ayarlar.HOST, port=ayarlar.PORT)
