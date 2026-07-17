"""VRAG web arayüzü — Flask.

Fotoğraf yükle → retrieval (DINOv2 + Qdrant) → VLM doğrulama (Qwen2.5-VL) → sonuç.
Modeller sunucu başlarken bir kez yüklenir ve sıcak tutulur (istekler arasında
yeniden yüklenmez). 8 GB VRAM'de DINOv2 + 4-bit Qwen birlikte sığar.

Çalıştırma:  python app.py   →  http://127.0.0.1:5000
"""
from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import quote

from flask import Flask, abort, jsonify, render_template, request, send_file

from vrag import config
from vrag.arama import ara
from vrag.dogrulama import aday_dogrula, vlm_dogrulayici_al
from vrag.gomleme import gomleme_modeli_al
from vrag.vektor_deposu import VektorDeposu

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB üst sınır

# Qdrant embedded + GPU'yu istekler arasında seri kullan (tek kullanıcılı demo).
_kilit = threading.Lock()

YUKLEME_DIZINI = config.PROJE_KOK / "yuklemeler"
YUKLEME_DIZINI.mkdir(exist_ok=True)

# /gorsel yalnızca bu kökler altındaki dosyaları servis eder (path traversal koruması).
_IZINLI_KOKLER = [
    (config.PROJE_KOK / "veriler").resolve(),
    config.REFERANS_DIZINI.resolve(),
    YUKLEME_DIZINI.resolve(),
]

_depo: VektorDeposu | None = None


def _depo_al() -> VektorDeposu:
    """Sunucu ömrü boyunca açık kalan tek Qdrant istemcisini döndürür."""
    global _depo
    if _depo is None:
        _depo = VektorDeposu()
    return _depo


def _izinli(p: Path) -> bool:
    try:
        p = p.resolve()
    except OSError:
        return False
    return any(p.is_relative_to(kok) for kok in _IZINLI_KOKLER)


def _gorsel_url(yol) -> str:
    return "/gorsel?yol=" + quote(str(yol))


def _model_listesi() -> list[str]:
    kok = config.PROJE_KOK / "veriler"
    if not kok.exists():
        kok = config.REFERANS_DIZINI
    if not kok.exists():
        return []
    return sorted(p.name for p in kok.iterdir() if p.is_dir())


@app.route("/")
def anasayfa():
    return render_template("index.html", modeller=_model_listesi())


@app.route("/tani", methods=["POST"])
def tani():
    dosya = request.files.get("gorsel")
    if dosya is None or dosya.filename == "":
        return jsonify({"hata": "Görsel seçilmedi."}), 400

    uzanti = Path(dosya.filename).suffix.lower()
    if uzanti not in config.DESTEKLENEN_UZANTILAR:
        return jsonify({"hata": f"Desteklenmeyen dosya türü: {uzanti}"}), 400

    hedef = YUKLEME_DIZINI / f"sorgu{uzanti}"
    dosya.save(hedef)

    try:
        with _kilit:
            adaylar = ara(hedef, topk=config.VARSAYILAN_TOPK, depo=_depo_al())
            sonuc = aday_dogrula(hedef, adaylar, vlm=True)
    except Exception as e:  # kullanıcıya anlaşılır hata dön
        return jsonify({"hata": str(e)}), 500

    return jsonify({
        "sorgu": _gorsel_url(hedef),
        "vlm": {
            "secilen_model": sonuc.secilen_model,
            "guven": round(sonuc.guven, 4),
            "gerekce": sonuc.gerekce,
        },
        "adaylar": [
            {
                "sira": i + 1,
                "model": a.model,
                "skor": round(a.skor, 4),
                "kategori": a.kategori,
                "aci": a.aci,
                "gorsel": _gorsel_url(a.referans_yolu),
            }
            for i, a in enumerate(adaylar)
        ],
    })


@app.route("/gorsel")
def gorsel():
    """Sorgu ve referans görsellerini (yalnızca izinli kökler altında) servis eder."""
    yol = request.args.get("yol", "")
    if not yol:
        abort(400)
    p = Path(yol)
    if not p.is_absolute():
        p = config.PROJE_KOK / p
    if not _izinli(p) or not p.is_file():
        abort(404)
    return send_file(p)


def _modelleri_isit() -> None:
    """Sunucu başlarken DINOv2, VLM ve Qdrant'ı yükleyip sıcak tutar."""
    print("Modeller yükleniyor (DINOv2 + Qwen2.5-VL)... ilk açılış ~30 sn sürebilir",
          flush=True)
    gomleme_modeli_al()
    vlm_dogrulayici_al()
    _depo_al()
    print("Modeller hazır → http://127.0.0.1:5000", flush=True)


if __name__ == "__main__":
    _modelleri_isit()
    app.run(host="127.0.0.1", port=5000, debug=False)
