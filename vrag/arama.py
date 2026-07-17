"""Arama — sorgu görselinden en benzer hava aracı modellerini bulur.

YOLO'dan gelen crop DINOv2 ile gömülür, Qdrant'ta aranır ve sonuçlar MODEL
BAZINDA tekilleştirilir: aynı modelin birden çok varyasyonu top-k'yı doldurmaz,
her modelin yalnızca en iyi skoru alınır.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import config
from .gomleme import gomleme_modeli_al
from .vektor_deposu import VektorDeposu


@dataclass
class AdaySonuc:
    """Tek bir aday modelin arama sonucu (VLM doğrulama katmanına girdi olur)."""

    model: str
    skor: float
    kategori: str
    referans_yolu: str
    ayirt_edici_ozellikler: str
    aci: str


def ara(
    sorgu_yolu: str | Path,
    topk: int = config.VARSAYILAN_TOPK,
    kategori: str | None = None,
    depo: VektorDeposu | None = None,
    haric_yollar: set[str] | None = None,
) -> list[AdaySonuc]:
    """Sorgu görseli için model bazında tekilleştirilmiş top-k aday döndürür.

    kategori verilirse arama yalnızca o kategorideki referanslarla sınırlanır.
    depo verilirse mevcut Qdrant istemcisi kullanılır (web sunucusu için); yoksa
    geçici bir tane açılıp kapatılır.
    haric_yollar verilirse bu referans dosyalarından gelen isabetler yok sayılır —
    "sorgu görseli veritabanında yokmuş gibi" ölçüm (leave-one-out) için kullanılır.
    """
    sorgu_yolu = Path(sorgu_yolu)
    if not sorgu_yolu.exists():
        raise FileNotFoundError(f"Sorgu görseli bulunamadı: {sorgu_yolu}")

    gomleyici = gomleme_modeli_al()
    with Image.open(sorgu_yolu) as im:
        vektor = gomleyici.gomle([im.convert("RGB")])[0]

    kendi_depo = depo is None
    d = depo if depo is not None else VektorDeposu()
    try:
        ham = d.ara(vektor, limit=config.HAM_ARAMA_LIMITI, kategori=kategori)
    finally:
        if kendi_depo:
            d.kapat()

    # Model bazında en iyi skoru tut (tekilleştirme / aggregation).
    en_iyi: dict[str, AdaySonuc] = {}
    for nokta in ham:
        pl = nokta.payload or {}
        if haric_yollar and pl.get("dosya_yolu") in haric_yollar:
            continue  # leave-one-out: sorgunun kendi referansını sayma
        model = pl.get("model", "bilinmiyor")
        skor = float(nokta.score)
        mevcut = en_iyi.get(model)
        if mevcut is None or skor > mevcut.skor:
            en_iyi[model] = AdaySonuc(
                model=model,
                skor=skor,
                kategori=pl.get("kategori", "bilinmiyor"),
                referans_yolu=pl.get("dosya_yolu", ""),
                ayirt_edici_ozellikler=pl.get("ayirt_edici_ozellikler", ""),
                aci=pl.get("aci", "bilinmiyor"),
            )

    adaylar = sorted(en_iyi.values(), key=lambda a: a.skor, reverse=True)
    return adaylar[:topk]
