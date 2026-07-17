"""Augmentation — referans görsellerden varyasyon üretir.

Drone kamerasıyla havadan çekim koşullarını taklit etmek için döndürme,
ölçekleme (uzaklaşma) ve hafif bulanıklık uygular. Böylece tek referans
fotoğraftan birkaç varyasyon indekslenir ve retrieval isabeti artar.
Her varyasyon, hata ayıklamada işe yarasın diye bir etiketle birlikte döner.
"""
from __future__ import annotations

from PIL import Image, ImageFilter

from . import config


def _dondur(goruntu: Image.Image, aci: int) -> Image.Image:
    # expand=True: döndürünce köşeler kırpılmasın (uçağın tamamı görünür kalsın).
    return goruntu.rotate(aci, resample=Image.BICUBIC, expand=True)


def _olcekle(goruntu: Image.Image, faktor: float) -> Image.Image:
    yeni_boyut = (
        max(1, int(goruntu.width * faktor)),
        max(1, int(goruntu.height * faktor)),
    )
    return goruntu.resize(yeni_boyut, resample=Image.BICUBIC)


def _bulaniklastir(goruntu: Image.Image, yaricap: float) -> Image.Image:
    return goruntu.filter(ImageFilter.GaussianBlur(radius=yaricap))


def varyasyonlar_uret(goruntu: Image.Image) -> list[tuple[Image.Image, str]]:
    """Bir görselden (varyasyon, etiket) çiftleri üretir.

    Orijinal daima listenin ilk elemanıdır. Dönüşüm parametreleri config'ten
    okunur; augmentation davranışını oradan ayarlayabilirsin.
    """
    goruntu = goruntu.convert("RGB")
    varyasyonlar: list[tuple[Image.Image, str]] = [(goruntu, "orijinal")]

    for aci in config.DONME_ACILARI:
        varyasyonlar.append((_dondur(goruntu, aci), f"donme_{aci:+d}"))

    for faktor in config.OLCEK_FAKTORLERI:
        varyasyonlar.append((_olcekle(goruntu, faktor), f"olcek_{faktor}"))

    varyasyonlar.append((_bulaniklastir(goruntu, config.BULANIKLIK_YARICAPI), "bulanik"))

    return varyasyonlar
