"""Ingest — referans görselleri Qdrant'a indeksler.

data/reference/<model_klasoru>/ altındaki her klasörü tarar:
  - metadata.json'dan model adı, kategori ve ayırt edici özellikleri okur,
  - her görselden augmentation ile varyasyonlar üretir,
  - hepsini DINOv2 ile gömer ve payload'larıyla Qdrant'a yazar.

Tekrar çalıştırıldığında koleksiyonu sıfırdan kurar (idempotent).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from . import config
from .artirma import varyasyonlar_uret
from .gomleme import gomleme_modeli_al
from .vektor_deposu import VektorDeposu


def _aci_cikar(dosya_adi: str) -> str:
    """Dosya adından çekim açısını tahmin eder (üstten / yandan)."""
    ad = dosya_adi.lower()
    if "ust" in ad or "üst" in ad or "top" in ad:
        return "üstten"
    if "yan" in ad or "side" in ad:
        return "yandan"
    return "bilinmiyor"


def _metadata_oku(klasor: Path) -> dict:
    """Klasördeki metadata.json'u okur; yoksa makul varsayılanlar döndürür."""
    meta_yolu = klasor / "metadata.json"
    if meta_yolu.exists():
        with open(meta_yolu, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
        print(f"  [uyarı] metadata.json yok: '{klasor.name}' -> klasör adı model kabul edildi")

    kategori = meta.get("kategori") or "bilinmiyor"
    if kategori != "bilinmiyor" and kategori not in config.BILINEN_KATEGORILER:
        print(f"  [uyarı] tanımsız kategori '{kategori}' ({klasor.name})")

    return {
        "model": meta.get("model") or klasor.name,
        "kategori": kategori,
        "ayirt_edici_ozellikler": meta.get("ayirt_edici_ozellikler", ""),
    }


def _gorselleri_bul(klasor: Path) -> list[Path]:
    """Yalnızca bu klasörün İÇİNDEKİ (alt klasörler hariç) görselleri döndürür."""
    return sorted(
        p for p in klasor.iterdir()
        if p.is_file() and p.suffix.lower() in config.DESTEKLENEN_UZANTILAR
    )


def _model_klasorlerini_bul(referans: Path) -> list[Path]:
    """Görsel içeren her klasörü bir model klasörü kabul eder.

    Böylece hem düz yapı (veriler/<model>/) hem kategorili yapı
    (veriler/<kategori>/<model>/) fark etmeksizin çalışır. Kategori klasörleri
    doğrudan görsel içermediği için model sayılmaz, sadece kapsayıcıdır.
    """
    return sorted(
        p for p in referans.rglob("*") if p.is_dir() and _gorselleri_bul(p)
    )


def calistir(sifirla: bool = True, dizin: Path | str | None = None) -> None:
    """Referans dizinini tarar ve Qdrant koleksiyonunu (yeniden) kurar.

    dizin verilmezse config.REFERANS_DIZINI (data/reference) kullanılır.
    """
    referans = Path(dizin) if dizin else config.REFERANS_DIZINI
    if not referans.exists():
        raise FileNotFoundError(
            f"Referans dizini bulunamadı: {referans}\n"
            "data/reference/<model>/ klasörlerini oluşturup fotoğrafları koyun."
        )

    model_klasorleri = _model_klasorlerini_bul(referans)
    if not model_klasorleri:
        raise FileNotFoundError(
            f"{referans} altında görsel içeren model klasörü yok.\n"
            "Beklenen: veriler/<model>/ veya veriler/<kategori>/<model>/"
        )

    print(f"Cihaz: {config.CIHAZ} | Model: {config.GOMLEME_MODELI}")
    gomleyici = gomleme_modeli_al()

    # Model boyutu ile config tutarlılığını doğrula (yanlış koleksiyon boyutunu önler).
    if gomleyici.boyut != config.VEKTOR_BOYUTU:
        raise ValueError(
            f"config.VEKTOR_BOYUTU ({config.VEKTOR_BOYUTU}) modelin boyutuyla "
            f"({gomleyici.boyut}) uyuşmuyor. config.py'yi güncelleyin."
        )

    toplam_gorsel = 0
    toplam_vektor = 0

    # ÖNEMLİ: Qdrant local (embedded) modda delete_collection eski noktaları
    # diskten temizlemiyor -> her ingest üzerine eklenip indeks şişiyor ve eski
    # (geçersiz) dosya yolları kalıyor. Gerçek "sıfırdan kurulum" için depoyu
    # fiziksel olarak siliyoruz.
    if sifirla and config.QDRANT_YOLU.exists():
        shutil.rmtree(config.QDRANT_YOLU)

    with VektorDeposu() as depo:
        depo.koleksiyon_kur(boyut=gomleyici.boyut, sifirla=sifirla)

        for klasor in model_klasorleri:
            meta = _metadata_oku(klasor)
            # Kategorili yapıda üst klasör = kullanıcının gruplaması; payload'a yazılır.
            grup = klasor.parent.name if klasor.parent != referans else ""
            gorseller = _gorselleri_bul(klasor)
            if not gorseller:
                print(f"- {meta['model']}: görsel yok, atlanıyor")
                continue

            # Bu modelin tüm görsel varyasyonlarını toplayıp topluca göm.
            goruntuler = []
            payloadlar = []
            for gorsel in gorseller:
                with Image.open(gorsel) as im:
                    im = im.convert("RGB")
                    for varyasyon, etiket in varyasyonlar_uret(im):
                        goruntuler.append(varyasyon)
                        payloadlar.append({
                            "model": meta["model"],
                            "kategori": meta["kategori"],
                            "grup": grup,
                            "ayirt_edici_ozellikler": meta["ayirt_edici_ozellikler"],
                            "dosya_yolu": str(gorsel),
                            "aci": _aci_cikar(gorsel.name),
                            "varyasyon": etiket,
                        })

            vektorler = gomleyici.gomle(goruntuler)
            eklenen = depo.ekle(vektorler, payloadlar)
            toplam_gorsel += len(gorseller)
            toplam_vektor += eklenen
            print(
                f"- {meta['model']} ({meta['kategori']}): "
                f"{len(gorseller)} görsel -> {eklenen} vektör"
            )

        koleksiyon_toplam = depo.sayim()

    print("\n== İndeksleme tamamlandı ==")
    print(f"Model klasörü       : {len(model_klasorleri)}")
    print(f"Görsel              : {toplam_gorsel}")
    print(f"Vektör (eklenen)    : {toplam_vektor}")
    # Koleksiyonun GERÇEK sayısı; eklenenden farklıysa artık kalıntı var demektir.
    print(f"Vektör (koleksiyon) : {koleksiyon_toplam}")
    print(f"Koleksiyon          : {config.KOLEKSIYON_ADI} @ {config.QDRANT_YOLU}")
