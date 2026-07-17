"""VLM doğrulama katmanı — boru hattının son halkası.

Retrieval'dan gelen adaylar arasından, sorgu görseli ile aday referans
görsellerini KARŞILAŞTIRARAK nihai SEÇİMİ bir görsel-dil modeli (VLM) yapar.
VLM sıfırdan tahmin ETMEZ; yalnızca adaylar arasından seçer.

Backend değiştirilebilir (VLMDogrulayici arayüzü). Varsayılan yerel/offline
Qwen2.5-VL-3B'dir.

Not — gerekçe hakkında: 3B model 4-bit'te güvenilir serbest-metin gerekçe
üretemiyor (istemi/örneği kopyalıyor). Bu yüzden gerekçe olarak, VLM'in seçtiği
modelin BİLİNEN ayırt edici özelliklerini gösteriyoruz (dürüst ve tutarlı).
Gerçek görsele-dayalı gerekçe için daha büyük bir model (Qwen2.5-VL-7B) veya API
backend gerekir — VLM_MODELI değiştirilerek yükseltilebilir.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from . import config
from .arama import AdaySonuc


@dataclass
class DogrulamaSonucu:
    """VLM doğrulama çıktısı."""

    secilen_model: str
    gerekce: str
    guven: float


# VLM'e verilecek talimat — 7B için: görsele dayalı OBSERVATION + SELECTION + REASON.
# İngilizce yazılıyor çünkü VLM İngilizce talimatları çok daha güvenilir izliyor
# (Türkçe istemde aday metnini kopyalama/Çince karakter sızıntısı görülmüştü).
_TALIMAT = (
    "Above you are first shown the QUERY image to identify, then the numbered "
    "CANDIDATES' reference images. Match the aircraft in the query image to one "
    "of the candidates. Give your answer as EXACTLY these three lines:\n"
    "OBSERVATION: describe ONLY what you see in the query image itself "
    "(number of engines, wing shape, tail structure, nose, overall silhouette).\n"
    "SELECTION: write the number of the candidate that best matches.\n"
    "REASON: based on your observation, explain in one sentence why you chose this candidate.\n"
    "Choose only among the candidates; do not copy the candidate labels verbatim, "
    "describe your own observation instead.\n"
    "Answer ENTIRELY IN ENGLISH; do not use any other language or characters."
)


class VLMDogrulayici(ABC):
    """Sorgu + adaylar alıp karşılaştırmalı seçim yapan soyut arayüz."""

    @abstractmethod
    def dogrula(self, sorgu_yolu: str | Path, adaylar: list[AdaySonuc]) -> DogrulamaSonucu:
        """Adaylar arasından nihai modeli seçer."""


def _secim_indeksi(cikti: str, adaylar: list[AdaySonuc]) -> int:
    """VLM metninden seçilen aday indeksini çözer (çok katmanlı yedekli)."""
    # 1) "SELECTION: <numara>" (birincil).
    m = re.search(r"SELECTION\s*[:\-]?\s*(\d+)", cikti, re.IGNORECASE)
    if m:
        i = int(m.group(1)) - 1
        if 0 <= i < len(adaylar):
            return i
    # 2) Metinde geçen model adı.
    dusuk = cikti.lower()
    for i, a in enumerate(adaylar):
        if a.model.lower() in dusuk:
            return i
    # 3) Metindeki ilk tekil 1..N rakamı (model adlarındaki 16/5E'yi kapmaz).
    m2 = re.search(r"(?<!\S)([1-9])(?!\S)", cikti)
    if m2:
        i = int(m2.group(1)) - 1
        if 0 <= i < len(adaylar):
            return i
    # 4) Hepsi başarısız: retrieval'ın en iyi adayı.
    return 0


def _alan(cikti: str, desen: str) -> str:
    """Etiketli bir satırın (ör. 'OBSERVATION:') değerini döndürür."""
    m = re.search(desen + r"\s*[:\-]?\s*(.+)", cikti, re.IGNORECASE)
    return m.group(1).strip().splitlines()[0].strip() if m else ""


def _sonucu_coz(cikti: str, adaylar: list[AdaySonuc]) -> DogrulamaSonucu:
    """Seçilen adayı ve VLM'in kendi gözlem/gerekçesini çözer."""
    secilen = adaylar[_secim_indeksi(cikti, adaylar)]
    gozlem = _alan(cikti, r"OBSERVATION")
    gerekce = _alan(cikti, r"REASON")
    parcalar = [p for p in (gozlem, gerekce) if p]
    if parcalar:
        goster = " — ".join(parcalar)          # VLM'in görsele dayalı kendi cümleleri
    elif secilen.ayirt_edici_ozellikler:        # VLM metni yoksa yedek
        goster = f"Ayırt edici özellikler: {secilen.ayirt_edici_ozellikler}"
    else:
        goster = "Görsel olarak en benzer aday."
    # Güven = seçilen adayın görsel benzerlik skoru.
    return DogrulamaSonucu(secilen.model, goster, secilen.skor)


class QwenVLDogrulayici(VLMDogrulayici):
    """Yerel/offline Qwen2.5-VL tabanlı doğrulayıcı (transformers)."""

    def __init__(self):
        # Ağır import'lar yalnızca model yüklenirken yapılır.
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cihaz = config.CIHAZ
        yukleme_ayarlari: dict = {}
        if config.VLM_4BIT:
            # Düşük VRAM için 4-bit (bitsandbytes gerektirir).
            from transformers import BitsAndBytesConfig

            yukleme_ayarlari["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            yukleme_ayarlari["device_map"] = "auto"
        else:
            yukleme_ayarlari["torch_dtype"] = (
                torch.bfloat16 if self.cihaz == "cuda" else torch.float32
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.VLM_MODELI, **yukleme_ayarlari
        )
        if not config.VLM_4BIT:
            self.model = self.model.to(self.cihaz)
        self.model.eval()

        # min/max_pixels: görselleri VRAM'e uygun boyuta indirger.
        self.islemci = AutoProcessor.from_pretrained(
            config.VLM_MODELI,
            min_pixels=config.VLM_MIN_PIKSEL,
            max_pixels=config.VLM_MAX_PIKSEL,
        )

    def _istem_kur(self, sorgu_yolu: Path, adaylar: list[AdaySonuc]):
        """Çok görselli mesajı ve görsel listesini (sırayla) oluşturur."""
        crop = Image.open(sorgu_yolu).convert("RGB")
        resimler = [crop]

        icerik = [
            {"type": "text", "text": "QUERY IMAGE TO IDENTIFY:"},
            {"type": "image", "image": crop},
            {"type": "text", "text": "CANDIDATES:"},
        ]
        for i, a in enumerate(adaylar, start=1):
            ref = Image.open(a.referans_yolu).convert("RGB")
            resimler.append(ref)
            # Kopyalanabilir "ayırt edici" metni verilmez; 7B görsele bakıp betimlesin.
            icerik.append({"type": "text", "text": f"Candidate {i}: {a.model} ({a.kategori})"})
            icerik.append({"type": "image", "image": ref})

        icerik.append({"type": "text", "text": _TALIMAT})
        mesajlar = [{"role": "user", "content": icerik}]
        return mesajlar, resimler

    @torch.no_grad()
    def dogrula(self, sorgu_yolu: str | Path, adaylar: list[AdaySonuc]) -> DogrulamaSonucu:
        mesajlar, resimler = self._istem_kur(Path(sorgu_yolu), adaylar)

        metin = self.islemci.apply_chat_template(
            mesajlar, tokenize=False, add_generation_prompt=True
        )
        girdiler = self.islemci(
            text=[metin], images=resimler, padding=True, return_tensors="pt"
        ).to(self.model.device)

        uretilen = self.model.generate(
            **girdiler, max_new_tokens=config.VLM_MAX_YENI_TOKEN, do_sample=False
        )
        # Girdi token'larını kırp, sadece üretilen kısmı çöz.
        kirpik = [c[len(g):] for g, c in zip(girdiler.input_ids, uretilen)]
        cikti = self.islemci.batch_decode(
            kirpik, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return _sonucu_coz(cikti, adaylar)


@lru_cache(maxsize=1)
def vlm_dogrulayici_al() -> VLMDogrulayici:
    """Yapılandırmaya göre tek (önbelleğe alınmış) VLM doğrulayıcı örneği döndürür.

    Model ağırdır; ilk çağrıda yüklenir. Farklı bir VLM'e geçmek için burayı değiştir.
    """
    return QwenVLDogrulayici()


def aday_dogrula(
    sorgu_yolu: str | Path,
    adaylar: list[AdaySonuc],
    vlm: bool = True,
) -> DogrulamaSonucu:
    """Adaylar arasından nihai modeli seçer.

    vlm=True: Qwen2.5-VL ile karşılaştırmalı seçim (VLM_ADAY_SAYISI kadar aday).
    vlm=False: VLM'i yüklemeden retrieval'ın en yüksek skorlu adayını döndürür.
    """
    if not adaylar:
        return DogrulamaSonucu("bilinmiyor", "Aday bulunamadı.", 0.0)

    if not vlm:
        en_iyi = adaylar[0]
        return DogrulamaSonucu(
            en_iyi.model, "VLM kapalı — retrieval'ın en yüksek skorlu adayı.", en_iyi.skor
        )

    dogrulayici = vlm_dogrulayici_al()
    return dogrulayici.dogrula(sorgu_yolu, adaylar[: config.VLM_ADAY_SAYISI])
