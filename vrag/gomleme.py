"""Embedding (gömme) modülü — görüntü -> vektör.

Embedding modeli soyutlanmıştır: `GomlemeModeli` arayüzünü uygulayan farklı
modeller (ör. ileride CLIP/SigLIP) eklenebilir. Varsayılan uygulama DINOv2'dir.
Model bir kez yüklenip önbelleğe alınır; batch ve GPU/CPU desteklidir.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from . import config


class GomlemeModeli(ABC):
    """Görüntüleri sabit boyutlu, L2-normalize vektörlere çeviren soyut arayüz."""

    @property
    @abstractmethod
    def boyut(self) -> int:
        """Üretilen vektörün boyutu."""

    @abstractmethod
    def gomle(self, goruntuler: list[Image.Image]) -> np.ndarray:
        """Görüntü listesini (N, boyut) float32 numpy dizisine çevirir.

        Dönen vektörler L2-normalize edilmiştir (kosinüs benzerliği için).
        """


class DINOv2Gomleyici(GomlemeModeli):
    """HuggingFace transformers üzerinden DINOv2 tabanlı gömme uygulaması."""

    def __init__(self, model_adi: str = config.GOMLEME_MODELI, cihaz: str = config.CIHAZ):
        # Ağır import'u (transformers) yalnızca örnek oluşturulurken yap.
        from transformers import AutoImageProcessor, AutoModel

        self.cihaz = cihaz
        # use_fast=True: hızlı görüntü işleyici (transformers v4.52+ varsayılanı).
        self.islemci = AutoImageProcessor.from_pretrained(model_adi, use_fast=True)
        self.model = AutoModel.from_pretrained(model_adi).to(cihaz).eval()

    @property
    def boyut(self) -> int:
        # Modelin gerçek gizli katman boyutu (dinov2-base için 768).
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def gomle(self, goruntuler: list[Image.Image]) -> np.ndarray:
        if not goruntuler:
            return np.empty((0, self.boyut), dtype=np.float32)

        parcalar: list[np.ndarray] = []
        for bas in range(0, len(goruntuler), config.TOPLU_BOYUT):
            grup = [g.convert("RGB") for g in goruntuler[bas:bas + config.TOPLU_BOYUT]]
            girdiler = self.islemci(images=grup, return_tensors="pt").to(self.cihaz)
            cikti = self.model(**girdiler)
            # DINOv2 CLS token temsili (pooler_output): silüet/şekil için genel özet.
            vektor = cikti.pooler_output
            # Kosinüs benzerliği için L2-normalize et.
            vektor = torch.nn.functional.normalize(vektor, p=2, dim=-1)
            parcalar.append(vektor.cpu().numpy().astype(np.float32))

        return np.concatenate(parcalar, axis=0)


@lru_cache(maxsize=1)
def gomleme_modeli_al() -> GomlemeModeli:
    """Yapılandırmaya göre tek (önbelleğe alınmış) gömme modeli örneği döndürür.

    Model ağırdır; ilk çağrıda yüklenir, sonraki çağrılar aynı örneği kullanır.
    Farklı bir embedding modeline geçmek için yalnızca burayı değiştir.
    """
    return DINOv2Gomleyici()


def bellegi_bosalt() -> None:
    """Önbellekteki embedding modelini bırakır ve GPU belleğini boşaltır.

    Retrieval bittikten sonra, ağır VLM'i yüklemeden önce çağrılır; böylece
    8 GB VRAM'de DINOv2 ile VLM aynı anda bellekte tutulmaz.
    """
    import gc

    gomleme_modeli_al.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
