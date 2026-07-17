"""Qdrant vektör deposu sarmalayıcı.

Docker'sız, embedded/local modda çalışır (QdrantClient(path=...)). Koleksiyon
kurma (idempotent), toplu ekleme ve kategori-filtreli arama sağlar. Kategori
filtresi, payload'daki "kategori" alanı üzerinden yapılır.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from . import config


class VektorDeposu:
    """Qdrant koleksiyonu üzerinde ince bir sarmalayıcı."""

    def __init__(
        self,
        yol: Path | str = config.QDRANT_YOLU,
        koleksiyon: str = config.KOLEKSIYON_ADI,
    ):
        # Embedded mod: veriler yerel klasörde tutulur, sunucu/Docker gerekmez.
        self.istemci = QdrantClient(path=str(yol))
        self.koleksiyon = koleksiyon

    def kapat(self) -> None:
        """Qdrant istemcisini kapatır (embedded modda dosya kilidini serbest bırakır)."""
        self.istemci.close()

    def __enter__(self) -> "VektorDeposu":
        return self

    def __exit__(self, *_) -> None:
        self.kapat()

    def koleksiyon_kur(self, boyut: int = config.VEKTOR_BOYUTU, sifirla: bool = True) -> None:
        """Koleksiyonu oluşturur. sifirla=True ise varsa önce siler (idempotent).

        Böylece ingest her çalıştığında koleksiyon sıfırdan ve tutarlı kurulur.
        """
        if self.istemci.collection_exists(self.koleksiyon):
            if sifirla:
                self.istemci.delete_collection(self.koleksiyon)
            else:
                return
        self.istemci.create_collection(
            collection_name=self.koleksiyon,
            vectors_config=VectorParams(size=boyut, distance=Distance.COSINE),
        )

    def ekle(self, vektorler: np.ndarray, payloadlar: list[dict]) -> int:
        """Vektörleri payload'larıyla toplu ekler; eklenen nokta sayısını döndürür."""
        if len(vektorler) != len(payloadlar):
            raise ValueError("vektör ve payload sayısı eşleşmiyor")
        if len(vektorler) == 0:
            return 0

        noktalar = [
            PointStruct(id=str(uuid.uuid4()), vector=vek.tolist(), payload=pl)
            for vek, pl in zip(vektorler, payloadlar)
        ]
        self.istemci.upsert(collection_name=self.koleksiyon, points=noktalar, wait=True)
        return len(noktalar)

    def ara(self, vektor: np.ndarray, limit: int, kategori: str | None = None):
        """Sorgu vektörüne en yakın noktaları döndürür (ScoredPoint listesi).

        kategori verilirse yalnızca o kategorideki referanslar arasında aranır.
        """
        suzgec = None
        if kategori:
            suzgec = Filter(
                must=[FieldCondition(key="kategori", match=MatchValue(value=kategori))]
            )
        yanit = self.istemci.query_points(
            collection_name=self.koleksiyon,
            query=vektor.tolist(),
            limit=limit,
            query_filter=suzgec,
            with_payload=True,
        )
        return yanit.points

    def sayim(self) -> int:
        """Koleksiyondaki toplam nokta (vektör) sayısı."""
        return self.istemci.count(collection_name=self.koleksiyon).count
