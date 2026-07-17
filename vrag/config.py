"""VRAG proje ayarları — tüm sabitler ve yollar tek yerde toplanır.

Model, Qdrant koleksiyonu, augmentation ve arama davranışını buradan tek
noktadan değiştirebilirsin. Kod içine sabit gömülmez; her şey buraya bakar.
"""
from __future__ import annotations

from pathlib import Path

import torch

# --- Yollar -----------------------------------------------------------------
# Bu dosya vrag/config.py içinde; proje kökü bir üst dizin.
PROJE_KOK = Path(__file__).resolve().parent.parent
REFERANS_DIZINI = PROJE_KOK / "data" / "reference"
QDRANT_YOLU = PROJE_KOK / "qdrant_db"

# --- Embedding modeli -------------------------------------------------------
# DINOv2: self-supervised olduğu için silüet/şekil benzerliğinde güçlü.
# Üstten çekilmiş uçak ayrımında asıl sinyal silüet olduğundan uygun.
GOMLEME_MODELI = "facebook/dinov2-base"
VEKTOR_BOYUTU = 768          # dinov2-base çıktı boyutu (small=384, large=1024)
CIHAZ = "cuda" if torch.cuda.is_available() else "cpu"
TOPLU_BOYUT = 16             # embedding çıkarımında batch boyutu (8 GB VRAM'e uygun)

# --- Qdrant -----------------------------------------------------------------
KOLEKSIYON_ADI = "hava_araclari"
# Vektörler L2-normalize edildiğinden kosinüs benzerliği kullanıyoruz.

# --- Arama ------------------------------------------------------------------
VARSAYILAN_TOPK = 5
# Model bazında tekilleştirmeden ÖNCE Qdrant'tan çekilecek ham sonuç sayısı.
# Aynı modelin çok sayıda varyasyonu olduğundan bol çekip sonra tekilleştiriyoruz.
HAM_ARAMA_LIMITI = 64

# --- Augmentation (drone kamerası varyasyonlarını simüle eder) --------------
# Her referans görselden orijinal + aşağıdaki dönüşümlerle varyasyonlar üretilir.
DONME_ACILARI = (-12, 12)          # derece cinsinden döndürme açıları
OLCEK_FAKTORLERI = (0.8,)          # küçültme / uzaklaşma simülasyonu
BULANIKLIK_YARICAPI = 1.2          # Gaussian blur yarıçapı (drone kamerası netsizliği)

# --- Kategoriler & dosya türleri --------------------------------------------
# metadata.json içindeki "kategori" alanı için beklenen değerler (uyarı amaçlı).
BILINEN_KATEGORILER = (
    "Savaş Uçağı",
    "Bombardıman Uçağı",
    "Nakliye/Özel Görev Uçağı",
    "Helikopter",
    "İHA",
    "SİHA",
    "Yolcu Uçağı",
    "Drone",
)
DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png")

# --- VLM doğrulama katmanı (Qwen2.5-VL) -------------------------------------
# Retrieval'dan gelen adaylar arasından karşılaştırmalı seçim yapan görsel-dil
# modeli. Yerel/offline çalışır. Backend değiştirilebilir (bkz. dogrulama.py).
VLM_MODELI = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_ADAY_SAYISI = 3          # VLM'e gönderilecek en iyi aday sayısı (VRAM/token sınırı)
VLM_MAX_YENI_TOKEN = 160     # VLM üretim uzunluğu (gerekçe kısa; hız için sınırlı)
# 8 GB VRAM için varsayılan 4-bit (bitsandbytes). >12 GB GPU'da False (bf16) yapılabilir.
VLM_4BIT = True
# Qwen görüntü işleyici piksel sınırları (patch = 28x28). Üst sınır VRAM'i kontrol eder;
# büyük görseller otomatik küçültülür.
# Düşük çözünürlük = daha az görsel token = 7B'de çok daha hızlı çıkarım.
VLM_MIN_PIKSEL = 128 * 28 * 28
VLM_MAX_PIKSEL = 256 * 28 * 28
