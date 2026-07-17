"""VRAG — Teknofest hava aracı tanıma için Visual RAG retrieval hattı.

Boru hattı: YOLO crop -> DINOv2 embedding -> Qdrant retrieval -> (VLM doğrulama).
Bu paket retrieval katmanını (embedding + Qdrant + arama) içerir.
"""

__version__ = "0.1.0"
