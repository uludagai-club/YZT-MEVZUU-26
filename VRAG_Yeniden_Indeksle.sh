#!/usr/bin/env bash
# VRAG/Qdrant yeniden indeksleme (Linux). Windows'taki VRAG_Yeniden_Indeksle.bat karsiligi.
# Artimli: yeni gorselleri embed eder, diskten silinenleri indeksten cikarir, numara
# desync'ini duzeltir. Backend KAPALI olmali (ikisi ayni Qdrant dosyasini kilitler).
set -euo pipefail

KOK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$KOK/LLM/.venv/bin/python"

echo "========================================================"
echo "  VRAG Veritabani Yeniden Indeksleme"
echo "========================================================"

# Backend 8000'de acik mi? (Qdrant dosya kilidi cakismasin)
if command -v lsof >/dev/null 2>&1 && lsof -ti tcp:8000 -s tcp:LISTEN >/dev/null 2>&1; then
  echo "[UYARI] Backend 8000'de CALISIYOR gibi. Indeksleme sirasinda KAPALI olmalidir."
  echo "        Once backend'i durdurun (Ctrl+C), sonra bu script'i tekrar calistirin."
  exit 1
fi

if [ ! -x "$VENV_PY" ]; then
  echo "[HATA] $VENV_PY bulunamadi. Once kurulumu yapin (bkz. KURULUM_UBUNTU.md)."
  exit 1
fi

echo "Referans veri seti indeksleniyor (veriler/)..."
echo "Yeni/degisen gorseller islenecek, indeksliler atlanacak, silinenler cikarilacak."
echo
cd "$KOK"
export PYTHONPATH="$KOK:${PYTHONPATH:-}"
"$VENV_PY" -m src.vrag.ingest --img_dir veriler
echo
echo "========================================================"
echo "  Indeksleme tamamlandi."
echo "========================================================"
