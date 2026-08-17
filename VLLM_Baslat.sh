#!/usr/bin/env bash
# Qwen3-VL-4B-Instruct'i vLLM ile OpenAI-uyumlu sunucu olarak baslatir (port 8002).
# Ollama'nin yerini alir; src/config.py'de VLM_API_URL bu adrese cevrilmelidir.
#
# 8 GB VRAM notu: vLLM Ollama'dan farkli olarak CPU'ya TASIYAMAZ -- sigmazsa
# dogrudan OOM verir. Bu yuzden 4-bit AWQ agirlik + dusuk gpu-memory-utilization
# kullaniyoruz. Backend (YOLO+SigLIP2) de ayni karti paylastigi icin, once
# backend'i baslatip VRAM'i olcmek, sonra buradaki orani ayarlamak gerekir.
set -euo pipefail

KOK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$KOK/vllm_servis/.venv/bin/python"

MODEL="${VLLM_MODEL:-cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit}"
PORT="${VLLM_PORT:-8002}"
# Kartin TOPLAM belleginin orani (backend'e yer birakacak sekilde disaridan ayarla):
#   VLLM_GPU_UTIL=0.45 ./VLLM_Baslat.sh
GPU_UTIL="${VLLM_GPU_UTIL:-0.41}"   # 0.36 KV cache icin YETMIYOR (-0.18 GiB); 0.41 olculmus calisan deger
MAX_LEN="${VLLM_MAX_LEN:-1536}"

if [ ! -x "$VENV_PY" ]; then
  echo "[HATA] $VENV_PY yok. Once: cd vllm_servis && uv venv --python 3.12 && uv pip install vllm"
  exit 1
fi

PIDS="$(lsof -ti tcp:"$PORT" -s tcp:LISTEN 2>/dev/null || true)"
[ -n "$PIDS" ] && { echo "Port $PORT kapatiliyor: $PIDS"; kill -9 $PIDS 2>/dev/null || true; sleep 1; }

echo "========================================================"
echo " vLLM baslatiliyor"
echo "   model    : $MODEL"
echo "   port     : $PORT   (OpenAI: http://127.0.0.1:$PORT/v1)"
echo "   gpu-util : $GPU_UTIL   max-model-len: $MAX_LEN"
echo "========================================================"

# flashinfer'in JIT sampling cekirdegi calisma aninda nvcc ile CUDA derlemeye
# calisiyor; sistemde CUDA toolkit yok (sadece surucu) -> "Could not find nvcc".
# Ornekleme zaten greedy (temperature=0), flashinfer'e ihtiyac yok.
export VLLM_USE_FLASHINFER_SAMPLER=0

# max_pixels: vLLM bellek profilini EN KOTU senaryodaki goruntu boyutuna gore
# cikariyor. Biz sadece 384px kolaj gonderiyoruz; sinir koymazsak profil tepesi
# tum butceyi yiyip KV cache'e yer birakmiyordu ("No available memory for the
# cache blocks"). 200704 = 448x448, kolajimiz icin fazlasiyla yeterli.
exec "$VENV_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 200704}' \
  --no-enable-log-requests
