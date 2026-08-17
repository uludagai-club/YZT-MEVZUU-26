#!/usr/bin/env bash
# UAV-VRAG sistemini bastan sona baslatir (Linux).
#
#   [1] vLLM        (8002) -- VLM servisi, OpenAI-uyumlu. Once bu, cunku backend
#                             acilirken VLM'i isitmaya calisiyor.
#   [2] Backend     (8000) -- FastAPI + web arayuzu. Kendi lifespan'inde
#   [3] LLM servisi (8001) -- operasyonel karari alt-surec olarak baslatir.
#
# Ollama (11434) harici calisir ve LLM modeli (llama3.2:1b, CPU) icin gereklidir.
#
# Ollama'daki eski VLM'e donmek istersen:  VLLM_ATLA=1 ./Sistemi_Baslat.sh
# (ayrica src/config.py'de VLM_API_URL/VLM_MODEL_NAME eski degerlerine cevrilmeli)
set -uo pipefail

KOK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$KOK/LLM/.venv/bin/python"
VLLM_LOG="$KOK/output/vllm.log"
VLLM_PORT="${VLLM_PORT:-8002}"
VLLM_BEKLEME="${VLLM_BEKLEME:-300}"   # vLLM icin azami bekleme (sn)

echo "========================================================"
echo " UAV-VRAG Sistemi Baslatiliyor..."
echo "========================================================"

# ---------------------------------------------------------------- [1/4] temizlik
echo "[1/4] Onceki oturum kapatiliyor (varsa)..."
if command -v lsof >/dev/null 2>&1; then
  for PORT in 8000 8001 "$VLLM_PORT"; do
    PIDS="$(lsof -ti tcp:"$PORT" -s tcp:LISTEN 2>/dev/null || true)"
    if [ -n "$PIDS" ]; then
      echo "  Port $PORT kapatiliyor: $PIDS"
      kill -9 $PIDS 2>/dev/null || true
    fi
  done
else
  echo "  (lsof yok - port temizligi atlaniyor. 'sudo apt install lsof' onerilir.)"
fi
# vLLM alt-surecleri ana surec olunce OKSUZ kalip VRAM'i tutmaya devam ediyor.
# EngineCore kendini "VLLM::EngineCore" olarak yeniden adlandirdigi icin
# "vllm.entrypoints" deseni onu ISKALIYOR -- iki desen de gerekli.
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f '^VLLM::' 2>/dev/null || true
sleep 2

if [ ! -x "$VENV_PY" ]; then
  echo "[HATA] $VENV_PY bulunamadi. Once kurulumu yapin (bkz. KURULUM_UBUNTU.md)."
  exit 1
fi

VLLM_PID=""
# Kapanista PID'e GUVENME: VLLM_Baslat.sh kendini exec ile python'a ceviriyor,
# EngineCore ise ayri bir surec olup kendini "VLLM::EngineCore" diye yeniden
# adlandiriyor. Saklanan tek PID'i oldurmek yetmiyordu -> vLLM ayakta kalip
# 3.8 GB VRAM'i rehin aliyordu. Bu yuzden PORT tabanli, surec adindan bagimsiz
# temizlik yapiyoruz (ayrica isim deseniyle oksuz EngineCore'u da suzuyoruz).
temizle() {
  echo
  echo "Kapatiliyor (backend + LLM + vLLM)..."
  if command -v lsof >/dev/null 2>&1; then
    for PORT in 8000 8001 "$VLLM_PORT"; do
      PIDS="$(lsof -ti tcp:"$PORT" -s tcp:LISTEN 2>/dev/null || true)"
      [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null || true
    done
  fi
  [ -n "$VLLM_PID" ] && kill -9 "$VLLM_PID" 2>/dev/null || true
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -f '^VLLM::' 2>/dev/null || true
  sleep 1
}
trap temizle EXIT INT TERM

# ---------------------------------------------------------------- [2/4] vLLM
if [ "${VLLM_ATLA:-0}" = "1" ]; then
  echo "[2/4] vLLM ATLANDI (VLLM_ATLA=1) - VLM icin Ollama kullanilacak."
else
  echo "[2/4] vLLM (VLM) baslatiliyor -> log: $VLLM_LOG"
  mkdir -p "$KOK/output"
  "$KOK/VLLM_Baslat.sh" > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  printf "      model yukleniyor (ilk acilis ~90sn: agirlik + CUDA graph)"
  HAZIR=0
  for ((i=0; i<VLLM_BEKLEME; i+=3)); do
    if curl -s -m 2 -o /dev/null "http://127.0.0.1:$VLLM_PORT/v1/models" 2>/dev/null; then
      HAZIR=1; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo; echo "[HATA] vLLM cikti. Son satirlar:"; tail -20 "$VLLM_LOG"; exit 1
    fi
    printf "."
    sleep 3
  done
  echo
  if [ "$HAZIR" != "1" ]; then
    echo "[HATA] vLLM $VLLM_BEKLEME sn icinde hazir olmadi. Son satirlar:"
    tail -20 "$VLLM_LOG"; exit 1
  fi
  echo "      vLLM hazir -> http://127.0.0.1:$VLLM_PORT/v1"
fi

# ---------------------------------------------------------------- [3/4] backend
echo "[3/4] Arka plan YZ servisleri (YOLO/VRAG/VLM/LLM) baslatiliyor..."
echo "[4/4] Hazir olunca web arayuzu: http://127.0.0.1:8000/goruntule/"
echo "      (Durdurmak icin Ctrl+C - vLLM de birlikte kapanir)"
echo "--------------------------------------------------------"
cd "$KOK/entegrasyon/backend"
export PYTHONPATH="$KOK:${PYTHONPATH:-}"
# expandable_segments: allocator parcalanmasini azaltir. vLLM ile ayni 8GB karti
# paylasirken backend'in bosuna onbellege ayirdigi bloklari geri vermesini saglar.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# exec KULLANILMIYOR: trap'in calisip vLLM'i kapatabilmesi icin bu kabuk hayatta kalmali.
#
# Backend ARKA PLANA alinip `wait` ile bekleniyor. Sebep: bash, ON PLANDA bir
# cocuk surec calisirken sinyal trap'ini ERTELER (cocuk bitene kadar calistirmaz)
# -> Ctrl+C'de temizle() hic calismiyor, vLLM ayakta kalip VRAM'i rehin aliyordu.
# `wait` ile beklerken sinyal aninda islenir ve temizlik gercekten yapilir.
"$VENV_PY" -m uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
wait "$BACKEND_PID"
