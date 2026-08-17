@echo off
title UAV-VRAG Sistem Baslatici
echo ========================================================
echo UAV-VRAG Sistemi Baslatiliyor...
echo Lutfen bekleyin...
echo ========================================================
echo.

echo [1/3] Onceki oturum kapatiliyor (varsa)...
REM ONEMLI: 8000 (backend) ve 8001 (LLM) portlarindaki ESKI surecleri once kapat.
REM Aksi halde yeni uvicorn "port dolu" deyip cikar, BAYAT backend calismaya
REM devam eder ve config/model degisiklikleri (or. yeni VLM) HIC devreye girmez.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Arka Plan Yapay Zeka (YOLO, VRAG, VLM, LLM) Servisleri Baslatiliyor...
start "UAV-VRAG Yapay Zeka Sunuculari" cmd /c "cd /d %~dp0entegrasyon\backend && ..\..\LLM\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo [3/3] Servislerin hazir olmasi bekleniyor (10 Saniye)...
timeout /t 10 /nobreak >nul

echo Web arayuzu taraycida aciliyor...
start "" "http://127.0.0.1:8000/goruntule/"

exit
