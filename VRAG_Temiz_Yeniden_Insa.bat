@echo off
chcp 65001 >nul
title VRAG Temiz Yeniden Insa (Full Rebuild)
cd /d %~dp0

echo ========================================================
echo   VRAG TEMIZ YENIDEN INSA  (Full Rebuild)
echo ========================================================
echo.
echo Bu islem mevcut indeksi ( output\qdrant_db ) TAMAMEN SILER ve
echo tum gorselleri SIFIRDAN yeniden embed eder (GPU fp16).
echo Birkac dakika surebilir.
echo.
echo Ne zaman gerekir: embedding cihazi/ayari degistiginde (or. CPU -^> GPU fp16)
echo tum vektorleri tutarli sekilde yeniden uretmek icin. Sadece yeni gorsel
echo eklediyseniz buna GEREK YOK - "VRAG_Yeniden_Indeksle.bat" yeterli.
echo.

REM --- Backend acik mi? (Qdrant dosya kilidi cakismasin) ---
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [UYARI] Backend 8000 portunda CALISIYOR. Once backend penceresini kapatin.
    echo.
    pause
    exit /b 1
)

REM --- venv Python var mi? ---
if not exist "LLM\.venv\Scripts\python.exe" (
    echo [HATA] LLM\.venv\Scripts\python.exe bulunamadi.
    echo.
    pause
    exit /b 1
)

REM --- Onay (kazara silmeyi onle) ---
set "CONFIRM="
set /p CONFIRM=Mevcut indeksi SILIP yeniden insa etmek icin buyuk harfle EVET yazin:
if /i not "%CONFIRM%"=="EVET" (
    echo.
    echo Iptal edildi. Hicbir sey silinmedi.
    echo.
    pause
    exit /b 0
)

echo.
echo Mevcut indeks siliniyor...  ( output\qdrant_db )
if exist "output\qdrant_db" rmdir /s /q "output\qdrant_db"

echo Sifirdan indeksleniyor (bu birkac dakika surebilir)...
echo.
LLM\.venv\Scripts\python.exe -m src.vrag.ingest --img_dir veriler
set EXITCODE=%errorlevel%

echo.
echo ========================================================
if %EXITCODE%==0 (
    echo   Temiz yeniden insa tamamlandi.
) else (
    echo   HATA ile bitti ^(kod %EXITCODE%^). Yukaridaki mesaji okuyun.
)
echo ========================================================
pause
