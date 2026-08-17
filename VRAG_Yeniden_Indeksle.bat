@echo off
chcp 65001 >nul
title VRAG Yeniden Indeksleme
cd /d %~dp0

echo ========================================================
echo   VRAG Veritabani Yeniden Indeksleme
echo ========================================================
echo.

REM --- Backend acik mi? (Qdrant dosya kilidi cakismasin) ---
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [UYARI] Backend 8000 portunda CALISIYOR gibi gorunuyor.
    echo.
    echo Indeksleme sirasinda backend KAPALI olmalidir - ikisi ayni Qdrant
    echo veritabani dosyasini kilitler, ayni anda acik olamazlar.
    echo Once backend penceresini kapatin, sonra bu dosyayi tekrar calistirin.
    echo.
    pause
    exit /b 1
)

REM --- venv Python var mi? ---
if not exist "LLM\.venv\Scripts\python.exe" (
    echo [HATA] LLM\.venv\Scripts\python.exe bulunamadi.
    echo Once kurulumu tamamlayin ^(README: uv sync^).
    echo.
    pause
    exit /b 1
)

echo Referans veri seti indeksleniyor...  ( veriler\ )
echo Yeni/degisen gorseller islenecek, zaten indeksli olanlar atlanacak,
echo diskten silinmis gorseller indeksten cikarilacak.
echo.

LLM\.venv\Scripts\python.exe -m src.vrag.ingest --img_dir veriler
set EXITCODE=%errorlevel%

echo.
echo ========================================================
if %EXITCODE%==0 (
    echo   Indeksleme tamamlandi.
) else (
    echo   Indeksleme HATA ile bitti ^(kod %EXITCODE%^). Yukaridaki mesaji okuyun.
)
echo ========================================================
pause
