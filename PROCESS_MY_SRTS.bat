@echo off
echo ===== Processing Your 15 SRT Files =====
echo.

REM Count actual files
for /f %%i in ('dir "data\raw_srts\*.srt" /b 2^>nul ^| find /c /v ""') do set count=%%i
if %count%==0 (
    echo No SRT files found in data\raw_srts\
    pause
    exit /b 1
)

echo Found %count% SRT files
echo Starting Epic 2.4 processing...
echo.

py -3.10 simple_batch.py

echo.
echo All done! Check data\processed_srts\ for your enhanced files
pause