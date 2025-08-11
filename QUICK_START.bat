@echo off
echo ===== Epic 2.4 Quick Start =====
echo.

REM Check if we have SRT files
if not exist "data\raw_srts\*.srt" (
    echo ERROR: No SRT files found in data\raw_srts\
    echo Please add your .srt files to data\raw_srts\ folder
    pause
    exit /b 1
)

REM Count SRT files
for /f %%i in ('dir "data\raw_srts\*.srt" /b ^| find /c /v ""') do set count=%%i
echo Found %count% SRT files to process
echo.

REM Install dependencies (quick)
echo Installing dependencies...
py -3.10 -m pip install --quiet pandas pyyaml pysrt fuzzywuzzy python-Levenshtein structlog sanskrit_parser tqdm

REM Process all files
echo.
echo Processing all SRT files...
py -3.10 simple_batch.py

echo.
echo ===== DONE =====
pause