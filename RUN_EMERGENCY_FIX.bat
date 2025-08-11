@echo off
echo ========================================
echo    EMERGENCY ANTI-HALLUCINATION FIX
echo         ASR Post-Processing Recovery
echo ========================================
echo.

echo [INFO] Starting emergency deployment...
echo [INFO] This will:
echo   1. Validate anti-hallucination fixes
echo   2. Process your 15 SRT files safely
echo   3. Generate quality report
echo.

REM Try different Python commands
python EMERGENCY_DEPLOYMENT_COMPLETE.py
if %ERRORLEVEL% EQU 0 goto SUCCESS

py EMERGENCY_DEPLOYMENT_COMPLETE.py
if %ERRORLEVEL% EQU 0 goto SUCCESS

python3 EMERGENCY_DEPLOYMENT_COMPLETE.py
if %ERRORLEVEL% EQU 0 goto SUCCESS

echo [ERROR] Python not found. Please run manually:
echo   python EMERGENCY_DEPLOYMENT_COMPLETE.py
pause
exit /b 1

:SUCCESS
echo.
echo ========================================
echo     EMERGENCY DEPLOYMENT COMPLETE
echo ========================================
echo.
echo Check the output above for results.
echo Processed files are in: data\processed_srts\
echo.
pause