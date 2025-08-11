@echo off

echo ======================================================
echo      Shruti Post-Processing Web Interface
echo ======================================================
echo.

REM Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in the system's PATH.
    echo Please install Python 3.10 or higher and try again.
    pause
    exit /b 1
)

REM Check if requirements are installed
python -m pip freeze | findstr /R "pandas pysrt fuzzywuzzy python-Levenshtein pyyaml" >nul
if %errorlevel% neq 0 (
    echo Warning: Some required packages might be missing.
    echo If the server fails, please run 'pip install -r requirements.txt' and try again.
    echo.
)

echo Starting the web server...
echo You can access the interface at: http://localhost:8000
echo.
echo Press CTRL+C in this window to stop the server.
echo.

set PYTHONPATH=%~dp0src
python web_server.py
