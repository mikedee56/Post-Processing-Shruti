@echo off

echo ======================================================
echo      Shruti Post-Processing Web Interface
echo ======================================================
echo.





echo Starting the web server...
echo You can access the interface at: http://localhost:8000
echo.
echo Press CTRL+C in this window to stop the server.
echo.

set PYTHONPATH=%~dp0src
python web_server.py
