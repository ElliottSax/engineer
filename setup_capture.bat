@echo off
echo ============================================================
echo FILMORA API CAPTURE SETUP
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Installing required packages...
echo.

REM Install mitmproxy
echo Installing mitmproxy for API capture...
pip install mitmproxy

REM Install other requirements
echo Installing additional packages...
pip install requests aiohttp asyncio

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo.
echo 1. EASY METHOD - Automatic Capture:
echo    - Run: python capture_filmora_live.py
echo    - Open Filmora and use any AI feature
echo    - Script will capture wsid automatically
echo.
echo 2. ADVANCED METHOD - mitmproxy:
echo    - Run: start_mitmproxy.bat
echo    - Configure Filmora to use proxy (127.0.0.1:8080)
echo    - Use Filmora AI features
echo    - Check captured_api_data folder
echo.
echo 3. MANUAL METHOD - If automatic fails:
echo    - Open Filmora Developer Tools (if available)
echo    - Look for wondershare.cc API calls
echo    - Copy wsid header value
echo    - Add to monitor_filmora_api.py
echo.
echo ============================================================
pause