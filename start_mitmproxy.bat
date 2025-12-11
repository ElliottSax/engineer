@echo off
echo ============================================================
echo STARTING FILMORA API CAPTURE WITH MITMPROXY
echo ============================================================
echo.
echo Instructions:
echo 1. This window will show captured API calls
echo 2. Configure Windows proxy: 127.0.0.1:8080
echo 3. Open Filmora and use AI features
echo 4. Watch for wsid capture messages
echo.
echo Starting mitmproxy...
echo.

REM Start mitmproxy with our capture script
mitmdump -s filmora_mitmproxy_capture.py --set confdir=./mitmproxy --set flow_detail=3

pause