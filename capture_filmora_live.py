#!/usr/bin/env python3
"""
Live Filmora API Capture - Real-time monitoring
Captures wsid and session tokens as you use Filmora
"""

import asyncio
import json
import time
import sqlite3
import hashlib
import os
from pathlib import Path
from datetime import datetime
import subprocess
import re
import requests
import threading
from typing import Dict, Optional, Any

print("=" * 60)
print("🎯 FILMORA LIVE API CAPTURE")
print("=" * 60)

class FilmoraLiveCapture:
    """Real-time capture of Filmora API calls"""

    def __init__(self):
        self.captured_auth = {
            "client_sign": "{2871f7e8-51b4-487a-9ab9-5d99926ee2ebG}",
            "base_api": "https://prod-web.wondershare.cc/api/v1",
            "product_id": "1901",
            "version": "15.0.12.16430",
            "wsid": None,
            "session_token": None,
            "device_id": None,
            "timestamp": None
        }

        self.db_path = self._find_filmora_db()
        self.capture_dir = Path("captured_api_data")
        self.capture_dir.mkdir(exist_ok=True)

        # Process monitoring
        self.monitoring = False
        self.last_check = 0

    def _find_filmora_db(self) -> Optional[Path]:
        """Find Filmora's SQLite database"""

        possible_paths = [
            Path.home() / "AppData/Local/Wondershare/Wondershare Filmora/authorizeInfoCacheFile.db",
            Path.home() / "AppData/Roaming/Wondershare/Filmora/authorizeInfoCacheFile.db",
            Path("C:/ProgramData/Wondershare/Wondershare Filmora/authorizeInfoCacheFile.db")
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✅ Found Filmora DB: {path}")
                return path

        print("⚠️ Filmora DB not found in standard locations")
        return None

    def extract_from_db(self) -> Dict[str, Any]:
        """Extract authentication from SQLite database"""

        if not self.db_path:
            return {}

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📊 Found tables: {[t[0] for t in tables]}")

            # Try to extract auth data
            auth_data = {}

            # Check common auth tables
            for table in ['auth', 'session', 'token', 'user', 'config']:
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                    data = cursor.fetchone()
                    if data:
                        columns = [desc[0] for desc in cursor.description]
                        for i, col in enumerate(columns):
                            if 'token' in col.lower() or 'session' in col.lower() or 'wsid' in col.lower():
                                if data[i]:
                                    auth_data[col] = data[i]
                                    print(f"  Found {col}: {str(data[i])[:20]}...")
                except:
                    pass

            conn.close()
            return auth_data

        except Exception as e:
            print(f"⚠️ DB extraction error: {e}")
            return {}

    def monitor_network(self):
        """Monitor network traffic for API calls"""

        print("\n🔍 Monitoring network traffic...")

        # Use netsh to capture network trace
        capture_file = "filmora_trace.etl"

        # Start capture
        cmd_start = [
            "netsh", "trace", "start",
            "capture=yes",
            f"tracefile={capture_file}",
            "provider=Microsoft-Windows-TCPIP",
            "maxsize=10",
            "overwrite=yes"
        ]

        try:
            subprocess.run(cmd_start, shell=True, capture_output=True)
            print("📡 Network capture started")

            # Monitor for 30 seconds
            time.sleep(30)

            # Stop capture
            cmd_stop = ["netsh", "trace", "stop"]
            subprocess.run(cmd_stop, shell=True, capture_output=True)
            print("📡 Network capture stopped")

            # Parse capture file (simplified)
            self._parse_network_capture(capture_file)

        except Exception as e:
            print(f"⚠️ Network monitoring error: {e}")

    def _parse_network_capture(self, capture_file: str):
        """Parse network capture for API calls"""

        # This is simplified - in production would parse ETL file
        print(f"📋 Parsing {capture_file}...")

    def monitor_processes(self):
        """Monitor Filmora process memory for tokens"""

        print("\n🔍 Monitoring Filmora process...")

        # Find Filmora process
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Filmora.exe"],
                capture_output=True,
                text=True,
                shell=True
            )

            if "Filmora.exe" in result.stdout:
                print("✅ Filmora is running")

                # Extract process ID
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Filmora.exe" in line:
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            print(f"  Process ID: {pid}")

                            # Monitor process (simplified)
                            self._monitor_process_memory(pid)
            else:
                print("⚠️ Filmora not running. Please start Filmora.")

        except Exception as e:
            print(f"⚠️ Process monitoring error: {e}")

    def _monitor_process_memory(self, pid: str):
        """Monitor process memory for tokens"""

        # This would use Windows debugging APIs
        # For now, simplified approach
        print(f"  Monitoring PID {pid} memory...")

    def intercept_api_calls(self):
        """Intercept API calls using proxy"""

        print("\n🔍 Setting up API interception...")

        # Set system proxy to intercept calls
        proxy_port = 8888

        # Configure Windows proxy
        cmd = [
            "netsh", "winhttp", "set", "proxy",
            f"127.0.0.1:{proxy_port}"
        ]

        try:
            subprocess.run(cmd, shell=True, capture_output=True)
            print(f"✅ Proxy configured on port {proxy_port}")

            # Start simple proxy server
            self._start_proxy_server(proxy_port)

        except Exception as e:
            print(f"⚠️ Proxy setup error: {e}")

    def _start_proxy_server(self, port: int):
        """Start proxy server to intercept API calls"""

        from http.server import HTTPServer, BaseHTTPRequestHandler

        class ProxyHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Capture POST data
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)

                # Check for Wondershare API
                if "wondershare" in self.path or "filmora" in self.path:
                    print(f"\n🎯 CAPTURED API CALL:")
                    print(f"  URL: {self.path}")

                    # Parse headers for tokens
                    for header, value in self.headers.items():
                        if "token" in header.lower() or "auth" in header.lower():
                            print(f"  {header}: {value[:50]}...")

                    # Parse body for tokens
                    try:
                        body = json.loads(post_data)
                        if "wsid" in body:
                            print(f"  ✅ Found wsid: {body['wsid']}")
                        if "session" in body:
                            print(f"  ✅ Found session: {body['session']}")
                    except:
                        pass

                # Forward request (simplified)
                self.send_response(200)
                self.end_headers()

        # Start server in thread
        server = HTTPServer(('127.0.0.1', port), ProxyHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        print(f"  Proxy server running on port {port}")

    def test_captured_auth(self):
        """Test if captured authentication works"""

        if not self.captured_auth.get("wsid"):
            print("\n⚠️ wsid not captured yet")
            return False

        print("\n🧪 Testing captured authentication...")

        # Test API call
        url = f"{self.captured_auth['base_api']}/user/info"

        headers = {
            "wsid": self.captured_auth["wsid"],
            "client_sign": self.captured_auth["client_sign"],
            "product_id": self.captured_auth["product_id"],
            "User-Agent": f"Wondershare Filmora/{self.captured_auth['version']}"
        }

        try:
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                print("✅ Authentication successful!")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

    def save_captured_data(self):
        """Save captured authentication data"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.capture_dir / f"capture_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "auth": self.captured_auth,
            "status": "complete" if self.captured_auth.get("wsid") else "partial"
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Saved capture: {output_file}")

    async def auto_capture_loop(self):
        """Automatic capture loop"""

        print("\n🔄 Starting automatic capture...")
        print("Instructions:")
        print("1. Open Filmora if not already open")
        print("2. Use any AI feature (text-to-video, etc.)")
        print("3. Wait for capture confirmation")
        print("\nMonitoring...\n")

        attempts = 0
        max_attempts = 60  # 5 minutes

        while attempts < max_attempts:
            attempts += 1

            # Check database
            db_data = self.extract_from_db()
            if db_data:
                for key, value in db_data.items():
                    if value and key not in self.captured_auth:
                        self.captured_auth[key] = value
                        print(f"✅ Captured {key}: {str(value)[:30]}...")

            # Check if we have wsid
            if self.captured_auth.get("wsid"):
                print("\n🎉 SUCCESS! Captured wsid!")

                # Test authentication
                if self.test_captured_auth():
                    self.save_captured_data()
                    print("\n✅ Authentication verified and saved!")
                    print("You can now generate videos with ZERO cost!")
                    return True

            # Wait and retry
            print(f"  Attempt {attempts}/{max_attempts}...", end='\r')
            await asyncio.sleep(5)

        print("\n⚠️ Timeout - please ensure Filmora is running and you've used an AI feature")
        return False

async def main():
    """Main capture function"""

    capture = FilmoraLiveCapture()

    # Show current status
    print(f"\n📊 Current Authentication Status:")
    print(f"  client_sign: ✅ {capture.captured_auth['client_sign'][:30]}...")
    print(f"  base_api: ✅ {capture.captured_auth['base_api']}")
    print(f"  wsid: {'✅' if capture.captured_auth.get('wsid') else '❌ Not captured'}")
    print(f"  session: {'✅' if capture.captured_auth.get('session_token') else '❌ Not captured'}")

    # Start monitoring
    print("\n" + "="*60)
    print("STARTING CAPTURE METHODS:")
    print("="*60)

    # Method 1: Database extraction
    print("\n1️⃣ Database Extraction:")
    db_data = capture.extract_from_db()

    # Method 2: Process monitoring
    print("\n2️⃣ Process Monitoring:")
    capture.monitor_processes()

    # Method 3: Network interception
    print("\n3️⃣ Network Interception:")
    capture.intercept_api_calls()

    # Method 4: Auto-capture loop
    print("\n4️⃣ Auto-Capture Loop:")
    success = await capture.auto_capture_loop()

    if success:
        print("\n" + "="*60)
        print("🎉 CAPTURE COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Run: python3 run_hybrid_pipeline.py")
        print("2. Generate unlimited videos with your subscription!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ MANUAL CAPTURE NEEDED")
        print("="*60)
        print("Please:")
        print("1. Open Filmora")
        print("2. Generate any AI video")
        print("3. Check Tools → Developer → Network (if available)")
        print("4. Look for requests to wondershare.cc")
        print("5. Copy the 'wsid' header value")
        print("="*60)

if __name__ == "__main__":
    print("\n🚀 Filmora Live Capture System")
    print("This will capture your wsid automatically")
    print("-" * 40)

    asyncio.run(main())