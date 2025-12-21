#!/usr/bin/env python3
"""
Quick Filmora wsid Capture
Simple script to check for authentication
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import os
import sys

print("=" * 60)
print("🎯 FILMORA QUICK CAPTURE")
print("=" * 60)

# Check if running on Windows/WSL
if sys.platform == "linux" and "microsoft" in os.uname().release.lower():
    print("✅ Running on WSL")
    # Look for Windows paths from WSL
    possible_db_paths = [
        Path("/mnt/c/Users/") / os.environ.get("USER", "elliott") / "AppData/Local/Wondershare/Wondershare Filmora/authorizeInfoCacheFile.db",
        Path("/mnt/c/ProgramData/Wondershare/Wondershare Filmora/authorizeInfoCacheFile.db"),
    ]
else:
    # Native Windows paths
    possible_db_paths = [
        Path.home() / "AppData/Local/Wondershare/Wondershare Filmora/authorizeInfoCacheFile.db",
        Path("C:/ProgramData/Wondershare/Wondershare Filmora/authorizeInfoCacheFile.db"),
    ]

# Known authentication
auth_data = {
    "client_sign": "REDACTED_WONDERSHARE_SIGN",
    "base_api": "https://prod-web.wondershare.cc/api/v1",
    "product_id": "1901",
    "version": "15.0.12.16430",
}

print("\n📊 Current Authentication:")
print(f"  ✅ client_sign: {auth_data['client_sign'][:30]}...")
print(f"  ✅ base_api: {auth_data['base_api']}")
print(f"  ❓ wsid: Not captured yet")
print(f"  ❓ session_token: Not captured yet")

# Look for database
print("\n🔍 Searching for Filmora database...")
db_found = False

for path in possible_db_paths:
    if path.exists():
        print(f"  ✅ Found: {path}")
        db_found = True

        try:
            # Try to read database
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()

            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            if tables:
                print(f"  📊 Tables: {[t[0] for t in tables]}")

            conn.close()
        except Exception as e:
            print(f"  ⚠️ Cannot read DB: {e}")

        break
    else:
        print(f"  ❌ Not found: {path}")

if not db_found:
    print("\n⚠️ Filmora database not found")
    print("\nPlease ensure Filmora is installed at:")
    print("  C:\\Program Files\\Wondershare\\Filmora")
    print("  or")
    print("  C:\\Program Files (x86)\\Wondershare\\Filmora")

# Create capture directory
capture_dir = Path("captured_api_data")
capture_dir.mkdir(exist_ok=True)

# Save what we have
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = capture_dir / f"auth_{timestamp}.json"

data = {
    "timestamp": timestamp,
    "auth": auth_data,
    "status": "partial - need wsid",
}

with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n💾 Saved: {output_file}")

print("\n" + "=" * 60)
print("NEXT STEPS TO GET WSID:")
print("=" * 60)
print()
print("Since we're on WSL, we need to capture wsid from Windows.")
print()
print("OPTION 1 - Use Fiddler (Recommended):")
print("  1. Install Fiddler on Windows: https://www.telerik.com/fiddler")
print("  2. Start Fiddler")
print("  3. Open Filmora and use any AI feature")
print("  4. In Fiddler, look for requests to wondershare.cc")
print("  5. Find the 'wsid' header value")
print()
print("OPTION 2 - Use Browser DevTools:")
print("  1. Open Filmora web version (if available)")
print("  2. Press F12 for DevTools")
print("  3. Go to Network tab")
print("  4. Use an AI feature")
print("  5. Look for wsid in request headers")
print()
print("OPTION 3 - Manual Entry:")
print("  1. Once you find the wsid value")
print("  2. Create a file: captured_api_data/wsid.txt")
print("  3. Paste the wsid value in it")
print()
print("For now, we can test with the cheap API pipeline")
print("Run: python3 test_cheap_apis.py")
print("=" * 60)