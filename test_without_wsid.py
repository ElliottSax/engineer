#!/usr/bin/env python3
"""
Test System Without wsid
Use alternative methods while we find the wsid
"""

import json
import requests
from pathlib import Path

print("=" * 60)
print("🔬 TESTING FILMORA API ENDPOINTS")
print("=" * 60)

# Known authentication
auth = {
    "client_sign": "REDACTED_WONDERSHARE_SIGN",
    "base_api": "https://prod-web.wondershare.cc/api/v1",
    "product_id": "1901",
    "version": "15.0.12.16430"
}

print("\n1️⃣ Testing base API connectivity...")

# Test 1: Basic connectivity
try:
    response = requests.get(
        auth["base_api"],
        timeout=5,
        headers={
            "User-Agent": f"Wondershare Filmora/{auth['version']}"
        }
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        print("  ✅ API is reachable")
    else:
        print(f"  ⚠️ API returned: {response.status_code}")
except Exception as e:
    print(f"  ❌ Cannot reach API: {e}")

print("\n2️⃣ Testing public endpoints...")

# Test public endpoints that might not need wsid
public_endpoints = [
    "/version",
    "/status",
    "/public/features",
    "/aigc/models",
    "/aigc/capabilities"
]

for endpoint in public_endpoints:
    url = auth["base_api"] + endpoint
    try:
        response = requests.get(
            url,
            timeout=3,
            headers={
                "User-Agent": f"Wondershare Filmora/{auth['version']}",
                "client_sign": auth["client_sign"],
                "product_id": auth["product_id"]
            }
        )
        if response.status_code == 200:
            print(f"  ✅ {endpoint}: Accessible")
            if response.content:
                try:
                    data = response.json()
                    print(f"     Response: {str(data)[:100]}...")
                except:
                    pass
        elif response.status_code == 404:
            print(f"  ❌ {endpoint}: Not found")
        else:
            print(f"  ⚠️ {endpoint}: Status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️ {endpoint}: {str(e)[:50]}")

print("\n3️⃣ Alternative: Use proxy detection...")

# Check if Filmora is running and using a local proxy
import subprocess

try:
    result = subprocess.run(
        ["netstat", "-an"],
        capture_output=True,
        text=True
    )

    # Look for common proxy ports
    proxy_ports = ["8080", "8888", "3128", "8866", "9090"]
    for port in proxy_ports:
        if f":{port}" in result.stdout and "LISTENING" in result.stdout:
            print(f"  ✅ Found local proxy on port {port}")
            print(f"     Try setting Fiddler to intercept port {port}")
except:
    pass

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)
print()
print("Since direct API calls aren't working, try:")
print()
print("1. In Filmora, go to Account → Sign Out")
print("2. Sign back in")
print("3. Watch Fiddler for login requests")
print()
print("OR")
print()
print("1. Use Process Monitor (ProcMon) on Windows")
print("2. Filter for: Process Name = Filmora.exe")
print("3. Look for network activity")
print()
print("OR")
print()
print("1. Just use the cheap API pipeline for now:")
print("   python3 test_cheap_apis.py")
print("=" * 60)