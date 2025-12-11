#!/usr/bin/env python3
"""
Test Wondershare API with authorization token from cache
"""

import requests
import json
import base64
import hashlib
import time

# Found encrypted token from authorizeInfoCacheFile.db
AUTH_TOKEN = "74f4889bd74c0493a12d279b1f825c35e1a444d48dfddbfbafbdc0abb927873a78e1a8cbb1a9152bca6b497b6c44066fb4fc8903f86de23b5f5e8aeb5b5e800ce342b1a7252ae26bf7b08a7f59305587752b7b4428b361fb7f2301b6062f2d0e9fe52b870accea74cd146e9d6b47da1cd530db9173cf8fd1f988d135abf39675cbfbfe1eae9a1bf88a01f6d6f66ea4301249d8d4da1c092649c8e3e1be97a23a35fcb6abeae6260122518ebd2115904f8bc8ff127e20eb212ea6197b0977544c2da6c4cb3fca566803a621f71cfe4f4a86139c03f7680d5c6121a19ab7db66015a58c6b4fd097b67020d9e5e364ece224e0eefc090767e57d3d35fd4ca553d93864ad979032a631af0fefc8fbbf92a0e0df9745a5efbb43dd3558eb5e0de6a2c29ff095d5ab704d1bcbb7323c032d31353943b192494d08f43bf89022e80960607e727fbb5c1ac8ca9f7695cfc06b65131e6298fc04d4dedc66b670ddef5c79196aaab6cd1b149fd43706aebf83b124f"

CONFIG = {
    "wsid": "REDACTED_WSID",
    "client_sign": "REDACTED_WONDERSHARE_SIGN",
    "product_id": "1901",
    "version": "15.0.12.16430"
}

print("="*60)
print("🔐 TESTING WITH AUTHORIZATION TOKEN")
print("="*60)
print(f"Token length: {len(AUTH_TOKEN)} chars")
print(f"WSID: {CONFIG['wsid']}")
print()

def test_with_bearer_token():
    """Test using token as Bearer auth."""
    print("1️⃣ Testing as Bearer Token...")

    url = "https://ai-api.wondershare.cc/v1/ai/user/info"

    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "User-Agent": f"Filmora/{CONFIG['version']}",
        "X-WSID": CONFIG["wsid"],
        "X-Client-Sign": CONFIG["client_sign"],
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:200]}...")
        return response
    except Exception as e:
        print(f"  Error: {e}")
        return None

def test_with_custom_header():
    """Test using token in custom header."""
    print("\n2️⃣ Testing as Custom Header...")

    url = "https://ai-api.wondershare.cc/v1/ai/innovation/google-text2video/batch"

    headers = {
        "X-Auth-Token": AUTH_TOKEN,
        "X-Session-Token": AUTH_TOKEN,
        "X-WSID": CONFIG["wsid"],
        "X-Client-Sign": CONFIG["client_sign"],
        "User-Agent": f"Filmora/{CONFIG['version']}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": "A serene mountain landscape",
        "model": "veo-3.0-fast-generate-preview",
        "duration": 8,
        "wsid": CONFIG["wsid"]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:200]}...")
        return response
    except Exception as e:
        print(f"  Error: {e}")
        return None

def test_wondershare_login_api():
    """Test standard Wondershare login API."""
    print("\n3️⃣ Testing Login API...")

    # Standard Wondershare auth endpoint
    url = "https://account-api.wondershare.cc/api/v1/user/login"

    headers = {
        "User-Agent": f"Filmora/{CONFIG['version']}",
        "X-Client-Sign": CONFIG["client_sign"],
        "Content-Type": "application/json"
    }

    # Try with cached token
    payload = {
        "token": AUTH_TOKEN,
        "wsid": CONFIG["wsid"],
        "product_id": CONFIG["product_id"]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:500]}...")

        if response.status_code == 200:
            data = response.json()
            if "data" in data and "token" in data["data"]:
                print(f"\n  ✅ Got session token: {data['data']['token'][:50]}...")
                return data["data"]["token"]
    except Exception as e:
        print(f"  Error: {e}")

    return None

def test_with_decrypted_token():
    """Try to decrypt and use the token."""
    print("\n4️⃣ Attempting Token Decryption...")

    # Try hex decode
    try:
        token_bytes = bytes.fromhex(AUTH_TOKEN)
        print(f"  Hex decoded: {len(token_bytes)} bytes")

        # Try as base64
        token_b64 = base64.b64encode(token_bytes).decode()
        print(f"  Base64 encoded: {token_b64[:50]}...")

        # Test with decoded token
        url = "https://ai-api.wondershare.cc/v1/ai/user/info"

        headers = {
            "Authorization": f"Bearer {token_b64}",
            "X-WSID": CONFIG["wsid"],
            "User-Agent": f"Filmora/{CONFIG['version']}"
        }

        response = requests.get(url, headers=headers, timeout=10)
        print(f"  API Status: {response.status_code}")

    except Exception as e:
        print(f"  Decode error: {e}")

def check_filmora_server():
    """Check if we can reach Filmora's account server."""
    print("\n5️⃣ Checking Account Server...")

    url = "https://account.wondershare.cc/api/v1/user/check"

    headers = {
        "User-Agent": f"Filmora/{CONFIG['version']}",
        "X-Product-Id": CONFIG["product_id"],
        "X-Product-Version": CONFIG["version"]
    }

    params = {
        "wsid": CONFIG["wsid"]
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    # Test different authentication methods
    test_with_bearer_token()
    test_with_custom_header()

    # Try login API
    session_token = test_wondershare_login_api()

    # Try decryption
    test_with_decrypted_token()

    # Check account server
    check_filmora_server()

    print("\n" + "="*60)
    print("📊 Authentication test complete")
    print("="*60)
    print("\nNext steps:")
    print("1. If any method worked, use that authentication")
    print("2. Otherwise, need to capture live session from running Filmora")
    print("3. May need to reverse engineer the encryption method")