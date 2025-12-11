#!/usr/bin/env python3
"""
Test Wondershare API with your captured credentials
Your WSID: 426498096
"""

import requests
import json
import time
from pathlib import Path

# YOUR COMPLETE AUTHENTICATION - REAL API ENDPOINTS DISCOVERED!
AUTH_CONFIG = {
    "client_sign": "{2871f7e8-51b4-487a-9ab9-5d99926ee2ebG}",
    "wsid": "426498096",  # Your Wondershare user ID!
    "base_api": "https://ai-api.wondershare.cc",  # REAL AI API HOST!
    "cloud_api": "https://cloud-api.wondershare.cc",
    "rc_api": "https://rc-api.wondershare.cc",
    "product_id": "1901",
    "version": "15.0.12.16430"
}

print("=" * 60)
print("🎯 WONDERSHARE API TEST")
print("=" * 60)
print()
print("✅ Authentication Configured:")
print(f"  Client Sign: {AUTH_CONFIG['client_sign'][:20]}...")
print(f"  WSID: {AUTH_CONFIG['wsid']}")
print(f"  Base API: {AUTH_CONFIG['base_api']}")
print(f"  Product: Filmora {AUTH_CONFIG['version']}")
print()

def test_basic_connection():
    """Test basic API connection with REAL endpoints."""
    print("1️⃣ Testing Basic Connection with AI API...")

    # Test the actual AI API host
    url = f"{AUTH_CONFIG['base_api']}/v1/app/task"  # Real endpoint from VBLCloudConfig

    headers = {
        "User-Agent": f"Filmora/{AUTH_CONFIG['version']}",
        "X-WSID": AUTH_CONFIG["wsid"],
        "X-Client-Sign": AUTH_CONFIG["client_sign"],
        "X-Product-Id": AUTH_CONFIG["product_id"],
        "X-Product-Version": AUTH_CONFIG["version"]
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:200]}...")
        return response.status_code in [200, 401, 403]  # Any response means API exists
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_ai_endpoints():
    """Test REAL AI endpoints from VBLCloudConfig.json."""
    print("\n2️⃣ Testing REAL AI Endpoints (from VBLCloudConfig)...")

    # ACTUAL endpoints from Filmora's config!
    ai_endpoints = [
        "/v1/ai/innovation/google-text2video/batch",  # Veo3 text-to-video
        "/v1/ai/capacity/task/klm_text2video",  # Keling text-to-video
        "/v2/ai/aigc/img2video/batch",  # Image-to-video batch
        "/v1/app/task/text2video",  # General text-to-video
        "/v1/ai/user/info",  # User info endpoint
        "/v1/ai/task/list"  # Task list endpoint
    ]

    headers = {
        "User-Agent": f"Filmora/{AUTH_CONFIG['version']}",
        "Content-Type": "application/json",
        "X-WSID": AUTH_CONFIG["wsid"],
        "X-Client-Sign": AUTH_CONFIG["client_sign"],
        "X-PID": AUTH_CONFIG["product_id"],
        "X-PVER": AUTH_CONFIG["version"]
    }

    for endpoint in ai_endpoints:
        url = AUTH_CONFIG["base_api"] + endpoint
        print(f"\n  Trying: {endpoint}")

        try:
            # Try OPTIONS request first (safer)
            response = requests.options(url, headers=headers, timeout=5)
            print(f"    OPTIONS Status: {response.status_code}")

            if response.status_code in [200, 204, 405]:
                print(f"    ✅ Endpoint exists!")

                # Try GET to see requirements
                response = requests.get(url, headers=headers, timeout=5)
                print(f"    GET Status: {response.status_code}")
                if response.text:
                    print(f"    Response: {response.text[:100]}...")

        except requests.exceptions.Timeout:
            print(f"    ⏱️ Timeout")
        except requests.exceptions.ConnectionError:
            print(f"    ❌ Connection error")
        except Exception as e:
            print(f"    ❌ Error: {type(e).__name__}")

def test_text_to_video_request():
    """Try to make a REAL text-to-video request using Veo3."""
    print("\n3️⃣ Testing Text-to-Video Generation with VEO3...")

    # REAL Veo3 endpoint from VBLCloudConfig!
    url = f"{AUTH_CONFIG['base_api']}/v1/ai/innovation/google-text2video/batch"

    headers = {
        "User-Agent": f"Filmora/{AUTH_CONFIG['version']}",
        "Content-Type": "application/json",
        "X-WSID": AUTH_CONFIG["wsid"],
        "X-Client-Sign": AUTH_CONFIG["client_sign"],
        "X-Product-Id": AUTH_CONFIG["product_id"],
        "X-Product-Version": AUTH_CONFIG["version"]
    }

    # Actual parameters from Filmora's Veo3 config
    payload = {
        "prompt": "A beautiful sunset over the ocean with waves",
        "model": "veo-3.0-fast-generate-preview",  # From TextToVideoVEO3Config.json
        "duration": 8,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "workflow_id": "46",  # From the config
        "wsid": AUTH_CONFIG["wsid"],
        "batch_size": 1,
        "task_type": "text2video"
    }

    print(f"  Endpoint: {url}")
    print(f"  Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"\n  Status: {response.status_code}")
        print(f"  Headers: {dict(response.headers)}")
        print(f"  Response: {response.text}")

        if response.status_code == 200:
            print("\n  🎉 SUCCESS! Text-to-video API works!")
            return response.json()
        else:
            print("\n  ℹ️ API responded but may need different parameters")

    except Exception as e:
        print(f"\n  ❌ Error: {e}")

def save_credentials():
    """Save working credentials to .env file."""
    env_content = f"""# Wondershare Filmora API Credentials
# Captured on: {time.strftime('%Y-%m-%d %H:%M:%S')}
# REAL API ENDPOINTS DISCOVERED!

WONDERSHARE_CLIENT_SIGN={AUTH_CONFIG['client_sign']}
WONDERSHARE_WSID={AUTH_CONFIG['wsid']}
WONDERSHARE_AI_API=https://ai-api.wondershare.cc
WONDERSHARE_CLOUD_API=https://cloud-api.wondershare.cc
WONDERSHARE_RC_API=https://rc-api.wondershare.cc
WONDERSHARE_PRODUCT_ID={AUTH_CONFIG['product_id']}
WONDERSHARE_VERSION={AUTH_CONFIG['version']}

# Key endpoints:
# Text-to-video (Veo3): /v1/ai/innovation/google-text2video/batch
# Text-to-video (Keling): /v1/ai/capacity/task/klm_text2video
# Image-to-video: /v2/ai/aigc/img2video/batch
"""

    env_file = Path("/mnt/e/wondershare/engineer/.env")
    with open(env_file, 'w') as f:
        f.write(env_content)

    print(f"\n✅ Credentials saved to: {env_file}")

if __name__ == "__main__":
    # Test connection
    if test_basic_connection():
        print("\n✅ Basic connection successful!")

    # Test AI endpoints
    test_ai_endpoints()

    # Try text-to-video
    test_text_to_video_request()

    # Save credentials
    save_credentials()

    print("\n" + "=" * 60)
    print("📊 TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review which endpoints responded")
    print("2. Adjust parameters based on responses")
    print("3. Use the working endpoint in production")
    print("\nYour credentials are saved in .env file")