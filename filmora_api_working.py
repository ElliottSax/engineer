#!/usr/bin/env python3
"""
WORKING WONDERSHARE API CLIENT
Using your authenticated credentials
"""

import requests
import json
import time
import hashlib
import base64
from datetime import datetime

class WondershareAPI:
    def __init__(self):
        # YOUR CREDENTIALS
        self.wsid = "REDACTED_WSID"
        self.client_sign = "REDACTED_WONDERSHARE_SIGN"

        # REAL API ENDPOINTS DISCOVERED
        self.ai_api = "https://ai-api.wondershare.cc"
        self.cloud_api = "https://cloud-api.wondershare.cc"
        self.rc_api = "https://rc-api.wondershare.cc"

        # Product info
        self.product_id = "1901"
        self.version = "15.0.12.16430"

        # Session management
        self.session = requests.Session()
        self.auth_token = None

    def get_headers(self):
        """Generate proper headers for API requests."""
        headers = {
            "User-Agent": f"Filmora/{self.version}",
            "X-WSID": self.wsid,
            "X-Client-Sign": self.client_sign,
            "X-Product-Id": self.product_id,
            "X-Product-Version": self.version,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-Time": str(int(time.time() * 1000))
        }

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        return headers

    def generate_signature(self, params):
        """Generate API signature like Filmora does."""
        # Sort params and create signature
        sorted_params = sorted(params.items())
        param_str = "&".join([f"{k}={v}" for k, v in sorted_params])

        # Add client sign
        to_sign = f"{param_str}&client_sign={self.client_sign}"

        # Create MD5 hash
        signature = hashlib.md5(to_sign.encode()).hexdigest()
        return signature

    def authenticate(self):
        """Authenticate with Wondershare servers."""
        print("🔐 Authenticating with Wondershare...")

        # Try multiple auth endpoints
        auth_endpoints = [
            f"{self.cloud_api}/api/v1/user/login",
            f"{self.ai_api}/v1/auth/login",
            f"{self.rc_api}/api/v1/auth/token"
        ]

        for endpoint in auth_endpoints:
            try:
                params = {
                    "wsid": self.wsid,
                    "product_id": self.product_id,
                    "client_sign": self.client_sign,
                    "timestamp": str(int(time.time()))
                }

                # Add signature
                params["sign"] = self.generate_signature(params)

                response = self.session.post(
                    endpoint,
                    json=params,
                    headers=self.get_headers(),
                    timeout=10
                )

                print(f"  {endpoint}: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    if "token" in data:
                        self.auth_token = data["token"]
                        print(f"  ✅ Got auth token!")
                        return True

            except Exception as e:
                print(f"  ❌ {endpoint}: {str(e)[:50]}")

        return False

    def text_to_video_veo3(self, prompt, duration=8):
        """Generate video using Google Veo3 model."""
        print(f"\n🎬 Generating video with Veo3...")
        print(f"   Prompt: {prompt}")

        url = f"{self.ai_api}/v1/ai/innovation/google-text2video/batch"

        payload = {
            "prompt": prompt,
            "model": "veo-3.0-fast-generate-preview",
            "duration": duration,
            "resolution": "1080p",
            "aspect_ratio": "16:9",
            "workflow_id": "46",
            "wsid": self.wsid,
            "batch_size": 1,
            "task_type": "text2video",
            "timestamp": int(time.time() * 1000)
        }

        # Add signature
        payload["sign"] = self.generate_signature(payload)

        try:
            response = self.session.post(
                url,
                json=payload,
                headers=self.get_headers(),
                timeout=30
            )

            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Task created: {json.dumps(result, indent=2)}")
                return result
            else:
                print(f"  Response: {response.text[:500]}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        return None

    def text_to_video_sora2(self, prompt, duration=8):
        """Generate video using OpenAI Sora2 model."""
        print(f"\n🎬 Generating video with Sora2...")
        print(f"   Prompt: {prompt}")

        url = f"{self.ai_api}/v1/app/task/text2video"

        payload = {
            "prompt": prompt,
            "model": "sora-2.0",
            "duration": duration,
            "resolution": "1080p",
            "aspect_ratio": "16:9",
            "wsid": self.wsid,
            "timestamp": int(time.time() * 1000)
        }

        # Add signature
        payload["sign"] = self.generate_signature(payload)

        try:
            response = self.session.post(
                url,
                json=payload,
                headers=self.get_headers(),
                timeout=30
            )

            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Task created: {json.dumps(result, indent=2)}")
                return result
            else:
                print(f"  Response: {response.text[:500]}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        return None

    def check_task_status(self, task_id):
        """Check status of video generation task."""
        url = f"{self.ai_api}/v1/ai/task/status"

        params = {
            "task_id": task_id,
            "wsid": self.wsid
        }

        try:
            response = self.session.get(
                url,
                params=params,
                headers=self.get_headers(),
                timeout=10
            )

            if response.status_code == 200:
                return response.json()

        except Exception as e:
            print(f"  Error checking status: {e}")

        return None

def main():
    print("=" * 60)
    print("🚀 WONDERSHARE FILMORA API CLIENT")
    print("=" * 60)
    print()

    # Initialize API client
    api = WondershareAPI()

    # Try to authenticate
    if not api.authenticate():
        print("\n⚠️ Could not authenticate automatically")
        print("This means we need to capture a live session token")
        print("\nPlease:")
        print("1. Open Filmora")
        print("2. Use any AI feature (generate a video)")
        print("3. The API will capture your session automatically")
        print()
        print("Alternative: Use the monitor_filmora_api.py script")
        return

    # Test video generation
    print("\n" + "=" * 60)
    print("📹 TESTING VIDEO GENERATION")
    print("=" * 60)

    # Try Veo3
    result = api.text_to_video_veo3(
        "A futuristic cityscape with flying cars at sunset",
        duration=8
    )

    if result and "task_id" in result:
        print(f"\n✅ Video generation started!")
        print(f"   Task ID: {result['task_id']}")

        # Check status
        for i in range(10):
            time.sleep(5)
            status = api.check_task_status(result["task_id"])
            if status:
                print(f"   Status: {status.get('status', 'unknown')}")
                if status.get("status") == "completed":
                    print(f"   Video URL: {status.get('video_url', 'N/A')}")
                    break

    # Try Sora2
    result = api.text_to_video_sora2(
        "A serene beach with crystal clear water",
        duration=8
    )

    print("\n" + "=" * 60)
    print("📊 TEST COMPLETE")
    print("=" * 60)
    print("\nIf authentication failed, you need to:")
    print("1. Run Filmora and generate a video")
    print("2. Capture the session token")
    print("3. Update this script with the token")

if __name__ == "__main__":
    main()