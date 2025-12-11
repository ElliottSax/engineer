#!/usr/bin/env python3
"""
Filmora API Capture using mitmproxy
Captures ALL Filmora API calls including wsid/session tokens
"""

import json
import time
from pathlib import Path
from datetime import datetime
from mitmproxy import http, ctx
from typing import Optional

class FilmoraAPICapture:
    """mitmproxy addon to capture Filmora API calls"""

    def __init__(self):
        self.captured_auth = {}
        self.api_calls = []
        self.output_dir = Path("captured_api_data")
        self.output_dir.mkdir(exist_ok=True)

    def request(self, flow: http.HTTPFlow) -> None:
        """Intercept HTTP requests"""

        # Check if it's a Wondershare/Filmora API call
        if any(domain in flow.request.pretty_host for domain in [
            "wondershare.cc", "wondershare.com", "filmora.com", "aigc.wondershare"
        ]):
            ctx.log.info(f"🎯 Captured Filmora API: {flow.request.pretty_url}")

            # Extract authentication headers
            headers = dict(flow.request.headers)

            # Look for wsid
            if "wsid" in headers:
                self.captured_auth["wsid"] = headers["wsid"]
                ctx.log.info(f"✅ Found wsid: {headers['wsid'][:30]}...")

            # Look for session token
            if "session" in headers or "session-token" in headers:
                token = headers.get("session") or headers.get("session-token")
                self.captured_auth["session_token"] = token
                ctx.log.info(f"✅ Found session: {token[:30]}...")

            # Look for authorization
            if "authorization" in headers:
                self.captured_auth["authorization"] = headers["authorization"]
                ctx.log.info(f"✅ Found auth: {headers['authorization'][:30]}...")

            # Look for client_sign
            if "client_sign" in headers or "client-sign" in headers:
                sign = headers.get("client_sign") or headers.get("client-sign")
                self.captured_auth["client_sign"] = sign
                ctx.log.info(f"✅ Found client_sign: {sign[:30]}...")

            # Extract device ID from headers or URL
            if "device_id" in headers or "device-id" in headers:
                device = headers.get("device_id") or headers.get("device-id")
                self.captured_auth["device_id"] = device

            # Store API call details
            api_call = {
                "timestamp": time.time(),
                "method": flow.request.method,
                "url": flow.request.pretty_url,
                "headers": headers,
                "body": flow.request.text if flow.request.content else None,
                "is_ai_call": self._is_ai_call(flow.request.pretty_url, flow.request.text)
            }

            self.api_calls.append(api_call)

            # Save immediately
            self._save_capture()

            # Log AI calls specially
            if api_call["is_ai_call"]:
                ctx.log.info(f"🤖 AI API Call: {flow.request.pretty_url}")
                if flow.request.text:
                    try:
                        body = json.loads(flow.request.text)
                        if "workflow_id" in body:
                            ctx.log.info(f"  Workflow: {body['workflow_id']}")
                        if "alg_code" in body:
                            ctx.log.info(f"  Algorithm: {body['alg_code']}")
                        if "prompt" in body.get("params", {}):
                            ctx.log.info(f"  Prompt: {body['params']['prompt'][:50]}...")
                    except:
                        pass

    def response(self, flow: http.HTTPFlow) -> None:
        """Intercept HTTP responses"""

        if any(domain in flow.request.pretty_host for domain in [
            "wondershare.cc", "wondershare.com", "filmora.com"
        ]):
            # Log successful responses
            if flow.response.status_code == 200:
                ctx.log.info(f"✅ Success: {flow.request.pretty_url}")

                # Check for task IDs in response
                if flow.response.text:
                    try:
                        data = json.loads(flow.response.text)
                        if "task_id" in data:
                            ctx.log.info(f"  Task ID: {data['task_id']}")
                        if "video_url" in data:
                            ctx.log.info(f"  Video URL: {data['video_url']}")
                    except:
                        pass

    def _is_ai_call(self, url: str, body: Optional[str]) -> bool:
        """Check if this is an AI-related API call"""

        ai_indicators = [
            "aigc", "text2video", "text_to_video", "img2video",
            "text2img", "workflow", "generation", "/ai/", "/ml/"
        ]

        # Check URL
        url_lower = url.lower()
        if any(indicator in url_lower for indicator in ai_indicators):
            return True

        # Check body
        if body:
            try:
                data = json.loads(body)
                if any(key in data for key in ["workflow_id", "alg_code", "prompt"]):
                    return True
            except:
                pass

        return False

    def _save_capture(self):
        """Save captured data to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save authentication
        if self.captured_auth:
            auth_file = self.output_dir / f"auth_{timestamp}.json"
            with open(auth_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "auth": self.captured_auth,
                    "api_calls_count": len(self.api_calls)
                }, f, indent=2)
            ctx.log.info(f"💾 Saved auth: {auth_file}")

        # Save API calls
        if self.api_calls:
            calls_file = self.output_dir / f"api_calls_{timestamp}.json"
            with open(calls_file, 'w') as f:
                json.dump(self.api_calls, f, indent=2, default=str)
            ctx.log.info(f"💾 Saved {len(self.api_calls)} API calls")

    def done(self):
        """Called when mitmproxy shuts down"""

        ctx.log.info("="*60)
        ctx.log.info("CAPTURE SUMMARY:")
        ctx.log.info(f"Total API calls: {len(self.api_calls)}")
        ctx.log.info(f"AI calls: {sum(1 for c in self.api_calls if c['is_ai_call'])}")

        if self.captured_auth:
            ctx.log.info("Captured Authentication:")
            for key, value in self.captured_auth.items():
                ctx.log.info(f"  {key}: {str(value)[:50]}...")

        ctx.log.info("="*60)

# Create addon instance
addons = [FilmoraAPICapture()]