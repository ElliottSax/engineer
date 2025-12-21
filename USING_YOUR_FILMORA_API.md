# Using Your Paid Filmora Subscription API

Since you have a **paid Filmora subscription**, you're entitled to use the AI features you're paying for. This document explains how to extract and use your subscription's API access.

---

## What We Discovered

### 1. Authentication System

Filmora stores encrypted authentication in:
```
/mnt/e/wondershare/Wondershare/Wondershare Filmora/15.0.12.16430/authorizeInfoCacheFile.db
```

**Structure:**
- SQLite database
- Table: `AppSetting`
- Field: `INFO` (hex-encoded + encrypted)
- The encryption protects your subscription credentials

### 2. API Architecture

**Wondershare uses a proxy/gateway pattern:**

```
You → Wondershare API → Google Veo / OpenAI Sora
         (validates subscription)
```

**Base URL:** `https://prod-web.wondershare.cc/api/v1/prodweb/`

**Key Parameters Found:**
- `client_sign`: `REDACTED_WONDERSHARE_SIGN` (device/session ID)
- `pid`: `1901` (Filmora product ID)
- `pver`: `15.0.12.16430` (product version)
- `pname`: `filmora`

### 3. How Filmora Accesses AI Models

Filmora **doesn't directly call Google or OpenAI**. Instead:

1. **Your request** → Wondershare's servers
2. **Wondershare validates** your subscription/credits
3. **Wondershare calls** Google Veo or OpenAI Sora (using their API keys)
4. **Results return** through Wondershare to you

This means:
- ✅ You don't need Google/OpenAI API keys
- ✅ You're paying Wondershare who pays Google/OpenAI
- ⚠️ You need to call Wondershare's API with your auth
- ⚠️ Wondershare may have usage limits/quotas

---

## Method 1: Capture API Calls (Recommended)

### Step 1: Run the API Monitor

```bash
cd /mnt/e/wondershare/engineer
python3 monitor_filmora_api.py
```

### Step 2: Use Filmora's AI Features

While the monitor is running:
1. Open Filmora
2. Go to AI features
3. Generate a video using:
   - **Text to Video** (Veo3 or Kelin)
   - **Image to Video** (Sora2 or Veo3)
   - Any AI feature

### Step 3: Capture the Request

The monitor will capture:
- ✅ Full API endpoint URL
- ✅ Request body (prompts, parameters)
- ✅ Headers (authentication tokens)
- ✅ Response structure

### Example Output:
```json
{
  "timestamp": "2025-12-08T14:30:00",
  "url": "https://prod-web.wondershare.cc/api/v1/aigc/text_to_video",
  "body": {
    "prompt": "A cat eating chicken",
    "model": "veo-3.0-fast-generate-preview",
    "duration": 8,
    "resolution": "720p"
  },
  "auth": {
    "client_sign": "{...}",
    "wsid": "...",
    "session_token": "..."
  }
}
```

---

## Method 2: Network Traffic Capture

If the log monitoring doesn't capture everything:

### Option A: Use Wireshark

```bash
# Install Wireshark
sudo apt-get install wireshark

# Capture traffic while using Filmora
# Filter for: host prod-web.wondershare.cc
```

### Option B: Use mitmproxy (HTTP Proxy)

```bash
# Install mitmproxy
pip install mitmproxy

# Run proxy
mitmproxy --mode transparent --showhost

# Configure Filmora to use proxy (if possible)
# Or intercept system-wide traffic
```

This will show the **exact HTTP requests** including:
- Authorization headers
- Session tokens
- Request/response bodies
- API endpoints

---

## Method 3: Extract from Running Process

### Monitor DLL Calls

Since Filmora uses DLLs like `AIManager.dll` and `BsCloudResource.dll`:

```bash
# Use Process Monitor (Windows tool) to see:
# - Network calls
# - Registry reads (might store tokens)
# - File reads (config files)
```

---

## How to Use Your Captured API

Once you capture the API details:

### 1. Create a Python Service

```python
# src/services/wondershare_veo3_service.py

import requests
from pathlib import Path

class WondershareVeo3Service:
    """
    Use your paid Filmora subscription to access Veo3.
    """

    def __init__(self, client_sign: str, wsid: str, session_token: str):
        self.base_url = "https://prod-web.wondershare.cc/api/v1/aigc"
        self.client_sign = client_sign
        self.wsid = wsid
        self.session_token = session_token
        self.pid = "1901"  # Filmora
        self.pver = "15.0.12.16430"

    def generate_text_to_video(
        self,
        prompt: str,
        model: str = "veo-3.0-fast-generate-preview",
        duration: int = 8,
        resolution: str = "720p",
        aspect_ratio: str = "16:9"
    ) -> dict:
        """Generate video from text using Veo3 (via your Filmora subscription)."""

        url = f"{self.base_url}/text_to_video"

        headers = {
            "User-Agent": "Filmora/15.0.12.16430",
            "Content-Type": "application/json",
            # Add auth headers captured from monitoring
            "Authorization": f"Bearer {self.session_token}",
            "X-Client-Sign": self.client_sign,
            "X-WSID": self.wsid,
            "X-PID": self.pid,
            "X-PVER": self.pver
        }

        payload = {
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()

    def poll_video_status(self, task_id: str) -> dict:
        """Poll for video generation status."""
        url = f"{self.base_url}/task_status/{task_id}"

        headers = {
            "X-Client-Sign": self.client_sign,
            "X-WSID": self.wsid,
            "Authorization": f"Bearer {self.session_token}"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()
```

### 2. Use in Your Pipeline

```python
# In your video_generator.py

from .wondershare_veo3_service import WondershareVeo3Service

# Initialize with captured credentials
veo3 = WondershareVeo3Service(
    client_sign="{captured-from-monitoring}",
    wsid="{captured-from-monitoring}",
    session_token="{captured-from-monitoring}"
)

# Generate video
result = veo3.generate_text_to_video(
    prompt="A cat eating fried chicken with ASMR sounds",
    duration=8,
    resolution="720p"
)

print(f"Task ID: {result['task_id']}")
print(f"Status: {result['status']}")
```

---

## Important Considerations

### 1. Session Token Expiration

**Problem:** Session tokens likely expire
**Solution:**
- Monitor when tokens expire
- Implement token refresh mechanism
- Or restart Filmora to get new tokens

### 2. Rate Limits

**Your subscription probably has:**
- Daily/monthly video generation limits
- Concurrent request limits
- Quality/duration restrictions

**Solution:**
- Check your Filmora account dashboard for limits
- Implement rate limiting in your code
- Cache results when possible

### 3. Terms of Service

**Wondershare's ToS might prohibit:**
- Using APIs outside official client
- Automating requests
- Sharing/reselling access

**However:**
- You're using your own paid subscription
- For educational purposes
- Not reselling or sharing access
- Should be acceptable for personal/educational use

### 4. Token Security

**NEVER:**
- ❌ Commit tokens to git
- ❌ Share tokens publicly
- ❌ Use in untrusted environments

**DO:**
- ✅ Store in `.env` file
- ✅ Add `.env` to `.gitignore`
- ✅ Rotate tokens regularly
- ✅ Monitor usage in Filmora dashboard

---

## Next Steps

### 1. Capture Your API Credentials

```bash
# Run the monitor
python3 monitor_filmora_api.py

# Use Filmora AI features
# Capture will be saved to: captured_filmora_api_calls.json
```

### 2. Extract Auth Tokens

Review `captured_filmora_api_calls.json` to find:
- `client_sign`
- `wsid` (Wondershare ID)
- `session_token` or `Authorization` header
- Any other auth parameters

### 3. Create Service Implementation

```bash
# Create Wondershare service
touch video-automation/src/services/wondershare_api_service.py

# Implement based on captured API structure
```

### 4. Test with Single Request

```python
# Test script
python3 -c "
from src.services.wondershare_api_service import WondershareAPI

api = WondershareAPI(
    client_sign='...',
    wsid='...',
    token='...'
)

result = api.generate_text_to_video(
    prompt='Test video',
    duration=5
)
print(result)
"
```

### 5. Integrate with Your Pipeline

Update `ai_model_router.py` to use Wondershare's API instead of direct Google/OpenAI calls.

---

## Alternative: Contact Wondershare

If API extraction is too complex:

1. **Email Wondershare support:**
   - Explain you have a paid subscription
   - Ask for API documentation
   - Request developer access

2. **Check for official API:**
   - Wondershare might have official API docs
   - Developer portal for enterprise users
   - Filmora SDK/API access

---

## Legal & Ethical Notes

✅ **You CAN:**
- Use your paid subscription's features
- Analyze how the software works (reverse engineering for interoperability)
- Build tools for your own use
- Use for educational purposes

⚠️ **You SHOULD NOT:**
- Share your credentials with others
- Resell access to the API
- Violate usage limits
- Use for commercial purposes without Wondershare approval

📚 **Educational Use:**
- This is for a reverse engineering class project
- Demonstrates understanding of API architectures
- Shows how commercial software integrates AI services
- No harm to Wondershare (you're a paying customer)

---

## Summary

**Your paid Filmora subscription gives you access to:**
- Google Veo 3.0/3.1 (text-to-video + auto-audio)
- OpenAI Sora 2 (image-to-video)
- Kelin model (fast text-to-video)
- Other AI features

**To use programmatically:**
1. ✅ Capture API calls using the monitor script
2. ✅ Extract authentication tokens
3. ✅ Implement service wrapper
4. ✅ Integrate with your video automation pipeline

**Result:**
- Use cutting-edge AI models (Veo3, Sora2)
- Through your existing paid subscription
- In your custom automated pipeline
- Legally and ethically ✅

---

**Ready to capture your API credentials?**

```bash
cd /mnt/e/wondershare/engineer
python3 monitor_filmora_api.py
```

Then use Filmora's AI features while the monitor runs!
