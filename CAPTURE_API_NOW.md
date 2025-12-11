# 🎯 CAPTURE FILMORA API - Step-by-Step Guide

## ✅ What We Have So Far

**Found from logs:**
- **client_sign:** `REDACTED_WONDERSHARE_SIGN`
- **Base API:** `https://prod-web.wondershare.cc/api/v1/prodweb/`
- **Product ID:** `1901` (Filmora)
- **Version:** `15.0.12.16430`

## 🎬 What We Need Next

**Missing (need to capture):**
- ✗ Actual video generation API endpoint
- ✗ Session/auth token (wsid or Bearer token)
- ✗ Request format for Veo3/Sora2
- ✗ Response format (task ID, polling, etc.)

---

## 📋 STEP-BY-STEP INSTRUCTIONS

### Option 1: Real-Time Monitoring (Recommended)

**Do this NOW:**

1. **Open WSL Terminal #1** (this one):
   ```bash
   cd /mnt/e/wondershare/engineer
   python3 monitor_filmora_api.py
   ```
   Leave this running!

2. **Open Filmora on Windows:**
   - Launch Filmora from Start Menu
   - Sign in with your account

3. **Use ANY AI Feature:**

   **Option A - Text to Video (easiest):**
   - Click "AI" in top menu
   - Select "Text to Video"
   - Choose "Google Veo3" or "Normal mode"
   - Enter prompt: "A cat eating chicken"
   - Click "Generate"

   **Option B - Image to Video:**
   - Import an image
   - Right-click → "AI Tools" → "Image to Video"
   - Select "Sora 2" or "Veo 3.1"
   - Add prompt describing motion
   - Click "Generate"

4. **Watch Terminal #1:**
   - You'll see API calls appear in real-time
   - Let it capture the full request/response
   - Press Ctrl+C when done

5. **Check Results:**
   ```bash
   cat captured_filmora_api_calls.json
   ```

---

### Option 2: Network Traffic Capture

If real-time monitoring doesn't work:

**Windows (easier):**
1. Download Fiddler: https://www.telerik.com/fiddler
2. Run Fiddler before launching Filmora
3. Use Filmora AI features
4. Search Fiddler for: `prod-web.wondershare.cc`
5. Export the captured requests

**WSL (advanced):**
```bash
# Install tcpdump
sudo apt-get install tcpdump

# Capture traffic (need to run before using Filmora)
sudo tcpdump -i any -w filmora_traffic.pcap host prod-web.wondershare.cc

# Use Filmora AI features

# Stop capture (Ctrl+C)

# Analyze with tshark
tshark -r filmora_traffic.pcap -Y "http" -T fields -e http.request.full_uri -e http.file_data
```

---

### Option 3: Check Windows AppData

Filmora might cache API responses:

```bash
# From WSL, check Windows AppData
ls -la /mnt/c/Users/ellio/AppData/Local/Wondershare/
ls -la /mnt/c/Users/ellio/AppData/Roaming/Wondershare/

# Look for cache files
find /mnt/c/Users/ellio/AppData -name "*filmora*" -name "*.json" -o -name "*.cache" 2>/dev/null

# Search for API-related files
grep -r "prod-web" /mnt/c/Users/ellio/AppData/Local/Wondershare/ 2>/dev/null
```

---

## 🔍 What to Look For

When you capture the API call, you need:

### Request Details:
```http
POST https://prod-web.wondershare.cc/api/v1/aigc/text_to_video
Headers:
  Authorization: Bearer <TOKEN>
  X-Client-Sign: REDACTED_WONDERSHARE_SIGN
  X-WSID: <YOUR_WSID>
  X-PID: 1901
  Content-Type: application/json

Body:
{
  "prompt": "A cat eating chicken",
  "model": "veo-3.0-fast-generate-preview",
  "duration": 8,
  "resolution": "720p",
  "aspect_ratio": "16:9"
}
```

### Response Details:
```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "task_id": "abc123...",
    "status": "processing",
    "poll_url": "https://prod-web.wondershare.cc/api/v1/aigc/task_status/abc123"
  }
}
```

---

## 💾 Expected Output Files

After capturing, you should have:

1. **`captured_filmora_api_calls.json`**
   - All captured API calls
   - Full request/response data

2. **`log_analysis_results.json`**
   - Summary of endpoints
   - Auth info extracted

3. **Windows log files updated:**
   - New entries in Filmora logs
   - Check timestamp to find your AI request

---

## 🚀 Quick Test

**Fastest way to test:**

1. Run monitor in background:
   ```bash
   python3 monitor_filmora_api.py &
   MONITOR_PID=$!
   ```

2. Use Filmora (generate 1 short video)

3. Stop monitor:
   ```bash
   kill $MONITOR_PID
   ```

4. Check what was captured:
   ```bash
   cat captured_filmora_api_calls.json | python3 -m json.tool
   ```

---

## ❓ Troubleshooting

### If monitoring doesn't capture anything:

1. **Check log file location:**
   ```bash
   ls -lt "/mnt/e/wondershare/Wondershare/Wondershare Filmora/15.0.12.16430/log/" | head
   ```

2. **Make sure Filmora is running:**
   ```bash
   tasklist.exe | grep -i filmora
   ```

3. **Manually check latest log:**
   ```bash
   tail -f "/mnt/e/wondershare/Wondershare/Wondershare Filmora/15.0.12.16430/log/APP@MAIN_Wondershare Filmora.exe_2025_12_08_*.log"
   ```

### If you see errors in Filmora:

- Make sure you're signed in
- Check your subscription is active
- Try a different AI feature
- Restart Filmora

---

## ✅ Success Criteria

You've succeeded when you have:
- [x] client_sign (already have this!)
- [ ] wsid or session token
- [ ] Text-to-video API endpoint
- [ ] Image-to-video API endpoint
- [ ] Full request/response format
- [ ] Task polling mechanism

---

## 🎯 Next Steps After Capture

Once you have the API details, I'll help you:
1. Create Python service wrapper
2. Test authentication
3. Generate test video via API
4. Integrate with your automation pipeline
5. Deploy the hybrid system!

---

**Ready? Start with Option 1 above!**

Open Filmora and use an AI feature while the monitor runs.
