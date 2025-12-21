# 🎯 Fiddler Capture Guide for Filmora

## What to Look For in Fiddler:

### 1. **Enable ALL HTTPS Decryption**
- Tools → Options → HTTPS
- ✅ Decrypt HTTPS traffic
- ✅ Ignore server certificate errors

### 2. **Clear Filters** (Important!)
- Remove ALL filters first
- View ALL traffic
- Then look for these patterns

### 3. **Domains to Watch For:**

#### Primary Wondershare Domains:
- `prod-web.wondershare.cc`
- `aigc.wondershare.cc`
- `api-ai.wondershare.cc`
- `account.wondershare.cc`
- `vip.wondershare.cc`

#### AI Service Endpoints (Filmora might connect directly):
- `*.googleapis.com` (Google Veo)
- `*.openai.com` (OpenAI Sora)
- `*.azure.com` (Microsoft Azure AI)
- `*.huggingface.co`
- `*.replicate.com`

#### CDN/Cloud Storage:
- `*.cloudfront.net`
- `*.s3.amazonaws.com`
- `*.blob.core.windows.net`

### 4. **Look for These Headers:**

In the REQUEST headers, look for:
- `wsid: `
- `X-Ws-Id: `
- `X-Session-Token: `
- `Authorization: Bearer `
- `X-Auth-Token: `
- `X-Client-Sign: `
- `X-Product-Id: `

### 5. **Trigger API Calls in Filmora:**

Try these specific actions:
1. **AI Copilot Editing** (most likely to trigger API)
2. **AI Text to Video**
3. **AI Image Generation**
4. **AI Music Generation**
5. **Cloud Sync** (if you have cloud features)
6. **Check for Updates** (Help menu)

### 6. **If You See NO Requests:**

Filmora might be:
- Using cached credentials (try signing out/in)
- Using a local proxy (check Task Manager for proxy processes)
- Using certificate pinning (try disabling Windows Firewall temporarily)

### 7. **Alternative Method - Wireshark:**

If Fiddler doesn't work:
1. Download Wireshark
2. Filter: `tcp.port == 443 && ip.dst != 127.0.0.1`
3. Look for TLS handshakes to wondershare domains

### 8. **Check These Specific URLs:**

Look for POST requests to:
- `https://prod-web.wondershare.cc/api/v1/aigc/workflow`
- `https://prod-web.wondershare.cc/api/v1/aigc/text2video`
- `https://account.wondershare.cc/api/v1/user/info`

### 9. **Manual Test:**

In Fiddler's Composer tab, try:
```
GET https://prod-web.wondershare.cc/api/v1/user/info HTTP/1.1
Host: prod-web.wondershare.cc
User-Agent: Wondershare Filmora/15.0.12.16430
client_sign: REDACTED_WONDERSHARE_SIGN
```

If this returns a response, you're on the right track!

## 📋 What to Copy:

When you find a request to wondershare.cc:
1. Click on it
2. Go to "Inspectors" → "Headers"
3. Copy ALL request headers
4. Save them to a file: `captured_headers.txt`

We specifically need:
- `wsid` value
- `Cookie` value (if present)
- Any `X-` headers