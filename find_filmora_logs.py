#!/usr/bin/env python3
"""
Find Filmora API calls in log files
"""

import os
import re
from pathlib import Path
import json

print("=" * 60)
print("🔍 SEARCHING FILMORA LOGS FOR API CALLS")
print("=" * 60)

# Possible log locations on Windows (accessed from WSL)
log_paths = [
    # User AppData
    Path("/mnt/c/Users") / os.environ.get("USER", "elliott") / "AppData/Roaming/Wondershare/Wondershare Filmora/logs",
    Path("/mnt/c/Users") / os.environ.get("USER", "elliott") / "AppData/Local/Wondershare/Wondershare Filmora/logs",
    Path("/mnt/c/Users") / os.environ.get("USER", "elliott") / "AppData/Local/Temp",

    # Program Files
    Path("/mnt/c/Program Files/Wondershare/Wondershare Filmora/logs"),
    Path("/mnt/c/Program Files (x86)/Wondershare/Wondershare Filmora/logs"),
    Path("/mnt/c/ProgramData/Wondershare/Wondershare Filmora/logs"),
]

# Patterns to search for
api_patterns = [
    r"wsid[:\s=]+([a-zA-Z0-9\-_]+)",
    r"token[:\s=]+([a-zA-Z0-9\-_]+)",
    r"https?://[^\s]*wondershare[^\s]*",
    r"https?://[^\s]*filmora[^\s]*",
    r"Authorization[:\s]+Bearer\s+([a-zA-Z0-9\-_]+)",
    r"X-Auth-Token[:\s]+([a-zA-Z0-9\-_]+)",
    r"session[:\s=]+([a-zA-Z0-9\-_]+)",
]

found_items = []

for log_path in log_paths:
    if log_path.exists():
        print(f"\n📁 Checking: {log_path}")

        # Look for log files
        for file in log_path.glob("**/*.log"):
            print(f"  📄 {file.name}")
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Search for patterns
                    for pattern in api_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            for match in matches[:3]:  # First 3 matches
                                if len(str(match)) > 10:  # Skip short matches
                                    print(f"    ✅ Found: {str(match)[:50]}...")
                                    found_items.append(str(match))
            except Exception as e:
                print(f"    ⚠️ Cannot read: {e}")

        # Also check .txt and .json files
        for ext in ['*.txt', '*.json']:
            for file in log_path.glob(f"**/{ext}"):
                if 'auth' in file.name.lower() or 'token' in file.name.lower() or 'session' in file.name.lower():
                    print(f"  📄 Interesting file: {file.name}")
                    try:
                        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:1000]  # First 1000 chars
                            if 'wsid' in content.lower() or 'token' in content.lower():
                                print(f"    ✅ Contains auth data!")
                    except:
                        pass

if found_items:
    print("\n" + "=" * 60)
    print("FOUND POTENTIAL API DATA:")
    print("=" * 60)
    for item in set(found_items):  # Unique items
        print(f"  {item[:80]}")
else:
    print("\n⚠️ No API calls found in logs")

print("\n" + "=" * 60)
print("ALTERNATIVE: Check Windows Event Viewer")
print("=" * 60)
print("1. Open Event Viewer on Windows")
print("2. Go to: Applications and Services Logs")
print("3. Look for Wondershare or Filmora")
print("4. Check recent events for API calls")
print("=" * 60)