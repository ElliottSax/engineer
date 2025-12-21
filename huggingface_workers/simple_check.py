#!/usr/bin/env python3
"""
Simple check for HuggingFace Spaces (no heavy dependencies)
"""

import os
from huggingface_hub import HfApi

USERNAME = "elliottsax"
# SECURITY: Token must come from environment variable only
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable")
    exit(1)

SPACES = [
    ("autocoder-analysis-worker", "🤖"),
    ("autocoder-test-planning-worker", "🧪"),
    ("autocoder-implementation-worker", "⚙️"),
    ("autocoder-dashboard", "📊")
]

print("""
╔══════════════════════════════════════════════════════════════╗
║  HuggingFace Spaces - Quick Status Check                     ║
╚══════════════════════════════════════════════════════════════╝
""")

api = HfApi(token=HF_TOKEN)

print(f"Username: {USERNAME}\n")

all_running = True

for space_name, emoji in SPACES:
    repo_id = f"{USERNAME}/{space_name}"

    try:
        runtime = api.get_space_runtime(repo_id=repo_id)
        status = runtime.stage

        if status == "RUNNING":
            print(f"✅ {emoji} {space_name}: RUNNING")
        elif status == "BUILDING":
            print(f"🔄 {emoji} {space_name}: BUILDING (wait 1-2 min)")
            all_running = False
        else:
            print(f"⚠️  {emoji} {space_name}: {status}")
            all_running = False

    except Exception as e:
        print(f"❌ {emoji} {space_name}: Need secrets configured")
        all_running = False

print(f"\n{'='*60}")

if all_running:
    print("🟢 ALL SYSTEMS OPERATIONAL")
    print(f"\n📊 Dashboard: https://huggingface.co/spaces/{USERNAME}/autocoder-dashboard")
else:
    print("⚠️  ACTION REQUIRED")
    print(f"\nAdd secrets to each Space:")
    print(f"1. Go to Space → Settings → Repository secrets")
    print(f"2. Add HF_TOKEN = {HF_TOKEN}")
    print(f"3. Add HF_USERNAME = {USERNAME}")
    print(f"\nDo this for all 4 Spaces, then run this check again.")

print(f"{'='*60}\n")
