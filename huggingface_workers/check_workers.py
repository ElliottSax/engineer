#!/usr/bin/env python3
"""
Check status of all deployed HuggingFace Spaces workers
"""

import os
import time
from huggingface_hub import HfApi, list_repo_files
from datasets import load_dataset

# Configuration
USERNAME = "elliottsax"
# SECURITY: Token should only come from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable not set")
    HF_TOKEN = None

# Spaces to check
SPACES = [
    {
        "name": "autocoder-analysis-worker",
        "dataset": f"{USERNAME}/autocoder-analysis-results",
        "emoji": "🤖"
    },
    {
        "name": "autocoder-test-planning-worker",
        "dataset": f"{USERNAME}/autocoder-test-planning-results",
        "emoji": "🧪"
    },
    {
        "name": "autocoder-implementation-worker",
        "dataset": f"{USERNAME}/autocoder-implementation-results",
        "emoji": "⚙️"
    },
    {
        "name": "autocoder-dashboard",
        "dataset": None,  # Dashboard doesn't create a dataset
        "emoji": "📊"
    }
]

def check_space_exists(api, repo_id: str) -> bool:
    """Check if Space exists"""
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        return True
    except (ConnectionError, ValueError, PermissionError):
        return False

def check_space_runtime(api, repo_id: str) -> str:
    """Check Space runtime status"""
    try:
        runtime = api.get_space_runtime(repo_id=repo_id)
        return runtime.stage
    except (ConnectionError, AttributeError) as e:
        return f"Error: {str(e)[:50]}"

def check_dataset_exists(dataset_name: str) -> tuple:
    """Check if dataset exists and has data"""
    try:
        dataset = load_dataset(dataset_name, split="train")
        return True, len(dataset)
    except (FileNotFoundError, ValueError, ConnectionError):
        return False, 0

def check_space_files(api, repo_id: str) -> tuple:
    """Check if all required files are uploaded"""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="space")
        required = ["README.md", "requirements.txt", "app.py"]
        has_all = all(f in files for f in required)
        return has_all, files
    except (ConnectionError, ValueError, PermissionError):
        return False, []

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  HuggingFace Workers Status Check                            ║
╚══════════════════════════════════════════════════════════════╝
""")

    api = HfApi(token=HF_TOKEN)

    print(f"Username: {USERNAME}")
    print(f"Checking {len(SPACES)} Spaces...\n")

    all_healthy = True
    secrets_needed = []
    datasets_created = 0

    for space in SPACES:
        space_name = space["name"]
        dataset_name = space["dataset"]
        repo_id = f"{USERNAME}/{space_name}"

        print(f"\n{'='*60}")
        print(f"{space['emoji']} {space_name}")
        print(f"{'='*60}")

        # Check if Space exists
        exists = check_space_exists(api, repo_id)
        if not exists:
            print(f"❌ Space does not exist")
            print(f"   URL: https://huggingface.co/spaces/{repo_id}")
            all_healthy = False
            continue

        print(f"✅ Space exists")
        print(f"   URL: https://huggingface.co/spaces/{repo_id}")

        # Check files
        has_files, files = check_space_files(api, repo_id)
        if has_files:
            print(f"✅ All files uploaded (README.md, requirements.txt, app.py)")
        else:
            print(f"⚠️  Missing files: {files}")

        # Check runtime status
        runtime = check_space_runtime(api, repo_id)
        print(f"   Runtime: {runtime}")

        if runtime == "RUNNING":
            print(f"✅ Space is RUNNING")
        elif runtime == "BUILDING":
            print(f"🔄 Space is BUILDING (wait 1-2 minutes)")
        elif runtime in ["STOPPED", "PAUSED"]:
            print(f"⚠️  Space is {runtime} - may need secrets configured")
            secrets_needed.append(repo_id)
            all_healthy = False
        elif "Error" in str(runtime):
            print(f"❌ Cannot check runtime: {runtime}")
            print(f"   → Likely needs secrets (HF_TOKEN, HF_USERNAME)")
            secrets_needed.append(repo_id)
            all_healthy = False

        # Check dataset (for workers, not dashboard)
        if dataset_name:
            exists, count = check_dataset_exists(dataset_name)
            if exists:
                print(f"✅ Dataset exists: {dataset_name}")
                print(f"   Iterations: {count}")
                datasets_created += 1

                if count > 0:
                    print(f"✅ Worker is generating data!")
                else:
                    print(f"⚠️  Dataset exists but no data yet (wait 1-2 minutes)")
            else:
                print(f"⚠️  Dataset not created yet: {dataset_name}")
                print(f"   → Worker may need secrets or hasn't started yet")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    if all_healthy and datasets_created == 3:
        print("🟢 ALL SYSTEMS OPERATIONAL")
        print(f"   ✅ All 4 Spaces deployed")
        print(f"   ✅ All 3 worker datasets created")
        print(f"   ✅ Workers are generating data")
        print(f"\n📊 Dashboard: https://huggingface.co/spaces/{USERNAME}/autocoder-dashboard")
    elif secrets_needed:
        print("🟡 SPACES NEED CONFIGURATION")
        print(f"\n⚠️  {len(secrets_needed)} Space(s) need secrets configured:\n")
        for repo_id in secrets_needed:
            print(f"   {repo_id}")
            print(f"   → https://huggingface.co/spaces/{repo_id}/settings")

        print(f"\n📝 Add these secrets to EACH Space:")
        print(f"   Name:  HF_TOKEN")
        print(f"   Value: {HF_TOKEN}")
        print(f"   ")
        print(f"   Name:  HF_USERNAME")
        print(f"   Value: {USERNAME}")

        print(f"\n🔄 After adding secrets:")
        print(f"   - Spaces will auto-restart")
        print(f"   - Wait 2-3 minutes")
        print(f"   - Run this script again to verify")
    else:
        print("🟡 SYSTEM STARTING UP")
        print(f"   ✅ Spaces deployed")
        print(f"   🔄 Workers starting (wait 2-3 minutes)")
        print(f"   ⏳ Datasets will be created soon")
        print(f"\n🔄 Run this script again in 2-3 minutes")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
