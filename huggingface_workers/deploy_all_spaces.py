#!/usr/bin/env python3
"""
Deploy all 4 HuggingFace Spaces for autonomous workers
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file
from pathlib import Path

# Configuration
USERNAME = "elliottsax"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this first: export HF_TOKEN="your_token"

# Workers to deploy
WORKERS = [
    {
        "name": "autocoder-analysis-worker",
        "directory": "1_analysis_worker",
        "emoji": "🤖",
        "description": "24/7 autonomous code analysis worker"
    },
    {
        "name": "autocoder-test-planning-worker",
        "directory": "2_test_planning_worker",
        "emoji": "🧪",
        "description": "24/7 autonomous test planning worker"
    },
    {
        "name": "autocoder-implementation-worker",
        "directory": "3_implementation_worker",
        "emoji": "⚙️",
        "description": "24/7 autonomous implementation worker"
    },
    {
        "name": "autocoder-dashboard",
        "directory": "4_monitoring_dashboard",
        "emoji": "📊",
        "description": "Unified monitoring dashboard for all workers"
    }
]

def deploy_space(worker_info):
    """Deploy a single Space"""
    space_name = worker_info["name"]
    directory = worker_info["directory"]
    repo_id = f"{USERNAME}/{space_name}"

    print(f"\n{'='*60}")
    print(f"Deploying: {space_name}")
    print(f"{'='*60}")

    # Create Space
    try:
        print(f"Creating Space: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            token=HF_TOKEN,
            exist_ok=True
        )
        print(f"✓ Space created: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"✓ Space already exists (or error: {e})")

    # Upload files
    base_path = Path(__file__).parent / directory
    files_to_upload = ["README.md", "requirements.txt", "app.py"]

    for file_name in files_to_upload:
        file_path = base_path / file_name

        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            continue

        try:
            print(f"Uploading {file_name}...")
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="space",
                token=HF_TOKEN,
                commit_message=f"Add {file_name}"
            )
            print(f"✓ Uploaded {file_name}")
        except Exception as e:
            print(f"✗ Error uploading {file_name}: {e}")

    print(f"\n✓ Space deployed: https://huggingface.co/spaces/{repo_id}")
    print(f"⚠️  IMPORTANT: Add secrets in Space settings:")
    print(f"   - HF_TOKEN = your token")
    print(f"   - HF_USERNAME = {USERNAME}")

    return repo_id

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  HuggingFace Spaces Deployment Script                        ║
║  Deploying 4 Autonomous Workers                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Check token
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set!")
        print("\nSet it with:")
        print("  export HF_TOKEN='your_token_here'")
        print("\nGet token from: https://huggingface.co/settings/tokens")
        return

    print(f"Username: {USERNAME}")
    print(f"Token: {HF_TOKEN[:10]}..." if HF_TOKEN else "NOT SET")
    print(f"Workers to deploy: {len(WORKERS)}")

    # Deploy each worker
    deployed_spaces = []
    for worker in WORKERS:
        try:
            repo_id = deploy_space(worker)
            deployed_spaces.append(repo_id)
        except Exception as e:
            print(f"✗ Failed to deploy {worker['name']}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nDeployed {len(deployed_spaces)} Spaces:")
    for repo_id in deployed_spaces:
        print(f"  ✓ https://huggingface.co/spaces/{repo_id}")

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("\nFor EACH Space, add secrets:")
    print("1. Go to Space → Settings → Repository secrets")
    print("2. Add secret:")
    print("   Name:  HF_TOKEN")
    print("   Value: (your HuggingFace token)")
    print("3. Add secret:")
    print("   Name:  HF_USERNAME")
    print(f"   Value: {USERNAME}")
    print("\nThen visit dashboard:")
    print(f"  https://huggingface.co/spaces/{USERNAME}/autocoder-dashboard")
    print("\nShould show: 🟢 All Systems Operational")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
