# Hugging Face Spaces Deployment Guide

Complete guide to deploying autonomous workers to Hugging Face Spaces for 24/7 operation.

## Overview

This deployment creates **4 Hugging Face Spaces** that work together as a 24/7 autonomous system:

1. **Analysis Worker** 🤖 - Continuous code analysis
2. **Test Planning Worker** 🧪 - Continuous test requirement analysis
3. **Implementation Worker** ⚙️ - Code implementation with deep thinking
4. **Monitoring Dashboard** 📊 - Unified monitoring for all workers

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Hugging Face Spaces Infrastructure             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Space 1: autocoder-analysis-worker                     │
│  ├─ Gradio UI for monitoring                            │
│  ├─ 6 rotating analysis perspectives                    │
│  └─ Pushes to: autocoder-analysis-results (Dataset)     │
│                                                          │
│  Space 2: autocoder-test-planning-worker                │
│  ├─ Gradio UI for monitoring                            │
│  ├─ 15 rotating test perspectives                       │
│  └─ Pushes to: autocoder-test-planning-results (Dataset)│
│                                                          │
│  Space 3: autocoder-implementation-worker               │
│  ├─ Gradio UI for monitoring                            │
│  ├─ 6 thinking stages                                   │
│  └─ Pushes to: autocoder-implementation-results (Dataset)│
│                                                          │
│  Space 4: autocoder-dashboard                           │
│  ├─ Unified monitoring UI                               │
│  ├─ Aggregates data from all workers                    │
│  └─ Reads from all datasets                             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Hugging Face Account
- Create account at https://huggingface.co/join
- Free tier is sufficient for this deployment

### 2. Hugging Face Access Token
- Go to https://huggingface.co/settings/tokens
- Click "New token"
- Name: "autocoder-workers"
- Type: **Write** (needed for workers to push datasets)
- Click "Generate"
- Copy and save the token securely

### 3. Git Setup (Optional)
If deploying via Git:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Deployment Methods

### Method 1: Web UI Upload (Easiest)

For each Space, follow these steps:

#### 1. Create New Space
1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Owner**: Your username
   - **Space name**: See "Space Names" section below
   - **License**: MIT
   - **Space SDK**: Gradio
   - **Visibility**: Public (or Private if preferred)
3. Click "Create Space"

#### 2. Upload Files
For each Space directory (`1_analysis_worker`, `2_test_planning_worker`, etc.):

1. Click "Files" tab in the Space
2. Click "Add file" → "Upload files"
3. Upload all 3 files from the directory:
   - `README.md`
   - `requirements.txt`
   - `app.py`
4. Commit message: "Initial deployment"
5. Click "Commit changes to main"

#### 3. Configure Secrets
1. Click "Settings" tab
2. Scroll to "Repository secrets"
3. Add secrets:
   - **Name**: `HF_TOKEN`
   - **Value**: Your HF access token (from Prerequisites)
   - Click "Add"

   - **Name**: `HF_USERNAME`
   - **Value**: Your HF username
   - Click "Add"

#### 4. Wait for Build
- Space will automatically build and start
- Check "Logs" tab for build progress
- Should be running in 1-2 minutes

### Method 2: Git CLI (Advanced)

For each Space:

```bash
# 1. Create Space on HF website first (see Method 1, Step 1)

# 2. Clone the Space repository
cd /tmp
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
cd SPACE_NAME

# 3. Copy files from worker directory
cp /mnt/e/projects/code/huggingface_workers/WORKER_DIR/* .

# 4. Commit and push
git add .
git commit -m "Deploy autonomous worker"
git push

# 5. Configure secrets via web UI (see Method 1, Step 3)
```

### Method 3: Hugging Face Hub Python API

```python
from huggingface_hub import HfApi, create_repo, upload_file
import os

# Setup
api = HfApi()
token = "YOUR_HF_TOKEN"
username = "YOUR_USERNAME"

# For each worker
workers = [
    ("autocoder-analysis-worker", "1_analysis_worker"),
    ("autocoder-test-planning-worker", "2_test_planning_worker"),
    ("autocoder-implementation-worker", "3_implementation_worker"),
    ("autocoder-dashboard", "4_monitoring_dashboard")
]

for space_name, worker_dir in workers:
    # Create Space
    repo_id = f"{username}/{space_name}"
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        token=token,
        exist_ok=True
    )

    # Upload files
    base_path = f"/mnt/e/projects/code/huggingface_workers/{worker_dir}"

    for file in ["README.md", "requirements.txt", "app.py"]:
        upload_file(
            path_or_fileobj=f"{base_path}/{file}",
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="space",
            token=token
        )

    print(f"✓ Deployed {space_name}")

# Note: Still need to configure secrets via web UI
```

## Space Names

Use these exact names for proper integration:

| Space Number | Directory | Space Name | URL Pattern |
|--------------|-----------|------------|-------------|
| 1 | `1_analysis_worker` | `autocoder-analysis-worker` | `https://huggingface.co/spaces/USERNAME/autocoder-analysis-worker` |
| 2 | `2_test_planning_worker` | `autocoder-test-planning-worker` | `https://huggingface.co/spaces/USERNAME/autocoder-test-planning-worker` |
| 3 | `3_implementation_worker` | `autocoder-implementation-worker` | `https://huggingface.co/spaces/USERNAME/autocoder-implementation-worker` |
| 4 | `4_monitoring_dashboard` | `autocoder-dashboard` | `https://huggingface.co/spaces/USERNAME/autocoder-dashboard` |

Replace `USERNAME` with your HuggingFace username.

## Configuration

### Required Secrets (All Workers)

For **Spaces 1, 2, 3** (workers that write data):

| Secret Name | Value | Purpose |
|-------------|-------|---------|
| `HF_TOKEN` | Your HF access token | Write access to datasets |
| `HF_USERNAME` | Your HF username | Dataset naming |

For **Space 4** (dashboard - read only):

| Secret Name | Value | Purpose |
|-------------|-------|---------|
| `HF_TOKEN` | Your HF access token | Read access to datasets |
| `HF_USERNAME` | Your HF username | Dataset naming |

### Optional: Custom Dataset Names

If you want different dataset names, update the code in each `app.py`:

```python
# Default (uses your username)
DATASET_NAME = f"{HF_USERNAME}/autocoder-analysis-results"

# Custom
DATASET_NAME = "my-org/custom-analysis-results"
```

## Verification

### 1. Check Each Space is Running

Visit each Space URL and verify:
- ✅ Space loads without errors
- ✅ Status shows "Running" or "Initializing"
- ✅ Iteration counter increases over time
- ✅ No error messages in UI

### 2. Check Datasets are Created

After 1-2 minutes, verify datasets exist:

1. Go to https://huggingface.co/datasets/YOUR_USERNAME
2. You should see 3 new datasets:
   - `autocoder-analysis-results`
   - `autocoder-test-planning-results`
   - `autocoder-implementation-results`

### 3. Check Data is Being Written

For each dataset:
1. Click on the dataset
2. Click "Files" tab
3. You should see JSON files like:
   - `iteration_1.json`
   - `test_plan_iteration_1.json`
   - `implementation_iteration_1.json`

### 4. Check Dashboard Aggregation

1. Visit your dashboard Space
2. Verify:
   - ✅ System Status shows "🟢 All Systems Operational"
   - ✅ Workers Online: 3/3
   - ✅ Total Iterations > 0
   - ✅ Worker table shows all workers as "✅ Running"

## Troubleshooting

### Space Shows "Application Startup Failed"

**Cause**: Build error in dependencies or code

**Fix**:
1. Click "Logs" tab in Space
2. Look for error messages
3. Common issues:
   - Missing file: Re-upload all files
   - Dependency error: Check `requirements.txt` syntax
   - Python syntax error: Check `app.py` for errors

### Worker Status Shows "No HF_TOKEN"

**Cause**: Secret not configured

**Fix**:
1. Go to Space Settings
2. Add `HF_TOKEN` and `HF_USERNAME` secrets
3. Space will auto-restart

### Dataset Not Created

**Cause**: Token lacks write permission

**Fix**:
1. Check token type at https://huggingface.co/settings/tokens
2. Token must be **Write** type
3. If Read-only, create new Write token
4. Update secret in Space settings

### Dashboard Shows "Workers Offline"

**Causes & Fixes**:

1. **Workers not started yet**
   - Wait 2-3 minutes after deployment
   - Workers auto-start on Space launch

2. **Dataset names mismatch**
   - Verify `HF_USERNAME` secret matches actual username
   - Check dataset naming in code

3. **Token permissions**
   - Dashboard needs Read access to all datasets
   - Workers need Write access

### Iteration Counter Stuck at 0

**Cause**: Worker loop not running

**Fix**:
1. Check Space logs for errors
2. Verify `asyncio.create_task(worker.run_continuous())` is in code
3. Restart Space: Settings → "Factory reboot"

## Monitoring

### Real-time Monitoring

**Via Dashboard**:
- Visit: `https://huggingface.co/spaces/YOUR_USERNAME/autocoder-dashboard`
- Auto-refreshes every 10 seconds
- Shows all worker status, iterations, insights

**Via Individual Spaces**:
- Each worker has its own monitoring UI
- Status updates every 5 seconds
- View worker-specific logs and results

### Programmatic Access

```python
from datasets import load_dataset

# Load worker results
analysis = load_dataset("YOUR_USERNAME/autocoder-analysis-results")
tests = load_dataset("YOUR_USERNAME/autocoder-test-planning-results")
impl = load_dataset("YOUR_USERNAME/autocoder-implementation-results")

# Get latest iteration
latest_analysis = analysis["train"][-1]
print(latest_analysis)

# Count total iterations
print(f"Total analysis iterations: {len(analysis['train'])}")
```

### Logs

**Space Logs**:
- Each Space: Settings → "Logs"
- Shows build logs, runtime logs, errors

**Dataset Activity**:
- Each Dataset: Activity tab
- Shows all commits (file uploads)

## Maintenance

### Updating Workers

**Method 1: Web UI**
1. Go to Space
2. Click "Files" tab
3. Click on file to edit (e.g., `app.py`)
4. Make changes
5. Commit
6. Space auto-rebuilds

**Method 2: Git**
```bash
git clone https://huggingface.co/spaces/USERNAME/SPACE_NAME
cd SPACE_NAME
# Edit files
git add .
git commit -m "Update worker"
git push
```

### Stopping Workers

**Temporary Stop**:
1. Go to Space UI
2. Click "⏸ Stop Worker" button
3. Worker pauses (Space stays running)

**Permanent Stop**:
1. Space Settings
2. Click "Pause Space"
3. Space shuts down (can restart later)

**Delete Space**:
1. Space Settings
2. Scroll to "Danger Zone"
3. Click "Delete this Space"

### Dataset Management

**View Data**:
- Go to dataset page
- Browse files or use Datasets viewer

**Download Data**:
```python
from datasets import load_dataset
dataset = load_dataset("USERNAME/DATASET_NAME")
dataset.save_to_disk("./local_copy")
```

**Clear Data** (start fresh):
1. Go to dataset
2. Files tab
3. Delete all `iteration_*.json` files
4. Workers will recreate from iteration 1

## Cost and Limits

### Free Tier Limits

- **CPU**: 2 vCPUs per Space
- **RAM**: 16 GB per Space
- **Storage**: 50 GB total (across all repos)
- **Runtime**: Unlimited (24/7)

### Resource Usage

Each worker uses approximately:
- **CPU**: <5% (mostly idle, spikes during iteration)
- **RAM**: ~500 MB
- **Storage**: ~100 MB + dataset growth

**Estimated Dataset Growth**:
- Per iteration: ~5-10 KB JSON
- Per day (60 sec intervals): ~1,440 iterations = ~14 MB
- Per month: ~420 MB

### Optimization

**Reduce dataset growth**:
```python
# In app.py, increase sleep time
await asyncio.sleep(300)  # 5 minutes instead of 1
```

**Reduce storage**:
- Periodically delete old iterations
- Keep only latest N iterations

## Integration with Once Project

### Consuming Worker Results

```python
# In your Once project
from datasets import load_dataset

# Load test specifications
test_plans = load_dataset("USERNAME/autocoder-test-planning-results")
latest_plan = test_plans["train"][-1]

# Use test specifications
failure_modes = latest_plan["cumulative_insights"]["failure_modes_identified"]
print(f"Failure modes to test: {failure_modes}")

# Load implementation guidance
implementations = load_dataset("USERNAME/autocoder-implementation-results")
latest_impl = implementations["train"][-1]

# Get implementation code
if latest_impl["stage_name"] == "Implementation Plan":
    code = latest_impl["analysis"]["code_changes"]["code"]
    print("Generated implementation:")
    print(code)
```

### CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
name: Test with Worker Insights

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Get latest test plans
        run: |
          pip install datasets
          python scripts/fetch_test_plans.py

      - name: Run tests
        run: pytest tests/
```

## Advanced Configuration

### Custom Analysis Perspectives

Edit `app.py` to add custom analysis:

```python
async def perspective_custom(self):
    """Your custom analysis"""
    return {
        "perspective": "Custom Analysis",
        "analysis": {
            # Your analysis here
        }
    }

# Add to perspectives list
perspectives = [
    # ... existing perspectives
    self.perspective_custom
]
```

### Inter-Worker Communication

Workers can read from each other's datasets:

```python
# In implementation_worker/app.py
from datasets import load_dataset

# Load test planning results
test_plans = load_dataset(f"{HF_USERNAME}/autocoder-test-planning-results")

# Use insights to guide implementation
edge_cases = test_plans["train"][-1]["cumulative_insights"]["edge_cases_discovered"]
```

### Webhooks and Notifications

Add Discord/Slack notifications:

```python
import aiohttp

async def notify_completion(self, result):
    webhook_url = os.getenv("DISCORD_WEBHOOK")
    if webhook_url:
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json={
                "content": f"Worker iteration {self.iteration} complete!"
            })
```

## Security

### Token Security

- ✅ **DO**: Use Secrets (encrypted)
- ❌ **DON'T**: Hardcode tokens in code
- ❌ **DON'T**: Commit tokens to git

### Private Spaces

For sensitive projects:
1. Create Spaces as **Private**
2. Create Datasets as **Private**
3. Only you can access

### Access Control

- **Spaces**: Public/Private per Space
- **Datasets**: Public/Private per Dataset
- **Mixed**: Public Spaces can read Private Datasets (if you own both)

## Support

### Documentation
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://gradio.app/docs/

### Community
- HF Forums: https://discuss.huggingface.co/
- HF Discord: https://hf.co/join/discord

### Issues
- For deployment issues: Check Space logs
- For code issues: Review `app.py` logic
- For integration issues: Check dataset permissions

## Summary Checklist

- [ ] Created HF account
- [ ] Generated HF access token (Write permission)
- [ ] Deployed Space 1: Analysis Worker
- [ ] Deployed Space 2: Test Planning Worker
- [ ] Deployed Space 3: Implementation Worker
- [ ] Deployed Space 4: Monitoring Dashboard
- [ ] Configured HF_TOKEN secret (all 4 Spaces)
- [ ] Configured HF_USERNAME secret (all 4 Spaces)
- [ ] Verified all Spaces running
- [ ] Verified 3 datasets created
- [ ] Verified dashboard shows "All Systems Operational"
- [ ] Verified iteration counters increasing
- [ ] Bookmarked dashboard URL for monitoring

## Next Steps

After successful deployment:

1. **Monitor** - Check dashboard regularly
2. **Analyze** - Review worker findings
3. **Implement** - Use generated test plans and code
4. **Iterate** - Let workers continue analyzing 24/7
5. **Integrate** - Pull insights into your development workflow

Your autonomous workers are now running 24/7 on Hugging Face! 🎉
