# Quick Deploy - HuggingFace Spaces

## One-Command Deployment

### 1. Set Your Token

```bash
export HF_TOKEN="hf_your_token_here"
```

Get token from: https://huggingface.co/settings/tokens (need **Write** permission)

### 2. Run Deployment Script

```bash
cd /mnt/e/projects/code/huggingface_workers
python3 deploy_all_spaces.py
```

This will:
- Create 4 new Spaces
- Upload all files (README.md, requirements.txt, app.py)
- Show URLs for each Space

### 3. Add Secrets to Each Space

The script **cannot** add secrets (security restriction), so you must do this manually:

For **each of the 4 Spaces**:

1. Go to Space Settings → Repository secrets
2. Add two secrets:
   - `HF_TOKEN` = your token
   - `HF_USERNAME` = `elliottsax`

**Spaces to configure**:
- https://huggingface.co/spaces/elliottsax/autocoder-analysis-worker
- https://huggingface.co/spaces/elliottsax/autocoder-test-planning-worker
- https://huggingface.co/spaces/elliottsax/autocoder-implementation-worker
- https://huggingface.co/spaces/elliottsax/autocoder-dashboard

### 4. Verify

Visit dashboard: https://huggingface.co/spaces/elliottsax/autocoder-dashboard

Should show:
- System Status: 🟢 All Systems Operational
- Workers Online: 3/3
- Total Iterations: (increasing)

### Done! 🎉

Workers are now running 24/7 analyzing your code.

---

## Troubleshooting

**"HF_TOKEN not set"**
```bash
export HF_TOKEN="your_token"
```

**"Space already exists"**
- Normal if re-running script
- Files will be updated

**"Workers show offline in dashboard"**
- Wait 2-3 minutes after adding secrets
- Spaces need to build and start

**"No data in datasets"**
- Check HF_TOKEN is **Write** type (not Read)
- Verify secrets added correctly
- Wait for first iteration (~1-2 minutes)
