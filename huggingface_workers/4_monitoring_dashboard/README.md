---
title: Autocoder Monitoring Dashboard
emoji: 📊
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - monitoring
  - dashboard
  - automation
  - worker
---

# Autocoder Monitoring Dashboard 📊

**24/7 Unified Worker Monitoring**

This Hugging Face Space provides a central dashboard for monitoring all autonomous workers in the Autocoder system.

## Features

- ✅ **Unified Monitoring** - All workers in one dashboard
- ✅ **Real-time Updates** - Live status from all workers
- ✅ **Result Aggregation** - Combined insights from all datasets
- ✅ **Visual Analytics** - Charts and metrics
- ✅ **Free Tier Compatible** - Runs on HF free infrastructure

## How It Works

1. **Data Aggregation** - Pulls data from all worker datasets
2. **Status Monitoring** - Checks health of all workers
3. **Result Analysis** - Combines insights from all perspectives
4. **Visualization** - Displays metrics and trends
5. **Real-time Updates** - Refreshes every 10 seconds

## Workers Monitored

### 1. Analysis Worker 🤖
- **Status**: Real-time operational status
- **Iterations**: Total analysis cycles completed
- **Dataset**: `autocoder-analysis-results`
- **Metrics**: Analysis quality, findings count

### 2. Test Planning Worker 🧪
- **Status**: Real-time operational status
- **Iterations**: Total test planning cycles
- **Dataset**: `autocoder-test-planning-results`
- **Metrics**: Tests specified, edge cases discovered

### 3. Implementation Worker ⚙️
- **Status**: Real-time operational status
- **Iterations**: Total implementation cycles
- **Dataset**: `autocoder-implementation-results`
- **Metrics**: Implementations generated, code quality

## Configuration

### Required Secrets

Add these in Space Settings → Repository Secrets:

- `HF_TOKEN` - Your Hugging Face access token (read permission)
- `HF_USERNAME` - Your HF username (optional, defaults to 'autocoder')

### Datasets Monitored

- `{HF_USERNAME}/autocoder-analysis-results`
- `{HF_USERNAME}/autocoder-test-planning-results`
- `{HF_USERNAME}/autocoder-implementation-results`

## Usage

### Via UI

1. Visit this Space
2. View unified worker status
3. Browse aggregated insights
4. Check system health

### Programmatically

```python
from datasets import load_dataset

# Load data from all workers
analysis = load_dataset("autocoder/autocoder-analysis-results")
tests = load_dataset("autocoder/autocoder-test-planning-results")
impl = load_dataset("autocoder/autocoder-implementation-results")

# Analyze combined insights
print(f"Total analysis iterations: {len(analysis['train'])}")
print(f"Total tests planned: {len(tests['train'])}")
print(f"Total implementations: {len(impl['train'])}")
```

## Dashboard Sections

### 1. System Overview
- Overall system health
- Total iterations across all workers
- Combined insights summary

### 2. Worker Status
- Individual worker health
- Iteration counts
- Last update timestamps

### 3. Analysis Insights
- Latest code analysis findings
- Quality metrics
- Improvement recommendations

### 4. Test Coverage
- Test specifications count
- Edge cases discovered
- Failure modes identified

### 5. Implementation Progress
- Code implementations generated
- Phases completed
- Risk assessments

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_token"
export HF_USERNAME="your_username"

# Run locally
python app.py
```

## Deployment

1. Fork this Space or create new
2. Configure secrets
3. Space will auto-start dashboard

## Monitoring

- **Status Updates**: Every 10 seconds
- **Data Refresh**: Every 30 seconds
- **Health Checks**: Continuous

## Metrics Tracked

- **System Health**: All workers operational
- **Total Iterations**: Combined iteration count
- **Insights Generated**: Total findings across workers
- **Code Quality**: Implementation quality scores
- **Test Coverage**: Test specification completeness
- **Performance**: Worker response times

## Cost

- **Free Tier**: Fully compatible
- **Read-only**: No write operations (lower resource usage)
- **Storage**: Minimal local storage needed

## Support

For issues or questions, open an issue in the main repository.

## License

MIT License - See LICENSE file
