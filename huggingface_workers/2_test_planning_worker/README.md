---
title: Autocoder Test Planning Worker
emoji: 🧪
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - autonomous
  - test-planning
  - automation
  - worker
---

# Autocoder Test Planning Worker 🧪

**24/7 Autonomous Test Planning Worker**

This Hugging Face Space runs continuous test planning analysis, discovering test requirements, edge cases, and failure modes in real-time.

## Features

- ✅ **Continuous Analysis** - Runs 24/7 automatically
- ✅ **15 Perspectives** - Analyzes from 15 different testing angles
- ✅ **Cloud Storage** - Results pushed to HF Datasets
- ✅ **Real-time Monitoring** - Live status dashboard
- ✅ **Free Tier Compatible** - Runs on HF free infrastructure

## How It Works

1. **Analysis Loop** - Worker runs continuous test planning iterations
2. **Perspective Rotation** - Cycles through 15 different analysis perspectives
3. **Result Storage** - Pushes test specifications to HF Dataset
4. **Monitoring** - UI updates every 5 seconds

## Analysis Perspectives

1. What Are We Testing? - Scope and boundaries
2. How Can It Fail? - Failure mode analysis
3. Integration Points - Where systems connect
4. Edge Cases - Boundary conditions
5. Performance - Speed and efficiency
6. Backwards Compatibility - Breaking changes
7. Data Integrity - Data correctness
8. Error Handling - Failure recovery
9. User Experience - UX implications
10. Security - Vulnerability analysis
11. Test Coverage - Coverage gaps
12. Test Pyramid - Test structure
13. Mocking Strategy - Test isolation
14. CI/CD Integration - Automation
15. Big Picture Review - Holistic analysis

## Configuration

### Required Secrets

Add these in Space Settings → Repository Secrets:

- `HF_TOKEN` - Your Hugging Face access token (write permission)
- `HF_USERNAME` - Your HF username (optional, defaults to 'autocoder')

### Dataset Output

Results are saved to: `{HF_USERNAME}/autocoder-test-planning-results`

## Usage

### Via UI

1. Visit this Space
2. View real-time status
3. Start/stop worker as needed
4. View test planning logs

### Programmatically

```python
from datasets import load_dataset

# Load all test planning results
dataset = load_dataset("autocoder/autocoder-test-planning-results")

# Get latest results
latest = dataset["train"][-1]
print(latest)
```

## Integration with Other Workers

This worker's output is used by:
- Analysis Worker (provides code insights)
- Implementation Worker (uses test plans to guide coding)
- Monitoring Dashboard (visualizes test coverage)

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
3. Space will auto-start worker

## Monitoring

- **Status Updates**: Every 5 seconds
- **Test Planning Iterations**: Every 60 seconds
- **Dataset Uploads**: After each iteration

## Test Specifications Generated

Each iteration produces:
- Failure mode analysis
- Integration point mapping
- Edge case catalog
- Test specifications with code
- Coverage analysis
- Risk assessment

## Cost

- **Free Tier**: Fully compatible
- **Persistent**: Space keeps running 24/7
- **Storage**: Dataset grows over time (monitor quota)

## Support

For issues or questions, open an issue in the main repository.

## License

MIT License - See LICENSE file
