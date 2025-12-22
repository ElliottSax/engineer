#!/bin/bash
# Continuous Training Workers for Autocoder
# Uses DeepSeek API ($0.14/M tokens)

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set"
    echo "Set it in .env file or export DEEPSEEK_API_KEY=your-key"
    exit 1
fi

# Default settings
WORKERS=${1:-1}
ITERATIONS=${2:-50}
DELAY=${3:-0.5}

echo "=============================================="
echo "DEEPSEEK CONTINUOUS TRAINING"
echo "=============================================="
echo "Workers: $WORKERS"
echo "Iterations per worker: $ITERATIONS"
echo "Delay between iterations: ${DELAY}s"
echo "Estimated cost: \$$(echo "$WORKERS * $ITERATIONS * 0.01" | bc)"
echo "=============================================="
echo ""

# Create workspace
mkdir -p /tmp/training_workspace
mkdir -p training_output

# Run training
python3 deepseek_trainer.py \
    --workers $WORKERS \
    --iterations $ITERATIONS \
    --delay $DELAY

echo ""
echo "Training complete. Results in training_output/"
