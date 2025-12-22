#!/usr/bin/env python3
"""
Simple runner to train autocoder with Claude

Usage:
    # Set your API key first
    export ANTHROPIC_API_KEY=your-key-here

    # Run training
    python train_autocoder.py

    # Or run specific modes
    python train_autocoder.py --generate-data     # Just generate training data
    python train_autocoder.py --evaluate          # Just run evaluation
    python train_autocoder.py --full-training     # Complete training pipeline
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from claude_trainer import (
    ClaudeTrainingPipeline,
    ClaudeClient,
    SyntheticDataGenerator,
    AutocoderEvaluator,
    CurriculumDesigner,
    TaskCategory,
    Difficulty,
    create_autocoder_training_fn
)


async def generate_training_data(count_per_category: int = 5):
    """Generate training data using Claude"""

    print("=" * 60)
    print("GENERATING TRAINING DATA")
    print("=" * 60)

    client = ClaudeClient()
    generator = SyntheticDataGenerator(client)

    categories = [
        (TaskCategory.CODE_GENERATION, Difficulty.BEGINNER),
        (TaskCategory.CODE_GENERATION, Difficulty.INTERMEDIATE),
        (TaskCategory.BUG_FIXING, Difficulty.BEGINNER),
        (TaskCategory.BUG_FIXING, Difficulty.INTERMEDIATE),
        (TaskCategory.REFACTORING, Difficulty.INTERMEDIATE),
        (TaskCategory.TESTING, Difficulty.INTERMEDIATE),
    ]

    all_examples = []
    for category, difficulty in categories:
        print(f"\nGenerating {category.value} examples at {difficulty.name} level...")
        examples = await generator.generate_examples(
            category=category,
            difficulty=difficulty,
            count=count_per_category
        )
        all_examples.extend(examples)
        print(f"  Generated {len(examples)} examples")

    print(f"\nTotal examples generated: {len(all_examples)}")
    print(f"Saved to: training_data/")
    print(f"API cost: ${client.total_cost:.4f}")

    return all_examples


async def run_evaluation():
    """Run evaluation on autocoder outputs"""

    print("=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    client = ClaudeClient()
    generator = SyntheticDataGenerator(client)
    evaluator = AutocoderEvaluator(client)

    # Get autocoder function
    autocoder_fn = await create_autocoder_training_fn()

    # Generate a few test examples
    print("\nGenerating test examples...")
    examples = await generator.generate_examples(
        category=TaskCategory.CODE_GENERATION,
        difficulty=Difficulty.BEGINNER,
        count=3
    )

    # Run autocoder on each and evaluate
    print("\nRunning autocoder and evaluating...")
    for example in examples:
        print(f"\nTask: {example.task[:80]}...")

        # Get autocoder solution
        solution = await autocoder_fn(example.task)
        print(f"Solution preview: {solution[:100]}...")

        # Evaluate
        result = await evaluator.evaluate(example, solution)
        print(f"Score: {result.score:.2f}")
        print(f"Feedback: {result.feedback[:150]}...")

    # Print stats
    stats = evaluator.get_aggregate_stats()
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Average score: {stats.get('avg_score', 0):.2f}")
    print(f"Passing rate: {stats.get('passing_rate', 0):.1%}")
    print(f"API cost: ${client.total_cost:.4f}")

    evaluator.save_evaluations()


async def run_full_training(iterations: int = 10):
    """Run complete training pipeline"""

    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)

    pipeline = ClaudeTrainingPipeline()
    await pipeline.initialize()

    autocoder_fn = await create_autocoder_training_fn()

    summary = await pipeline.run_training_session(
        autocoder_fn=autocoder_fn,
        num_iterations=iterations
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    import json
    print(json.dumps(summary, indent=2))

    return summary


async def demo_mode():
    """Run in demo mode without API calls"""

    print("=" * 60)
    print("DEMO MODE (No API calls)")
    print("=" * 60)

    print("\nThis demonstrates the training pipeline structure.")
    print("\nComponents:")
    print("  1. SyntheticDataGenerator - Generates training examples")
    print("  2. AutocoderEvaluator - Evaluates autocoder outputs")
    print("  3. CurriculumDesigner - Designs progressive learning")
    print("  4. ClaudeTrainingPipeline - Orchestrates everything")

    print("\nTraining Flow:")
    print("  Claude generates task → Autocoder solves → Claude evaluates")
    print("  → Feedback improves autocoder → Curriculum advances")

    print("\nTo run actual training:")
    print("  1. Set ANTHROPIC_API_KEY environment variable")
    print("  2. Run: python train_autocoder.py --full-training")

    print("\nEstimated costs per session:")
    print("  - Data generation (30 examples): ~$0.50")
    print("  - Evaluation (10 iterations): ~$0.30")
    print("  - Full training (10 iterations): ~$1.00")


def main():
    parser = argparse.ArgumentParser(
        description="Train autocoder using Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_autocoder.py                    # Demo mode
  python train_autocoder.py --generate-data    # Generate training data
  python train_autocoder.py --evaluate         # Run evaluation
  python train_autocoder.py --full-training    # Complete training
  python train_autocoder.py --full-training --iterations 20
        """
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic training data"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on autocoder"
    )
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="Run complete training pipeline"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations (default: 10)"
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=5,
        help="Number of examples per category (default: 5)"
    )

    args = parser.parse_args()

    # Check API key for non-demo modes
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    if args.generate_data:
        if not has_api_key:
            print("ERROR: ANTHROPIC_API_KEY not set")
            print("Set it with: export ANTHROPIC_API_KEY=your-key")
            sys.exit(1)
        asyncio.run(generate_training_data(args.examples_per_category))

    elif args.evaluate:
        if not has_api_key:
            print("ERROR: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        asyncio.run(run_evaluation())

    elif args.full_training:
        if not has_api_key:
            print("ERROR: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        asyncio.run(run_full_training(args.iterations))

    else:
        # Demo mode
        asyncio.run(demo_mode())


if __name__ == "__main__":
    main()
