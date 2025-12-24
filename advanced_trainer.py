#!/usr/bin/env python3
"""
Advanced Training for Autocoder

More challenging tasks that test the limits of pattern-based code generation.
Includes:
- Algorithm implementations
- Data structure operations
- Mathematical computations
- String manipulations
- Multi-step problems
"""

import os
import sys
import asyncio
import random
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from unified_coding_agent import UnifiedCodingAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingTask:
    name: str
    description: str
    test_cases: List[Dict]
    difficulty: str  # easy, medium, hard, expert
    category: str


# Advanced training tasks - more challenging than basic patterns
ADVANCED_TASKS = [
    # === ALGORITHM TASKS ===
    TrainingTask(
        name="merge_sorted_lists",
        description="merges two sorted lists into one sorted list",
        difficulty="medium",
        category="algorithm",
        test_cases=[
            {"args": [[1, 3, 5], [2, 4, 6]], "expected": [1, 2, 3, 4, 5, 6]},
            {"args": [[1, 2], [3, 4, 5]], "expected": [1, 2, 3, 4, 5]},
            {"args": [[], [1, 2, 3]], "expected": [1, 2, 3]},
        ]
    ),
    TrainingTask(
        name="quick_sort",
        description="sorts a list using quicksort algorithm",
        difficulty="hard",
        category="algorithm",
        test_cases=[
            {"args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected": [1, 1, 2, 3, 4, 5, 6, 9]},
            {"args": [[5, 4, 3, 2, 1]], "expected": [1, 2, 3, 4, 5]},
            {"args": [[1]], "expected": [1]},
        ]
    ),
    TrainingTask(
        name="bubble_sort",
        description="sorts a list using bubble sort algorithm",
        difficulty="medium",
        category="algorithm",
        test_cases=[
            {"args": [[64, 34, 25, 12, 22, 11, 90]], "expected": [11, 12, 22, 25, 34, 64, 90]},
            {"args": [[5, 1, 4, 2, 8]], "expected": [1, 2, 4, 5, 8]},
            {"args": [[]], "expected": []},
        ]
    ),
    TrainingTask(
        name="insertion_sort",
        description="sorts a list using insertion sort algorithm",
        difficulty="medium",
        category="algorithm",
        test_cases=[
            {"args": [[12, 11, 13, 5, 6]], "expected": [5, 6, 11, 12, 13]},
            {"args": [[4, 3, 2, 1]], "expected": [1, 2, 3, 4]},
            {"args": [[1, 2, 3]], "expected": [1, 2, 3]},
        ]
    ),

    # === MATH TASKS ===
    TrainingTask(
        name="factorial",
        description="calculates the factorial of a number",
        difficulty="easy",
        category="math",
        test_cases=[
            {"args": [5], "expected": 120},
            {"args": [0], "expected": 1},
            {"args": [1], "expected": 1},
            {"args": [10], "expected": 3628800},
        ]
    ),
    TrainingTask(
        name="lcm",
        description="calculates the least common multiple of two numbers",
        difficulty="medium",
        category="math",
        test_cases=[
            {"args": [4, 6], "expected": 12},
            {"args": [3, 5], "expected": 15},
            {"args": [12, 18], "expected": 36},
        ]
    ),
    TrainingTask(
        name="is_perfect_square",
        description="checks if a number is a perfect square",
        difficulty="easy",
        category="math",
        test_cases=[
            {"args": [16], "expected": True},
            {"args": [14], "expected": False},
            {"args": [1], "expected": True},
            {"args": [0], "expected": True},
        ]
    ),
    TrainingTask(
        name="sum_of_digits",
        description="calculates the sum of digits in a number",
        difficulty="easy",
        category="math",
        test_cases=[
            {"args": [123], "expected": 6},
            {"args": [9999], "expected": 36},
            {"args": [0], "expected": 0},
        ]
    ),
    TrainingTask(
        name="nth_prime",
        description="returns the nth prime number",
        difficulty="hard",
        category="math",
        test_cases=[
            {"args": [1], "expected": 2},
            {"args": [5], "expected": 11},
            {"args": [10], "expected": 29},
        ]
    ),

    # === STRING TASKS ===
    TrainingTask(
        name="longest_word",
        description="finds the longest word in a string",
        difficulty="easy",
        category="string",
        test_cases=[
            {"args": ["The quick brown fox"], "expected": "quick"},
            {"args": ["Hello world"], "expected": "Hello"},
            {"args": ["a"], "expected": "a"},
        ]
    ),
    TrainingTask(
        name="count_vowels",
        description="counts the number of vowels in a string",
        difficulty="easy",
        category="string",
        test_cases=[
            {"args": ["hello"], "expected": 2},
            {"args": ["AEIOU"], "expected": 5},
            {"args": ["xyz"], "expected": 0},
        ]
    ),
    TrainingTask(
        name="remove_vowels",
        description="removes all vowels from a string",
        difficulty="easy",
        category="string",
        test_cases=[
            {"args": ["hello"], "expected": "hll"},
            {"args": ["AEIOU"], "expected": ""},
            {"args": ["python"], "expected": "pythn"},
        ]
    ),
    TrainingTask(
        name="compress_string",
        description="compresses a string using run-length encoding",
        difficulty="hard",
        category="string",
        test_cases=[
            {"args": ["aabcccccaaa"], "expected": "a2b1c5a3"},
            {"args": ["abc"], "expected": "a1b1c1"},
            {"args": [""], "expected": ""},
        ]
    ),
    TrainingTask(
        name="is_rotation",
        description="checks if one string is a rotation of another",
        difficulty="medium",
        category="string",
        test_cases=[
            {"args": ["waterbottle", "erbottlewat"], "expected": True},
            {"args": ["hello", "lohel"], "expected": True},
            {"args": ["hello", "world"], "expected": False},
        ]
    ),

    # === LIST/ARRAY TASKS ===
    TrainingTask(
        name="rotate_list",
        description="rotates a list by k positions to the right",
        difficulty="medium",
        category="list",
        test_cases=[
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [4, 5, 1, 2, 3]},
            {"args": [[1, 2, 3], 1], "expected": [3, 1, 2]},
            {"args": [[1, 2, 3], 3], "expected": [1, 2, 3]},
        ]
    ),
    TrainingTask(
        name="chunk_list",
        description="splits a list into chunks of size n",
        difficulty="medium",
        category="list",
        test_cases=[
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]},
            {"args": [[1, 2, 3, 4], 2], "expected": [[1, 2], [3, 4]]},
            {"args": [[1, 2, 3], 5], "expected": [[1, 2, 3]]},
        ]
    ),
    TrainingTask(
        name="intersection",
        description="finds the intersection of two lists",
        difficulty="easy",
        category="list",
        test_cases=[
            {"args": [[1, 2, 3, 4], [3, 4, 5, 6]], "expected": [3, 4]},
            {"args": [[1, 2], [3, 4]], "expected": []},
            {"args": [[1, 1, 2], [1, 1, 3]], "expected": [1]},
        ]
    ),
    TrainingTask(
        name="union",
        description="finds the union of two lists without duplicates",
        difficulty="easy",
        category="list",
        test_cases=[
            {"args": [[1, 2, 3], [3, 4, 5]], "expected": [1, 2, 3, 4, 5]},
            {"args": [[1, 2], [1, 2]], "expected": [1, 2]},
            {"args": [[], [1, 2]], "expected": [1, 2]},
        ]
    ),
    TrainingTask(
        name="second_largest",
        description="finds the second largest element in a list",
        difficulty="easy",
        category="list",
        test_cases=[
            {"args": [[1, 2, 3, 4, 5]], "expected": 4},
            {"args": [[10, 5, 8, 12]], "expected": 10},
            {"args": [[1, 1, 2]], "expected": 1},
        ]
    ),
    TrainingTask(
        name="move_zeros",
        description="moves all zeros to the end of a list",
        difficulty="medium",
        category="list",
        test_cases=[
            {"args": [[0, 1, 0, 3, 12]], "expected": [1, 3, 12, 0, 0]},
            {"args": [[1, 2, 3]], "expected": [1, 2, 3]},
            {"args": [[0, 0, 0]], "expected": [0, 0, 0]},
        ]
    ),
]


class AdvancedTrainer:
    """Advanced training for the autocoder."""

    def __init__(self, workspace: str = "/tmp/advanced_training"):
        self.workspace = workspace
        self.agent = None
        self.results: Dict[str, Dict] = {}
        self.stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "by_difficulty": {},
            "by_category": {},
        }

    def _get_agent(self) -> UnifiedCodingAgent:
        if self.agent is None:
            self.agent = UnifiedCodingAgent(repo_path=self.workspace)
            self.agent.auto_commit = False
            self.agent.auto_test = False
        return self.agent

    def _run_tests(self, code: str, func_name: str, test_cases: List[Dict]) -> Tuple[int, int]:
        """Run tests on generated code."""
        passed = 0
        total = len(test_cases)

        try:
            namespace = {"__name__": "__test__"}
            exec(code, namespace)

            if func_name not in namespace:
                return 0, total

            func = namespace[func_name]

            for test in test_cases:
                try:
                    result = func(*test["args"])
                    if result == test["expected"]:
                        passed += 1
                except Exception:
                    pass

        except Exception:
            pass

        return passed, total

    async def train_task(self, task: TrainingTask) -> Dict:
        """Train on a single task."""
        agent = self._get_agent()
        prompt = f"Create a Python function called '{task.name}' that {task.description}"

        start_time = time.time()
        result = await agent.solve_task(prompt)
        gen_time = time.time() - start_time

        if not result.success or not result.edits:
            return {
                "success": False,
                "passed": 0,
                "total": len(task.test_cases),
                "time": gen_time,
                "error": "Generation failed"
            }

        code = result.edits[0].modified
        passed, total = self._run_tests(code, task.name, task.test_cases)

        success = passed == total
        return {
            "success": success,
            "passed": passed,
            "total": total,
            "time": gen_time,
            "code": code if not success else None,
        }

    async def run_training(
        self,
        iterations: int = 100,
        tasks: List[TrainingTask] = None
    ):
        """Run training iterations."""
        if tasks is None:
            tasks = ADVANCED_TASKS

        print(f"\n{'='*60}")
        print("ADVANCED AUTOCODER TRAINING")
        print(f"{'='*60}")
        print(f"Tasks: {len(tasks)}")
        print(f"Iterations: {iterations}")
        print(f"{'='*60}\n")

        for i in range(iterations):
            task = random.choice(tasks)

            result = await self.train_task(task)

            self.stats["total"] += 1
            if result["success"]:
                self.stats["passed"] += 1
                status = "✅"
            else:
                self.stats["failed"] += 1
                status = "❌"

            # Track by difficulty
            diff = task.difficulty
            if diff not in self.stats["by_difficulty"]:
                self.stats["by_difficulty"][diff] = {"passed": 0, "failed": 0}
            if result["success"]:
                self.stats["by_difficulty"][diff]["passed"] += 1
            else:
                self.stats["by_difficulty"][diff]["failed"] += 1

            # Track by category
            cat = task.category
            if cat not in self.stats["by_category"]:
                self.stats["by_category"][cat] = {"passed": 0, "failed": 0}
            if result["success"]:
                self.stats["by_category"][cat]["passed"] += 1
            else:
                self.stats["by_category"][cat]["failed"] += 1

            # Track task-specific results
            if task.name not in self.results:
                self.results[task.name] = {"passed": 0, "failed": 0, "difficulty": task.difficulty}
            if result["success"]:
                self.results[task.name]["passed"] += 1
            else:
                self.results[task.name]["failed"] += 1

            logger.info(
                f"[{i+1}/{iterations}] {task.name} ({task.difficulty}) "
                f"{status} {result['passed']}/{result['total']} ({result['time']:.2f}s)"
            )

        self._print_report()

    def _print_report(self):
        """Print training report."""
        print(f"\n{'='*60}")
        print("TRAINING REPORT")
        print(f"{'='*60}")

        total = self.stats["total"]
        passed = self.stats["passed"]
        rate = passed / total * 100 if total > 0 else 0

        print(f"\n--- Overall ---")
        print(f"Total: {total}")
        print(f"Passed: {passed} ({rate:.1f}%)")
        print(f"Failed: {self.stats['failed']}")

        print(f"\n--- By Difficulty ---")
        for diff in ["easy", "medium", "hard", "expert"]:
            if diff in self.stats["by_difficulty"]:
                data = self.stats["by_difficulty"][diff]
                t = data["passed"] + data["failed"]
                r = data["passed"] / t * 100 if t > 0 else 0
                print(f"  {diff}: {data['passed']}/{t} ({r:.1f}%)")

        print(f"\n--- By Category ---")
        for cat, data in sorted(self.stats["by_category"].items()):
            t = data["passed"] + data["failed"]
            r = data["passed"] / t * 100 if t > 0 else 0
            print(f"  {cat}: {data['passed']}/{t} ({r:.1f}%)")

        print(f"\n--- By Task ---")
        # Sort by failure rate (worst first)
        sorted_tasks = sorted(
            self.results.items(),
            key=lambda x: x[1]["passed"] / (x[1]["passed"] + x[1]["failed"]) if x[1]["passed"] + x[1]["failed"] > 0 else 0
        )

        for name, data in sorted_tasks:
            t = data["passed"] + data["failed"]
            r = data["passed"] / t * 100 if t > 0 else 0
            status = "✅" if r == 100 else "⚠️" if r >= 50 else "❌"
            print(f"  {status} {name} ({data['difficulty']}): {data['passed']}/{t} ({r:.0f}%)")

        print(f"\n{'='*60}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Autocoder Training")
    parser.add_argument("-n", "--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "expert"], help="Filter by difficulty")
    parser.add_argument("--category", help="Filter by category")

    args = parser.parse_args()

    tasks = ADVANCED_TASKS

    if args.difficulty:
        tasks = [t for t in tasks if t.difficulty == args.difficulty]
    if args.category:
        tasks = [t for t in tasks if t.category == args.category]

    trainer = AdvancedTrainer()
    await trainer.run_training(iterations=args.iterations, tasks=tasks)


if __name__ == "__main__":
    asyncio.run(main())
