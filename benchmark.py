#!/usr/bin/env python3
"""
Autocoder Benchmark Suite

Comprehensive benchmark for testing the unified coding agent across
multiple categories of coding tasks with detailed performance metrics.

Usage:
    python benchmark.py                    # Run full benchmark
    python benchmark.py --category basic   # Run specific category
    python benchmark.py --quick            # Quick benchmark (subset)
    python benchmark.py --report           # Generate detailed report
"""

import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent))

from unified_coding_agent import UnifiedCodingAgent
from code_validator import CodeValidator, TestCase


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    name: str
    description: str
    category: str
    test_cases: List[TestCase]
    difficulty: str = "easy"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of running a single task."""
    task_name: str
    success: bool
    generation_time: float
    validation_time: float
    tests_passed: int
    tests_total: int
    error: Optional[str] = None
    code_length: int = 0


@dataclass
class BenchmarkResult:
    """Overall benchmark results."""
    timestamp: str
    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    total_time: float
    avg_generation_time: float
    by_category: Dict[str, Dict] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    task_results: List[TaskResult] = field(default_factory=list)


# Benchmark task definitions
BENCHMARK_TASKS = [
    # === BASIC TASKS ===
    BenchmarkTask(
        name="is_palindrome",
        description="checks if a string is a palindrome",
        category="basic",
        difficulty="easy",
        tags=["string", "boolean"],
        test_cases=[
            TestCase("basic_true", ["racecar"], expected=True),
            TestCase("basic_false", ["hello"], expected=False),
            TestCase("single_char", ["a"], expected=True),
            TestCase("empty", [""], expected=True),
        ]
    ),
    BenchmarkTask(
        name="fibonacci",
        description="returns the first n fibonacci numbers",
        category="basic",
        difficulty="easy",
        tags=["math", "sequence"],
        test_cases=[
            TestCase("five", [5], expected=[0, 1, 1, 2, 3]),
            TestCase("one", [1], expected=[0]),
            TestCase("zero", [0], expected=[]),
            TestCase("ten", [10], expected=[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        ]
    ),
    BenchmarkTask(
        name="is_prime",
        description="checks if a number is prime",
        category="basic",
        difficulty="easy",
        tags=["math", "boolean"],
        test_cases=[
            TestCase("prime_7", [7], expected=True),
            TestCase("not_prime_4", [4], expected=False),
            TestCase("prime_2", [2], expected=True),
            TestCase("not_prime_1", [1], expected=False),
            TestCase("prime_97", [97], expected=True),
        ]
    ),

    # === LIST OPERATIONS ===
    BenchmarkTask(
        name="find_max",
        description="finds the maximum value in a list",
        category="list",
        difficulty="easy",
        tags=["list", "search"],
        test_cases=[
            TestCase("basic", [[1, 5, 3, 9, 2]], expected=9),
            TestCase("single", [[42]], expected=42),
            TestCase("negative", [[-5, -1, -10]], expected=-1),
        ]
    ),
    BenchmarkTask(
        name="find_min",
        description="finds the minimum value in a list",
        category="list",
        difficulty="easy",
        tags=["list", "search"],
        test_cases=[
            TestCase("basic", [[1, 5, 3, 9, 2]], expected=1),
            TestCase("single", [[42]], expected=42),
            TestCase("negative", [[-5, -1, -10]], expected=-10),
        ]
    ),
    BenchmarkTask(
        name="sum_list",
        description="calculates the sum of all elements in a list",
        category="list",
        difficulty="easy",
        tags=["list", "math"],
        test_cases=[
            TestCase("basic", [[1, 2, 3, 4, 5]], expected=15),
            TestCase("empty", [[]], expected=0),
            TestCase("negative", [[-1, -2, 3]], expected=0),
        ]
    ),
    BenchmarkTask(
        name="remove_duplicates",
        description="removes duplicate elements from a list preserving order",
        category="list",
        difficulty="medium",
        tags=["list", "unique"],
        test_cases=[
            TestCase("basic", [[1, 2, 2, 3]], expected=[1, 2, 3]),
            TestCase("empty", [[]], expected=[]),
            TestCase("all_same", [[1, 1, 1]], expected=[1]),
            TestCase("no_dups", [[1, 2, 3]], expected=[1, 2, 3]),
        ]
    ),
    BenchmarkTask(
        name="find_duplicates",
        description="finds duplicate elements in a list and returns them sorted",
        category="list",
        difficulty="medium",
        tags=["list", "search"],
        test_cases=[
            TestCase("basic", [[1, 2, 3, 2, 4, 3]], expected=[2, 3]),
            TestCase("no_dups", [[1, 2, 3]], expected=[]),
            TestCase("all_same", [[1, 1, 1]], expected=[1]),
        ]
    ),
    BenchmarkTask(
        name="flatten_list",
        description="flattens a nested list into a single list",
        category="list",
        difficulty="medium",
        tags=["list", "recursion"],
        test_cases=[
            TestCase("nested", [[[1, 2], [3, [4, 5]]]], expected=[1, 2, 3, 4, 5]),
            TestCase("simple", [[[1], [2], [3]]], expected=[1, 2, 3]),
            TestCase("deep", [[[[1]], [[2]]]], expected=[1, 2]),
        ]
    ),

    # === STRING OPERATIONS ===
    BenchmarkTask(
        name="reverse_string",
        description="reverses a string",
        category="string",
        difficulty="easy",
        tags=["string"],
        test_cases=[
            TestCase("basic", ["hello"], expected="olleh"),
            TestCase("single", ["a"], expected="a"),
            TestCase("palindrome", ["racecar"], expected="racecar"),
        ]
    ),
    BenchmarkTask(
        name="count_words",
        description="counts the number of words in a string",
        category="string",
        difficulty="easy",
        tags=["string", "counting"],
        test_cases=[
            TestCase("two_words", ["hello world"], expected=2),
            TestCase("one_word", ["hello"], expected=1),
            TestCase("multiple", ["one two three four"], expected=4),
        ]
    ),
    BenchmarkTask(
        name="capitalize_words",
        description="capitalizes the first letter of each word",
        category="string",
        difficulty="easy",
        tags=["string", "transform"],
        test_cases=[
            TestCase("basic", ["hello world"], expected="Hello World"),
            TestCase("single", ["python"], expected="Python"),
            TestCase("already", ["Hello"], expected="Hello"),
        ]
    ),
    BenchmarkTask(
        name="is_anagram",
        description="checks if two strings are anagrams",
        category="string",
        difficulty="medium",
        tags=["string", "boolean"],
        test_cases=[
            TestCase("true", ["listen", "silent"], expected=True),
            TestCase("false", ["hello", "world"], expected=False),
            TestCase("case", ["Listen", "Silent"], expected=True),
        ]
    ),

    # === MATH OPERATIONS ===
    BenchmarkTask(
        name="get_average",
        description="calculates the average of numbers in a list",
        category="math",
        difficulty="easy",
        tags=["math", "list"],
        test_cases=[
            TestCase("basic", [[2, 4, 6]], expected=4.0),
            TestCase("single", [[5]], expected=5.0),
            TestCase("decimal", [[1, 2, 3, 4]], expected=2.5),
        ]
    ),
    BenchmarkTask(
        name="calculate_power",
        description="calculates base raised to the power of exponent",
        category="math",
        difficulty="easy",
        tags=["math"],
        test_cases=[
            TestCase("2_to_3", [2, 3], expected=8),
            TestCase("5_to_2", [5, 2], expected=25),
            TestCase("any_to_0", [10, 0], expected=1),
        ]
    ),
    BenchmarkTask(
        name="calculate_gcd",
        description="calculates the greatest common divisor",
        category="math",
        difficulty="medium",
        tags=["math", "algorithm"],
        test_cases=[
            TestCase("12_8", [12, 8], expected=4),
            TestCase("17_13", [17, 13], expected=1),
            TestCase("48_18", [48, 18], expected=6),
        ]
    ),

    # === SEARCH/FILTER ===
    BenchmarkTask(
        name="filter_even",
        description="filters and returns only even numbers from a list",
        category="filter",
        difficulty="easy",
        tags=["list", "filter"],
        test_cases=[
            TestCase("mixed", [[1, 2, 3, 4, 5, 6]], expected=[2, 4, 6]),
            TestCase("none", [[1, 3, 5]], expected=[]),
            TestCase("all", [[2, 4, 6]], expected=[2, 4, 6]),
        ]
    ),
    BenchmarkTask(
        name="filter_odd",
        description="filters and returns only odd numbers from a list",
        category="filter",
        difficulty="easy",
        tags=["list", "filter"],
        test_cases=[
            TestCase("mixed", [[1, 2, 3, 4, 5, 6]], expected=[1, 3, 5]),
            TestCase("none", [[2, 4, 6]], expected=[]),
            TestCase("all", [[1, 3, 5]], expected=[1, 3, 5]),
        ]
    ),
    BenchmarkTask(
        name="get_unique",
        description="returns unique elements preserving order",
        category="filter",
        difficulty="medium",
        tags=["list", "unique"],
        test_cases=[
            TestCase("basic", [[1, 2, 2, 3, 3, 3]], expected=[1, 2, 3]),
            TestCase("same", [[1, 1, 1]], expected=[1]),
            TestCase("unique", [[1, 2, 3]], expected=[1, 2, 3]),
        ]
    ),
    BenchmarkTask(
        name="binary_search",
        description="performs binary search on a sorted array",
        category="search",
        difficulty="medium",
        tags=["list", "algorithm", "search"],
        test_cases=[
            TestCase("found", [[1, 2, 3, 4, 5], 3], expected=2),
            TestCase("not_found", [[1, 2, 3, 4, 5], 6], expected=-1),
            TestCase("first", [[1, 2, 3, 4, 5], 1], expected=0),
            TestCase("last", [[1, 2, 3, 4, 5], 5], expected=4),
        ]
    ),
]


class Benchmark:
    """Benchmark runner for the autocoder."""

    def __init__(self, workspace: str = "/tmp/benchmark_workspace"):
        self.workspace = workspace
        self.agent = None
        self.validator = CodeValidator()

    def _get_agent(self) -> UnifiedCodingAgent:
        if self.agent is None:
            self.agent = UnifiedCodingAgent(repo_path=self.workspace)
            self.agent.auto_commit = False
            self.agent.auto_test = False
        return self.agent

    async def run_task(self, task: BenchmarkTask) -> TaskResult:
        """Run a single benchmark task."""
        agent = self._get_agent()
        prompt = f"Create a Python function called '{task.name}' that {task.description}"

        # Generate code
        gen_start = time.time()
        result = await agent.solve_task(prompt)
        gen_time = time.time() - gen_start

        if not result.success or not result.edits:
            return TaskResult(
                task_name=task.name,
                success=False,
                generation_time=gen_time,
                validation_time=0,
                tests_passed=0,
                tests_total=len(task.test_cases),
                error="Code generation failed"
            )

        code = result.edits[0].modified

        # Validate code
        val_start = time.time()
        validation = self.validator.validate(code, task.name, task.test_cases)
        val_time = time.time() - val_start

        return TaskResult(
            task_name=task.name,
            success=validation.valid,
            generation_time=gen_time,
            validation_time=val_time,
            tests_passed=validation.tests_passed,
            tests_total=validation.tests_passed + validation.tests_failed,
            error=validation.errors[0] if validation.errors else None,
            code_length=len(code)
        )

    async def run(
        self,
        tasks: Optional[List[BenchmarkTask]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> BenchmarkResult:
        """Run benchmark on selected tasks."""
        if tasks is None:
            tasks = BENCHMARK_TASKS

        # Filter tasks
        if category:
            tasks = [t for t in tasks if t.category == category]
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]

        print(f"\n{'='*60}")
        print(f"AUTOCODER BENCHMARK")
        print(f"{'='*60}")
        print(f"Tasks: {len(tasks)}")
        print(f"{'='*60}\n")

        start_time = time.time()
        results = []
        by_category: Dict[str, Dict] = {}
        by_difficulty: Dict[str, Dict] = {}

        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task.name} ({task.category}/{task.difficulty})", end=" ... ")
            result = await self.run_task(task)
            results.append(result)

            status = "✅" if result.success else "❌"
            print(f"{status} ({result.generation_time:.2f}s)")

            # Track by category
            if task.category not in by_category:
                by_category[task.category] = {"passed": 0, "failed": 0, "total_time": 0}
            if result.success:
                by_category[task.category]["passed"] += 1
            else:
                by_category[task.category]["failed"] += 1
            by_category[task.category]["total_time"] += result.generation_time

            # Track by difficulty
            if task.difficulty not in by_difficulty:
                by_difficulty[task.difficulty] = {"passed": 0, "failed": 0}
            if result.success:
                by_difficulty[task.difficulty]["passed"] += 1
            else:
                by_difficulty[task.difficulty]["failed"] += 1

        total_time = time.time() - start_time
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed

        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            total_tasks=len(results),
            passed=passed,
            failed=failed,
            pass_rate=passed / len(results) if results else 0,
            total_time=total_time,
            avg_generation_time=sum(r.generation_time for r in results) / len(results) if results else 0,
            by_category=by_category,
            by_difficulty=by_difficulty,
            task_results=results
        )

    def print_report(self, result: BenchmarkResult):
        """Print detailed benchmark report."""
        print(f"\n{'='*60}")
        print("BENCHMARK REPORT")
        print(f"{'='*60}")
        print(f"Timestamp: {result.timestamp}")
        print(f"\n--- Overall Results ---")
        print(f"Total Tasks: {result.total_tasks}")
        print(f"Passed: {result.passed} ({result.pass_rate*100:.1f}%)")
        print(f"Failed: {result.failed}")
        print(f"Total Time: {result.total_time:.2f}s")
        print(f"Avg Generation Time: {result.avg_generation_time:.3f}s")

        print(f"\n--- By Category ---")
        for cat, data in sorted(result.by_category.items()):
            total = data["passed"] + data["failed"]
            rate = data["passed"] / total * 100 if total > 0 else 0
            print(f"  {cat}: {data['passed']}/{total} ({rate:.0f}%)")

        print(f"\n--- By Difficulty ---")
        for diff in ["easy", "medium", "hard"]:
            if diff in result.by_difficulty:
                data = result.by_difficulty[diff]
                total = data["passed"] + data["failed"]
                rate = data["passed"] / total * 100 if total > 0 else 0
                print(f"  {diff}: {data['passed']}/{total} ({rate:.0f}%)")

        # Failed tasks
        failed_tasks = [r for r in result.task_results if not r.success]
        if failed_tasks:
            print(f"\n--- Failed Tasks ---")
            for r in failed_tasks:
                print(f"  ❌ {r.task_name}: {r.error or 'Unknown error'}")

        print(f"\n{'='*60}")

    def save_report(self, result: BenchmarkResult, filepath: str):
        """Save benchmark report to JSON file."""
        data = {
            "timestamp": result.timestamp,
            "summary": {
                "total_tasks": result.total_tasks,
                "passed": result.passed,
                "failed": result.failed,
                "pass_rate": result.pass_rate,
                "total_time": result.total_time,
                "avg_generation_time": result.avg_generation_time,
            },
            "by_category": result.by_category,
            "by_difficulty": result.by_difficulty,
            "task_results": [asdict(r) for r in result.task_results]
        }

        Path(filepath).write_text(json.dumps(data, indent=2))
        print(f"\nReport saved to: {filepath}")


async def main():
    parser = argparse.ArgumentParser(description="Autocoder Benchmark Suite")
    parser.add_argument("--category", help="Run only specific category")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (easy tasks only)")
    parser.add_argument("--report", help="Save report to file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.quiet:
        import logging
        logging.getLogger("unified_coding_agent").setLevel(logging.WARNING)

    benchmark = Benchmark()

    difficulty = "easy" if args.quick else args.difficulty
    result = await benchmark.run(
        category=args.category,
        difficulty=difficulty
    )

    benchmark.print_report(result)

    if args.report:
        benchmark.save_report(result, args.report)


if __name__ == "__main__":
    asyncio.run(main())
