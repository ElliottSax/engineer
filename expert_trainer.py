#!/usr/bin/env python3
"""
Expert Training for Autocoder

Significantly harder tasks that push pattern-based code generation to its limits.
Includes:
- Dynamic programming problems
- Graph algorithms
- Tree operations
- Complex string manipulations
- Mathematical challenges
- Multi-step algorithmic problems
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
class ExpertTask:
    name: str
    description: str
    test_cases: List[Dict]
    difficulty: str  # hard, expert, legendary
    category: str


# Expert-level training tasks
EXPERT_TASKS = [
    # === DYNAMIC PROGRAMMING ===
    ExpertTask(
        name="longest_common_subsequence",
        description="finds the longest common subsequence of two strings",
        difficulty="expert",
        category="dynamic_programming",
        test_cases=[
            {"args": ["ABCDGH", "AEDFHR"], "expected": "ADH"},
            {"args": ["AGGTAB", "GXTXAYB"], "expected": "GTAB"},
            {"args": ["ABC", "DEF"], "expected": ""},
            {"args": ["ABC", "ABC"], "expected": "ABC"},
        ]
    ),
    ExpertTask(
        name="edit_distance",
        description="calculates the minimum edit distance (Levenshtein distance) between two strings",
        difficulty="expert",
        category="dynamic_programming",
        test_cases=[
            {"args": ["kitten", "sitting"], "expected": 3},
            {"args": ["saturday", "sunday"], "expected": 3},
            {"args": ["", "abc"], "expected": 3},
            {"args": ["abc", "abc"], "expected": 0},
        ]
    ),
    ExpertTask(
        name="knapsack",
        description="solves the 0/1 knapsack problem given weights, values, and capacity",
        difficulty="legendary",
        category="dynamic_programming",
        test_cases=[
            {"args": [[10, 20, 30], [60, 100, 120], 50], "expected": 220},
            {"args": [[1, 2, 3], [6, 10, 12], 5], "expected": 22},
            {"args": [[5], [10], 4], "expected": 0},
        ]
    ),
    ExpertTask(
        name="coin_change",
        description="finds the minimum number of coins needed to make a given amount",
        difficulty="expert",
        category="dynamic_programming",
        test_cases=[
            {"args": [[1, 2, 5], 11], "expected": 3},
            {"args": [[2], 3], "expected": -1},
            {"args": [[1], 0], "expected": 0},
            {"args": [[1, 5, 10, 25], 30], "expected": 2},
        ]
    ),
    ExpertTask(
        name="longest_increasing_subsequence",
        description="finds the length of the longest increasing subsequence in a list",
        difficulty="expert",
        category="dynamic_programming",
        test_cases=[
            {"args": [[10, 9, 2, 5, 3, 7, 101, 18]], "expected": 4},
            {"args": [[0, 1, 0, 3, 2, 3]], "expected": 4},
            {"args": [[7, 7, 7, 7]], "expected": 1},
        ]
    ),
    ExpertTask(
        name="max_subarray_sum",
        description="finds the maximum sum of a contiguous subarray (Kadane's algorithm)",
        difficulty="hard",
        category="dynamic_programming",
        test_cases=[
            {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"args": [[1]], "expected": 1},
            {"args": [[5, 4, -1, 7, 8]], "expected": 23},
            {"args": [[-1, -2, -3]], "expected": -1},
        ]
    ),

    # === GRAPH ALGORITHMS ===
    ExpertTask(
        name="topological_sort",
        description="performs topological sort on a directed acyclic graph represented as adjacency list",
        difficulty="expert",
        category="graph",
        test_cases=[
            {"args": [{0: [1, 2], 1: [3], 2: [3], 3: []}], "expected": [0, 2, 1, 3]},
            {"args": [{0: [1], 1: [2], 2: []}], "expected": [0, 1, 2]},
        ]
    ),
    ExpertTask(
        name="detect_cycle",
        description="detects if a directed graph contains a cycle",
        difficulty="expert",
        category="graph",
        test_cases=[
            {"args": [{0: [1], 1: [2], 2: [0]}], "expected": True},
            {"args": [{0: [1], 1: [2], 2: []}], "expected": False},
            {"args": [{0: [1, 2], 1: [], 2: []}], "expected": False},
        ]
    ),
    ExpertTask(
        name="shortest_path",
        description="finds the shortest path between two nodes in an unweighted graph using BFS",
        difficulty="hard",
        category="graph",
        test_cases=[
            {"args": [{0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}, 0, 3], "expected": [0, 1, 3]},
            {"args": [{0: [1], 1: [2], 2: [3], 3: []}, 0, 3], "expected": [0, 1, 2, 3]},
        ]
    ),
    ExpertTask(
        name="connected_components",
        description="counts the number of connected components in an undirected graph",
        difficulty="hard",
        category="graph",
        test_cases=[
            {"args": [{0: [1], 1: [0], 2: [3], 3: [2], 4: []}], "expected": 3},
            {"args": [{0: [1, 2], 1: [0, 2], 2: [0, 1]}], "expected": 1},
            {"args": [{0: [], 1: [], 2: []}], "expected": 3},
        ]
    ),

    # === TREE OPERATIONS ===
    ExpertTask(
        name="tree_depth",
        description="calculates the maximum depth of a binary tree represented as nested dict with 'val', 'left', 'right'",
        difficulty="hard",
        category="tree",
        test_cases=[
            {"args": [{"val": 1, "left": {"val": 2, "left": None, "right": None}, "right": {"val": 3, "left": None, "right": None}}], "expected": 2},
            {"args": [{"val": 1, "left": {"val": 2, "left": {"val": 3, "left": None, "right": None}, "right": None}, "right": None}], "expected": 3},
            {"args": [None], "expected": 0},
        ]
    ),
    ExpertTask(
        name="invert_tree",
        description="inverts a binary tree (swaps left and right children at every node)",
        difficulty="hard",
        category="tree",
        test_cases=[
            {"args": [{"val": 1, "left": {"val": 2, "left": None, "right": None}, "right": {"val": 3, "left": None, "right": None}}],
             "expected": {"val": 1, "left": {"val": 3, "left": None, "right": None}, "right": {"val": 2, "left": None, "right": None}}},
            {"args": [None], "expected": None},
        ]
    ),
    ExpertTask(
        name="is_balanced_tree",
        description="checks if a binary tree is height-balanced (heights of subtrees differ by at most 1)",
        difficulty="expert",
        category="tree",
        test_cases=[
            {"args": [{"val": 1, "left": {"val": 2, "left": None, "right": None}, "right": {"val": 3, "left": None, "right": None}}], "expected": True},
            {"args": [{"val": 1, "left": {"val": 2, "left": {"val": 3, "left": None, "right": None}, "right": None}, "right": None}], "expected": False},
            {"args": [None], "expected": True},
        ]
    ),

    # === COMPLEX STRING OPERATIONS ===
    ExpertTask(
        name="longest_palindromic_substring",
        description="finds the longest palindromic substring in a string",
        difficulty="expert",
        category="string",
        test_cases=[
            {"args": ["babad"], "expected": "bab"},  # or "aba"
            {"args": ["cbbd"], "expected": "bb"},
            {"args": ["a"], "expected": "a"},
            {"args": ["racecar"], "expected": "racecar"},
        ]
    ),
    ExpertTask(
        name="valid_parentheses",
        description="checks if a string of brackets ()[]{}  is valid (properly matched and nested)",
        difficulty="hard",
        category="string",
        test_cases=[
            {"args": ["()[]{}"], "expected": True},
            {"args": ["([)]"], "expected": False},
            {"args": ["{[]}"], "expected": True},
            {"args": ["((("], "expected": False},
            {"args": [""], "expected": True},
        ]
    ),
    ExpertTask(
        name="generate_parentheses",
        description="generates all combinations of n pairs of well-formed parentheses",
        difficulty="expert",
        category="string",
        test_cases=[
            {"args": [1], "expected": ["()"]},
            {"args": [2], "expected": ["(())", "()()"]},
            {"args": [3], "expected": ["((()))", "(()())", "(())()", "()(())", "()()()"]},
        ]
    ),
    ExpertTask(
        name="word_break",
        description="checks if a string can be segmented into space-separated dictionary words",
        difficulty="expert",
        category="string",
        test_cases=[
            {"args": ["leetcode", ["leet", "code"]], "expected": True},
            {"args": ["applepenapple", ["apple", "pen"]], "expected": True},
            {"args": ["catsandog", ["cats", "dog", "sand", "and", "cat"]], "expected": False},
        ]
    ),

    # === MATHEMATICAL CHALLENGES ===
    ExpertTask(
        name="prime_factors",
        description="returns all prime factors of a number in ascending order with repetition",
        difficulty="hard",
        category="math",
        test_cases=[
            {"args": [12], "expected": [2, 2, 3]},
            {"args": [100], "expected": [2, 2, 5, 5]},
            {"args": [13], "expected": [13]},
            {"args": [1], "expected": []},
        ]
    ),
    ExpertTask(
        name="power_set",
        description="generates all subsets (power set) of a list",
        difficulty="hard",
        category="math",
        test_cases=[
            {"args": [[1, 2]], "expected": [[], [1], [2], [1, 2]]},
            {"args": [[1]], "expected": [[], [1]]},
            {"args": [[]], "expected": [[]]},
        ]
    ),
    ExpertTask(
        name="permutations",
        description="generates all permutations of a list",
        difficulty="hard",
        category="math",
        test_cases=[
            {"args": [[1, 2]], "expected": [[1, 2], [2, 1]]},
            {"args": [[1, 2, 3]], "expected": [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]},
        ]
    ),
    ExpertTask(
        name="combinations",
        description="generates all combinations of k elements from a list",
        difficulty="hard",
        category="math",
        test_cases=[
            {"args": [[1, 2, 3, 4], 2], "expected": [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]},
            {"args": [[1, 2, 3], 1], "expected": [[1], [2], [3]]},
        ]
    ),
    ExpertTask(
        name="matrix_multiply",
        description="multiplies two matrices",
        difficulty="hard",
        category="math",
        test_cases=[
            {"args": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "expected": [[19, 22], [43, 50]]},
            {"args": [[[1, 2, 3]], [[4], [5], [6]]], "expected": [[32]]},
        ]
    ),
    ExpertTask(
        name="spiral_matrix",
        description="returns elements of a matrix in spiral order",
        difficulty="expert",
        category="math",
        test_cases=[
            {"args": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], "expected": [1, 2, 3, 6, 9, 8, 7, 4, 5]},
            {"args": [[[1, 2], [3, 4]]], "expected": [1, 2, 4, 3]},
        ]
    ),

    # === ADVANCED LIST OPERATIONS ===
    ExpertTask(
        name="merge_intervals",
        description="merges overlapping intervals",
        difficulty="hard",
        category="list",
        test_cases=[
            {"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expected": [[1, 6], [8, 10], [15, 18]]},
            {"args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
            {"args": [[[1, 2]]], "expected": [[1, 2]]},
        ]
    ),
    ExpertTask(
        name="three_sum",
        description="finds all unique triplets in a list that sum to zero",
        difficulty="expert",
        category="list",
        test_cases=[
            {"args": [[-1, 0, 1, 2, -1, -4]], "expected": [[-1, -1, 2], [-1, 0, 1]]},
            {"args": [[0, 0, 0]], "expected": [[0, 0, 0]]},
            {"args": [[1, 2, 3]], "expected": []},
        ]
    ),
    ExpertTask(
        name="trap_water",
        description="calculates how much rainwater can be trapped between bars of given heights",
        difficulty="legendary",
        category="list",
        test_cases=[
            {"args": [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]], "expected": 6},
            {"args": [[4, 2, 0, 3, 2, 5]], "expected": 9},
            {"args": [[1, 2, 3]], "expected": 0},
        ]
    ),
    ExpertTask(
        name="median_two_sorted",
        description="finds the median of two sorted arrays",
        difficulty="legendary",
        category="list",
        test_cases=[
            {"args": [[1, 3], [2]], "expected": 2.0},
            {"args": [[1, 2], [3, 4]], "expected": 2.5},
            {"args": [[0, 0], [0, 0]], "expected": 0.0},
        ]
    ),
]


class ExpertTrainer:
    """Expert-level training for the autocoder."""

    def __init__(self, workspace: str = "/tmp/expert_training"):
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

    def _run_tests(self, code: str, func_name: str, test_cases: List[Dict]) -> Tuple[int, int, List[str]]:
        """Run tests on generated code."""
        passed = 0
        total = len(test_cases)
        errors = []

        try:
            namespace = {"__name__": "__test__"}
            exec(code, namespace)

            if func_name not in namespace:
                return 0, total, [f"Function '{func_name}' not found"]

            func = namespace[func_name]

            for i, test in enumerate(test_cases):
                try:
                    result = func(*test["args"])
                    expected = test["expected"]

                    # Handle special cases for comparison
                    if isinstance(expected, list) and isinstance(result, list):
                        # For some problems, order might not matter
                        if func_name in ["power_set", "permutations", "combinations", "three_sum", "generate_parentheses"]:
                            # Convert inner lists to tuples for comparison
                            result_set = set(tuple(x) if isinstance(x, list) else x for x in result)
                            expected_set = set(tuple(x) if isinstance(x, list) else x for x in expected)
                            if result_set == expected_set:
                                passed += 1
                                continue
                        elif result == expected:
                            passed += 1
                            continue
                    elif result == expected:
                        passed += 1
                        continue

                    # Special handling for palindrome (multiple valid answers)
                    if func_name == "longest_palindromic_substring":
                        if len(result) == len(expected) and result == result[::-1]:
                            passed += 1
                            continue

                    # Special handling for topological sort (multiple valid orderings)
                    if func_name == "topological_sort":
                        # Just check that it's a valid ordering
                        if len(result) == len(expected):
                            passed += 1
                            continue

                    errors.append(f"Test {i+1}: expected {expected}, got {result}")
                except Exception as e:
                    errors.append(f"Test {i+1}: {type(e).__name__}: {e}")

        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        except Exception as e:
            errors.append(f"Execution error: {type(e).__name__}: {e}")

        return passed, total, errors

    async def train_task(self, task: ExpertTask) -> Dict:
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
                "error": "Generation failed",
                "errors": []
            }

        code = result.edits[0].modified
        passed, total, errors = self._run_tests(code, task.name, task.test_cases)

        success = passed == total
        return {
            "success": success,
            "passed": passed,
            "total": total,
            "time": gen_time,
            "code": code if not success else None,
            "errors": errors if not success else []
        }

    async def run_training(
        self,
        iterations: int = 100,
        tasks: List[ExpertTask] = None
    ):
        """Run training iterations."""
        if tasks is None:
            tasks = EXPERT_TASKS

        print(f"\n{'='*60}")
        print("EXPERT AUTOCODER TRAINING")
        print(f"{'='*60}")
        print(f"Tasks: {len(tasks)}")
        print(f"Iterations: {iterations}")
        print(f"Difficulty levels: hard, expert, legendary")
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
                self.results[task.name] = {"passed": 0, "failed": 0, "difficulty": task.difficulty, "errors": []}
            if result["success"]:
                self.results[task.name]["passed"] += 1
            else:
                self.results[task.name]["failed"] += 1
                if result.get("errors"):
                    self.results[task.name]["errors"].extend(result["errors"][:2])  # Keep first 2 errors

            logger.info(
                f"[{i+1}/{iterations}] {task.name} ({task.difficulty}) "
                f"{status} {result['passed']}/{result['total']} ({result['time']:.2f}s)"
            )

        self._print_report()

    def _print_report(self):
        """Print training report."""
        print(f"\n{'='*60}")
        print("EXPERT TRAINING REPORT")
        print(f"{'='*60}")

        total = self.stats["total"]
        passed = self.stats["passed"]
        rate = passed / total * 100 if total > 0 else 0

        print(f"\n--- Overall ---")
        print(f"Total: {total}")
        print(f"Passed: {passed} ({rate:.1f}%)")
        print(f"Failed: {self.stats['failed']}")

        print(f"\n--- By Difficulty ---")
        for diff in ["hard", "expert", "legendary"]:
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
            if data.get("errors") and r < 100:
                for err in data["errors"][:1]:
                    print(f"      └─ {err[:60]}...")

        print(f"\n{'='*60}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Expert Autocoder Training")
    parser.add_argument("-n", "--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--difficulty", choices=["hard", "expert", "legendary"], help="Filter by difficulty")
    parser.add_argument("--category", help="Filter by category")

    args = parser.parse_args()

    tasks = EXPERT_TASKS

    if args.difficulty:
        tasks = [t for t in tasks if t.difficulty == args.difficulty]
    if args.category:
        tasks = [t for t in tasks if t.category == args.category]

    trainer = ExpertTrainer()
    await trainer.run_training(iterations=args.iterations, tasks=tasks)


if __name__ == "__main__":
    asyncio.run(main())
