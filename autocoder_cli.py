#!/usr/bin/env python3
"""
Autocoder CLI - Command-line interface for the unified coding agent.

Usage:
    python autocoder_cli.py "Create a function that calculates factorial"
    python autocoder_cli.py --file task.txt
    python autocoder_cli.py --interactive
    python autocoder_cli.py --benchmark

Examples:
    # Generate a single function
    python autocoder_cli.py "Create a function is_prime that checks if a number is prime"

    # Run in interactive mode
    python autocoder_cli.py -i

    # Run benchmark
    python autocoder_cli.py --benchmark

    # Generate code and save to file
    python autocoder_cli.py "Create a sorting function" --output sort.py
"""

import os
import sys
import asyncio
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_coding_agent import UnifiedCodingAgent, AgentResult


class AutocoderCLI:
    """Command-line interface for the autocoder."""

    def __init__(self, workspace: str = "/tmp/autocoder_workspace"):
        self.workspace = workspace
        self.agent = None
        self.history: List[Dict] = []

    def _get_agent(self) -> UnifiedCodingAgent:
        """Get or create agent instance."""
        if self.agent is None:
            self.agent = UnifiedCodingAgent(repo_path=self.workspace)
            self.agent.auto_commit = False
            self.agent.auto_test = False
        return self.agent

    async def generate(self, task: str, output_file: Optional[str] = None) -> AgentResult:
        """Generate code for a task."""
        agent = self._get_agent()

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        start_time = time.time()
        result = await agent.solve_task(task)
        elapsed = time.time() - start_time

        if result.success and result.edits:
            code = result.edits[0].modified

            print(f"\nGenerated code ({elapsed:.2f}s):")
            print("-" * 40)
            print(code)
            print("-" * 40)

            if output_file:
                Path(output_file).write_text(code)
                print(f"\nSaved to: {output_file}")

            # Record in history
            self.history.append({
                "task": task,
                "success": True,
                "time": elapsed,
                "timestamp": datetime.now().isoformat()
            })
        else:
            print(f"\nFailed: {result.message}")
            self.history.append({
                "task": task,
                "success": False,
                "error": result.message,
                "timestamp": datetime.now().isoformat()
            })

        return result

    async def interactive(self):
        """Run in interactive mode."""
        print("\n" + "="*60)
        print("Autocoder Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  <task>     - Generate code for task")
        print("  /test      - Test last generated code")
        print("  /save <f>  - Save last code to file")
        print("  /history   - Show generation history")
        print("  /clear     - Clear history")
        print("  /help      - Show this help")
        print("  /quit      - Exit")
        print("="*60 + "\n")

        last_code = None

        while True:
            try:
                user_input = input("autocoder> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()
                args = user_input[len(cmd):].strip()

                if cmd == "/quit" or cmd == "/exit":
                    print("Goodbye!")
                    break
                elif cmd == "/help":
                    print("Commands: /test, /save <file>, /history, /clear, /quit")
                elif cmd == "/history":
                    if not self.history:
                        print("No history yet.")
                    else:
                        for i, h in enumerate(self.history[-10:], 1):
                            status = "OK" if h.get("success") else "FAIL"
                            print(f"{i}. [{status}] {h['task'][:50]}...")
                elif cmd == "/clear":
                    self.history.clear()
                    print("History cleared.")
                elif cmd == "/save":
                    if not last_code:
                        print("No code to save. Generate something first.")
                    elif not args:
                        print("Usage: /save <filename>")
                    else:
                        Path(args).write_text(last_code)
                        print(f"Saved to {args}")
                elif cmd == "/test":
                    if not last_code:
                        print("No code to test. Generate something first.")
                    else:
                        print("Testing generated code...")
                        try:
                            namespace = {}
                            exec(last_code, namespace)
                            funcs = [k for k in namespace.keys() if not k.startswith("_") and callable(namespace[k])]
                            print(f"Code executed successfully!")
                            print(f"Defined functions: {', '.join(funcs)}")
                        except Exception as e:
                            print(f"Error: {e}")
                else:
                    print(f"Unknown command: {cmd}")
            else:
                result = await self.generate(user_input)
                if result.success and result.edits:
                    last_code = result.edits[0].modified

    async def benchmark(self):
        """Run benchmark against all patterns."""
        print("\n" + "="*60)
        print("Autocoder Benchmark")
        print("="*60 + "\n")

        TASKS = [
            ("is_palindrome", "checks if a string is a palindrome", ["racecar"], True),
            ("fibonacci", "returns the first n fibonacci numbers", [5], [0, 1, 1, 2, 3]),
            ("is_prime", "checks if a number is prime", [7], True),
            ("find_max", "finds the maximum value in a list", [[1, 5, 3]], 5),
            ("find_min", "finds the minimum value in a list", [[1, 5, 3]], 1),
            ("reverse_string", "reverses a string", ["hello"], "olleh"),
            ("sum_list", "calculates the sum of elements", [[1, 2, 3]], 6),
            ("count_words", "counts words in a string", ["hello world"], 2),
            ("find_duplicates", "finds duplicate elements", [[1, 2, 2, 3]], [2]),
            ("remove_duplicates", "removes duplicates from list", [[1, 2, 2, 3]], [1, 2, 3]),
            ("flatten_list", "flattens a nested list", [[[1], [2, 3]]], [1, 2, 3]),
            ("get_average", "calculates the average", [[2, 4, 6]], 4.0),
            ("filter_even", "filters even numbers", [[1, 2, 3, 4]], [2, 4]),
            ("filter_odd", "filters odd numbers", [[1, 2, 3, 4]], [1, 3]),
            ("calculate_power", "calculates base to the power of exp", [2, 3], 8),
            ("calculate_gcd", "calculates greatest common divisor", [12, 8], 4),
            ("is_anagram", "checks if strings are anagrams", ["listen", "silent"], True),
            ("binary_search", "binary search on sorted array", [[1, 2, 3, 4, 5], 3], 2),
            ("capitalize_words", "capitalizes each word", ["hello world"], "Hello World"),
            ("get_unique", "gets unique elements", [[1, 1, 2, 2]], [1, 2]),
        ]

        passed = 0
        failed = 0
        total_time = 0

        for func_name, desc, args, expected in TASKS:
            task = f"Create a function {func_name} that {desc}"

            start = time.time()
            result = await self.generate(task)
            elapsed = time.time() - start
            total_time += elapsed

            if result.success and result.edits:
                code = result.edits[0].modified
                try:
                    namespace = {}
                    exec(code, namespace)
                    actual = namespace[func_name](*args)
                    if actual == expected:
                        print(f"  PASS ({elapsed:.2f}s)")
                        passed += 1
                    else:
                        print(f"  FAIL: expected {expected}, got {actual}")
                        failed += 1
                except Exception as e:
                    print(f"  ERROR: {e}")
                    failed += 1
            else:
                print(f"  FAIL: {result.message}")
                failed += 1

        print("\n" + "="*60)
        print(f"Benchmark Results")
        print("="*60)
        print(f"Passed: {passed}/{passed+failed} ({passed/(passed+failed)*100:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per task: {total_time/(passed+failed):.3f}s")
        print("="*60)

        return passed, failed

    async def run_file(self, filepath: str, output_dir: Optional[str] = None):
        """Run tasks from a file (one task per line)."""
        tasks = Path(filepath).read_text().strip().split("\n")
        tasks = [t.strip() for t in tasks if t.strip() and not t.startswith("#")]

        print(f"Running {len(tasks)} tasks from {filepath}\n")

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] {task}")
            output = None
            if output_dir:
                output = os.path.join(output_dir, f"task_{i}.py")
            await self.generate(task, output)


def main():
    parser = argparse.ArgumentParser(
        description="Autocoder CLI - Generate code using the unified coding agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="The coding task to perform"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "-f", "--file",
        help="Read tasks from file (one per line)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file to save generated code"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for multi-task generation"
    )

    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        help="Run benchmark on all patterns"
    )

    parser.add_argument(
        "-w", "--workspace",
        default="/tmp/autocoder_workspace",
        help="Workspace directory (default: /tmp/autocoder_workspace)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        import logging
        logging.getLogger("unified_coding_agent").setLevel(logging.WARNING)

    cli = AutocoderCLI(workspace=args.workspace)

    if args.interactive:
        asyncio.run(cli.interactive())
    elif args.benchmark:
        asyncio.run(cli.benchmark())
    elif args.file:
        asyncio.run(cli.run_file(args.file, args.output_dir))
    elif args.task:
        asyncio.run(cli.generate(args.task, args.output))
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python autocoder_cli.py "Create a function is_prime that checks if prime"')
        print('  python autocoder_cli.py --interactive')
        print('  python autocoder_cli.py --benchmark')


if __name__ == "__main__":
    main()
