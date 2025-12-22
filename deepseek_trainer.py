#!/usr/bin/env python3
"""
DeepSeek-Based Continuous Training for Autocoder

Uses DeepSeek API ($0.14/M tokens) for:
1. Generating training tasks
2. Evaluating autocoder outputs
3. Providing feedback and corrections
4. Continuous improvement loop

Cost estimate: ~$0.01 per training iteration (vs ~$0.10 with Claude)
"""

import os
import json
import asyncio
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DEEPSEEK CLIENT
# =============================================================================

class DeepSeekClient:
    """Client for DeepSeek API - Ultra cheap at $0.14/M tokens"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"

        self.total_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 2000) -> str:
        """Send completion request to DeepSeek"""

        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY not set")
            return "Error: No API key"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                # Track usage
                usage = result.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                self.total_tokens += tokens
                self.total_cost += (tokens / 1_000_000) * 0.14
                self.request_count += 1

                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"

        except Exception as e:
            logger.error(f"DeepSeek request failed: {e}")
            return f"Error: {e}"

    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
            "request_count": self.request_count,
            "avg_cost_per_request": f"${self.total_cost/max(1,self.request_count):.4f}"
        }


# =============================================================================
# TASK GENERATOR
# =============================================================================

class TaskGenerator:
    """Generates training tasks using DeepSeek"""

    TASK_TEMPLATES = [
        # Code generation
        "Create a Python function called '{func_name}' that {description}",
        "Implement a function '{func_name}' in Python that {description}",
        "Write a Python function named '{func_name}' to {description}",

        # Algorithms
        "Implement {algorithm} algorithm in Python",
        "Create a function that performs {operation} on {data_structure}",

        # Data structures
        "Implement a {data_structure} class in Python with {methods}",
        "Create a {data_structure} with methods for {operations}",
    ]

    FUNCTION_IDEAS = [
        ("find_max", "finds the maximum value in a list of numbers"),
        ("find_min", "finds the minimum value in a list of numbers"),
        ("calculate_average", "calculates the average of a list of numbers"),
        ("count_occurrences", "counts how many times each element appears in a list"),
        ("remove_duplicates", "removes duplicate elements from a list"),
        ("reverse_string", "reverses a string"),
        ("is_anagram", "checks if two strings are anagrams"),
        ("binary_search", "performs binary search on a sorted list"),
        ("merge_sorted_lists", "merges two sorted lists into one sorted list"),
        ("find_common_elements", "finds common elements between two lists"),
        ("flatten_list", "flattens a nested list into a single list"),
        ("rotate_list", "rotates a list by n positions"),
        ("find_missing_number", "finds the missing number in a sequence"),
        ("validate_parentheses", "checks if parentheses in a string are balanced"),
        ("compress_string", "compresses a string using run-length encoding"),
    ]

    def __init__(self, deepseek: DeepSeekClient):
        self.deepseek = deepseek
        self.generated_tasks = []

    def generate_simple_task(self) -> Dict:
        """Generate a simple task without API call"""
        func_name, description = random.choice(self.FUNCTION_IDEAS)
        template = random.choice(self.TASK_TEMPLATES[:3])

        task = template.format(func_name=func_name, description=description)

        return {
            "id": hashlib.md5(f"{func_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "task": task,
            "func_name": func_name,
            "difficulty": "beginner",
            "category": "code_generation"
        }

    async def generate_task_with_tests(self) -> Dict:
        """Generate a task with test cases using DeepSeek"""

        base_task = self.generate_simple_task()

        prompt = f"""Generate test cases for this Python function task:

Task: {base_task['task']}

Provide 3 test cases as a JSON array:
```json
[
  {{"input": "example input", "expected": "expected output"}},
  {{"input": "edge case", "expected": "expected output"}},
  {{"input": "another case", "expected": "expected output"}}
]
```

Make sure test cases cover normal cases and edge cases."""

        response = await self.deepseek.complete(prompt, system="You are a Python testing expert. Generate practical test cases.")

        # Parse test cases
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                tests = json.loads(response[json_start:json_end])
                base_task['tests'] = tests
        except json.JSONDecodeError:
            base_task['tests'] = []

        self.generated_tasks.append(base_task)
        return base_task


# =============================================================================
# EVALUATOR
# =============================================================================

class DeepSeekEvaluator:
    """Evaluates autocoder outputs using DeepSeek"""

    def __init__(self, deepseek: DeepSeekClient):
        self.deepseek = deepseek
        self.evaluations = []

    async def evaluate(self, task: str, code: str, test_results: List[Dict]) -> Dict:
        """Evaluate generated code"""

        # Calculate test score
        passed = sum(1 for t in test_results if t.get('passed', False))
        total = len(test_results) if test_results else 1
        test_score = passed / total

        # Quick evaluation for high-scoring code
        if test_score >= 0.9:
            return {
                "score": test_score,
                "feedback": "Code passes all tests. Good implementation.",
                "improvements": [],
                "grade": "A"
            }

        # Get detailed feedback from DeepSeek for failing code
        prompt = f"""Evaluate this Python code:

TASK: {task}

CODE:
```python
{code}
```

TEST RESULTS: {passed}/{total} passed

Provide brief feedback as JSON:
```json
{{
  "score": 0.0 to 1.0,
  "feedback": "one sentence summary",
  "improvements": ["improvement 1", "improvement 2"],
  "grade": "A/B/C/D/F"
}}
```"""

        response = await self.deepseek.complete(
            prompt,
            system="You are a code reviewer. Be concise.",
            max_tokens=500
        )

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
                result['test_score'] = test_score
                self.evaluations.append(result)
                return result
        except json.JSONDecodeError:
            pass

        return {
            "score": test_score,
            "feedback": "Evaluation parsing failed",
            "improvements": [],
            "grade": "C" if test_score >= 0.5 else "F"
        }

    def get_summary(self) -> Dict:
        """Get evaluation summary"""
        if not self.evaluations:
            return {"count": 0}

        scores = [e.get('score', 0) for e in self.evaluations]
        return {
            "count": len(self.evaluations),
            "avg_score": sum(scores) / len(scores),
            "passing_rate": sum(1 for s in scores if s >= 0.7) / len(scores)
        }


# =============================================================================
# TRAINING WORKER
# =============================================================================

class TrainingWorker:
    """Continuous training worker"""

    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id
        self.deepseek = DeepSeekClient()
        self.task_gen = TaskGenerator(self.deepseek)
        self.evaluator = DeepSeekEvaluator(self.deepseek)

        self.iterations = 0
        self.successes = 0
        self.failures = 0
        self.improvements_made = []

        self.output_dir = Path("training_output")
        self.output_dir.mkdir(exist_ok=True)

        self.running = False

    async def get_autocoder(self):
        """Get autocoder instance"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        # Reload to get latest version
        if 'unified_coding_agent' in sys.modules:
            del sys.modules['unified_coding_agent']

        from unified_coding_agent import UnifiedCodingAgent

        agent = UnifiedCodingAgent(repo_path='/tmp/training_workspace')
        agent.auto_commit = False
        agent.auto_test = False
        return agent

    async def run_iteration(self) -> Dict:
        """Run a single training iteration"""

        self.iterations += 1
        logger.info(f"[Worker {self.worker_id}] Iteration {self.iterations}")

        # Generate task
        task_info = self.task_gen.generate_simple_task()
        task = task_info['task']
        func_name = task_info['func_name']

        logger.info(f"  Task: {func_name}")

        # Run autocoder
        try:
            agent = await self.get_autocoder()
            result = await agent.solve_task(task)

            if not result.edits:
                self.failures += 1
                return {"success": False, "error": "No code generated"}

            code = result.edits[0].modified

        except Exception as e:
            self.failures += 1
            logger.error(f"  Autocoder error: {e}")
            return {"success": False, "error": str(e)}

        # Test the code
        test_results = await self._test_code(code, func_name)
        passed = sum(1 for t in test_results if t.get('passed', False))
        total = len(test_results)

        logger.info(f"  Tests: {passed}/{total} passed")

        # Evaluate
        evaluation = await self.evaluator.evaluate(task, code, test_results)

        if evaluation['score'] >= 0.7:
            self.successes += 1
            logger.info(f"  ✅ Score: {evaluation['score']:.0%}")
        else:
            self.failures += 1
            logger.info(f"  ❌ Score: {evaluation['score']:.0%}")

        return {
            "success": evaluation['score'] >= 0.7,
            "task": func_name,
            "score": evaluation['score'],
            "tests_passed": passed,
            "tests_total": total,
            "feedback": evaluation.get('feedback', '')
        }

    async def _test_code(self, code: str, func_name: str) -> List[Dict]:
        """Test generated code"""
        results = []

        # Basic test cases based on function type
        test_cases = self._get_test_cases(func_name)

        exec_globals = {}
        try:
            exec(code, exec_globals)
            func = exec_globals.get(func_name)

            if not func:
                return [{"passed": False, "error": "Function not found"}]

            for test in test_cases:
                try:
                    actual = func(*test['args'])
                    passed = actual == test['expected']
                    results.append({
                        "passed": passed,
                        "input": test['args'],
                        "expected": test['expected'],
                        "actual": actual
                    })
                except Exception as e:
                    results.append({"passed": False, "error": str(e)})

        except Exception as e:
            results.append({"passed": False, "error": f"Execution error: {e}"})

        return results

    def _get_test_cases(self, func_name: str) -> List[Dict]:
        """Get test cases for common functions"""

        test_bank = {
            "find_max": [
                {"args": [[1, 5, 3, 9, 2]], "expected": 9},
                {"args": [[-1, -5, -3]], "expected": -1},
                {"args": [[42]], "expected": 42},
            ],
            "find_min": [
                {"args": [[1, 5, 3, 9, 2]], "expected": 1},
                {"args": [[-1, -5, -3]], "expected": -5},
                {"args": [[42]], "expected": 42},
            ],
            "calculate_average": [
                {"args": [[1, 2, 3, 4, 5]], "expected": 3.0},
                {"args": [[10]], "expected": 10.0},
            ],
            "remove_duplicates": [
                {"args": [[1, 2, 2, 3, 3, 3]], "expected": [1, 2, 3]},
                {"args": [[1, 1, 1]], "expected": [1]},
                {"args": [[]], "expected": []},
            ],
            "reverse_string": [
                {"args": ["hello"], "expected": "olleh"},
                {"args": ["a"], "expected": "a"},
                {"args": [""], "expected": ""},
            ],
            "find_duplicates": [
                {"args": [[1, 2, 3, 2, 4, 3, 5]], "expected": [2, 3]},
                {"args": [[1, 2, 3]], "expected": []},
                {"args": [[]], "expected": []},
            ],
        }

        return test_bank.get(func_name, [
            {"args": [[1, 2, 3]], "expected": None}  # Generic test
        ])

    async def run_continuous(self, max_iterations: int = 100, delay: float = 1.0):
        """Run continuous training"""

        self.running = True
        logger.info(f"[Worker {self.worker_id}] Starting continuous training")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Delay between iterations: {delay}s")

        while self.running and self.iterations < max_iterations:
            try:
                result = await self.run_iteration()

                # Save periodic checkpoints
                if self.iterations % 10 == 0:
                    self._save_checkpoint()

                await asyncio.sleep(delay)

            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Iteration error: {e}")
                await asyncio.sleep(5)  # Wait before retry

        self._save_checkpoint()
        self._print_summary()

    def stop(self):
        """Stop the worker"""
        self.running = False

    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "iterations": self.iterations,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.successes / max(1, self.iterations),
            "deepseek_stats": self.deepseek.get_stats(),
            "evaluation_summary": self.evaluator.get_summary()
        }

        filepath = self.output_dir / f"worker_{self.worker_id}_checkpoint.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"  Checkpoint saved: {filepath}")

    def _print_summary(self):
        """Print training summary"""
        stats = self.deepseek.get_stats()

        print("\n" + "=" * 60)
        print(f"WORKER {self.worker_id} TRAINING SUMMARY")
        print("=" * 60)
        print(f"Iterations: {self.iterations}")
        print(f"Successes: {self.successes} ({self.successes/max(1,self.iterations):.0%})")
        print(f"Failures: {self.failures}")
        print(f"API Cost: {stats['total_cost']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print("=" * 60)


# =============================================================================
# WORKER MANAGER
# =============================================================================

class WorkerManager:
    """Manages multiple training workers"""

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self.workers: List[TrainingWorker] = []
        self.start_time = None

    async def start(self, iterations_per_worker: int = 50):
        """Start all workers"""

        logger.info(f"Starting {self.num_workers} training workers")
        self.start_time = datetime.now()

        # Create workers
        self.workers = [
            TrainingWorker(worker_id=i)
            for i in range(self.num_workers)
        ]

        # Run all workers concurrently
        tasks = [
            worker.run_continuous(max_iterations=iterations_per_worker)
            for worker in self.workers
        ]

        await asyncio.gather(*tasks)

        self._print_final_summary()

    def stop_all(self):
        """Stop all workers"""
        for worker in self.workers:
            worker.stop()

    def _print_final_summary(self):
        """Print final summary across all workers"""

        total_iterations = sum(w.iterations for w in self.workers)
        total_successes = sum(w.successes for w in self.workers)
        total_cost = sum(w.deepseek.total_cost for w in self.workers)

        duration = (datetime.now() - self.start_time).total_seconds()

        print("\n" + "=" * 60)
        print("FINAL TRAINING SUMMARY")
        print("=" * 60)
        print(f"Workers: {self.num_workers}")
        print(f"Total Iterations: {total_iterations}")
        print(f"Total Successes: {total_successes} ({total_successes/max(1,total_iterations):.0%})")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Duration: {duration:.1f}s")
        print(f"Cost per iteration: ${total_cost/max(1,total_iterations):.4f}")
        print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek Continuous Training")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--iterations", type=int, default=20, help="Iterations per worker")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between iterations")
    args = parser.parse_args()

    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set")
        print("Set it with: export DEEPSEEK_API_KEY=your-key")
        return

    print("=" * 60)
    print("DEEPSEEK CONTINUOUS TRAINING")
    print("=" * 60)
    print(f"Workers: {args.workers}")
    print(f"Iterations per worker: {args.iterations}")
    print(f"Estimated cost: ${args.workers * args.iterations * 0.01:.2f}")
    print("=" * 60)

    manager = WorkerManager(num_workers=args.workers)

    try:
        await manager.start(iterations_per_worker=args.iterations)
    except KeyboardInterrupt:
        print("\nStopping workers...")
        manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
