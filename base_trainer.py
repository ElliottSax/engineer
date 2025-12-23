#!/usr/bin/env python3
"""
Base Trainer Module - Common abstractions for all training implementations.

This module provides base classes that reduce code duplication across
claude_trainer.py, deepseek_trainer.py, multi_provider_trainer.py, etc.

Usage:
    from base_trainer import BaseLLMClient, BaseTrainer, BaseEvaluator

    class MyClient(BaseLLMClient):
        async def complete(self, prompt, system="", max_tokens=1000):
            # Implementation
            ...
"""

import os
import json
import asyncio
import logging
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from logging.handlers import RotatingFileHandler

# Configure logger with rotation
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrainingTask:
    """A training task with optional test cases."""
    id: str
    task: str
    func_name: str = ""
    difficulty: str = "beginner"
    category: str = "code_generation"
    test_cases: List[Dict] = field(default_factory=list)
    expected_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationResult:
    """Result of evaluating generated code."""
    task_id: str
    code: str
    score: float  # 0.0 to 1.0
    tests_passed: int = 0
    tests_total: int = 0
    feedback: str = ""
    improvements: List[str] = field(default_factory=list)
    grade: str = ""  # A, B, C, D, F
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingStats:
    """Statistics for a training session."""
    iterations: int = 0
    successes: int = 0
    failures: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    scores: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.iterations)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / max(1, len(self.scores))

    def to_dict(self) -> Dict:
        return {
            "iterations": self.iterations,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_score": f"{self.avg_score:.2f}",
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}"
        }


# =============================================================================
# BASE LLM CLIENT
# =============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    name: str = "base"
    cost_per_1k_tokens: float = 0.0

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1000
    ) -> Optional[str]:
        """Send a completion request."""
        pass

    def is_available(self) -> bool:
        """Check if the client is available (e.g., API key set)."""
        return True

    def track_usage(self, tokens: int):
        """Track token usage and cost."""
        self.total_tokens += tokens
        self.total_cost += (tokens / 1000) * self.cost_per_1k_tokens
        self.request_count += 1

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "provider": self.name,
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
            "request_count": self.request_count,
            "avg_tokens_per_request": self.total_tokens // max(1, self.request_count)
        }


# =============================================================================
# BASE TASK GENERATOR
# =============================================================================

class BaseTaskGenerator:
    """Base class for generating training tasks."""

    DEFAULT_TASKS = [
        ("find_max", "finds the maximum value in a list"),
        ("find_min", "finds the minimum value in a list"),
        ("is_palindrome", "checks if a string is a palindrome"),
        ("fibonacci", "returns the first n fibonacci numbers"),
        ("is_prime", "checks if a number is prime"),
        ("reverse_string", "reverses a string"),
        ("count_words", "counts the number of words in a string"),
        ("remove_duplicates", "removes duplicate elements from a list"),
        ("sum_list", "calculates the sum of all elements in a list"),
        ("find_duplicates", "finds duplicate elements in a list"),
        ("flatten_list", "flattens a nested list into a single list"),
        ("get_average", "calculates the average of numbers in a list"),
        ("filter_even", "filters and returns only even numbers from a list"),
        ("binary_search", "performs binary search on a sorted array"),
        ("calculate_gcd", "calculates the greatest common divisor of two numbers"),
    ]

    DEFAULT_TEST_CASES = {
        "find_max": [
            {"args": [[1, 5, 3, 9, 2]], "expected": 9},
            {"args": [[42]], "expected": 42},
        ],
        "find_min": [
            {"args": [[1, 5, 3, 9, 2]], "expected": 1},
            {"args": [[42]], "expected": 42},
        ],
        "is_palindrome": [
            {"args": ["racecar"], "expected": True},
            {"args": ["hello"], "expected": False},
        ],
        "fibonacci": [
            {"args": [5], "expected": [0, 1, 1, 2, 3]},
            {"args": [1], "expected": [0]},
        ],
        "is_prime": [
            {"args": [7], "expected": True},
            {"args": [4], "expected": False},
        ],
        "reverse_string": [
            {"args": ["hello"], "expected": "olleh"},
            {"args": [""], "expected": ""},
        ],
        "count_words": [
            {"args": ["hello world"], "expected": 2},
            {"args": ["one"], "expected": 1},
        ],
        "remove_duplicates": [
            {"args": [[1, 2, 2, 3]], "expected": [1, 2, 3]},
            {"args": [[]], "expected": []},
        ],
        "sum_list": [
            {"args": [[1, 2, 3, 4, 5]], "expected": 15},
            {"args": [[]], "expected": 0},
        ],
        "find_duplicates": [
            {"args": [[1, 2, 3, 2, 4, 3]], "expected": [2, 3]},
            {"args": [[1, 2, 3]], "expected": []},
        ],
        "flatten_list": [
            {"args": [[[1, 2], [3, [4, 5]]]], "expected": [1, 2, 3, 4, 5]},
        ],
        "get_average": [
            {"args": [[1, 2, 3, 4, 5]], "expected": 3.0},
        ],
        "filter_even": [
            {"args": [[1, 2, 3, 4, 5, 6]], "expected": [2, 4, 6]},
        ],
        "binary_search": [
            {"args": [[1, 2, 3, 4, 5], 3], "expected": 2},
            {"args": [[1, 2, 3, 4, 5], 6], "expected": -1},
        ],
        "calculate_gcd": [
            {"args": [12, 8], "expected": 4},
            {"args": [17, 13], "expected": 1},
        ],
    }

    def __init__(self):
        self.generated_tasks: List[TrainingTask] = []

    def generate_task(self) -> TrainingTask:
        """Generate a simple task from templates."""
        import random

        func_name, description = random.choice(self.DEFAULT_TASKS)
        task_text = f"Create a Python function called '{func_name}' that {description}"

        task_id = hashlib.md5(
            f"{func_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        task = TrainingTask(
            id=task_id,
            task=task_text,
            func_name=func_name,
            test_cases=self.DEFAULT_TEST_CASES.get(func_name, [])
        )

        self.generated_tasks.append(task)
        return task

    def get_test_cases(self, func_name: str) -> List[Dict]:
        """Get test cases for a function."""
        return self.DEFAULT_TEST_CASES.get(func_name, [])


# =============================================================================
# BASE EVALUATOR
# =============================================================================

class BaseEvaluator:
    """Base class for evaluating generated code."""

    def __init__(self):
        self.evaluations: List[EvaluationResult] = []

    def run_tests(
        self,
        code: str,
        func_name: str,
        test_cases: List[Dict]
    ) -> Tuple[int, int, List[Dict]]:
        """
        Run test cases against generated code.

        Returns:
            Tuple of (passed_count, total_count, detailed_results)
        """
        if not test_cases:
            return 0, 0, []

        passed = 0
        results = []
        exec_globals = {}

        try:
            exec(code, exec_globals)
            func = exec_globals.get(func_name)

            if not func:
                return 0, len(test_cases), [{"passed": False, "error": "Function not found"}]

            for test in test_cases:
                try:
                    actual = func(*test['args'])
                    is_passed = actual == test['expected']
                    if is_passed:
                        passed += 1
                    results.append({
                        "passed": is_passed,
                        "input": test['args'],
                        "expected": test['expected'],
                        "actual": actual
                    })
                except (TypeError, ValueError, IndexError, KeyError, RuntimeError) as e:
                    results.append({"passed": False, "error": str(e)})

        except (SyntaxError, NameError) as e:
            return 0, len(test_cases), [{"passed": False, "error": f"Execution error: {e}"}]

        return passed, len(test_cases), results

    def evaluate(
        self,
        task: TrainingTask,
        code: str,
        test_results: Optional[List[Dict]] = None
    ) -> EvaluationResult:
        """Evaluate generated code."""

        # Run tests if not provided
        if test_results is None:
            passed, total, test_results = self.run_tests(
                code, task.func_name, task.test_cases
            )
        else:
            passed = sum(1 for t in test_results if t.get('passed', False))
            total = len(test_results)

        # Calculate score
        score = passed / max(1, total)

        # Determine grade
        if score >= 0.9:
            grade = "A"
        elif score >= 0.7:
            grade = "B"
        elif score >= 0.5:
            grade = "C"
        elif score >= 0.3:
            grade = "D"
        else:
            grade = "F"

        result = EvaluationResult(
            task_id=task.id,
            code=code,
            score=score,
            tests_passed=passed,
            tests_total=total,
            feedback=f"{passed}/{total} tests passed",
            grade=grade
        )

        self.evaluations.append(result)
        return result

    def get_summary(self) -> Dict:
        """Get evaluation summary."""
        if not self.evaluations:
            return {"count": 0}

        scores = [e.score for e in self.evaluations]
        return {
            "count": len(self.evaluations),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "passing_rate": sum(1 for s in scores if s >= 0.7) / len(scores),
            "grade_distribution": self._get_grade_distribution()
        }

    def _get_grade_distribution(self) -> Dict[str, int]:
        """Get distribution of grades."""
        from collections import Counter
        grades = [e.grade for e in self.evaluations]
        return dict(Counter(grades))


# =============================================================================
# BASE TRAINER
# =============================================================================

class BaseTrainer(ABC):
    """Abstract base class for training workers."""

    def __init__(
        self,
        worker_id: int = 0,
        output_dir: str = "training_output"
    ):
        self.worker_id = worker_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.stats = TrainingStats()
        self.running = False
        self.results: List[Dict] = []

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging with rotation."""
        log_file = self.output_dir / f"worker_{self.worker_id}.log"

        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)

    @abstractmethod
    async def run_iteration(self) -> Dict:
        """Run a single training iteration."""
        pass

    @abstractmethod
    async def get_agent(self) -> Any:
        """Get the agent/autocoder instance."""
        pass

    async def run_continuous(
        self,
        max_iterations: int = 50,
        delay: float = 1.0,
        checkpoint_interval: int = 10
    ):
        """Run continuous training."""
        self.running = True
        logger.info(f"[Worker {self.worker_id}] Starting - {max_iterations} iterations")

        while self.running and self.stats.iterations < max_iterations:
            try:
                result = await self.run_iteration()
                self.results.append(result)

                # Update stats
                self.stats.iterations += 1
                if result.get('success', False):
                    self.stats.successes += 1
                else:
                    self.stats.failures += 1

                if 'score' in result:
                    self.stats.scores.append(result['score'])

                # Periodic checkpoint
                if self.stats.iterations % checkpoint_interval == 0:
                    self.save_checkpoint()

                await asyncio.sleep(delay)

            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"[Worker {self.worker_id}] Error: {e}")
                await asyncio.sleep(2)

        self.save_checkpoint()
        self.print_summary()

    def stop(self):
        """Stop the trainer."""
        self.running = False

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats.to_dict(),
            "recent_results": self.results[-20:]
        }

        filepath = self.output_dir / f"worker_{self.worker_id}_checkpoint.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved: {filepath}")

    def print_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print(f"WORKER {self.worker_id} TRAINING SUMMARY")
        print(f"{'='*60}")
        for key, value in self.stats.to_dict().items():
            print(f"  {key}: {value}")
        print(f"{'='*60}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from an LLM response."""
    try:
        # Try to find JSON object
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            return json.loads(response[json_start:json_end])

        # Try to find JSON array
        json_start = response.find('[')
        json_end = response.rfind(']') + 1

        if json_start >= 0 and json_end > json_start:
            return json.loads(response[json_start:json_end])

    except json.JSONDecodeError:
        pass

    return None


def create_training_task(
    func_name: str,
    description: str,
    test_cases: Optional[List[Dict]] = None
) -> TrainingTask:
    """Create a training task."""
    task_id = hashlib.md5(
        f"{func_name}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:8]

    return TrainingTask(
        id=task_id,
        task=f"Create a Python function called '{func_name}' that {description}",
        func_name=func_name,
        test_cases=test_cases or []
    )


# =============================================================================
# EXAMPLE IMPLEMENTATION
# =============================================================================

class ExampleTrainer(BaseTrainer):
    """Example trainer implementation for reference."""

    def __init__(self, worker_id: int = 0):
        super().__init__(worker_id)
        self.task_generator = BaseTaskGenerator()
        self.evaluator = BaseEvaluator()

    async def get_agent(self):
        """Get autocoder agent."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        try:
            from unified_coding_agent import UnifiedCodingAgent
            agent = UnifiedCodingAgent(repo_path='/tmp/training_workspace')
            agent.auto_commit = False
            agent.auto_test = False
            return agent
        except ImportError:
            return None

    async def run_iteration(self) -> Dict:
        """Run a single training iteration."""
        self.stats.iterations += 1

        # Generate task
        task = self.task_generator.generate_task()
        logger.info(f"[W{self.worker_id}] Task: {task.func_name}")

        # Get agent and solve
        agent = await self.get_agent()
        if not agent:
            return {"success": False, "error": "Agent not available"}

        try:
            result = await agent.solve_task(task.task)

            if not result.edits:
                return {"success": False, "task": task.func_name, "error": "No code generated"}

            code = result.edits[0].modified

        except Exception as e:
            return {"success": False, "task": task.func_name, "error": str(e)}

        # Evaluate
        evaluation = self.evaluator.evaluate(task, code)

        success = evaluation.score >= 0.7

        return {
            "success": success,
            "task": task.func_name,
            "score": evaluation.score,
            "tests_passed": evaluation.tests_passed,
            "tests_total": evaluation.tests_total,
            "grade": evaluation.grade
        }


if __name__ == "__main__":
    # Demo the base classes
    print("Base Trainer Module")
    print("=" * 40)

    # Demo task generator
    gen = BaseTaskGenerator()
    task = gen.generate_task()
    print(f"Generated task: {task.task}")
    print(f"Test cases: {len(task.test_cases)}")

    # Demo evaluator with sample code
    evaluator = BaseEvaluator()
    sample_code = """
def find_max(lst):
    if not lst:
        return None
    return max(lst)
"""

    task = create_training_task(
        "find_max",
        "finds the maximum value in a list",
        BaseTaskGenerator.DEFAULT_TEST_CASES["find_max"]
    )

    result = evaluator.evaluate(task, sample_code)
    print(f"\nEvaluation result:")
    print(f"  Score: {result.score:.1%}")
    print(f"  Grade: {result.grade}")
    print(f"  Tests: {result.tests_passed}/{result.tests_total}")
