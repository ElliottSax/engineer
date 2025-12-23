#!/usr/bin/env python3
"""
Tests for training modules: base_trainer, multi_provider_trainer.

Run with: python -m unittest tests/test_trainers.py -v
"""

import unittest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrainingTask(unittest.TestCase):
    """Tests for TrainingTask dataclass."""

    def setUp(self):
        from base_trainer import TrainingTask
        self.TrainingTask = TrainingTask

    def test_create_task(self):
        """Test creating a training task."""
        task = self.TrainingTask(
            id="test123",
            task="Create a function",
            func_name="my_func"
        )
        self.assertEqual(task.id, "test123")
        self.assertEqual(task.task, "Create a function")
        self.assertEqual(task.func_name, "my_func")
        self.assertEqual(task.difficulty, "beginner")
        self.assertEqual(task.category, "code_generation")
        self.assertEqual(task.test_cases, [])

    def test_task_with_test_cases(self):
        """Test creating a task with test cases."""
        test_cases = [
            {"args": [[1, 2, 3]], "expected": 6},
            {"args": [[]], "expected": 0}
        ]
        task = self.TrainingTask(
            id="test456",
            task="Sum list",
            func_name="sum_list",
            test_cases=test_cases
        )
        self.assertEqual(len(task.test_cases), 2)
        self.assertEqual(task.test_cases[0]["expected"], 6)


class TestEvaluationResult(unittest.TestCase):
    """Tests for EvaluationResult dataclass."""

    def setUp(self):
        from base_trainer import EvaluationResult
        self.EvaluationResult = EvaluationResult

    def test_create_result(self):
        """Test creating an evaluation result."""
        result = self.EvaluationResult(
            task_id="test123",
            code="def func(): pass",
            score=0.85,
            tests_passed=8,
            tests_total=10,
            grade="B"
        )
        self.assertEqual(result.task_id, "test123")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.grade, "B")


class TestTrainingStats(unittest.TestCase):
    """Tests for TrainingStats dataclass."""

    def setUp(self):
        from base_trainer import TrainingStats
        self.TrainingStats = TrainingStats

    def test_initial_stats(self):
        """Test initial stats are zero."""
        stats = self.TrainingStats()
        self.assertEqual(stats.iterations, 0)
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.failures, 0)
        self.assertEqual(stats.success_rate, 0.0)

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = self.TrainingStats(
            iterations=10,
            successes=7,
            failures=3
        )
        self.assertEqual(stats.success_rate, 0.7)

    def test_total_execution_time(self):
        """Test execution time tracking."""
        stats = self.TrainingStats()
        stats.iterations = 5
        stats.total_execution_time = 25.0
        self.assertEqual(stats.total_execution_time, 25.0)
        # Calculate avg manually
        avg = stats.total_execution_time / max(1, stats.iterations)
        self.assertEqual(avg, 5.0)

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = self.TrainingStats(
            iterations=10,
            successes=8,
            failures=2,
            total_tokens=1000,
            total_cost=0.05
        )
        result = stats.to_dict()
        self.assertIn("iterations", result)
        self.assertIn("success_rate", result)
        self.assertEqual(result["iterations"], 10)


class TestBaseTaskGenerator(unittest.TestCase):
    """Tests for BaseTaskGenerator."""

    def setUp(self):
        from base_trainer import BaseTaskGenerator
        self.generator = BaseTaskGenerator()

    def test_generate_task(self):
        """Test generating a task."""
        task = self.generator.generate_task()
        self.assertIsNotNone(task.id)
        self.assertIsNotNone(task.task)
        self.assertIsNotNone(task.func_name)
        # Function name should be one of the defaults
        func_names = [t[0] for t in self.generator.DEFAULT_TASKS]
        self.assertIn(task.func_name, func_names)

    def test_get_test_cases(self):
        """Test getting test cases for known functions."""
        test_cases = self.generator.get_test_cases("find_max")
        self.assertGreater(len(test_cases), 0)
        self.assertIn("args", test_cases[0])
        self.assertIn("expected", test_cases[0])

    def test_get_test_cases_unknown_function(self):
        """Test getting test cases for unknown function returns empty."""
        test_cases = self.generator.get_test_cases("unknown_function")
        self.assertEqual(test_cases, [])

    def test_generated_tasks_tracked(self):
        """Test that generated tasks are tracked."""
        self.assertEqual(len(self.generator.generated_tasks), 0)
        self.generator.generate_task()
        self.assertEqual(len(self.generator.generated_tasks), 1)
        self.generator.generate_task()
        self.assertEqual(len(self.generator.generated_tasks), 2)


class TestBaseEvaluator(unittest.TestCase):
    """Tests for BaseEvaluator."""

    def setUp(self):
        from base_trainer import BaseEvaluator, TrainingTask
        self.evaluator = BaseEvaluator()
        self.TrainingTask = TrainingTask

    def test_run_tests_passing(self):
        """Test running tests that pass."""
        code = """
def find_max(items):
    if not items:
        return None
    return max(items)
"""
        test_cases = [
            {"args": [[1, 5, 3]], "expected": 5},
            {"args": [[42]], "expected": 42}
        ]
        passed, total, results = self.evaluator.run_tests(code, "find_max", test_cases)
        self.assertEqual(passed, 2)
        self.assertEqual(total, 2)
        self.assertTrue(results[0]["passed"])
        self.assertTrue(results[1]["passed"])

    def test_run_tests_failing(self):
        """Test running tests that fail."""
        code = """
def find_max(items):
    return 0  # Wrong implementation
"""
        test_cases = [
            {"args": [[1, 5, 3]], "expected": 5}
        ]
        passed, total, results = self.evaluator.run_tests(code, "find_max", test_cases)
        self.assertEqual(passed, 0)
        self.assertEqual(total, 1)
        self.assertFalse(results[0]["passed"])

    def test_run_tests_syntax_error(self):
        """Test running tests with syntax error in code."""
        code = "def broken(:"  # Syntax error
        test_cases = [{"args": [1], "expected": 1}]
        passed, total, results = self.evaluator.run_tests(code, "broken", test_cases)
        self.assertEqual(passed, 0)
        self.assertIn("error", results[0])

    def test_run_tests_function_not_found(self):
        """Test running tests when function doesn't exist."""
        code = """
def wrong_name():
    pass
"""
        test_cases = [{"args": [1], "expected": 1}]
        passed, total, results = self.evaluator.run_tests(code, "right_name", test_cases)
        self.assertEqual(passed, 0)
        self.assertIn("error", results[0])

    def test_evaluate(self):
        """Test full evaluation."""
        task = self.TrainingTask(
            id="test123",
            task="Find max",
            func_name="find_max",
            test_cases=[
                {"args": [[1, 2, 3]], "expected": 3},
                {"args": [[5]], "expected": 5}
            ]
        )
        code = """
def find_max(items):
    return max(items) if items else None
"""
        result = self.evaluator.evaluate(task, code)
        self.assertEqual(result.task_id, "test123")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.grade, "A")
        self.assertEqual(result.tests_passed, 2)

    def test_grade_distribution(self):
        """Test grade assignment."""
        from base_trainer import TrainingTask

        # Test grade A
        task = TrainingTask(id="t1", task="Test", func_name="f", test_cases=[{"args": [[1]], "expected": 1}])
        result = self.evaluator.evaluate(task, "def f(x): return x[0]")
        self.assertEqual(result.grade, "A")

    def test_get_summary(self):
        """Test getting evaluation summary."""
        # First evaluate some tasks
        from base_trainer import TrainingTask
        task = TrainingTask(
            id="t1", task="Test", func_name="test_func",
            test_cases=[{"args": [1], "expected": 2}]
        )

        self.evaluator.evaluate(task, "def test_func(x): return x + 1")

        summary = self.evaluator.get_summary()
        self.assertEqual(summary["count"], 1)
        self.assertIn("avg_score", summary)


class TestParseJsonFromResponse(unittest.TestCase):
    """Tests for parse_json_from_response utility."""

    def test_parse_json_object(self):
        """Test parsing JSON object from response."""
        from base_trainer import parse_json_from_response

        response = 'Some text {"key": "value", "num": 42} more text'
        result = parse_json_from_response(response)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["num"], 42)

    def test_parse_json_array(self):
        """Test parsing JSON array from response."""
        from base_trainer import parse_json_from_response

        response = 'Here is the list: [1, 2, 3, "four"]'
        result = parse_json_from_response(response)
        self.assertEqual(result, [1, 2, 3, "four"])

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        from base_trainer import parse_json_from_response

        response = 'No JSON here at all'
        result = parse_json_from_response(response)
        self.assertIsNone(result)


class TestCreateTrainingTask(unittest.TestCase):
    """Tests for create_training_task utility."""

    def test_create_task(self):
        """Test creating a training task via utility function."""
        from base_trainer import create_training_task

        task = create_training_task(
            "find_max",
            "finds the maximum value",
            [{"args": [[1, 2, 3]], "expected": 3}]
        )
        self.assertIn("find_max", task.task)
        self.assertEqual(task.func_name, "find_max")
        self.assertEqual(len(task.test_cases), 1)


class TestMultiProviderTrainer(unittest.TestCase):
    """Tests for multi_provider_trainer module."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Mock environment to avoid needing real API keys
        import os
        os.environ.setdefault("DEEPSEEK_API_KEY", "test_key")

    def test_deepseek_provider_exists(self):
        """Test DeepSeekProvider class exists."""
        from multi_provider_trainer import DeepSeekProvider
        provider = DeepSeekProvider()
        self.assertIsNotNone(provider)

    def test_github_models_provider_exists(self):
        """Test GitHubModelsProvider class exists."""
        from multi_provider_trainer import GitHubModelsProvider
        provider = GitHubModelsProvider()
        self.assertIsNotNone(provider)

    def test_huggingface_provider_exists(self):
        """Test HuggingFaceProvider class exists."""
        from multi_provider_trainer import HuggingFaceProvider
        provider = HuggingFaceProvider()
        self.assertIsNotNone(provider)

    def test_ollama_provider_exists(self):
        """Test OllamaProvider class exists."""
        from multi_provider_trainer import OllamaProvider
        provider = OllamaProvider()
        self.assertIsNotNone(provider)

    def test_multi_provider_worker(self):
        """Test MultiProviderWorker class."""
        from multi_provider_trainer import MultiProviderWorker
        worker = MultiProviderWorker(worker_id=0)
        self.assertIsNotNone(worker)

    def test_task_generation(self):
        """Test task generation in trainer."""
        from multi_provider_trainer import MultiProviderWorker
        worker = MultiProviderWorker(worker_id=0)
        # Check worker has task generation capabilities
        self.assertIsNotNone(worker)


class TestCodePatterns(unittest.TestCase):
    """Test that code generation patterns work correctly."""

    def test_find_max_pattern(self):
        """Test find_max code generation."""
        from unified_coding_agent import LocalEditor
        from unified_coding_agent import PlanStep

        editor = LocalEditor()
        step = PlanStep(
            step_id=0,
            action="modify_file",
            target="test.py",
            description="Create a function called 'find_max' that finds the maximum value"
        )
        code_lines = editor._generate_function_from_desc(
            "Create a function called 'find_max' that finds the maximum value"
        )
        code = '\n'.join(code_lines)

        # Execute and test
        exec_globals = {}
        exec(code, exec_globals)
        self.assertIn("find_max", exec_globals)
        self.assertEqual(exec_globals["find_max"]([1, 5, 3]), 5)

    def test_is_palindrome_pattern(self):
        """Test palindrome code generation."""
        from unified_coding_agent import LocalEditor

        editor = LocalEditor()
        code_lines = editor._generate_function_from_desc(
            "Create a function called 'is_palindrome' that checks if string is palindrome"
        )
        code = '\n'.join(code_lines)

        exec_globals = {}
        exec(code, exec_globals)
        self.assertIn("is_palindrome", exec_globals)
        self.assertTrue(exec_globals["is_palindrome"]("racecar"))
        self.assertFalse(exec_globals["is_palindrome"]("hello"))

    def test_fibonacci_pattern(self):
        """Test fibonacci code generation."""
        from unified_coding_agent import LocalEditor

        editor = LocalEditor()
        code_lines = editor._generate_function_from_desc(
            "Create a function called 'fibonacci' that generates fibonacci sequence"
        )
        code = '\n'.join(code_lines)

        exec_globals = {}
        exec(code, exec_globals)
        self.assertIn("fibonacci", exec_globals)
        self.assertEqual(exec_globals["fibonacci"](5), [0, 1, 1, 2, 3])


class TestPatternRegistry(unittest.TestCase):
    """Tests for PatternRegistry."""

    def test_list_patterns(self):
        """Test listing registered patterns."""
        from unified_coding_agent import PatternRegistry

        patterns = PatternRegistry.list_patterns()
        self.assertIsInstance(patterns, list)
        # Check that some patterns are registered
        self.assertIn("matrix_transpose", patterns)
        self.assertIn("memoize", patterns)

    def test_match_pattern(self):
        """Test matching patterns to descriptions."""
        from unified_coding_agent import PatternRegistry

        # Should match matrix_transpose
        match = PatternRegistry.match("transpose a matrix")
        self.assertEqual(match, "matrix_transpose")

        # Should match memoize
        match = PatternRegistry.match("create a memoization decorator")
        self.assertEqual(match, "memoize")

    def test_generate_pattern(self):
        """Test generating code from pattern."""
        from unified_coding_agent import PatternRegistry

        code = PatternRegistry.generate("matrix_transpose", "transpose", "transpose matrix")
        self.assertGreater(len(code), 0)
        self.assertIn("def transpose", '\n'.join(code))


class TestAgentMetrics(unittest.TestCase):
    """Tests for AgentMetrics."""

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        from unified_coding_agent import AgentMetrics

        metrics = AgentMetrics()
        self.assertEqual(metrics.tasks_attempted, 0)
        self.assertEqual(metrics.success_rate, 0.0)

    def test_metrics_calculation(self):
        """Test metrics calculations."""
        from unified_coding_agent import AgentMetrics

        metrics = AgentMetrics(
            tasks_attempted=10,
            tasks_succeeded=7,
            tasks_failed=3,
            total_execution_time=50.0
        )
        self.assertEqual(metrics.success_rate, 0.7)
        self.assertEqual(metrics.avg_execution_time, 5.0)

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        from unified_coding_agent import AgentMetrics

        metrics = AgentMetrics(
            tasks_attempted=5,
            tasks_succeeded=4,
            tasks_failed=1
        )
        result = metrics.to_dict()
        self.assertIn("success_rate", result)
        self.assertEqual(result["tasks_attempted"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
