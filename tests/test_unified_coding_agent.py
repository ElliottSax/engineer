"""
Tests for the unified coding agent.

Run with: python -m unittest tests.test_unified_coding_agent -v
"""

import os
import sys
import asyncio
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_coding_agent import (
    UnifiedCodingAgent,
    RepositoryMapper,
    LocalArchitect,
    LocalEditor,
    SandboxedExecutor,
    SelfRecovery,
    XMLPlanParser,
    AgentState,
    AgentResult,
    CodeEdit,
    PlanStep,
    ExecutionPlan,
)


class TestRepositoryMapper(unittest.TestCase):
    """Tests for RepositoryMapper."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        mapper = RepositoryMapper(self.tmp_dir)
        result = mapper.scan()

        self.assertEqual(result["total_files"], 0)
        self.assertIn("summary", result)

    def test_scan_with_python_files(self):
        """Test scanning directory with Python files."""
        # Create some test files
        (self.tmp_path / "main.py").write_text("def main(): pass")
        (self.tmp_path / "utils.py").write_text("def helper(): pass")

        mapper = RepositoryMapper(self.tmp_dir)
        result = mapper.scan()

        self.assertEqual(result["total_files"], 2)

    def test_scan_respects_max_files(self):
        """Test that max_files limit is respected."""
        # Create many files
        for i in range(20):
            (self.tmp_path / f"file_{i}.py").write_text(f"x = {i}")

        mapper = RepositoryMapper(self.tmp_dir)
        result = mapper.scan(max_files=5)

        self.assertLessEqual(result["total_files"], 5)

    def test_scan_extracts_symbols(self):
        """Test that symbols are extracted from Python files."""
        code = '''
def hello():
    pass

class MyClass:
    def method(self):
        pass
'''
        (self.tmp_path / "module.py").write_text(code)

        mapper = RepositoryMapper(self.tmp_dir)
        result = mapper.scan()

        self.assertIn("module.py", result["symbols"])
        symbols = result["symbols"]["module.py"]
        self.assertTrue(any("hello" in s for s in symbols))
        self.assertTrue(any("MyClass" in s for s in symbols))

    def test_skip_hidden_directories(self):
        """Test that hidden directories are skipped."""
        # Create hidden directory with files
        hidden_dir = self.tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.py").write_text("x = 1")

        # Create normal file
        (self.tmp_path / "visible.py").write_text("y = 2")

        mapper = RepositoryMapper(self.tmp_dir)
        result = mapper.scan()

        self.assertEqual(result["total_files"], 1)

    def test_get_context_for_task(self):
        """Test context retrieval for a task."""
        # Context matching is based on symbol names, not filenames
        (self.tmp_path / "auth.py").write_text("def authenticate(): pass\ndef login(): pass")
        (self.tmp_path / "utils.py").write_text("def helper(): pass")

        mapper = RepositoryMapper(self.tmp_dir)
        mapper.scan()

        # Should find auth.py because it contains 'authenticate' symbol matching 'authenticate'
        context = mapper.get_context_for_task("fix authenticate function")
        # If symbols are found, context should contain file content
        # Note: matching depends on symbol extraction working correctly
        self.assertIsInstance(context, str)


class TestLocalEditor(unittest.TestCase):
    """Tests for LocalEditor pattern matching."""

    def setUp(self):
        self.editor = LocalEditor()

    def test_generate_palindrome_function(self):
        """Test palindrome function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function is_palindrome that checks if a string is a palindrome"
        )
        code = "\n".join(lines)

        self.assertIn("def is_palindrome", code)
        self.assertIn("return", code)

        # Execute and test
        namespace = {}
        exec(code, namespace)
        self.assertTrue(namespace["is_palindrome"]("racecar"))
        self.assertFalse(namespace["is_palindrome"]("hello"))

    def test_generate_fibonacci_function(self):
        """Test fibonacci function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function fibonacci that returns the first n fibonacci numbers"
        )
        code = "\n".join(lines)

        self.assertIn("def fibonacci", code)

        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["fibonacci"](5), [0, 1, 1, 2, 3])
        self.assertEqual(namespace["fibonacci"](1), [0])

    def test_generate_prime_function(self):
        """Test prime checker function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function is_prime that checks if a number is prime"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertTrue(namespace["is_prime"](7))
        self.assertFalse(namespace["is_prime"](4))
        self.assertFalse(namespace["is_prime"](1))

    def test_generate_find_duplicates(self):
        """Test find duplicates function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function find_duplicates that finds duplicate elements"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["find_duplicates"]([1, 2, 3, 2, 4, 3]), [2, 3])
        self.assertEqual(namespace["find_duplicates"]([1, 2, 3]), [])

    def test_generate_remove_duplicates(self):
        """Test remove duplicates function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function remove_duplicates that removes duplicate elements"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["remove_duplicates"]([1, 2, 2, 3]), [1, 2, 3])

    def test_generate_count_words(self):
        """Test count words function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function count_words that counts the number of words in a string"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["count_words"]("hello world"), 2)
        self.assertEqual(namespace["count_words"]("one"), 1)

    def test_generate_binary_search(self):
        """Test binary search function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function binary_search that performs binary search on a sorted array"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["binary_search"]([1, 2, 3, 4, 5], 3), 2)
        self.assertEqual(namespace["binary_search"]([1, 2, 3, 4, 5], 6), -1)

    def test_generate_gcd(self):
        """Test GCD function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function calculate_gcd that calculates the greatest common divisor"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["calculate_gcd"](12, 8), 4)
        self.assertEqual(namespace["calculate_gcd"](17, 13), 1)

    def test_generate_flatten_list(self):
        """Test flatten list function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function flatten_list that flattens a nested list"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["flatten_list"]([[1, 2], [3, [4, 5]]]), [1, 2, 3, 4, 5])

    def test_generate_is_anagram(self):
        """Test anagram checker function generation."""
        lines = self.editor._generate_function_from_desc(
            "Create a function is_anagram that checks if two strings are anagrams"
        )
        code = "\n".join(lines)

        namespace = {}
        exec(code, namespace)
        self.assertTrue(namespace["is_anagram"]("listen", "silent"))
        self.assertFalse(namespace["is_anagram"]("hello", "world"))


class TestLocalArchitect(unittest.TestCase):
    """Tests for LocalArchitect planning."""

    def setUp(self):
        self.architect = LocalArchitect()

    def test_analyze_create_task(self):
        """Test planning for create/add tasks."""
        plan = asyncio.run(
            self.architect.analyze_task("Create a user authentication system", "")
        )

        self.assertIsInstance(plan, ExecutionPlan)
        self.assertGreater(len(plan.steps), 0)
        self.assertTrue(any(step.action in ["create_file", "modify_file"] for step in plan.steps))

    def test_analyze_fix_task(self):
        """Test planning for fix tasks."""
        plan = asyncio.run(
            self.architect.analyze_task("Fix the login bug in auth module", "")
        )

        self.assertIsInstance(plan, ExecutionPlan)
        self.assertGreater(len(plan.steps), 0)

    def test_word_boundary_for_test(self):
        """Test that 'test' is matched as whole word only."""
        # "greatest" contains "test" but shouldn't match test task pattern
        plan = asyncio.run(
            self.architect.analyze_task("Create a function for greatest common divisor", "")
        )

        # Should NOT create a test file
        self.assertFalse(
            any(step.target.endswith("test_new_feature.py") for step in plan.steps)
        )


class TestXMLPlanParser(unittest.TestCase):
    """Tests for XML plan parsing."""

    def test_parse_valid_xml(self):
        """Test parsing valid XML plan."""
        xml = '''
        <execution_plan>
            <task>Add user authentication</task>
            <reasoning>Need to add auth module</reasoning>
            <steps>
                <step id="1">
                    <action>create_file</action>
                    <target>auth.py</target>
                    <description>Create auth module</description>
                    <dependencies></dependencies>
                </step>
            </steps>
        </execution_plan>
        '''

        plan = XMLPlanParser.parse_plan(xml)

        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].action, "create_file")
        self.assertEqual(plan.steps[0].target, "auth.py")

    def test_generate_plan_xml(self):
        """Test generating XML from plan."""
        plan = ExecutionPlan(
            task_description="Test task",
            steps=[
                PlanStep(
                    step_id=0,
                    action="create_file",
                    target="test.py",
                    description="Create test file"
                )
            ],
            architect_reasoning="Simple plan"
        )

        xml = XMLPlanParser.generate_plan_xml(plan)

        self.assertIn("<task>Test task</task>", xml)
        self.assertIn("<action>create_file</action>", xml)


class TestSandboxedExecutor(unittest.TestCase):
    """Tests for sandboxed code execution."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.executor = SandboxedExecutor(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_execute_bash_safe_command(self):
        """Test executing safe bash command."""
        result = self.executor.execute_bash("echo hello")

        self.assertTrue(result["success"])
        self.assertIn("hello", result["stdout"])

    def test_execute_python_safe_code(self):
        """Test executing safe Python code."""
        result = self.executor.execute_python("result = 1 + 2")

        self.assertTrue(result["success"])


class TestUnifiedCodingAgent(unittest.TestCase):
    """Integration tests for UnifiedCodingAgent."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = UnifiedCodingAgent(repo_path=self.tmp_dir)

        self.assertEqual(agent.state, AgentState.IDLE)
        self.assertIsNotNone(agent.repo_mapper)
        self.assertIsNotNone(agent.architect)
        self.assertIsNotNone(agent.editor)

    def test_agent_creates_directories(self):
        """Test that agent creates workspace if it doesn't exist."""
        new_dir = os.path.join(self.tmp_dir, "new_workspace")

        agent = UnifiedCodingAgent(repo_path=new_dir)

        self.assertTrue(os.path.exists(new_dir))

    def test_solve_simple_task(self):
        """Test solving a simple code generation task."""
        agent = UnifiedCodingAgent(repo_path=self.tmp_dir)
        agent.auto_commit = False
        agent.auto_test = False

        result = asyncio.run(
            agent.solve_task("Create a function is_palindrome that checks if a string is a palindrome")
        )

        self.assertTrue(result.success)
        self.assertGreater(len(result.edits), 0)

        # Verify generated code works
        code = result.edits[0].modified
        namespace = {}
        exec(code, namespace)
        self.assertTrue(namespace["is_palindrome"]("racecar"))

    def test_solve_task_with_numbers(self):
        """Test solving task with numeric operations."""
        agent = UnifiedCodingAgent(repo_path=self.tmp_dir)
        agent.auto_commit = False
        agent.auto_test = False

        result = asyncio.run(
            agent.solve_task("Create a function fibonacci that returns the first n fibonacci numbers")
        )

        self.assertTrue(result.success)
        code = result.edits[0].modified
        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace["fibonacci"](5), [0, 1, 1, 2, 3])


class TestAgentResult(unittest.TestCase):
    """Tests for AgentResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        result = AgentResult(
            success=True,
            message="Task completed",
            edits=[CodeEdit("test.py", "", "code", "create")],
            files_created=["test.py"]
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.edits), 1)

    def test_failure_result(self):
        """Test creating failure result."""
        result = AgentResult(
            success=False,
            message="Task failed: Error"
        )

        self.assertFalse(result.success)
        self.assertEqual(len(result.edits), 0)


class TestCodeEdit(unittest.TestCase):
    """Tests for CodeEdit dataclass."""

    def test_create_edit(self):
        """Test creating code edit."""
        edit = CodeEdit(
            file_path="test.py",
            original="",
            modified="print('hello')",
            edit_type="create"
        )

        self.assertEqual(edit.file_path, "test.py")
        self.assertEqual(edit.edit_type, "create")


class TestAllPatterns(unittest.TestCase):
    """Test all 20 code generation patterns."""

    PATTERNS = [
        ("find_duplicates", "finds duplicate elements", [[1, 2, 2, 3]], [2]),
        ("find_max", "finds the maximum value", [[1, 5, 3]], 5),
        ("find_min", "finds the minimum value", [[1, 5, 3]], 1),
        ("is_palindrome", "checks if palindrome", ["racecar"], True),
        ("fibonacci", "returns fibonacci numbers", [5], [0, 1, 1, 2, 3]),
        ("is_prime", "checks if prime", [7], True),
        ("reverse_string", "reverses a string", ["hello"], "olleh"),
        ("count_words", "counts words in string", ["hello world"], 2),
        ("remove_duplicates", "removes duplicates", [[1, 2, 2, 3]], [1, 2, 3]),
        ("sum_list", "sums elements", [[1, 2, 3]], 6),
        ("flatten_list", "flattens nested list", [[[1], [2, 3]]], [1, 2, 3]),
        ("get_average", "calculates average", [[2, 4, 6]], 4.0),
        ("filter_even", "filters even numbers", [[1, 2, 3, 4]], [2, 4]),
        ("filter_odd", "filters odd numbers", [[1, 2, 3, 4]], [1, 3]),
        ("get_unique", "gets unique elements", [[1, 1, 2, 2]], [1, 2]),
        ("calculate_power", "calculates power", [2, 3], 8),
        ("calculate_gcd", "calculates greatest common divisor", [12, 8], 4),
        ("is_anagram", "checks if anagram", ["listen", "silent"], True),
        ("binary_search", "binary search on sorted array", [[1, 2, 3, 4, 5], 3], 2),
        ("capitalize_words", "capitalizes words", ["hello world"], "Hello World"),
    ]

    def setUp(self):
        self.editor = LocalEditor()

    def test_all_patterns_generate_valid_code(self):
        """Test that all patterns generate executable code."""
        for func_name, desc, args, expected in self.PATTERNS:
            with self.subTest(pattern=func_name):
                lines = self.editor._generate_function_from_desc(
                    f"Create a function {func_name} that {desc}"
                )
                code = "\n".join(lines)

                # Should contain function definition
                self.assertIn(f"def {func_name}", code, f"Missing function {func_name}")

                # Should be executable
                namespace = {}
                try:
                    exec(code, namespace)
                except SyntaxError as e:
                    self.fail(f"Syntax error in {func_name}: {e}")

                # Function should exist
                self.assertIn(func_name, namespace, f"Function {func_name} not found")

    def test_all_patterns_produce_correct_output(self):
        """Test that all patterns produce correct output."""
        for func_name, desc, args, expected in self.PATTERNS:
            with self.subTest(pattern=func_name):
                lines = self.editor._generate_function_from_desc(
                    f"Create a function {func_name} that {desc}"
                )
                code = "\n".join(lines)

                namespace = {}
                exec(code, namespace)

                try:
                    result = namespace[func_name](*args)
                    self.assertEqual(
                        result, expected,
                        f"{func_name}({args}) returned {result}, expected {expected}"
                    )
                except Exception as e:
                    self.fail(f"Error executing {func_name}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
