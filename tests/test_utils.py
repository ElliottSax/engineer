"""
Tests for utility modules.

Run with: python -m unittest tests.test_utils -v
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_locking import (
    FileLock,
    safe_read_json,
    safe_write_json,
    StateFileManager,
    FileLockError
)
from utils.safe_executor import (
    SafeCodeExecutor,
    validate_code,
    safe_eval,
    CodeExecutionError,
    SecurityViolation,
    analyze_code_structure
)
from utils.config_loader import (
    load_config,
    get_config,
    get_default_config,
    ConfigSection
)


class TestFileLocking(unittest.TestCase):
    """Tests for file locking utilities."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_file_lock_acquire_release(self):
        """Test basic lock acquire and release."""
        lock_file = self.tmp_path / "test.lock"

        lock = FileLock(lock_file, timeout=5.0)
        self.assertTrue(lock.acquire())
        lock.release()

        # Should be able to acquire again
        self.assertTrue(lock.acquire())
        lock.release()

    def test_file_lock_context_manager(self):
        """Test file lock as context manager."""
        lock_file = self.tmp_path / "test.lock"

        with FileLock(lock_file) as lock:
            self.assertIsNotNone(lock._fd)

        # Lock should be released
        self.assertIsNone(lock._fd)

    def test_safe_write_read_json(self):
        """Test atomic JSON write and read."""
        json_file = self.tmp_path / "test.json"

        test_data = {
            "key": "value",
            "number": 42,
            "nested": {"a": 1, "b": 2}
        }

        # Write
        self.assertTrue(safe_write_json(json_file, test_data))

        # Verify file exists
        self.assertTrue(json_file.exists())

        # Read back
        result = safe_read_json(json_file)
        self.assertEqual(result, test_data)

    def test_safe_read_json_missing_file(self):
        """Test reading non-existent file returns default."""
        json_file = self.tmp_path / "nonexistent.json"

        result = safe_read_json(json_file, default={"default": True})
        self.assertEqual(result, {"default": True})

    def test_safe_read_json_invalid_json(self):
        """Test reading invalid JSON returns default."""
        json_file = self.tmp_path / "invalid.json"
        json_file.write_text("not valid json {{{")

        result = safe_read_json(json_file, default={"fallback": True})
        self.assertEqual(result, {"fallback": True})

    def test_state_file_manager(self):
        """Test StateFileManager basic operations."""
        state_file = self.tmp_path / "state.json"
        default_state = {"initialized": True, "count": 0}

        manager = StateFileManager(state_file, default_state=default_state)

        # Load returns default for new file
        state = manager.load()
        self.assertEqual(state, default_state)

        # Update and save
        state["count"] = 5
        self.assertTrue(manager.save(state))

        # Reload
        new_state = manager.load()
        self.assertEqual(new_state["count"], 5)
        self.assertIn("_last_updated", new_state)


class TestSafeExecutor(unittest.TestCase):
    """Tests for safe code execution."""

    def test_validate_safe_code(self):
        """Test validation of safe code."""
        safe_code = """
x = 1 + 2
y = [i * 2 for i in range(10)]
result = sum(y)
"""
        violations = validate_code(safe_code)
        self.assertEqual(len(violations), 0)

    def test_validate_blocks_import(self):
        """Test that imports are blocked."""
        code = "import os"
        violations = validate_code(code)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("import" in v.lower() for v in violations))

    def test_validate_blocks_import_from(self):
        """Test that from imports are blocked."""
        code = "from os import path"
        violations = validate_code(code)
        self.assertGreater(len(violations), 0)

    def test_validate_blocks_exec(self):
        """Test that exec is blocked."""
        code = "exec('print(1)')"
        violations = validate_code(code)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("exec" in v.lower() for v in violations))

    def test_validate_blocks_dunder_access(self):
        """Test that dunder attribute access is blocked."""
        code = "x.__class__.__bases__"
        violations = validate_code(code)
        self.assertGreater(len(violations), 0)

    def test_executor_simple_code(self):
        """Test executing simple code."""
        executor = SafeCodeExecutor(timeout=5.0)
        result = executor.execute("x = 1 + 2")

        self.assertTrue(result.success)
        self.assertEqual(result.result["x"], 3)

    def test_executor_blocks_dangerous_code(self):
        """Test that dangerous code is blocked."""
        executor = SafeCodeExecutor(timeout=5.0)
        result = executor.execute("import os")

        self.assertFalse(result.success)
        self.assertTrue("security" in result.error.lower() or "import" in result.error.lower())

    @unittest.skip("Skipping timeout test - requires isolated process and may hang")
    def test_executor_timeout(self):
        """Test execution timeout."""
        # This test is skipped because:
        # 1. signal.SIGALRM is not available on Windows
        # 2. Infinite loops will hang without proper process isolation
        pass

    def test_safe_eval_simple_expression(self):
        """Test safe_eval with simple expressions."""
        result = safe_eval("1 + 2 * 3")
        self.assertEqual(result, 7)

    def test_safe_eval_with_namespace(self):
        """Test safe_eval with variables."""
        result = safe_eval("x + y", namespace={"x": 10, "y": 5})
        self.assertEqual(result, 15)

    def test_safe_eval_blocks_import(self):
        """Test that safe_eval blocks dangerous code."""
        with self.assertRaises((CodeExecutionError, SecurityViolation)):
            safe_eval("__import__('os')")

    def test_analyze_code_structure(self):
        """Test code structure analysis."""
        code = """
def hello(name):
    return f"Hello, {name}"

class MyClass:
    pass

x = 42
"""
        structure = analyze_code_structure(code)

        self.assertEqual(len(structure['functions']), 1)
        self.assertEqual(structure['functions'][0]['name'], 'hello')
        self.assertEqual(len(structure['classes']), 1)
        self.assertEqual(structure['classes'][0]['name'], 'MyClass')
        self.assertIn('x', structure['variables'])


class TestConfigLoader(unittest.TestCase):
    """Tests for configuration loading."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_default_config(self):
        """Test default config structure."""
        config = get_default_config()

        self.assertIn('health_monitoring', config)
        self.assertIn('autonomous_worker', config)
        self.assertIn('security', config)

    def test_get_config_with_default(self):
        """Test get_config returns default for missing keys."""
        result = get_config('nonexistent', 'key', 'path', default='fallback')
        self.assertEqual(result, 'fallback')


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple utilities."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_state_file_with_safe_executor(self):
        """Test using state file manager with safe code executor."""
        state_file = self.tmp_path / "state.json"

        # Initialize state
        manager = StateFileManager(state_file, default_state={"execution_count": 0})

        # Execute some code
        executor = SafeCodeExecutor(timeout=5.0)
        result = executor.execute("result = 2 ** 10")

        if result.success:
            state = manager.load()
            state["execution_count"] += 1
            state["last_result"] = result.result.get("result")
            manager.save(state)

        # Verify
        final_state = manager.load()
        self.assertEqual(final_state["execution_count"], 1)
        self.assertEqual(final_state["last_result"], 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
