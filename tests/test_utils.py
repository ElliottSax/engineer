"""
Tests for utility modules.

Run with: pytest tests/test_utils.py -v
"""

import os
import sys
import json
import pytest
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


class TestFileLocking:
    """Tests for file locking utilities."""

    def test_file_lock_acquire_release(self, tmp_path):
        """Test basic lock acquire and release."""
        lock_file = tmp_path / "test.lock"

        lock = FileLock(lock_file, timeout=5.0)
        assert lock.acquire()
        lock.release()

        # Should be able to acquire again
        assert lock.acquire()
        lock.release()

    def test_file_lock_context_manager(self, tmp_path):
        """Test file lock as context manager."""
        lock_file = tmp_path / "test.lock"

        with FileLock(lock_file) as lock:
            assert lock._fd is not None

        # Lock should be released
        assert lock._fd is None

    def test_safe_write_read_json(self, tmp_path):
        """Test atomic JSON write and read."""
        json_file = tmp_path / "test.json"

        test_data = {
            "key": "value",
            "number": 42,
            "nested": {"a": 1, "b": 2}
        }

        # Write
        assert safe_write_json(json_file, test_data)

        # Verify file exists
        assert json_file.exists()

        # Read back
        result = safe_read_json(json_file)
        assert result == test_data

    def test_safe_read_json_missing_file(self, tmp_path):
        """Test reading non-existent file returns default."""
        json_file = tmp_path / "nonexistent.json"

        result = safe_read_json(json_file, default={"default": True})
        assert result == {"default": True}

    def test_safe_read_json_invalid_json(self, tmp_path):
        """Test reading invalid JSON returns default."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json {{{")

        result = safe_read_json(json_file, default={"fallback": True})
        assert result == {"fallback": True}

    def test_state_file_manager(self, tmp_path):
        """Test StateFileManager basic operations."""
        state_file = tmp_path / "state.json"
        default_state = {"initialized": True, "count": 0}

        manager = StateFileManager(state_file, default_state=default_state)

        # Load returns default for new file
        state = manager.load()
        assert state == default_state

        # Update and save
        state["count"] = 5
        assert manager.save(state)

        # Reload
        new_state = manager.load()
        assert new_state["count"] == 5
        assert "_last_updated" in new_state


class TestSafeExecutor:
    """Tests for safe code execution."""

    def test_validate_safe_code(self):
        """Test validation of safe code."""
        safe_code = """
x = 1 + 2
y = [i * 2 for i in range(10)]
result = sum(y)
"""
        violations = validate_code(safe_code)
        assert len(violations) == 0

    def test_validate_blocks_import(self):
        """Test that imports are blocked."""
        code = "import os"
        violations = validate_code(code)
        assert len(violations) > 0
        assert any("import" in v.lower() for v in violations)

    def test_validate_blocks_import_from(self):
        """Test that from imports are blocked."""
        code = "from os import path"
        violations = validate_code(code)
        assert len(violations) > 0

    def test_validate_blocks_exec(self):
        """Test that exec is blocked."""
        code = "exec('print(1)')"
        violations = validate_code(code)
        assert len(violations) > 0
        assert any("exec" in v.lower() for v in violations)

    def test_validate_blocks_dunder_access(self):
        """Test that dunder attribute access is blocked."""
        code = "x.__class__.__bases__"
        violations = validate_code(code)
        assert len(violations) > 0

    def test_executor_simple_code(self):
        """Test executing simple code."""
        executor = SafeCodeExecutor(timeout=5.0)
        result = executor.execute("x = 1 + 2")

        assert result.success
        assert result.result["x"] == 3

    def test_executor_blocks_dangerous_code(self):
        """Test that dangerous code is blocked."""
        executor = SafeCodeExecutor(timeout=5.0)
        result = executor.execute("import os")

        assert not result.success
        assert "security" in result.error.lower() or "import" in result.error.lower()

    def test_executor_timeout(self):
        """Test execution timeout."""
        executor = SafeCodeExecutor(timeout=0.5)

        # Infinite loop (will timeout)
        code = """
i = 0
while True:
    i += 1
"""
        result = executor.execute(code)
        assert not result.success

    def test_safe_eval_simple_expression(self):
        """Test safe_eval with simple expressions."""
        result = safe_eval("1 + 2 * 3")
        assert result == 7

    def test_safe_eval_with_namespace(self):
        """Test safe_eval with variables."""
        result = safe_eval("x + y", namespace={"x": 10, "y": 5})
        assert result == 15

    def test_safe_eval_blocks_import(self):
        """Test that safe_eval blocks dangerous code."""
        with pytest.raises((CodeExecutionError, SecurityViolation)):
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

        assert len(structure['functions']) == 1
        assert structure['functions'][0]['name'] == 'hello'
        assert len(structure['classes']) == 1
        assert structure['classes'][0]['name'] == 'MyClass'
        assert 'x' in structure['variables']


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_get_default_config(self):
        """Test default config structure."""
        config = get_default_config()

        assert 'health_monitoring' in config
        assert 'autonomous_worker' in config
        assert 'security' in config

    def test_get_config_nested_key(self, tmp_path):
        """Test getting nested config values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
health_monitoring:
  check_interval_seconds: 600
  alert_thresholds:
    response_time_warning_ms: 5000
""")

        with patch('utils.config_loader.find_config_file', return_value=config_file):
            from utils.config_loader import load_config, get_config
            load_config(config_file, reload=True)

            interval = get_config('health_monitoring', 'check_interval_seconds')
            assert interval == 600

            warning = get_config(
                'health_monitoring', 'alert_thresholds', 'response_time_warning_ms'
            )
            assert warning == 5000

    def test_get_config_with_default(self):
        """Test get_config returns default for missing keys."""
        result = get_config('nonexistent', 'key', 'path', default='fallback')
        assert result == 'fallback'

    def test_config_section(self, tmp_path):
        """Test ConfigSection attribute access."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
test_section:
  value1: 100
  value2: "hello"
  nested:
    deep: true
""")

        with patch('utils.config_loader.find_config_file', return_value=config_file):
            from utils.config_loader import load_config
            load_config(config_file, reload=True)

            section = ConfigSection('test_section')
            assert section.get('value1') == 100
            assert section.get('value2') == "hello"


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_state_file_with_safe_executor(self, tmp_path):
        """Test using state file manager with safe code executor."""
        state_file = tmp_path / "state.json"

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
        assert final_state["execution_count"] == 1
        assert final_state["last_result"] == 1024


# Pytest fixtures
@pytest.fixture
def tmp_config_file(tmp_path):
    """Create a temporary config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
health_monitoring:
  check_interval_seconds: 300
  history_max_items: 50

security:
  code_execution:
    enabled: false
    timeout_seconds: 10
""")
    return config_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
