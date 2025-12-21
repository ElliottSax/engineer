# Utilities Module

Core utility modules for the Autonomous Worker System providing security, reliability, and configuration management.

## Modules

### `safe_executor.py` - Sandboxed Code Execution

Provides secure code execution with AST validation and restricted builtins.

```python
from utils.safe_executor import SafeCodeExecutor, safe_eval

# Execute code in sandbox
executor = SafeCodeExecutor(timeout=5.0)
result = executor.execute("x = 1 + 2")
print(result.result['x'])  # 3

# Safe expression evaluation
value = safe_eval("x + y * 2", namespace={"x": 10, "y": 5})
print(value)  # 20

# Validate code without executing
from utils.safe_executor import validate_code
violations = validate_code("import os")
print(violations)  # ["Import blocked: os"]
```

**Security Features:**
- Blocks `import`, `exec`, `eval`, `open`
- Blocks dunder attribute access (`__class__`, `__globals__`, etc.)
- Execution timeout
- Memory limits (on Unix)
- Process isolation option

### `file_locking.py` - Thread-Safe File Operations

Provides atomic file operations with cross-process locking.

```python
from utils.file_locking import safe_read_json, safe_write_json, FileLock

# Atomic JSON read/write
data = safe_read_json("/path/to/file.json", default={})
data["key"] = "value"
safe_write_json("/path/to/file.json", data, backup=True)

# File locking
with FileLock("/path/to/file.lock"):
    # Exclusive access to resource
    pass

# State file management
from utils.file_locking import StateFileManager

manager = StateFileManager("/path/to/state.json", default_state={"count": 0})
state = manager.load()
state["count"] += 1
manager.save(state)
```

**Features:**
- Cross-process locking via `fcntl`
- Atomic writes (write-to-temp-then-rename)
- Automatic backup creation
- Backup recovery on corruption

### `config_loader.py` - Centralized Configuration

Loads configuration from `config.yaml` with nested key access.

```python
from utils.config_loader import get_config, load_config

# Load config (cached)
config = load_config()

# Get nested values with defaults
interval = get_config('health_monitoring', 'check_interval_seconds', default=300)
threshold = get_config('health_monitoring', 'alert_thresholds', 'error_rate_warning', default=0.05)

# Attribute-style access
from utils.config_loader import ConfigSection
health = ConfigSection('health_monitoring')
print(health.check_interval_seconds)
```

## Configuration (`config.yaml`)

The main configuration file at `/mnt/e/projects/code/config.yaml` contains:

- `health_monitoring` - Check intervals, thresholds, history limits
- `autonomous_worker` - Worker count, timeouts, retention
- `script_analysis` - Pacing thresholds, cliche detection
- `security` - Code execution settings
- `file_operations` - Locking and encoding settings
- `logging` - Log level and format

## Running Tests

```bash
cd /mnt/e/projects/code
python -m pytest tests/test_utils.py -v
```

## Dependencies

- Python 3.10+
- pyyaml (optional, for YAML config)
- No external dependencies for core functionality
