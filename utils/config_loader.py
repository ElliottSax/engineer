"""
Configuration Loader

Centralizes configuration management by loading from config.yaml
and providing easy access to configuration values.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# yaml is optional - fall back to JSON if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global configuration cache
_config: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None


def find_config_file() -> Path:
    """Find the config.yaml file by searching upward from current directory."""
    # Check environment variable first
    if os.getenv('CONFIG_PATH'):
        return Path(os.getenv('CONFIG_PATH'))

    # Search for config.yaml
    search_paths = [
        Path(__file__).parent.parent / 'config.yaml',  # /utils/../config.yaml
        Path.cwd() / 'config.yaml',
        Path('/mnt/e/projects/code/config.yaml'),  # Fallback absolute path
    ]

    for path in search_paths:
        if path.exists():
            return path

    raise FileNotFoundError("config.yaml not found in any search path")


def load_config(config_path: Optional[Path] = None, reload: bool = False) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (auto-detected if not provided)
        reload: Force reload even if already cached

    Returns:
        Configuration dictionary
    """
    global _config, _config_path

    if _config is not None and not reload and config_path == _config_path:
        return _config

    if config_path is None:
        config_path = find_config_file()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try YAML first, then JSON
        if YAML_AVAILABLE and (str(config_path).endswith('.yaml') or str(config_path).endswith('.yml')):
            _config = yaml.safe_load(content)
        else:
            # Try JSON as fallback
            try:
                _config = json.loads(content)
            except json.JSONDecodeError:
                if YAML_AVAILABLE:
                    _config = yaml.safe_load(content)
                else:
                    raise

        _config_path = config_path
        logger.info(f"Loaded configuration from {config_path}")
        return _config

    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        _config = get_default_config()
        return _config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        _config = get_default_config()
        return _config


def get_config(
    *keys: str,
    default: Any = None
) -> Any:
    """
    Get a configuration value by key path.

    Args:
        keys: Sequence of keys to traverse (e.g., 'health_monitoring', 'check_interval_seconds')
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        # Get nested value
        interval = get_config('health_monitoring', 'check_interval_seconds')

        # With default
        timeout = get_config('api', 'timeout', default=30)
    """
    config = load_config()

    try:
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def get_default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        'health_monitoring': {
            'check_interval_seconds': 300,
            'history_max_items': 100,
            'alert_thresholds': {
                'response_time_warning_ms': 3000,
                'response_time_critical_ms': 10000,
                'error_rate_warning': 0.05,
                'error_rate_critical': 0.20,
                'uptime_warning': 0.95,
                'uptime_critical': 0.80,
            }
        },
        'autonomous_worker': {
            'num_workers': 4,
            'task_timeout_seconds': 600,
            'results_retention': {
                'max_history_items': 1000,
            }
        },
        'script_analysis': {
            'pacing': {
                'words_per_second_max': 4.0,
                'words_per_second_min': 1.5,
            }
        },
        'security': {
            'code_execution': {
                'enabled': False,
                'timeout_seconds': 30,
            }
        },
        'file_operations': {
            'locking': {
                'enabled': True,
                'timeout_seconds': 30,
            },
            'encoding': 'utf-8',
        },
        'logging': {
            'level': 'INFO',
        }
    }


class ConfigSection:
    """
    Provides attribute-style access to configuration sections.

    Example:
        config = ConfigSection('health_monitoring')
        print(config.check_interval_seconds)
    """

    def __init__(self, *section_keys: str):
        self._section_keys = section_keys
        self._data = get_config(*section_keys, default={})

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)

        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigSection(*self._section_keys, name)
            return value

        raise AttributeError(f"Configuration key not found: {'.'.join(self._section_keys)}.{name}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()


# Convenience functions for common config sections
def get_health_config() -> ConfigSection:
    """Get health monitoring configuration."""
    return ConfigSection('health_monitoring')


def get_worker_config() -> ConfigSection:
    """Get autonomous worker configuration."""
    return ConfigSection('autonomous_worker')


def get_security_config() -> ConfigSection:
    """Get security configuration."""
    return ConfigSection('security')


def is_code_execution_enabled() -> bool:
    """Check if code execution is enabled in config."""
    return get_config('security', 'code_execution', 'enabled', default=False)
