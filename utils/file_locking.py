"""
File Locking Utilities

Provides thread-safe and process-safe file operations with proper locking
to prevent race conditions and data corruption.
"""

import os
import json
import time
import fcntl
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class FileLockError(Exception):
    """Raised when file locking fails."""
    pass


class FileLock:
    """
    A file-based lock for cross-process synchronization.

    Usage:
        with FileLock('/path/to/file.lock'):
            # Do something with exclusive access
            pass
    """

    def __init__(
        self,
        lock_file: Union[str, Path],
        timeout: float = 30.0,
        retry_delay: float = 0.1
    ):
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.retry_delay = retry_delay
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        """Acquire the file lock with timeout."""
        start_time = time.time()

        # Ensure parent directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # Open lock file (create if doesn't exist)
                self._fd = os.open(
                    str(self.lock_file),
                    os.O_CREAT | os.O_RDWR
                )

                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Write PID to lock file for debugging
                os.write(self._fd, f"{os.getpid()}\n".encode())
                os.fsync(self._fd)

                return True

            except (OSError, IOError) as e:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    raise FileLockError(
                        f"Could not acquire lock on {self.lock_file} "
                        f"within {self.timeout} seconds"
                    )

                # Wait and retry
                time.sleep(self.retry_delay)

    def release(self) -> None:
        """Release the file lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except (OSError, IOError) as e:
                logger.warning(f"Error releasing lock: {e}")
            finally:
                self._fd = None

    def __enter__(self) -> 'FileLock':
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


@contextmanager
def atomic_write_json(
    file_path: Union[str, Path],
    data: Any,
    indent: int = 2,
    backup: bool = True
):
    """
    Atomically write JSON data to a file with locking.

    Uses a write-to-temp-then-rename strategy to ensure the file
    is never left in a corrupted state.

    Args:
        file_path: Path to the JSON file
        data: Data to serialize to JSON
        indent: JSON indentation level
        backup: Whether to create a backup before writing

    Yields:
        None - use as context manager

    Example:
        with atomic_write_json('/path/to/file.json', data):
            pass  # File is written after context exits
    """
    file_path = Path(file_path)
    lock_path = file_path.with_suffix(file_path.suffix + '.lock')
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')

    with FileLock(lock_path):
        try:
            # Create backup if file exists
            if backup and file_path.exists():
                shutil.copy2(file_path, backup_path)

            # Write to temporary file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str)
                f.flush()
                os.fsync(f.fileno())

            # Atomically replace the original file
            temp_path.replace(file_path)

            yield

        except Exception as e:
            # Attempt to restore from backup on failure
            if backup_path.exists():
                try:
                    shutil.copy2(backup_path, file_path)
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")

            # Clean up temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

            raise

        finally:
            # Clean up temp file if it still exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass


def safe_read_json(
    file_path: Union[str, Path],
    default: Any = None,
    encoding: str = 'utf-8'
) -> Any:
    """
    Safely read JSON from a file with locking.

    Args:
        file_path: Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid
        encoding: File encoding

    Returns:
        Parsed JSON data or default value
    """
    file_path = Path(file_path)
    lock_path = file_path.with_suffix(file_path.suffix + '.lock')

    if not file_path.exists():
        return default

    with FileLock(lock_path, timeout=10.0):
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")

            # Try to recover from backup
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            if backup_path.exists():
                try:
                    with open(backup_path, 'r', encoding=encoding) as f:
                        data = json.load(f)
                    logger.info(f"Recovered from backup: {backup_path}")
                    return data
                except Exception:
                    pass

            return default
        except FileNotFoundError:
            return default
        except IOError as e:
            logger.error(f"IO error reading {file_path}: {e}")
            return default


def safe_write_json(
    file_path: Union[str, Path],
    data: Any,
    indent: int = 2,
    backup: bool = True,
    encoding: str = 'utf-8'
) -> bool:
    """
    Safely write JSON to a file with locking.

    Args:
        file_path: Path to the JSON file
        data: Data to serialize to JSON
        indent: JSON indentation level
        backup: Whether to create a backup
        encoding: File encoding

    Returns:
        True if successful, False otherwise
    """
    try:
        with atomic_write_json(file_path, data, indent=indent, backup=backup):
            pass
        return True
    except Exception as e:
        logger.error(f"Failed to write {file_path}: {e}")
        return False


class StateFileManager:
    """
    Manages state files with automatic locking, backup, and recovery.

    Example:
        state_manager = StateFileManager('/path/to/state.json')

        # Read state
        state = state_manager.load()

        # Update state
        state['key'] = 'value'
        state_manager.save(state)
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        default_state: Optional[Dict] = None,
        max_backups: int = 5,
        auto_backup: bool = True
    ):
        self.file_path = Path(file_path)
        self.default_state = default_state or {}
        self.max_backups = max_backups
        self.auto_backup = auto_backup
        self._lock = FileLock(self.file_path.with_suffix('.lock'))

    def load(self) -> Dict:
        """Load state from file, returning default if not found."""
        return safe_read_json(self.file_path, default=self.default_state.copy())

    def save(self, state: Dict) -> bool:
        """Save state to file with backup."""
        # Rotate old backups if enabled
        if self.auto_backup:
            self._rotate_backups()

        # Add metadata
        state['_last_updated'] = datetime.now().isoformat()

        return safe_write_json(
            self.file_path,
            state,
            backup=self.auto_backup
        )

    def _rotate_backups(self) -> None:
        """Rotate backup files, keeping only max_backups."""
        backup_dir = self.file_path.parent / '.backups'
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{self.file_path.stem}_{timestamp}{self.file_path.suffix}"

        if self.file_path.exists():
            shutil.copy2(self.file_path, backup_dir / backup_name)

        # Remove old backups
        backups = sorted(backup_dir.glob(f"{self.file_path.stem}_*"))
        while len(backups) > self.max_backups:
            oldest = backups.pop(0)
            oldest.unlink()

    def update(self, updates: Dict) -> bool:
        """Atomically update state with new values."""
        state = self.load()
        state.update(updates)
        return self.save(state)
