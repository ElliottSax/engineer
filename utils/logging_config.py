#!/usr/bin/env python3
"""
Centralized logging configuration with rotation support.

Usage:
    from utils.logging_config import setup_logging, get_logger

    # Setup logging once at application start
    setup_logging(log_dir="logs", app_name="my_app")

    # Get logger in any module
    logger = get_logger(__name__)
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
from datetime import datetime


# Default configuration
DEFAULT_LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    log_dir: str = "logs",
    app_name: str = "autocoder",
    level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    console_output: bool = True,
    rotate_by_time: bool = False,
    when: str = "midnight",
    interval: int = 1,
) -> logging.Logger:
    """
    Setup centralized logging with rotation.

    Args:
        log_dir: Directory for log files
        app_name: Application name for log files
        level: Logging level (default: INFO)
        max_bytes: Max size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        log_format: Log message format
        date_format: Timestamp format
        console_output: Whether to output to console
        rotate_by_time: If True, rotate by time instead of size
        when: Time rotation interval ('midnight', 'H', 'D', 'W0'-'W6')
        interval: Time rotation interval count

    Returns:
        Root logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Main application log file with rotation
    main_log_file = log_path / f"{app_name}.log"

    if rotate_by_time:
        # Rotate by time (e.g., daily at midnight)
        file_handler = TimedRotatingFileHandler(
            main_log_file,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.suffix = "%Y-%m-%d"
    else:
        # Rotate by size
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Error log file (separate file for errors only)
    error_log_file = log_path / f"{app_name}_error.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Console output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Log startup info
    root_logger.info(f"Logging initialized: {main_log_file}")
    root_logger.info(f"Log rotation: {'time-based' if rotate_by_time else 'size-based'} "
                    f"(max_bytes={max_bytes}, backup_count={backup_count})")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def add_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> RotatingFileHandler:
    """
    Add a rotating file handler to an existing logger.

    Args:
        logger: Logger instance
        log_file: Path to log file
        level: Logging level
        max_bytes: Max size before rotation
        backup_count: Number of backup files

    Returns:
        The created handler
    """
    # Ensure directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT))
    logger.addHandler(handler)

    return handler


class LoggingContext:
    """Context manager for temporary logging configuration."""

    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        handler: Optional[logging.Handler] = None,
    ):
        self.logger = logger
        self.old_level = logger.level
        self.new_level = level
        self.handler = handler

    def __enter__(self):
        if self.new_level is not None:
            self.logger.setLevel(self.new_level)
        if self.handler is not None:
            self.logger.addHandler(self.handler)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.new_level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
        return False


# Convenience function for quick setup
def quick_setup(
    app_name: str = "autocoder",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Quick logging setup with sensible defaults.

    Args:
        app_name: Application name
        level: Logging level
        console: Enable console output

    Returns:
        Root logger
    """
    return setup_logging(
        log_dir="logs",
        app_name=app_name,
        level=level,
        console_output=console,
        max_bytes=10 * 1024 * 1024,  # 10 MB
        backup_count=5,
    )


if __name__ == "__main__":
    # Demo usage
    setup_logging(app_name="demo", level=logging.DEBUG)

    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("\nLog files created in ./logs/")
