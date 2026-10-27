"""
Logging utilities for the Transformers NLP Suite

Provides structured logging with rotation, formatting, and performance tracking.
"""

import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color support for terminal output
    """

    def __init__(self, *args, use_colors: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors

    def format(self, record):
        if self.use_colors and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
                record.name = f"\033[34m{record.name}{COLORS['RESET']}"  # Blue

        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter for production log ingestion."""

    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in [
            "request_id",
            "method",
            "path",
            "query",
            "client_ip",
            "status_code",
            "processing_time_ms",
            "user",
            "auth_method",
            "error",
            "model",
            "task",
        ]:
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class PerformanceLogger:
    """
    Context manager for logging performance metrics

    Example:
        with PerformanceLogger("model_inference", logger=logger):
            result = model.predict(text)
    """

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
        self.level = level
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.operation} (took {duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} after {duration:.2f}s - {exc_val}")

        return False  # Don't suppress exceptions


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
    use_colors: bool = True,
    json_logs: bool = False,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        format_string: Custom format string
        use_colors: Use colored output for console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = JsonFormatter() if json_logs else ColoredFormatter(format_string, use_colors=use_colors)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (if log_file specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = JsonFormatter() if json_logs else logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, level: Optional[str] = None, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Get or create a logger

    Args:
        name: Logger name (usually __name__)
        level: Optional logging level override
        log_file: Optional log file path

    Returns:
        Logger instance
    """
    # Try to get settings, but don't fail if not available
    try:
        from config.settings import get_settings

        settings = get_settings()
        if level is None:
            level = settings.log_level
        if log_file is None and settings.logs_dir:
            log_file = settings.logs_dir / f"{name.replace('.', '_')}.log"
        json_logs = settings.json_logs
    except Exception:
        # Fallback if settings not available
        if level is None:
            level = "INFO"
        json_logs = False

    return setup_logger(name=name, level=level, log_file=log_file, json_logs=json_logs)


def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments and return values

    Args:
        logger: Logger to use (defaults to module logger)
        level: Logging level

    Example:
        @log_function_call()
        def process_text(text, model_name):
            # ... implementation
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            # Log function call
            args_str = ", ".join([repr(arg)[:100] for arg in args])
            kwargs_str = ", ".join([f"{k}={repr(v)[:100]}" for k, v in kwargs.items()])
            params = ", ".join(filter(None, [args_str, kwargs_str]))

            logger.log(level, f"Calling {func.__name__}({params})")

            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log(level, f"{func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise

        return wrapper

    return decorator


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds context to all log messages

    Example:
        adapter = LoggerAdapter(logger, {"model": "bert-base", "task": "sentiment"})
        adapter.info("Processing text")
        # Output: Processing text [model=bert-base, task=sentiment]
    """

    def process(self, msg, kwargs):
        # Add context to message
        if self.extra:
            context = ", ".join([f"{k}={v}" for k, v in self.extra.items()])
            msg = f"{msg} [{context}]"
        return msg, kwargs


def configure_root_logger(level: str = "INFO", log_dir: Optional[Path] = None, json_logs: bool = False):
    """
    Configure the root logger for the entire application

    Args:
        level: Logging level
        log_dir: Directory for log files
    """
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "nlp_app.log"
    else:
        log_file = None

    setup_logger(
        name="",  # Root logger
        level=level,
        log_file=log_file,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        use_colors=True,
        json_logs=json_logs,
    )


# Convenience function for quick setup
def init_logging(debug: bool = False):
    """
    Initialize logging for the application

    Args:
        debug: Enable debug mode
    """
    level = "DEBUG" if debug else "INFO"

    try:
        from config.settings import get_settings

        settings = get_settings()
        configure_root_logger(level=settings.log_level, log_dir=settings.logs_dir, json_logs=settings.json_logs)
    except Exception:
        # Fallback if settings not available
        configure_root_logger(level=level)
