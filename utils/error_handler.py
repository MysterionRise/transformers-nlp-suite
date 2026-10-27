"""
Error handling utilities for the Transformers NLP Suite

Provides custom exceptions and error handling decorators for graceful degradation.
"""

import functools
import traceback
from typing import Any, Callable, Optional, TypeVar

from .logger import get_logger

logger = get_logger(__name__)

# Type variable for generic function typing
F = TypeVar("F", bound=Callable[..., Any])


# Custom Exception Classes
class NLPError(Exception):
    """Base exception for all NLP-related errors"""

    pass


class ModelLoadError(NLPError):
    """Raised when a model fails to load"""

    def __init__(self, model_name: str, original_error: Optional[Exception] = None):
        self.model_name = model_name
        self.original_error = original_error
        message = f"Failed to load model: {model_name}"
        if original_error:
            message += f"\nOriginal error: {str(original_error)}"
        super().__init__(message)


class ModelNotFoundError(NLPError):
    """Raised when a requested model is not found in the registry"""

    def __init__(self, model_name: str, category: Optional[str] = None):
        self.model_name = model_name
        self.category = category
        message = f"Model not found: {model_name}"
        if category:
            message += f" in category: {category}"
        super().__init__(message)


class InferenceError(NLPError):
    """Raised when model inference fails"""

    def __init__(
        self,
        model_name: str,
        input_text: str,
        original_error: Optional[Exception] = None,
    ):
        self.model_name = model_name
        self.input_text = input_text[:100]  # Limit text length
        self.original_error = original_error
        message = f"Inference failed for model: {model_name}"
        if original_error:
            message += f"\nOriginal error: {str(original_error)}"
        super().__init__(message)


class ConfigurationError(NLPError):
    """Raised when configuration is invalid or missing"""

    pass


class InvalidInputError(NLPError):
    """Raised when user input is invalid"""

    def __init__(self, message: str, input_value: Any = None):
        self.input_value = input_value
        super().__init__(message)


class ResourceError(NLPError):
    """Raised when system resources are insufficient"""

    pass


# Error Handler Decorators
def handle_errors(
    default_return: Any = None,
    raise_error: bool = False,
    log_error: bool = True,
    user_message: Optional[str] = None,
):
    """
    Decorator for handling errors in functions with graceful degradation

    Args:
        default_return: Value to return if error occurs (if raise_error=False)
        raise_error: Whether to re-raise the error after handling
        log_error: Whether to log the error
        user_message: Custom user-friendly error message

    Returns:
        Decorated function

    Example:
        @handle_errors(default_return={}, user_message="Failed to analyze text")
        def analyze_text(text):
            # ... implementation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={
                            "function": func.__name__,
                            "function_args": str(args)[:100],
                            "function_kwargs": str(kwargs)[:100],
                        },
                    )

                if raise_error:
                    raise

                # Return user-friendly error message
                if user_message:
                    return {"error": user_message, "details": str(e)}
                else:
                    return default_return

        return wrapper

    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to retry a function on error with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exception types to catch

    Returns:
        Decorated function

    Example:
        @retry_on_error(max_retries=3, delay=1.0)
        def load_model(model_name):
            # ... implementation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise last_exception

        return wrapper

    return decorator


def validate_input(
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    allow_empty: bool = False,
    strip: bool = True,
):
    """
    Decorator to validate text input

    Args:
        max_length: Maximum allowed text length
        min_length: Minimum required text length
        allow_empty: Whether to allow empty text
        strip: Whether to strip whitespace

    Returns:
        Decorated function

    Raises:
        InvalidInputError: If validation fails

    Example:
        @validate_input(max_length=512, min_length=10)
        def process_text(text):
            # ... implementation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(text: str, *args, **kwargs):
            if strip:
                text = text.strip()

            if not allow_empty and not text:
                raise InvalidInputError("Text cannot be empty")

            if min_length and len(text) < min_length:
                raise InvalidInputError(f"Text must be at least {min_length} characters (got {len(text)})")

            if max_length and len(text) > max_length:
                raise InvalidInputError(f"Text must be at most {max_length} characters (got {len(text)})")

            return func(text, *args, **kwargs)

        return wrapper

    return decorator


class ErrorContext:
    """
    Context manager for error handling with logging

    Example:
        with ErrorContext("Loading model", model_name="bert-base"):
            model = load_model("bert-base")
    """

    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.logger = get_logger(__name__)

    def __enter__(self):
        self.logger.debug(f"Starting: {self.operation}", extra=self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.debug(f"Completed: {self.operation}", extra=self.context)
        else:
            self.logger.error(
                f"Failed: {self.operation} - {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
                extra=self.context,
            )
        # Don't suppress the exception
        return False


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Format an error message for display to users

    Args:
        error: Exception to format
        include_traceback: Whether to include full traceback

    Returns:
        Formatted error message
    """
    if isinstance(error, NLPError):
        # Custom errors already have good messages
        message = str(error)
    elif isinstance(error, ValueError):
        message = f"Invalid value: {str(error)}"
    elif isinstance(error, FileNotFoundError):
        message = f"File not found: {str(error)}"
    elif isinstance(error, MemoryError):
        message = "Out of memory. Try using a smaller model or reducing batch size."
    elif isinstance(error, KeyboardInterrupt):
        message = "Operation cancelled by user"
    else:
        message = f"An error occurred: {type(error).__name__}: {str(error)}"

    if include_traceback:
        tb = traceback.format_exc()
        message += f"\n\nTraceback:\n{tb}"

    return message


def get_error_suggestion(error: Exception) -> Optional[str]:
    """
    Get a helpful suggestion based on the error type

    Args:
        error: Exception that occurred

    Returns:
        Suggestion string or None
    """
    suggestions = {
        MemoryError: "Try using a smaller model or reducing the batch size. "
        "Close other applications to free up memory.",
        FileNotFoundError: "Check that all required files are present and paths are correct.",
        ModelLoadError: (
            "Ensure you have internet connection for downloading models. " "Check if the model name is correct."
        ),
        InvalidInputError: "Review the input requirements and try again with valid input.",
        ConnectionError: ("Check your internet connection. " "Some models need to be downloaded from HuggingFace."),
    }

    for error_type, suggestion in suggestions.items():
        if isinstance(error, error_type):
            return suggestion

    return None
