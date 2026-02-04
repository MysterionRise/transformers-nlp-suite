"""
Prometheus metrics instrumentation for NLP API

Provides metrics for:
- Model inference latency
- Cache hits/misses
- Request counts by endpoint
- Active requests gauge
- Error rates
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

# Application info
app_info = Info("nlp_api", "NLP API application information")
app_info.info({"version": "1.0.0", "name": "Enterprise NLP API"})

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

active_requests = Gauge(
    "active_requests",
    "Number of currently active requests",
    ["endpoint"],
)

# Model inference metrics
model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference time in seconds",
    ["model", "task"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

model_inference_total = Counter(
    "model_inference_total",
    "Total number of model inferences",
    ["model", "task", "status"],
)

# Model cache metrics
model_cache_hits_total = Counter(
    "model_cache_hits_total",
    "Total number of model cache hits",
    ["model"],
)

model_cache_misses_total = Counter(
    "model_cache_misses_total",
    "Total number of model cache misses",
    ["model"],
)

model_cache_size = Gauge(
    "model_cache_size",
    "Current number of models in cache",
)

model_cache_evictions_total = Counter(
    "model_cache_evictions_total",
    "Total number of models evicted from cache",
)

# Memory metrics
model_memory_bytes = Gauge(
    "model_memory_bytes",
    "Approximate memory used by loaded models",
    ["model"],
)

# Error metrics
nlp_errors_total = Counter(
    "nlp_errors_total",
    "Total number of NLP processing errors",
    ["error_type", "endpoint"],
)

# Authentication metrics
auth_attempts_total = Counter(
    "auth_attempts_total",
    "Total authentication attempts",
    ["method", "status"],
)

# Rate limiting metrics
rate_limit_hits_total = Counter(
    "rate_limit_hits_total",
    "Total number of rate limit hits",
    ["api_key_prefix"],
)


@contextmanager
def track_inference_time(model: str, task: str):
    """
    Context manager to track model inference time

    Args:
        model: Model identifier
        task: Task type (sentiment, summarization, etc.)

    Example:
        with track_inference_time("bert-base", "sentiment"):
            result = model.predict(text)
    """
    start_time = time.time()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        model_inference_duration_seconds.labels(model=model, task=task).observe(duration)
        model_inference_total.labels(model=model, task=task, status=status).inc()


def track_request(endpoint: str):
    """
    Decorator to track HTTP request metrics

    Args:
        endpoint: Endpoint path for labeling

    Example:
        @track_request("/api/v1/sentiment")
        async def sentiment_endpoint():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            active_requests.labels(endpoint=endpoint).inc()
            start_time = time.time()
            status_code = 200

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                # Try to get status code from HTTPException
                status_code = getattr(e, "status_code", 500)
                raise
            finally:
                duration = time.time() - start_time
                active_requests.labels(endpoint=endpoint).dec()
                http_request_duration_seconds.labels(method="POST", endpoint=endpoint).observe(duration)
                http_requests_total.labels(method="POST", endpoint=endpoint, status_code=status_code).inc()

        return wrapper

    return decorator


def record_cache_hit(model: str):
    """Record a cache hit for a model"""
    model_cache_hits_total.labels(model=model).inc()


def record_cache_miss(model: str):
    """Record a cache miss for a model"""
    model_cache_misses_total.labels(model=model).inc()


def record_cache_eviction():
    """Record a model cache eviction"""
    model_cache_evictions_total.inc()


def update_cache_size(size: int):
    """Update the current cache size gauge"""
    model_cache_size.set(size)


def record_error(error_type: str, endpoint: str):
    """Record an NLP processing error"""
    nlp_errors_total.labels(error_type=error_type, endpoint=endpoint).inc()


def record_auth_attempt(method: str, success: bool):
    """Record an authentication attempt"""
    status = "success" if success else "failure"
    auth_attempts_total.labels(method=method, status=status).inc()


def record_rate_limit_hit(api_key: Optional[str]):
    """Record a rate limit hit"""
    # Use prefix for privacy
    prefix = api_key[:8] if api_key else "anonymous"
    rate_limit_hits_total.labels(api_key_prefix=prefix).inc()


def get_metrics() -> bytes:
    """
    Generate Prometheus metrics output

    Returns:
        Prometheus metrics in exposition format
    """
    return generate_latest()


class MetricsCollector:
    """
    Helper class to collect and expose metrics

    Can be used as a FastAPI dependency or standalone.
    """

    def __init__(self):
        self._start_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self._start_time

    def get_summary(self) -> dict:
        """
        Get a summary of current metrics

        Returns:
            Dictionary with metric summaries
        """
        return {
            "uptime_seconds": self.uptime_seconds,
            "cache_size": model_cache_size._value.get(),
            "active_requests": sum(g._value.get() for g in active_requests._metrics.values()),
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
