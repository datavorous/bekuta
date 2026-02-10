# LLM Generated Code : True

import functools
import os
import statistics
import time


def profile(func):
    """Basic timing decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        print(f"[PROFILE] {func.__name__} took {elapsed_ms:.3f}ms")
        return result
    return wrapper


def timed(func):
    """Decorator that returns (result, elapsed_seconds)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, (end - start)
    return wrapper


def memory_usage_mb():
    """Return current memory usage in MB."""
    try:
        import resource
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return usage_kb / 1024
        return usage_kb / 1024
    except Exception:
        return 0.0


def percentile(values, pct):
    """Calculate percentile from list of values."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((pct / 100) * (len(values_sorted) - 1)))
    return values_sorted[idx]


def recall_at_k(exact_results, approx_results, k):
    """Calculate recall@k between exact and approximate results."""
    exact_ids = {id_val for id_val, _ in exact_results[:k]}
    approx_ids = {id_val for id_val, _ in approx_results[:k]}
    if not exact_ids:
        return 0.0
    return len(exact_ids & approx_ids) / len(exact_ids)


def benchmark_queries(index, queries, k, exact_map=None):
    """
    Run queries against index and return metrics.
    
    Returns: (mean_ms, p99_ms, qps, recall_pct)
    """
    latencies = []
    recalls = []
    for q in queries:
        start = time.perf_counter()
        approx = index.search(q, k)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        if exact_map is not None:
            recalls.append(recall_at_k(exact_map[id(q)], approx, k))
    
    mean_ms = statistics.mean(latencies)
    p99_ms = percentile(latencies, 99)
    qps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    recall_pct = statistics.mean(recalls) * 100 if recalls else 0.0
    return mean_ms, p99_ms, qps, recall_pct
