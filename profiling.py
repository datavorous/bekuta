import functools
import time


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        print(f"[PROFILE] {func.__name__} took {elapsed_ms:.3f}ms")
        return result

    return wrapper
