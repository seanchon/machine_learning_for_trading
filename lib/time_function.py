from time import time


def how_long(func, *args):
    """Execute function with given arguments and measure execution time."""
    t0 = time()
    result = func(*args)  # all arguments are passed in as-is
    t1 = time()
    return result, t1 - t0
