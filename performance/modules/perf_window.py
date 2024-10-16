from __future__ import annotations

import timeit

import numpy as np

from postproc.modules.window import (
    define_windows,
    generate_map,
    generate_windowed_hits,
    subtract_smallest_time,
)

rng = np.random.default_rng(12345)

# Sample data for testing
t_all = rng.integers(low=0, high=10000, size=40000).reshape(200, 200)
t = t_all[::2]
t_sub = subtract_smallest_time(t, t_all)
edep = t
dT = 1e1

para = {"dT": dT}
input = {"t_all": "t_all", "t": "t", "edep": "edep"}
output = {"w_t": "w_t", "t_sub": "t_sub", "edep": "w_edep"}
pv = {"t_all": t_all, "t": t, "edep": t}


def benchmark(func, *args, **kwargs):
    def wrapper():
        return func(*args, **kwargs)

    # Get the result of the function
    result = func(*args, **kwargs)

    # Benchmark the time
    execution_time = timeit.timeit(wrapper, number=10)

    return result, execution_time


# Centralized benchmarking script
def main():
    # Benchmark define_windows
    w_t, define_windows_time = benchmark(define_windows, t, dT)
    print(f"define_windows execution time: {define_windows_time:.6f} seconds")  # noqa: T201

    # Benchmark generate_map
    mapping, generate_map_time = benchmark(generate_map, t_sub, w_t)
    print(f"generate_map execution time: {generate_map_time:.6f} seconds")  # noqa: T201

    # Benchmark generate_map
    _, generate_windowed_hits_time = benchmark(generate_windowed_hits, mapping, edep)
    print(  # noqa: T201
        f"generate_windowed_hits execution time: {generate_windowed_hits_time:.6f} seconds"
    )

    # Benchmark m_window
    # m_window_time = benchmark(m_window, para, input, output, pv)
    # print(f"m_window execution time: {m_window_time:.6f} seconds")


if __name__ == "__main__":
    main()
