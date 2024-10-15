from __future__ import annotations

import timeit

import numpy as np

from postproc.modules.window import define_windows

# Sample data for testing
t_all = np.random.Generator(10000).reshape(100, 100)
t = t_all[::2]
edep = t
dT = 1e4


para = {"dT": dT}
input = {"t_all": "t_all", "t": "t", "edep": "edep"}
output = {"w_t": "w_t", "t_sub": "t_sub", "edep": "w_edep"}
pv = {"t_all": t_all, "t": t, "edep": t}


# Function to benchmark another function
def benchmark(func, *args, **kwargs):
    def wrapper():
        return func(*args, **kwargs)

    return timeit.timeit(wrapper, number=10)


# Centralized benchmarking script
def main():
    # Benchmark define_windows
    define_windows_time = benchmark(define_windows, t, dT)
    print(f"define_windows execution time: {define_windows_time:.6f} seconds")  # noqa: T201

    # Benchmark subtract_smallest_time
    # subtract_smallest_time_time = benchmark(subtract_smallest_time, t, t_all)
    # print(f"subtract_smallest_time execution time: {subtract_smallest_time_time:.6f} seconds")

    # Benchmark m_window
    # m_window_time = benchmark(m_window, para, input, output, pv)
    # print(f"m_window execution time: {m_window_time:.6f} seconds")


if __name__ == "__main__":
    main()
