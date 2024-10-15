from __future__ import annotations

import timeit

import awkward as ak
import numpy as np

from postproc.modules.group_sensitive_volume import m_group_sensitive_volume

# Sample data for testing
v_in = ak.Array(np.random.Generator(100000).reshape(1000, 100))
v_voln_hw = ak.Array(np.random.Generator(1, 6, 100000).reshape(1000, 100))

para = {
    "group": 1,
    "sensitive_volumes": {"sensVolID": [1, 2, 3, 4, 5], "group": [1, 1, 2, 2, 3]},
}

para = {"some_parameter": "value"}
input_data = {"vol": "v_voln_hw", "edep": "v_in"}
output_data = {"vol": "test", "edep": "test2"}
pv = {"v_voln_hw": v_voln_hw, "v_in": v_in}


# Function to benchmark another function
def benchmark(func, *args, **kwargs):
    def wrapper():
        return func(*args, **kwargs)

    return timeit.timeit(wrapper, number=10)


# Centralized benchmarking script
def main():
    # Benchmark m_group_sensitive_volume
    m_group_sensitive_volume_time = benchmark(
        m_group_sensitive_volume, para, input_data, output_data, pv
    )
    print(  # noqa: T201
        f"m_group_sensitive_volume execution time: {m_group_sensitive_volume_time:.6f} seconds"
    )
    print(pv["test"])  # noqa: T201


if __name__ == "__main__":
    main()
