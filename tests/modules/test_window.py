from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.window import (
    define_windows,
    generate_map,
    generate_windowed_hits,
    subtract_smallest_time,
)


def test_subtract_smallest_time():
    t = [5, 10, 15]
    t_all = [0, 5, 10, 15]
    expected = ak.Array([5, 10, 15])
    result = subtract_smallest_time(t, t_all)
    assert result.to_list() == expected.to_list()

    t = [[5, 10], [15, 20]]
    t_all = [[0, 5], [10, 15]]
    expected = ak.Array([[5, 10], [5, 10]])
    result = subtract_smallest_time(t, t_all)
    assert result.to_list() == expected.to_list()


def test_define_windows():
    t_sub = [1, 2, 3, 10, 11, 12]
    dT = 5
    expected = ak.Array([1, 10])
    result = define_windows(t_sub, dT)
    assert result.to_list() == expected.to_list()

    t_sub = [[1, 2, 3], [10, 11, 12]]
    dT = 5
    expected = ak.Array([[1], [10]])
    result = define_windows(t_sub, dT)
    assert result.to_list() == expected.to_list()


def test_generate_map():
    t_sub = [[1, 2, 3], [10, 11, 12]]
    w_t = [[0, 5], [5, 10]]
    expected = ak.Array([[0, 0, 0], [1, 1, 1]])
    result = generate_map(t_sub, w_t)
    assert result.tolist() == expected.to_list()


def test_generate_windowed_hits():
    mapping = [0, 1, 0, 1]
    v_in = [10, 20, 30, 40]
    expected = ak.Array([[10, 30], [20, 40]])
    result = generate_windowed_hits(mapping, v_in)
    assert result.tolist() == expected.to_list()

    mapping = [[0, 1], [0, 1]]
    v_in = [[10, 20], [30, 40]]
    expected = ak.Array([[[10], [20]], [[30], [40]]])
    result = generate_windowed_hits(mapping, v_in)
    assert result.tolist() == expected.to_list()


if __name__ == "__main__":
    pytest.main()
