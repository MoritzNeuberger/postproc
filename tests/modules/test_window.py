from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.window import (
    define_windows,
    generate_map,
    generate_windowed_hits,
    m_window,
    subtract_smallest_time,
)


def test_subtract_smallest_time():
    t = ak.Array([5, 10, 15])
    t_all = ak.Array([0, 5, 10, 15])
    expected = ak.Array([5, 10, 15])
    result = subtract_smallest_time(t, t_all)
    assert result.to_list() == expected.to_list()

    t = ak.Array([[5, 10], [15, 20]])
    t_all = ak.Array([[0, 5], [10, 15]])
    expected = ak.Array([[5, 10], [5, 10]])
    result = subtract_smallest_time(t, t_all)
    assert result.to_list() == expected.to_list()


def test_define_windows():
    t_sub = [1, 2, 3, 10, 11, 12]
    dT = 5
    expected = [1, 10]
    result = define_windows(t_sub, dT)
    assert ak.Array(result).to_list() == expected

    t_sub = [[1, 2, 3], [10, 11, 12]]
    dT = 5
    expected = [[1], [10]]
    result = define_windows(t_sub, dT)
    assert ak.Array(result).to_list() == expected


def test_generate_map():
    t_sub = [[1, 2, 3], [10, 11, 12]]
    w_t = [[0, 5], [5, 10]]
    expected = [[0, 0, 0], [1, 1, 1]]
    result = generate_map(t_sub, w_t)
    assert ak.Array(result).to_list() == expected


def test_generate_windowed_hits():
    mapping = [0, 1, 0, 1]
    v_in = [10, 20, 30, 40]
    expected = [[10, 30], [20, 40]]
    result = generate_windowed_hits(mapping, v_in)
    assert result == expected

    mapping = [[0, 1], [0, 1]]
    v_in = [[10, 20], [30, 40]]
    expected = [[[10], [20]], [[30], [40]]]
    result = generate_windowed_hits(mapping, v_in)
    assert result == expected


def test_m_window():
    para = {"dT": 5}
    input = {
        "t_all": "t_all",
        "t": "t",
        "edep": "edep",
        "vol": "vol",
        "posx": "posx",
        "posy": "posy",
        "posz": "posz",
    }
    output = {
        "w_t": "w_t",
        "t_sub": "t_sub",
        "edep": "w_edep",
        "vol": "w_vol",
        "posx": "w_posx",
        "posy": "w_posy",
        "posz": "w_posz",
    }
    pv = {
        "t_all": ak.Array([0, 10, 20, 30, 40, 50]),
        "t": ak.Array([5, 15, 25, 35, 45, 55]),
        "edep": ak.Array([1, 2, 3, 4, 5, 6]),
        "vol": ak.Array([1, 1, 2, 2, 3, 3]),
        "posx": ak.Array([0, 1, 2, 3, 4, 5]),
        "posy": ak.Array([0, -1, -2, -3, -4, -5]),
        "posz": ak.Array([1, 2, 3, 4, 5, 6]),
    }

    m_window(para, input, output, pv)

    expected_w_t = ak.Array([5, 15, 25, 35, 45, 55])
    expected_t_sub = ak.Array([[5], [15], [25], [35], [45], [55]])
    expected_edep = ak.Array([[1], [2], [3], [4], [5], [6]])
    expected_vol = ak.Array([[1], [1], [2], [2], [3], [3]])
    expected_posx = ak.Array([[0], [1], [2], [3], [4], [5]])
    expected_posy = ak.Array([[0], [-1], [-2], [-3], [-4], [-5]])
    expected_posz = ak.Array([[1], [2], [3], [4], [5], [6]])

    assert ak.to_list(pv["w_t"]) == ak.to_list(expected_w_t)
    assert ak.to_list(pv["t_sub"]) == ak.to_list(expected_t_sub)
    assert ak.to_list(pv["w_edep"]) == ak.to_list(expected_edep)
    assert ak.to_list(pv["w_vol"]) == ak.to_list(expected_vol)
    assert ak.to_list(pv["w_posx"]) == ak.to_list(expected_posx)
    assert ak.to_list(pv["w_posy"]) == ak.to_list(expected_posy)
    assert ak.to_list(pv["w_posz"]) == ak.to_list(expected_posz)


if __name__ == "__main__":
    pytest.main()
