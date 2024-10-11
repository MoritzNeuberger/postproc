from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.coincidence_window import generate_output, m_coincidence_window


def test_generate_output():
    wt_m1 = ak.Array([1, 2, 3])
    wt_m2 = ak.Array([1.1, 1.3, 3.5])
    val = ak.Array([10, 20, 30])
    para = {"t_min": -0.4, "t_max": 0.6}

    expected = ak.Array([[10, 20], [], [30]])
    result = generate_output(wt_m1, wt_m2, val, para)
    assert ak.to_list(result) == ak.to_list(expected)

    wt_m1 = ak.Array([[1, 2], [3]])
    wt_m2 = ak.Array([[1.1, 1.3], [3.5]])
    val = ak.Array([[10, 20], [30]])
    para = {"t_min": -0.4, "t_max": 0.6}

    expected = ak.Array([[[10, 20], []], [[30]]])
    result = generate_output(wt_m1, wt_m2, val, para)
    assert ak.to_list(result) == ak.to_list(expected)


def test_m_coincidence_window():
    para = {"t_min": -0.4, "t_max": 0.4}
    input = ["w_t_1", "w_t_2", "val"]
    output = ["coincident_events"]
    pv = {
        "w_t_1": ak.Array([1, 2, 3]),
        "w_t_2": ak.Array([1.5, 2.5, 3.5]),
        "val": ak.Array([10, 20, 30]),
    }

    m_coincidence_window(para, input, output, pv)
    expected = ak.Array([[], [], []])
    result = pv["coincident_events"]
    assert ak.to_list(result) == ak.to_list(expected)

    para = {"t_min": -0.4, "t_max": 0.6}
    pv = {
        "w_t_1": ak.Array([1, 2, 3]),
        "w_t_2": ak.Array([1.5, 2.5, 3.5]),
        "val": ak.Array([10, 20, 30]),
    }

    m_coincidence_window(para, input, output, pv)
    expected = ak.Array([[10], [20], [30]])
    result = pv["coincident_events"]
    assert ak.to_list(result) == ak.to_list(expected)


if __name__ == "__main__":
    pytest.main()
