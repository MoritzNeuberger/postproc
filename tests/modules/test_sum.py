from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.sum import m_sum


def test_m_sum():
    para = {}
    input = {"val": "val"}
    output = {"val": "total_val"}
    pv = {"val": ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}

    m_sum(para, input, output, pv)
    expected = ak.Array([6, 15, 24])
    result = pv["total_val"]

    assert ak.to_list(result) == ak.to_list(expected)


def test_m_sum_empty():
    para = {}
    input = {"val": "val"}
    output = {"val": "total_val"}
    pv = {"val": ak.Array([[], [], []])}

    m_sum(para, input, output, pv)
    expected = ak.Array([0, 0, 0])
    result = pv["total_val"]

    assert ak.to_list(result) == ak.to_list(expected)


def test_m_sum_single_element():
    para = {}
    input = {"val": "val"}
    output = {"val": "total_val"}
    pv = {"val": ak.Array([[1], [2], [3]])}

    m_sum(para, input, output, pv)
    expected = ak.Array([1, 2, 3])
    result = pv["total_val"]

    assert ak.to_list(result) == ak.to_list(expected)


if __name__ == "__main__":
    pytest.main()
