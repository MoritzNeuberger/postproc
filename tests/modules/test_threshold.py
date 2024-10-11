from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.threshold import m_threshold


def test_m_threshold():
    para = {"thr": [10, 20]}
    input = ["val"]
    output = ["val"]
    pv = {"val": ak.Array([5, 15, 25, 10, 20, 12, 18])}

    m_threshold(para, input, output, pv)

    expected = ak.Array([False, True, False, False, False, True, True])
    result = pv["val"]

    assert ak.to_list(result) == ak.to_list(expected)

    # Test with different thresholds
    para = {"thr": [15, 25]}
    pv = {"val": ak.Array([5, 15, 25, 10, 20, 12, 18])}

    m_threshold(para, input, output, pv)

    expected = ak.Array([False, False, False, False, True, False, True])
    result = pv["val"]

    assert ak.to_list(result) == ak.to_list(expected)


if __name__ == "__main__":
    pytest.main()
