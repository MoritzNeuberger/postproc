from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.acceptance_range import m_acceptance_range


def test_m_acceptance_range():
    para = {"thr": [10, 20]}
    input = {"val": "val"}  # ["val"]
    output = {"val": "mask"}
    pv = {"val": ak.Array([5, 15, 25, 10, 20, 12, 18])}

    m_acceptance_range(para, input, output, pv)

    expected = ak.Array([False, True, False, True, True, True, True])
    result = pv["mask"]

    assert ak.to_list(result) == ak.to_list(expected)

    # Test with different thresholds
    para = {"thr": [15, 25]}
    pv = {"val": ak.Array([5, 15, 25, 10, 20, 12, 18])}

    m_acceptance_range(para, input, output, pv)

    expected = ak.Array([False, True, True, False, True, False, True])
    result = pv["mask"]

    assert ak.to_list(result) == ak.to_list(expected)


if __name__ == "__main__":
    pytest.main()
