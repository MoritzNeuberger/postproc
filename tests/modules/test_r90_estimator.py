from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from postproc.modules.r90_estimator import (
    calculate_R90,
    get_R90_per_detector,
    m_r90_estimator,
)


def test_get_R90_per_detector():
    v_dist = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    v_edep = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    expected = 9
    result = get_R90_per_detector(v_dist, v_edep)
    assert result == expected


def test_calculate_R90():
    v_edep_hwd = ak.Array([[[90, 10], [90, 10]], [[90, 10], [90, 10]]])
    v_posx_hwd = ak.Array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    v_posy_hwd = ak.Array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    v_posz_hwd = ak.Array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])

    expected = ak.Array([[0.173, 0.173], [0.173, 0.173]])
    result = calculate_R90(v_edep_hwd, v_posx_hwd, v_posy_hwd, v_posz_hwd)
    assert all(
        pytest.approx(a, 0.01) == b
        for a, b in zip(ak.to_list(result), ak.to_list(expected))
    )


def test_m_r90_estimator():
    para = {}
    input = {"edep": "edep", "posx": "posx", "posy": "posy", "posz": "posz"}
    output = {"r90": "r90"}
    pv = {
        "edep": ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "posx": ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "posy": ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "posz": ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    }

    m_r90_estimator(para, input, output, pv)
    expected = calculate_R90(pv["edep"], pv["posx"], pv["posy"], pv["posz"])
    result = pv["r90"]

    assert ak.to_list(result) == ak.to_list(expected)


if __name__ == "__main__":
    pytest.main()
