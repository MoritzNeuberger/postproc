from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.detector_active_time import m_detector_active_time


def test_m_detector_active_time_valid():
    para = {}
    input = {"edep": "edep", "vol": "vol"}
    output = {"edep": "processed_edep"}
    pv = {
        "edep": ak.Array([[[0.03, 0.02], [0.04, 0.01]]]),
        "vol": ak.Array([[[1010101, 1010209], [1010101, 1010209]]]),
    }

    m_detector_active_time(para, input, output, pv)

    expected_processed_edep = ak.Array([[[0.03, 0], [0.04, 0]]])
    assert ak.to_list(pv["processed_edep"]) == ak.to_list(expected_processed_edep)


if __name__ == "__main__":
    pytest.main()
