from __future__ import annotations

import awkward as ak
import pytest

from postproc.modules.group_sensitive_volume import (
    generate_group_mask,
    group_all_in_detector_ids,
    m_group_sensitive_volume,
)


def test_generate_group_mask():
    vol = ak.Array([1, 2, 3, 4, 5])
    group = "HPGe"
    sensitive_volumes = {
        "sensVolID": [1, 2, 3, 4, 5],
        "group": ["HPGe", "HPGe", "LAr", "LAr", "WC"],
    }
    expected = ak.Array([1, 1, 0, 0, 0])
    result = generate_group_mask(vol, group, sensitive_volumes)
    assert ak.to_list(result) == ak.to_list(expected)

    group = "LAr"
    expected = ak.Array([0, 0, 1, 1, 0])
    result = generate_group_mask(vol, group, sensitive_volumes)
    assert ak.to_list(result) == ak.to_list(expected)

    vol = ak.Array([[1, 2, 3], [4, 5]])
    group = "LAr"
    expected = ak.Array([[0, 0, 1], [1, 0]])
    result = generate_group_mask(vol, group, sensitive_volumes)
    assert ak.to_list(result) == ak.to_list(expected)


def test_group_all_in_detector_ids():
    v_voln_hw = ak.Array([1, 2, 1, 3, 2])
    v_in = ak.Array([10, 20, 30, 40, 50])
    expected = ak.Array([[10, 30], [20, 50], [40]])
    result = group_all_in_detector_ids(v_voln_hw, v_in)
    assert ak.to_list(result) == ak.to_list(expected)

    v_voln_hw = ak.Array([[1, 2], [1, 3]])
    v_in = ak.Array([[10, 20], [30, 40]])
    expected = ak.Array([[[10], [20]], [[30], [40]]])
    result = group_all_in_detector_ids(v_voln_hw, v_in)
    assert ak.to_list(result) == ak.to_list(expected)


def test_m_group_sensitive_volume():
    para = {
        "group": 1,
        "sensitive_volumes": {"sensVolID": [1, 2, 3, 4, 5], "group": [1, 1, 2, 2, 3]},
    }
    input = {
        "t": "t",
        "edep": "edep",
        "vol": "vol",
        "posx": "posx",
        "posy": "posy",
        "posz": "posz",
    }  # ["t", "edep", "vol", "posx", "posy", "posz"]
    output = {
        "t": "grouped_t",
        "edep": "grouped_edep",
        "vol": "grouped_vol",
        "posx": "grouped_posx",
        "posy": "grouped_posy",
        "posz": "grouped_posz",
    }

    pv = {
        "t": ak.Array([1, 2, 3, 4, 5]),
        "edep": ak.Array([10, 20, 30, 40, 50]),
        "vol": ak.Array([1, 2, 3, 4, 5]),
        "posx": ak.Array([1, 2, 3, 4, 5]),
        "posy": ak.Array([1, 2, 3, 4, 5]),
        "posz": ak.Array([1, 2, 3, 4, 5]),
    }
    m_group_sensitive_volume(para, input, output, pv)
    assert ak.to_list(pv["grouped_edep"]) == [10, 20]
    assert ak.to_list(pv["grouped_vol"]) == [1, 2]

    para = {
        "group": 2,
        "sensitive_volumes": {"sensVolID": [1, 2, 3, 4, 5], "group": [1, 1, 2, 2, 3]},
    }
    m_group_sensitive_volume(para, input, output, pv)
    assert ak.to_list(pv["grouped_edep"]) == [30, 40]
    assert ak.to_list(pv["grouped_vol"]) == [3, 4]


if __name__ == "__main__":
    pytest.main()
