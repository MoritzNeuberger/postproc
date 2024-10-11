from __future__ import annotations

import json
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from numba import types
from numba.typed import Dict

from postproc.modules.active_volume import (
    generate_mask_cylinder,
    generate_mask_deadlayer,
    is_in_active_volume_polycone,
    m_active_volume,
)


def test_generate_mask_cylinder():
    para = {"conditions": {"r": 1, "h_top": 1, "h_bottom": -1}}

    x = ak.Array([[0.5, 1.5], [0.5]])
    y = ak.Array([[0.5, 1.5], [0.5]])
    z = ak.Array([[0.5, 1.5], [0.5]])

    assert (
        generate_mask_cylinder(x, y, z, para).to_list()
        == ak.Array([[True, False], [True]]).to_list()
    )


def test_is_in_active_volume_polycone():
    dl_input = {
        "1": {
            "name": "tmp1",
            "center": [0, 0, 0],
            "surface_mesh": {
                "orig": {"r": [0, 1, 1, 0], "z": [-1, -1, 1, 1]},
                "dl": {"r": [0, 0.9, 0.9, 0], "z": [-0.9, -0.9, 0.9, 0.9]},
            },
        },
        "2": {
            "name": "tmp2",
            "center": [10, 10, 10],
            "surface_mesh": {
                "orig": {"r": [0, 1, 1, 0], "z": [-1, -1, 1, 1]},
                "dl": {"r": [0, 0.9, 0.9, 0], "z": [-0.9, -0.9, 0.9, 0.9]},
            },
        },
    }

    def convert_to_numba_dict(py_dict):
        # Create the outer numba.typed.Dict
        nb_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.DictType(types.unicode_type, types.float64[:]),
        )

        for vol_key, vol_value in py_dict.items():
            # Create the inner dictionary for each volume (flatten surface_mesh fields)
            vol_dict = Dict.empty(
                key_type=types.unicode_type, value_type=types.float64[:]
            )

            # Convert "center" to a numpy array
            center_array = np.array(vol_value["center"], dtype=np.float64)
            vol_dict["center"] = center_array

            # Flatten "surface_mesh.orig.r", "surface_mesh.orig.z", "surface_mesh.dl.r", and "surface_mesh.dl.z"
            r_orig_array = np.array(
                vol_value["surface_mesh"]["orig"]["r"], dtype=np.float64
            )
            z_orig_array = np.array(
                vol_value["surface_mesh"]["orig"]["z"], dtype=np.float64
            )
            r_dl_array = np.array(
                vol_value["surface_mesh"]["dl"]["r"], dtype=np.float64
            )
            z_dl_array = np.array(
                vol_value["surface_mesh"]["dl"]["z"], dtype=np.float64
            )

            # Store the flattened arrays into the vol_dict
            vol_dict["r_orig"] = r_orig_array
            vol_dict["z_orig"] = z_orig_array
            vol_dict["r_dl"] = r_dl_array
            vol_dict["z_dl"] = z_dl_array

            # Store vol_dict inside the nb_dict with the integer key
            nb_dict[int(vol_key)] = vol_dict

        return nb_dict

    dl_input_numba = convert_to_numba_dict(dl_input)

    assert is_in_active_volume_polycone(0.5, 0.5, 0.5, 1, dl_input_numba)
    assert not is_in_active_volume_polycone(0.5, 0.5, 0.5, 2, dl_input_numba)


def test_generate_mask_deadlayer():
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        json_content = {
            1: {
                "name": "tmp1",
                "center": [0, 0, 0],
                "surface_mesh": {
                    "orig": {"r": [0, 1, 1, 0], "z": [-1, -1, 1, 1]},
                    "dl": {"r": [0, 0.9, 0.9, 0], "z": [-0.9, -0.9, 0.9, 0.9]},
                },
            }
        }

        json.dump(json_content, tf)

        tf.flush()

        x = np.array([0, 1])
        y = np.array([0, 1])
        z = np.array([0, 1])
        vol = np.array([1, 1])
        para = {"file": Path(tf.name)}

        result = generate_mask_deadlayer(x, y, z, vol, para)
        expected = [True, False]

        assert result == expected


def test_m_active_volume():
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        json_content = {
            1: {
                "name": "tmp1",
                "center": [0, 0, 0],
                "surface_mesh": {
                    "orig": {"r": [0, 1, 1, 0], "z": [-1, -1, 1, 1]},
                    "dl": {"r": [0, 0.9, 0.9, 0], "z": [-0.9, -0.9, 0.9, 0.9]},
                },
            }
        }

        json.dump(json_content, tf)

        tf.flush()

        para = {"type": "deadlayer", "file": Path(tf.name)}
        input = ["t", "edep", "vol", "x", "y", "z"]
        output = ["t_a", "edep_a", "vol_a", "x_a", "y_a", "z_a", "vol_red"]
        pv = {
            "t": ak.Array([[0, 1], [0, 1]]),
            "edep": ak.Array([[1, 1], [0, 1]]),
            "x": ak.Array([[0, 1], [0, 1]]),
            "y": ak.Array([[0, 1], [0, 1]]),
            "z": ak.Array([[0, 1], [0, 1]]),
            "vol": ak.Array([[1, 1], [1, 1]]),
        }

        m_active_volume(para, input, output, pv)
        expected_edep = ak.Array([[1], [0]])
        result = pv["edep_a"]

        assert ak.to_list(result) == ak.to_list(expected_edep)


if __name__ == "__main__":
    pytest.main()
