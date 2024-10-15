from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from .misc import python_list_to_numba_list


def generate_mask_cylinder(x, y, z, para):
    r = np.sqrt(x**2 + y**2)
    mask = ak.ones_like(x)
    mask = mask - (r > para["conditions"]["r"])
    mask = mask - (
        (z > para["conditions"]["h_top"]) + (z < para["conditions"]["h_bottom"])
    )

    inverse = False
    if "inverse" in para:
        inverse = para["inverse"]

    if inverse:
        return mask < 1
    return mask > 0


@njit
def is_point_inside_polycone(x, y, z, r_val, z_val):
    r = np.sqrt(x**2 + y**2)
    n = len(r_val)
    inside = False
    p2r = 0.0
    p2z = 0.0
    rints = 0.0
    p1r, p1z = r_val[0], z_val[0]
    for i in range(n + 1):
        p2r, p2z = r_val[i % n], z_val[i % n]
        if z > min(p1z, p2z) and z <= max(p1z, p2z) and r <= max(p1r, p2r):
            if p1z != p2z:
                rints = (z - p1z) * (p2r - p1r) / (p2z - p1z) + p1r
            if p1r == p2r or r <= rints:
                inside = not inside
        p1r, p1z = p2r, p2z

    return inside


@njit
def is_in_active_volume_polycone(x, y, z, vol, dl_input):
    pos = dl_input[vol]["center"]
    r_dl = dl_input[vol]["r_dl"]
    z_dl = dl_input[vol]["z_dl"]
    return is_point_inside_polycone(x - pos[0], y - pos[1], z - pos[2], r_dl, z_dl)


def generate_mask_deadlayer(x, y, z, vol, para):
    if isinstance(para["file"], str):
        para["file"] = Path(para["file"])

    with Path.open(para["file"]) as f:
        dl_input = json.load(f)

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

    @njit
    def _internal(x, y, z, vol, dl_input):
        output = []
        for i in range(len(x)):
            output.append(
                is_in_active_volume_polycone(x[i], y[i], z[i], vol[i], dl_input)
            )
        return output

    @njit
    def recursion_function(x, y, z, vol, dl_input_numba):
        if isinstance(x[0], List):
            return List(
                [
                    recursion_function(x[i], y[i], z[i], vol[i], dl_input_numba)
                    for i in range(len(x))
                ]
            )
        return _internal(x, y, z, vol, dl_input_numba)

    x = python_list_to_numba_list(ak.Array(x).to_list())
    y = python_list_to_numba_list(ak.Array(y).to_list())
    z = python_list_to_numba_list(ak.Array(z).to_list())
    vol = python_list_to_numba_list(ak.Array(vol).to_list())

    return recursion_function(x, y, z, vol, dl_input_numba)


def m_active_volume(para, input, output, pv):
    """
    Active Volume module for the postprocessing pipeline.

    Given a condition on the location, the module filters the input data based on the condition.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        required:
        - type (str): Type of active volume. Options are 'cylinder' or 'deadlayer'.

        required for 'cylinder':
        - conditions (dict): Dictionary containing the conditions for the cylinder.
            - r (float): Radius of the cylinder.
            - h_top (float): Top boundary of the cylinder.
            - h_bottom (float): Bottom boundary of the cylinder.

        required for 'deadlayer':
        - file (str): Path to the deadlayer input file.

    input (dict): Dictionary containing input parameters.
        required:
        - posx: Name of the x positions array.
        - posy: Name of the y positions array.
        - posz: Name of the z positions array.
        - vol: Name of the volume array.

        additional:
        - arbitrary number of input arrays to be filtered.

    output (dict): Dictionary containing output parameters.
        required:
        - posx: Name of the filtered x positions array.
        - posy: Name of the filtered y positions array.
        - posz: Name of the filtered z positions array.
        - vol: Name of the filtered volume array.

        required for deadlayer type:
        - vol_red: Name of the reduced volume array.

        additional:
        - arbitrary number of output arrays. One for each additional input array.

    pv (dict): Dictionary to store the processed values.

    """

    required_input = ["posx", "posy", "posz", "vol"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    if "type" not in para:
        text = (
            "Parameter 'type' not found in para. Options are 'cylinder' or 'deadlayer'"
        )
        raise ValueError(text)

    for r in input:
        if r not in output:
            text = f"For each input parameter, there must be a corresponding output parameter. {r} is in input but not in output."
            raise ValueError(text)

    if para["type"] == "deadlayer" and "vol_red" not in output:
        text = "Output parameter 'vol_red' is required for deadlayer type."
        raise ValueError(text)

    if para["type"] == "cylinder":
        mask = generate_mask_cylinder(
            pv[input["posx"]], pv[input["posy"]], pv[input["posz"]], para
        )

        for key in input:
            pv[output[key]] = pv[input[key]][mask]

    if para["type"] == "deadlayer":
        mask = generate_mask_deadlayer(
            pv[input["posx"]],
            pv[input["posy"]],
            pv[input["posz"]],
            pv[input["vol"]],
            para,
        )
        pv[output["vol_red"]] = ak.firsts(pv[input["vol"]], axis=-1)

        for key in input:
            pv[output[key]] = pv[input[key]][mask]
