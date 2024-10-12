from __future__ import annotations

import awkward as ak
import numpy as np
from numba import njit


def generate_group_mask(vol, group, sensitive_volumes):
    sv_in_group = np.array(sensitive_volumes["sensVolID"])[
        np.array(sensitive_volumes["group"]) == group
    ]
    mask = ak.zeros_like(vol)

    if isinstance(sv_in_group, int):
        mask = mask + (vol == sv_in_group)
    else:
        for sv in sv_in_group:
            mask = mask + (vol == sv)
    return mask > 0


def group_all_in_detector_ids(v_voln_hw, v_in):
    if isinstance(v_voln_hw[0], (list, ak.Array)):
        return ak.Array(
            [
                group_all_in_detector_ids(v_voln_hw[i], v_in[i])
                for i in range(len(v_voln_hw))
            ]
        )

    @njit
    def _internal(v_voln_hw, v_in):
        list_indices = []
        for i in range(len(v_voln_hw)):
            if v_voln_hw[i] not in list_indices and v_voln_hw[i] > -1:
                list_indices.append(v_voln_hw[i])

        output = []
        for i in range(len(list_indices)):
            tmp = []
            for j in range(len(v_in)):
                if list_indices[i] == v_voln_hw[j]:
                    tmp.append(v_in[j])
            output.append(tmp)

        return output

    return ak.Array(_internal(v_voln_hw, v_in))


def m_group_sensitive_volume(para, input, output, pv):
    """
    Group Sensitive Volume module for the postprocessing pipeline.

    Groups the input data into sensitive volumes according to the vol array.

    When sellecting "group" mode, only hits with sensitive volumes in the selected group are kept. The number of dimensions stays the same.
    Otherwise, the grouping is done for all sensitive volumes and a dimension is added to the output arrays for each group.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        required:
        - group (string,int): Group name/number to select.
        - sensitive_volumes (dict): Dictionary containing the sensitive volumes.

    input (dict): Dictionary containing input parameters.
        required:
        - vol: Name of the sensitive volume array.
        additional:
        - arbitrary number of input arrays to be grouped.

    output (dict): Dictionary containing output parameters.
        required:
        - vol: Name of the sensitive volume array.
        additional:
        - arbitrary number of output arrays to store the grouped. One output array for each input array.


    pv (dict): Dictionary to store the processed values.

    """

    required_input = ["vol"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    if len(output) != len(input):
        text = "Number of input and output parameters must be the same."
        raise ValueError(text)

    for r in input:
        if r not in output:
            text = f"All input parameters must have an output parameter. {r} not found in output"
            raise ValueError(text)

    if "group" in para:
        mask = generate_group_mask(
            pv[input["vol"]], para["group"], para["sensitive_volumes"]
        )
        for key, value in output.items():
            pv[value] = pv[input[key]][mask]

    else:
        for key, value in output.items():
            pv[value] = group_all_in_detector_ids(pv[input["vol"]], pv[input[key]])
