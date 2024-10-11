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
        - group (string,int): Group name/number to select.

    input (list): List of input parameters in the following order:
        - t: Name of time array.
        - edep: Name of energy depositions array.
        - vol: Name of volumes array.
        - posx: Name of x positions array.
        - posy: Name of y positions array.
        - posz: Name of z positions array.

    output (list): List of output parameters in the following order:
        - grouped_t: Array of grouped times.
        - grouped_edep: Array of grouped energy depositions.
        - grouped_vol: Array of grouped volumes.
        - grouped_posx: Array of grouped x positions.
        - grouped_posy: Array of grouped y positions.
        - grouped_posz: Array of grouped


    pv (dict): Dictionary to store the processed values.

    """
    in_n = {
        "t": input[0],
        "edep": input[1],
        "vol": input[2],
        "posx": input[3],
        "posy": input[4],
        "posz": input[5],
    }

    out_n = {
        "t": output[0],
        "edep": output[1],
        "vol": output[2],
        "posx": output[3],
        "posy": output[4],
        "posz": output[5],
    }

    if "group" in para:
        mask = generate_group_mask(
            pv[in_n["vol"]], para["group"], para["sensitive_volumes"]
        )
        for key, value in out_n.items():
            pv[value] = pv[in_n[key]][mask]

    else:
        for key, value in out_n.items():
            pv[value] = group_all_in_detector_ids(pv[in_n["vol"]], pv[in_n[key]])
