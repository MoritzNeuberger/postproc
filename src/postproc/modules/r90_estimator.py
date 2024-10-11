from __future__ import annotations

import awkward as ak
import numpy as np


def get_R90_per_detector(v_dist, v_edep):
    tot_e = np.sum(v_edep)
    mask_sort = np.array(v_dist).argsort()
    v_dist_sort = v_dist[mask_sort]
    v_edep_sort = v_edep[mask_sort]
    v_cumsum_edep_sort = np.cumsum(v_edep_sort)
    if len(v_cumsum_edep_sort):
        pos = np.argmax(v_cumsum_edep_sort >= 0.9 * tot_e)
        return v_dist_sort[pos]
    return 0


def calculate_R90(v_edep_hwd, v_posx_hwd, v_posy_hwd, v_posz_hwd):
    if isinstance(v_edep_hwd[0], (list, ak.Array)):
        return ak.Array(
            [
                calculate_R90(
                    v_edep_hwd[i], v_posx_hwd[i], v_posy_hwd[i], v_posz_hwd[i]
                )
                for i in range(len(v_edep_hwd))
            ]
        )

    v_posx_edep = v_edep_hwd * v_posx_hwd
    v_posy_edep = v_edep_hwd * v_posy_hwd
    v_posz_edep = v_edep_hwd * v_posz_hwd
    v_edep_e = ak.sum(v_edep_hwd, axis=-1)

    v_meanx = ak.sum(v_posx_edep, axis=-1) / v_edep_e
    v_meany = ak.sum(v_posy_edep, axis=-1) / v_edep_e
    v_meanz = ak.sum(v_posz_edep, axis=-1) / v_edep_e

    v_dist = np.sqrt(
        (v_posx_hwd - v_meanx) ** 2
        + (v_posy_hwd - v_meany) ** 2
        + (v_posz_hwd - v_meanz) ** 2
    )

    return get_R90_per_detector(v_dist.to_numpy(), v_edep_hwd.to_numpy())


def m_r90_estimator(para, input, output, pv):  # noqa: ARG001
    """
    R90 Estimator module for the postprocessing pipeline.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        None necessary.

    input (list): List of input parameters in the following order:
        - edep: Name of energy depositions array.
        - posx: Name of x positions array.
        - posy: Name of y positions array.
        - posz: Name of z positions array.

    output (list): List of output parameters in the following order:
        - r90: R90 estimation.

    pv (dict): Dictionary to store the processed values.

    """
    in_n = {"edep": input[0], "x": input[1], "y": input[2], "z": input[3]}

    out_n = {"r90": output[0]}
    pv[out_n["r90"]] = calculate_R90(
        pv[in_n["edep"]], pv[in_n["x"]], pv[in_n["y"]], pv[in_n["z"]]
    )
