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

    input (dict): Dictionary containing input parameters.
        required:
        - edep: Name of the energy deposition array.
        - posx: Name of the x positions array.
        - posy: Name of the y positions array.
        - posz: Name of the z positions array.

    output (dict): Dictionary containing output parameters.
        required:
        - r90: Name of the R90 array.

    pv (dict): Dictionary to store the processed values.

    """

    required_input = ["edep", "posx", "posy", "posz"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    required_output = ["r90"]
    for r in required_output:
        if r not in output:
            text = f"Required output {r} not found in output. All required outputs are {required_output}."
            raise ValueError(text)

    pv[output["r90"]] = calculate_R90(
        pv[input["edep"]], pv[input["posx"]], pv[input["posy"]], pv[input["posz"]]
    )
