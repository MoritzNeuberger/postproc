from __future__ import annotations

import awkward as ak
from numba import njit


def generate_output(wt_m1, wt_m2, val, para):
    if isinstance(wt_m1[0], (list, ak.Array)):
        return ak.Array(
            [
                generate_output(wt_m1[i], wt_m2[i], val[i], para)
                for i in range(len(wt_m1))
            ]
        )

    t_min = para["t_min"]
    t_max = para["t_max"]

    @njit
    def _internal(wt_m1, wt_m2, val, t_min, t_max):
        output = []
        for i in range(len(wt_m1)):
            tmp = []
            for k in range(len(wt_m2)):
                if (wt_m2[k] > wt_m1[i] + t_min) & (wt_m2[k] < wt_m1[i] + t_max):
                    tmp.append(val[k])
            output.append(tmp)
        return output

    return ak.Array(_internal(wt_m1, wt_m2, val, t_min, t_max))


def m_coincidence_window(para, input, output, pv):
    """
    Coincidence Window module for the postprocessing pipeline.

    Given two lists of windowed data and a list of values associated with the second windowed data, the module compares the time windows of the second list with the first list and  generates lists of values that are within the coincidence region relative to the first time window.
    A dimension is added to the output arrays.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        - t_min (float): minimum time difference for coincidence.
        - t_max (float): maximum time difference for coincidence.

    input (list): List of input parameters in the following order:
        - w_t_1: Name of first windowed time array.
        - w_t_2: Name of second windowed time array.
        - val: Name of value array.

    output (list): List of output parameters in the following order:
        - val: Array of coincident events values.

    pv (dict): Dictionary to store the processed values.

    """
    in_n = {
        "w_t_1": input[0],
        "w_t_2": input[1],
        "val": input[2],
    }

    out_n = {"val": output[0]}

    pv[out_n["val"]] = generate_output(
        pv[in_n["w_t_1"]], pv[in_n["w_t_2"]], pv[in_n["val"]], para
    )
