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

    t_min = para["coincidence_gate"][0]
    t_max = para["coincidence_gate"][1]

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
        - coincidence_gate (list): List containing two float values representing the lower and upper time thresholds for the coincidence region.

    input (dict): Dictionary containing input parameters.
        required:
        - w_t_1: Name of the first time windowed data.
        - w_t_2: Name of the second time windowed data.
        additional:
        - additional values: Name of the additional values associated with the second time windowed data.

    output (dict): Dictionary containing output parameters.
        additional:
        - additional values: Name of the output values that are within the coincidence region. One output array for each additional input array.

    pv (dict): Dictionary to store the processed values.

    """

    required_input = ["w_t_1", "w_t_2"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    if len(input) <= 2:
        text = "Required at least one input val in input."
        raise ValueError(text)

    if len(output) != len(input) - 2:
        text = "For each val in input, a corresponding val in output is required."
        raise ValueError(text)

    for r in output:
        if r not in input:
            text = f"Output {r} not found in input."
            raise ValueError(text)

        pv[output[r]] = generate_output(
            pv[input["w_t_1"]], pv[input["w_t_2"]], pv[input[r]], para
        )
