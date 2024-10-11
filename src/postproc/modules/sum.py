from __future__ import annotations

import awkward as ak


def m_sum(para, input, output, pv):  # noqa: ARG001
    """
    Sum module for the postprocessing pipeline.

    Calculates the sum of the lowest dimension of the input array.
    Reduces one dimension of the array.

    Parameters:
    para (dict): Dictionary containing parameters for the module.

    input (list): List of input parameters in the following order:
        - val: Name of array.

    output (list): List of output parameters in the following order:
        - total_val: Total value.

    pv (dict): Dictionary to store the processed values.

    """
    # Module implementation
    in_n = {"val": input[0]}

    out_n = {"val": output[0]}

    pv[out_n["val"]] = ak.sum(pv[in_n["val"]], axis=-1)
