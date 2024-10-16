from __future__ import annotations

import awkward as ak


def m_max(para, input, output, pv):  # noqa: ARG001
    """
    Max module for the postprocessing pipeline.

    Given an awkward array, it returns a awkward array with the maximum values of the lowest dimension.
    Reduces the dimension of the array by one.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        None necessary.

    input (dict): Dictionary containing input parameters.
        required:
        - val: Name of the values array.

    output (dict): Dictionary containing output parameters.
        required:
        - val: Name of the summed values array.

    pv (dict): Dictionary to store the processed values.

    """

    required_input = ["val"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    required_output = ["val"]
    for r in required_output:
        if r not in output:
            text = f"Required output {r} not found in output. All required outputs are {required_output}."
            raise ValueError(text)

    pv[output["val"]] = ak.max(pv[input["val"]], axis=-1)
