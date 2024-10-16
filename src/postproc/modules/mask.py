from __future__ import annotations


def m_mask(para, input, output, pv):  # noqa: ARG001
    """
    Mask module for the postprocessing pipeline.

    Given an awkward array, it returns a awkward array with the maximum values of the lowest dimension.
    Reduces the dimension of the array by one.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        None necessary.

    input (dict): Dictionary containing input parameters.
        required:
        - mask: Name of array of boolean values indicating whether the input values are within the threshold.
        additional:
        - val: Names of arbitrary number of array.

    output (dict): Dictionary containing output parameters.
        additional:
        - val: Names arbitrary number of array.

    pv (dict): Dictionary to store the processed values.

    """

    required_input = ["mask"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    additional_input = {i: input[i] for i in input if i not in required_input}

    for r in additional_input:
        if r not in output:
            text = f"All additional input parameters must have an output parameter. {r} not found in output."
            raise ValueError(text)

    for r in additional_input:
        pv[output[r]] = pv[input[r]][pv[input["mask"]]]
