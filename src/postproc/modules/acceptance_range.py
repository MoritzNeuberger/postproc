from __future__ import annotations


def m_acceptance_range(para, input, output, pv):
    """
    Threshold module for the postprocessing pipeline.

    Returns a awkward array of boolean values in the same shape as the input array, indicating whether the input values are within the threshold range.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        - thr (list): List containing two float values representing the lower and upper energy thresholds.

    input (dict): Dictionary containing input parameters.
        required:
        - val: Name of the values array.

    output (dict): Dictionary containing output parameters.
        required:
        - val: Array of boolean values indicating whether the input values are within the threshold.

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

    required_para = ["thr"]
    for r in required_para:
        if r not in para:
            text = f"Required parameter {r} not found in para. All required parameters are {required_para}."
            raise ValueError(text)

    pv[output["val"]] = (pv[input["val"]] > para["thr"][0]) * (
        pv[input["val"]] < para["thr"][1]
    ) > 0
