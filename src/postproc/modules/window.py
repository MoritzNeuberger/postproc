from __future__ import annotations

import awkward as ak
from numba import njit
from numba.typed import List


def subtract_smallest_time(t, t_all):
    if isinstance(t[0], (list, ak.Array)):
        return ak.Array([subtract_smallest_time(t[i], t_all[i]) for i in range(len(t))])

    min_t_all = min(t_all)
    return ak.Array([time - min_t_all for time in t])


def define_windows(t_sub, dT=1e4):
    if isinstance(t_sub[0], (list, ak.Array)):
        return ak.Array([define_windows(t_sub[i], dT) for i in range(len(t_sub))])

    t_sub_sort = ak.sort(t_sub)
    output = []
    for i in range(len(t_sub_sort)):
        if len(output) == 0 or t_sub_sort[i] > output[-1] + dT:
            output.append(t_sub_sort[i])
    return ak.Array(output)


def generate_map(t_sub, w_t):
    if isinstance(t_sub[0], (list, ak.Array)):
        return ak.Array([generate_map(t_sub[i], w_t[i]) for i in range(len(t_sub))])

    @njit
    def _internal(t_sub, w_t):
        output = []
        for k in range(len(t_sub)):
            for j in range(len(w_t) - 1):
                if t_sub[k] >= w_t[j] and t_sub[k] < w_t[j + 1]:
                    output.append(j)
            if t_sub[k] >= w_t[-1]:
                output.append(len(w_t) - 1)
        return output

    return _internal(List(t_sub), List(w_t))


def generate_windowed_hits(mapping, v_in):
    if isinstance(v_in[0], (list, ak.Array)):
        return ak.Array(
            [generate_windowed_hits(mapping[i], v_in[i]) for i in range(len(mapping))]
        )

    @njit
    def _internal(mapping, v_in):
        list_in_indices = []
        for i in range(len(mapping)):
            if mapping[i] not in list_in_indices:
                list_in_indices.append(mapping[i])

        output = []
        for i in range(len(list_in_indices)):
            output.append([v_in[j] for j in range(len(mapping)) if mapping[j] == i])

        return output

    return ak.Array(_internal(List(mapping), List(v_in)))


def m_window(para, input, output, pv):
    """
    Windowing module for the postprocessing pipeline.

    Groups the input data into time windows according to the t array.
    A dimension is added to the output arrays.

    Parameters:
    para (dict): Dictionary containing parameters for the windowing module.
        required:
        - dT (float): Time window duration.

    input (dict): Dictionary containing input parameters for the windowing module.
        required:
        - t_all: Name of the full time array.
        - t: Name of the time array.
        additional:
        - arbitrary many additional input arrays.

    output (dict): Dictionary containing output parameters for the windowing module.
        required:
        - w_t: Name of the time window array.
        - t_sub: Name of the subtracted time array.
        additional:
        - arbitrary many additional output arrays. One for each additional input array.



    pv (dict): Dictionary to store the processed values.

    """

    required_input = ["t_all", "t"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    required_output = ["w_t", "t_sub"]
    for r in required_output:
        if r not in output:
            text = f"Required output {r} not found in output. All required outputs are {required_output}."
            raise ValueError(text)

    required_para = ["dT"]
    for r in required_para:
        if r not in para:
            text = f"Required parameter {r} not found in para. All required parameters are {required_para}."
            raise ValueError(text)

    if len(output) != len(input):
        text = "Number of input and output parameters must be the same."
        raise ValueError(text)

    for r in input:
        if r not in output and r not in required_input:
            text = f"All additional input parameters must have an output parameter. {r} not found in output."
            raise ValueError(text)

    t_sub = subtract_smallest_time(pv[input["t"]], pv[input["t_all"]])
    w_t = define_windows(t_sub, para["dT"])
    map = generate_map(t_sub, w_t)

    pv[output["t_sub"]] = generate_windowed_hits(map, t_sub)
    pv[output["w_t"]] = w_t

    for key in input:
        if key in output:
            pv[output[key]] = generate_windowed_hits(map, pv[input[key]])
