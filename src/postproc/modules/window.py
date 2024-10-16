from __future__ import annotations

import awkward as ak
from numba import jit
from numba.typed import List


def subtract_smallest_time(t, t_all):
    if t.ndim == 1:
        return t - ak.min(t_all)
    t_min = ak.min(t_all, axis=1)
    return t - t_min


def define_windows(t_sub, dT):
    """
    Define time windows for the given time array.
    Assumes t_sub is a numba typed list.
    """

    @jit(forceobj=True, looplift=True)
    def _internal(t_sub, dT):
        output = []
        last_time = 0.0
        for i in range(len(t_sub)):
            if len(output) == 0:
                output.append(t_sub[i])
                last_time = t_sub[i]
            else:
                diff = last_time + dT
                if t_sub[i] > diff:
                    output.append(t_sub[i])
                    last_time = t_sub[i]
        return output

    @jit(forceobj=True, looplift=True)
    def recursion_function(t_sub, dT):
        if isinstance(t_sub[0], list):
            output = []
            for i in range(len(t_sub)):
                output.append(_internal(t_sub[i], dT))
            return output
        return _internal(t_sub, dT)

    return recursion_function(t_sub, dT)


def generate_map(t_sub, w_t):
    """
    Define time windows for the given time array.
    Assumes t_sub and w_t are numba typed lists.
    """

    @jit(forceobj=True, looplift=True)
    def _internal(t_sub, w_t):
        output = []
        for k in range(len(t_sub)):
            for j in range(len(w_t) - 1):
                if t_sub[k] >= w_t[j] and t_sub[k] < w_t[j + 1]:
                    output.append(j)
            if t_sub[k] >= w_t[-1]:
                output.append(len(w_t) - 1)
        return output

    @jit(forceobj=True, looplift=True)
    def _recursion_function(t_sub, w_t):
        if isinstance(t_sub[0], (list, List)):
            return [_recursion_function(t_sub[i], w_t[i]) for i in range(len(t_sub))]
        return _internal(t_sub, w_t)

    return _recursion_function(t_sub, w_t)


def generate_windowed_hits(mapping, v_in):
    """
    Define time windows for the given time array.
    Assumes mapping and v_in are numba typed lists.
    """

    @jit(looplift=True)
    def _internal(mapping, v_in):
        list_in_indices = []
        for i in range(len(mapping)):
            if mapping[i] not in list_in_indices:
                list_in_indices.append(mapping[i])
        output = []
        for i in range(len(list_in_indices)):
            output.append([v_in[j] for j in range(len(mapping)) if mapping[j] == i])
        return output

    @jit(forceobj=True, looplift=True)
    def _recursion_function(mapping, v_in):
        if isinstance(v_in[0], list):
            return [
                _recursion_function(mapping[i], v_in[i]) for i in range(len(mapping))
            ]
        return _internal(mapping, v_in)

    return _recursion_function(mapping, v_in)


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
    w_t = define_windows(ak.sort(t_sub), para["dT"])
    map = generate_map(t_sub, w_t)

    pv[output["t_sub"]] = ak.Array(generate_windowed_hits(map, t_sub))
    pv[output["w_t"]] = ak.Array(w_t)

    for key in input:
        if key in output:
            pv[output[key]] = ak.Array(generate_windowed_hits(map, pv[input[key]]))
