from __future__ import annotations

import awkward as ak
import numpy as np
from numba import njit
from numba.typed import List

from .misc import python_list_to_numba_list


def subtract_smallest_time(t, t_all):
    @njit
    def _internal(t, min_t_all):
        return [time - min_t_all for time in t]

    @njit
    def recursion_function(t, t_all):
        if isinstance(t[0], (list, List)):
            return List([recursion_function(t[i], t_all[i]) for i in range(len(t))])
        min_t_all = min(t_all)
        return _internal(t, min_t_all)

    t = python_list_to_numba_list(ak.Array(t).to_list())
    t_all = python_list_to_numba_list(ak.Array(t_all).to_list())
    return ak.Array(recursion_function(t, t_all))


def define_windows(t_sub, dT):
    @njit
    def _internal(t_sub, dT):
        t_sub_sort = List(np.sort(t_sub))
        output = List()
        last_time = 0.0
        for i in range(len(t_sub_sort)):
            if len(output) == 0:
                output.append(t_sub_sort[i])
                last_time = t_sub_sort[i]
            else:
                diff = last_time + dT
                if t_sub_sort[i] > diff:
                    output.append(t_sub_sort[i])
                    last_time = t_sub_sort[i]
        return output

    @njit
    def recursion_function(t_sub, dT):
        if isinstance(t_sub[0], List):
            output = List()
            for i in range(len(t_sub)):
                output.append(_internal(t_sub[i], dT))
            return output
        return _internal(t_sub, dT)

    t_sub_list = python_list_to_numba_list(ak.to_list(t_sub))

    return ak.Array(recursion_function(t_sub_list, dT))


def generate_map(t_sub, w_t):
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

    @njit
    def _recursion_function(t_sub, w_t):
        if isinstance(t_sub[0], (list, List)):
            return List(
                [_recursion_function(t_sub[i], w_t[i]) for i in range(len(t_sub))]
            )
        return _internal(t_sub, w_t)

    t_sub = python_list_to_numba_list(ak.Array(t_sub).to_list())
    w_t = python_list_to_numba_list(ak.Array(w_t).to_list())

    return ak.Array(_recursion_function(t_sub, w_t))


def generate_windowed_hits(mapping, v_in):
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

    @njit
    def _recursion_function(mapping, v_in):
        if isinstance(v_in[0], (list, List)):
            return List(
                [_recursion_function(mapping[i], v_in[i]) for i in range(len(mapping))]
            )
        return _internal(mapping, v_in)

    mapping = python_list_to_numba_list(ak.Array(mapping).to_list())
    v_in = python_list_to_numba_list(ak.Array(v_in).to_list())

    return ak.Array(_recursion_function(mapping, v_in))


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
