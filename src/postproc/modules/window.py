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
        - dT (float): Time window duration.

    input (list): List of input parameters in the following order:
        - t_all: Name of all times array.
        - t: Name of times array.
        - edep: Name of energy depositions array.
        - vol: Name of volumes array.
        - posx: Name of x positions array.
        - posy: Name of y positions array.
        - posz: Name of z positions array.

    output (list): List of output parameters in the following order:
        - w_t: Array of window times array.
        - t_sub: Array of subtracted times array.
        - edep: Array of energy depositions array.
        - vol: Array of volumes array.
        - posx: Array of x positions array.
        - posy: Array of y positions array.
        - posz: Array of z positions array.

    pv (dict): Dictionary to store the processed values.

    """

    in_n = {
        "t_all": input[0],
        "t": input[1],
        "edep": input[2],
        "vol": input[3],
        "posx": input[4],
        "posy": input[5],
        "posz": input[6],
    }

    out_n = {
        "w_t": output[0],
        "t_sub": output[1],
        "edep": output[2],
        "vol": output[3],
        "posx": output[4],
        "posy": output[5],
        "posz": output[6],
    }

    t_sub = subtract_smallest_time(pv[in_n["t"]], pv[in_n["t_all"]])
    w_t = define_windows(t_sub, para["dT"])
    map = generate_map(t_sub, w_t)

    pv[out_n["t_sub"]] = generate_windowed_hits(map, t_sub)
    pv[out_n["w_t"]] = w_t

    for key in list(out_n.keys())[2:]:
        pv[out_n[key]] = generate_windowed_hits(map, pv[in_n[key]])
