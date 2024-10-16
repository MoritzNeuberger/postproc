from __future__ import annotations

import awkward as ak
import numpy as np
from legendmeta import LegendMetadata

lmeta = LegendMetadata()
chmap = lmeta.channelmap()


def sensVolID_to_detName(sensVolID):
    sensVolID = sensVolID % 10000
    pos = sensVolID % 100
    string = (sensVolID - pos) // 100
    try:
        return (
            chmap.group("system")
            .geds.group("location.string")[string]
            .group("location.position")[pos]
            .name
        )
    except KeyError:
        return ""


def get_analysis_runs():
    output = []
    for key in lmeta.dataprod.config.analysis_runs:
        for run in lmeta.dataprod.config.analysis_runs[key]:
            output.append({"p": key, "r": run})
    return output


def run_info(runType="phy"):
    ana_run = get_analysis_runs()
    output = []
    for run in ana_run:
        if runType in lmeta.dataprod.runinfo[run["p"]][run["r"]]:
            output.append(
                {
                    "p": run["p"],
                    "r": run["r"],
                    "start_key": lmeta.dataprod.runinfo[run["p"]][run["r"]][runType][
                        "start_key"
                    ],
                    "livetime_in_s": lmeta.dataprod.runinfo[run["p"]][run["r"]][
                        runType
                    ]["livetime_in_s"],
                }
            )

    return output


def format_run_info(r_info):
    total_livetime = np.sum([run["livetime_in_s"] for run in r_info])
    cum_rel_livetime = 0
    for i in range(len(r_info)):
        r_info[i]["relative_livetime"] = r_info[i]["livetime_in_s"] / total_livetime
        r_info[i]["relative_livetime_start"] = cum_rel_livetime / total_livetime
        cum_rel_livetime += r_info[i]["livetime_in_s"]


def get_start_key_from_relative(rel, r_info):
    for i in range(len(r_info) - 1):
        if (rel >= r_info[i]["relative_livetime_start"]) and (
            rel < r_info[i + 1]["relative_livetime_start"]
        ):
            return r_info[i]["start_key"]
    return None


def get_run_from_relative(rel, r_info):
    for i in range(len(r_info) - 1):
        if (rel >= r_info[i]["relative_livetime_start"]) and (
            rel < r_info[i + 1]["relative_livetime_start"]
        ):
            return r_info[i]
    return r_info[-1]


def generate_usability_maps(r_info):
    def generate_usability_map(run):
        output = {}
        lmeta = LegendMetadata()
        chmap = lmeta.channelmap(run["start_key"])
        det_names = chmap.group("system").geds.group("name").keys()
        for det in det_names:
            output[det] = chmap[det].analysis.usability
        return output

    for i in range(len(r_info)):
        r_info[i]["usability_map"] = generate_usability_map(r_info[i])


# @jit (forceobj=True, liftloops=False)
def process_off_ac_one_det(edep, voln, evt_rpos, r_info):
    detName = sensVolID_to_detName(voln)
    usability = get_run_from_relative(evt_rpos, r_info)["usability_map"][detName]
    if usability == "off":
        return 0
    # elif usability == "ac":
    #    if edep > 0.025:
    #        return -1
    #    else:
    #        return 0
    return edep


# @jit (forceobj=True, liftloops=False)
def process_off_ac_one_run(v_edep_e, v_voln_e, evt_rpos, r_info):
    output = []
    for i_wind in range(len(v_edep_e)):
        tmp = []
        for i_det in range(len(v_edep_e[i_wind])):
            tmp.append(
                process_off_ac_one_det(
                    v_edep_e[i_wind][i_det], v_voln_e[i_wind][i_det], evt_rpos, r_info
                )
            )
        output.append(tmp)
    return output


# @jit (forceobj=True, liftloops=False)
def process_off_ac_all_runs(v_edep_e, v_voln_e, v_evt_rpos, r_info):
    v_edep_em = []
    for i in range(len(v_edep_e)):
        v_edep_em.append(
            process_off_ac_one_run(v_edep_e[i], v_voln_e[i], v_evt_rpos[i], r_info)
        )
    return ak.Array(v_edep_em)


class RunInfoSingleton:
    _instance = None
    _r_info = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_r_info(cls):
        if cls._r_info is None:
            cls._r_info = run_info()
            format_run_info(cls._r_info)
            generate_usability_maps(cls._r_info)
        return cls._r_info


def m_detector_active_time(para, input, output, pv):  # noqa: ARG001
    """
    Processes the detector active time by generating usability maps and applying it to the energy.
    Parameters:
    para (dict): Parameters for the detector active time processing.
    input (dict): Dictionary containing required input data with keys "edep" and "vol".
    output (dict): Dictionary containing required output data with key "edep".
    pv (dict): Dictionary containing the processed values for input and output data.
    Raises:
    ValueError: If required inputs or outputs are not found in the provided dictionaries.
    RuntimeError: If run information is not properly initialized.
    Returns:
    None: The function updates the `pv` dictionary in place with processed output data.
    """

    required_input = ["edep", "vol"]
    for r in required_input:
        if r not in input:
            text = f"Required input {r} not found in input. All required inputs are {required_input}."
            raise ValueError(text)

    required_output = ["edep"]
    for r in required_output:
        if r not in output:
            text = f"Required output {r} not found in output. All required outputs are {required_output}."
            raise ValueError(text)

    r_info = RunInfoSingleton.get_instance().get_r_info()

    if "usability_map" not in r_info[0]:
        text = "r_info was not properly initialized."
        raise RuntimeError(text)

    v_id = np.arange(len(pv[input["edep"]]))
    v_relpos = v_id / len(v_id)
    pv[output["edep"]] = process_off_ac_all_runs(
        pv[input["edep"]], pv[input["vol"]], v_relpos, r_info
    )
