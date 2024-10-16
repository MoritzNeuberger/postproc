from __future__ import annotations

import gc

import awkward as ak
import h5py
import numpy as np
import uproot
from tqdm import tqdm


class data_manager:
    def __init__(self, inst, infile, outfile, pm, task_id):
        self.inst = inst
        self.infile = infile
        self.infile_format = inst["io"]["input"]["format"]
        self.outfile = outfile
        self.module_manager = pm
        self.output_dict = {}
        self.task_id = task_id
        for key in self.inst["output"]:
            self.output_dict[key] = []
        if self.infile_format == "root":
            self.ttree = uproot.open(self.infile)[self.inst["input"]["tree"]]
        elif self.infile_format == "hdf5":
            with h5py.File(self.infile, "r") as f:
                group = f[self.inst["input"]["base_name"]]
                reconstituted = ak.from_buffers(
                    ak.forms.from_json(group.attrs["form"]),
                    group.attrs["length"],
                    {k: np.asarray(v) for k, v in group.items()},
                )
            self.ttree = ak.Array(reconstituted)

    def process_data(self):
        if self.infile_format == "root":
            n_entries = self.ttree.num_entries
            pbar = tqdm(total=n_entries, position=self.task_id)
            for batch, report in self.ttree.iterate(
                step_size=self.inst["para"]["step_size"], report=True
            ):
                processing_variables = {
                    key: batch[value.rsplit("/")[-1]]
                    for key, value in self.inst["input"]["var"].items()
                }
                self.module_manager.run(processing_variables, pbar, self.task_id)
                for key in self.output_dict:
                    self.output_dict[key].extend(processing_variables[key])
                del processing_variables
                gc.collect()
                pbar.update(report.stop - report.start)
            self.output_dict = ak.Array(self.output_dict)
            pbar.close()

        elif self.infile_format == "hdf5":
            n_entries = len(self.ttree)
            pbar = tqdm(total=n_entries, position=self.task_id)
            # for entry in self.ttree:
            #    pbar.update(1)
            processing_variables = {
                key: self.ttree[value]
                for key, value in self.inst["input"]["var"].items()
            }
            self.module_manager.run(processing_variables, pbar, self.task_id)
            for key in self.output_dict:
                self.output_dict[key].extend(processing_variables[key])
            del processing_variables
            gc.collect()
            self.output_dict = ak.Array(self.output_dict)
            pbar.close()

    def write_output(self):
        with h5py.File(self.outfile, "w") as f:
            group = f.create_group("awkward")
            form, length, container = ak.to_buffers(
                ak.to_packed(self.output_dict), container=group
            )
            group.attrs["form"] = form.to_json()
            group.attrs["length"] = length

        del self.output_dict
        gc.collect()
