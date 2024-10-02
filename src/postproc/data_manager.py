from __future__ import annotations

import awkward as ak
import h5py
import uproot
from tqdm import tqdm


class data_manager:
    def __init__(self, inst, infile, outfile, pm, task_id):
        self.inst = inst
        self.infile = infile
        self.outfile = outfile
        self.processor_manager = pm
        self.output_dict = {}
        self.task_id = task_id
        for key in self.inst["output"]:
            self.output_dict[key] = []
        self.ttree = uproot.open(self.infile)[self.inst["input"]["tree"]]

    def process_data(self):
        n_entries = self.ttree.num_entries
        pbar = tqdm(total=n_entries, position=self.task_id)
        for batch, report in self.ttree.iterate(
            step_size=self.inst["para"]["step_size"], report=True
        ):
            pbar.update(report.stop - report.start)
            processing_variables = {
                key: batch[value.rsplit("/")[-1]]
                for key, value in self.inst["input"]["var"].items()
            }
            self.processor_manager.run(processing_variables, pbar, self.task_id)
            for key in self.output_dict:
                self.output_dict[key].extend(processing_variables[key])
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