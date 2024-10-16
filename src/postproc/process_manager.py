from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
from process import run_post_proc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# logging.basicConfig(level=logging.DEBUG)


class process_manager:
    def __init__(self, inst, overwrite=False):
        self.in_folder = inst["io"]["input"]["folder"]
        self.in_format = inst["io"]["input"]["format"]
        self.out = inst["io"]["output"]
        self.overwrite = overwrite
        self.threads = inst["para"]["threads"]
        self.mode = inst["para"].get("mode", "")

        self.log_initialization()

        # Get input files and corresponding output files
        self.input_files = list(Path(self.in_folder).glob("*." + self.in_format))
        if self.mode != "summarize":
            self.output_files = [
                Path(self.out).joinpath(infile.stem + ".hdf5")
                for infile in self.input_files
            ]
        else:
            self.tmp_dir = tempfile.mkdtemp()
            self.output_files = [
                Path(self.tmp_dir).joinpath(infile.stem + ".hdf5")
                for infile in self.input_files
            ]
            # self.output_files = self.out

        # Filter out files that already exist if overwrite is False
        if not self.overwrite:
            mask_doesnt_exist = np.array(
                [not Path.exists(f) for f in self.output_files]
            )
            self.input_files = np.array(self.input_files)[mask_doesnt_exist].tolist()
            self.output_files = np.array(self.output_files)[mask_doesnt_exist].tolist()

        # Create a list of arguments: each is a tuple (input_file, output_file, inst)
        self.args = list(
            zip(
                self.input_files,
                self.output_files,
                [inst] * len(self.input_files),
                np.arange(len(self.input_files)),
            )
        )

    def log_initialization(self):
        logging.info("Process manager initialized with the following parameters:")
        logging.info("Input folder: %s", self.in_folder)
        logging.info("Input format: %s", self.in_format)
        logging.info("Output folder: %s", self.out)
        logging.info("Overwrite: %s", self.overwrite)
        logging.info("Threads: %s", self.threads)
        logging.info("Mode: %s", self.mode)

    def summarize(self):
        def gen_files():
            for file in self.output_files:
                with h5py.File(file, "r") as f:
                    group = f["awkward"]
                    reconstituted = ak.from_buffers(
                        ak.forms.from_json(group.attrs["form"]),
                        group.attrs["length"],
                        {k: np.asarray(v) for k, v in group.items()},
                    )
                    yield ak.Array(reconstituted)

        with h5py.File(self.out, "w") as f:
            group = f.create_group("awkward")
            form, length, container = ak.to_buffers(
                ak.to_packed(ak.concatenate(ak.from_iter(gen_files()))), container=group
            )
            group.attrs["form"] = form.to_json()
            group.attrs["length"] = length

    def run_processes(self):
        if self.threads > 1:
            logging.debug(
                "Running with multiprocessing. Number of threads: %d", self.threads
            )
            # Use multiprocessing to run run_post_proc with the arguments
            with ProcessPoolExecutor(
                max_workers=self.threads, max_tasks_per_child=1
            ) as executor:
                # Unpack the arguments using *args
                futures = [executor.submit(run_post_proc, arg) for arg in self.args]
                # iterate over all submitted tasks and get results as they are available
                for future in as_completed(futures):
                    try:
                        result = future.result()  # blocks
                        logging.debug("Process completed with result: %s", result)
                    except Exception as e:
                        logging.error("Process raised an exception: %s", e)
            # results = executor.map(lambda args: run_post_proc(*args), self.args)
        else:
            for i in range(len(self.args)):
                run_post_proc(self.args[i])

        if self.mode == "summarize":
            self.summarize()
