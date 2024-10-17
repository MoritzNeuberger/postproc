from __future__ import annotations

import argparse

import process_manager
from misc import load_inst


def main(infile, overwrite):
    inst = load_inst(infile)
    pm = process_manager.process_manager(inst, overwrite=overwrite)
    pm.run_processes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="post_proc", description="A generic post processor for Geant4 simulation"
    )
    parser.add_argument("input_file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()
    main(args.input_file, args.overwrite)
