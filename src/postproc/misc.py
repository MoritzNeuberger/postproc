from __future__ import annotations

import json
from pathlib import Path


def load_inst(file):
    with Path.open(file) as f:
        return json.load(f)