from __future__ import annotations

from pathlib import Path
import importlib.util


ALGORITHM_NAME = "Novelity Search"


def train(context):
    path = Path(__file__).with_name("novelty_search.py")
    spec = importlib.util.spec_from_file_location("tele_neuron_novelty_alias", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train(context)
