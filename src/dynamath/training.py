"""Shared deterministic setup, evidence serialization and address loss."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def dump(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")


def setup(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def prediction_loss(model, state, targets):
    logits = model.logits(state)
    selected = model.output(state).gather(1, targets[:, None]).squeeze(1)
    # A fixed 0.25 margin teaches positive output-address threshold eligibility.
    return F.cross_entropy(logits, targets) + 0.1 * F.relu(0.25 - selected).mean()
