"""Check that the opt-in objective resists sacrificing an operation."""
import json
from pathlib import Path

import numpy as np
import pytest

from tele_neuron.endophalon3.config import parse_config
from tele_neuron.endophalon3.core import operation_balanced_loss


def test_balanced_selection_can_reject_a_better_pooled_loss():
    # Candidate A fits three operations exactly and sacrifices the fourth.
    # Candidate B has a modest, uniform residual on every operation.
    ops = np.repeat(np.arange(4), 4)
    targets = np.zeros(16)
    a = np.repeat([0.0, 0.0, 0.0, .8], 4)
    b = np.full(16, .45)
    old_a, _, _ = operation_balanced_loss(a, targets, ops)
    old_b, _, _ = operation_balanced_loss(b, targets, ops)
    new_a, _, _ = operation_balanced_loss(a, targets, ops, .5)
    new_b, _, _ = operation_balanced_loss(b, targets, ops, .5)
    assert old_a < old_b
    assert new_b < new_a
    # The default is exactly the original pooled expression, not an approximation.
    assert old_a == float(np.mean((a - targets) ** 2))


def test_operation_frequency_does_not_hide_the_weakest_loss():
    raw = np.array([.1] * 99 + [.9])
    ops = np.array([0] * 99 + [1])
    _, pooled, worst = operation_balanced_loss(raw, np.zeros(100), ops, .5)
    assert pooled < .02
    assert worst == pytest.approx(.81)


@pytest.mark.parametrize('weight', [-.01, 1.01])
def test_invalid_loss_weight_is_rejected_before_training(weight):
    path = Path(__file__).resolve().parents[1] / 'configs/endophalon3/level_a.json'
    raw = json.loads(path.read_text())
    raw['objective']['worst_op_weight'] = weight
    with pytest.raises(ValueError, match='worst_op_weight'):
        parse_config(raw)
