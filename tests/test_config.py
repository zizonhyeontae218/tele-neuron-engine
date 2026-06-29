from pathlib import Path

import pytest

from tele_neuron.config import load_config, parse_config


ROOT = Path(__file__).resolve().parents[1]


def test_loads_tiny_configs() -> None:
    and_config = load_config(ROOT / "configs" / "and_tiny.json")
    or_config = load_config(ROOT / "configs" / "or_tiny.json")
    xor_config = load_config(ROOT / "configs" / "xor_tiny.json")
    first_model_config = load_config(ROOT / "configs" / "xor_first_model.json")

    assert and_config.task == "AND"
    assert or_config.task == "OR"
    assert xor_config.task == "XOR"
    assert first_model_config.name == "tele_neuron_001_xor"
    assert len(and_config.cases) == 4
    assert len(xor_config.input.points) == 2


def test_config_validation_rejects_bad_bits() -> None:
    raw = {
        "name": "bad",
        "task": "BAD",
        "space": {"size": [4, 4, 3]},
        "simulation": {
            "steps": 1,
            "dt": 1,
            "damping": 1,
            "restitution": 1,
            "cell_size": 1,
            "seed": 1,
        },
        "balls": {"count": 1, "mass": 1, "radius": 0.1, "initial_speed": 0},
        "input": {"points": [[0, 0, 0]], "strength": 0, "epsilon": 0.001},
        "readout": [
            {
                "name": "out",
                "center": [0, 0, 0],
                "half_extents": [1, 1, 1],
                "threshold": 1,
            }
        ],
        "cases": [{"bits": [2], "target": 0}],
    }

    with pytest.raises(ValueError, match="bits"):
        parse_config(raw)
