import numpy as np

from tele_neuron.config import parse_config
from tele_neuron.readout import readout_bits, readout_counts
from tele_neuron.trainer import run_cases


def _multi_config() -> dict:
    return {
        "name": "multi",
        "task": "MULTI",
        "space": {"size": [4, 4, 3]},
        "simulation": {
            "steps": 1,
            "dt": 1,
            "damping": 1,
            "restitution": 1,
            "cell_size": 0.5,
            "seed": 1,
        },
        "balls": {"count": 8, "mass": 1, "radius": 0.05, "initial_speed": 0},
        "input": {"points": [[0.5, 0.5, 1], [3.5, 0.5, 1]], "strength": 0, "epsilon": 0.001},
        "readout": [
            {"name": "out1", "center": [1, 1, 1], "half_extents": [0.5, 0.5, 1], "threshold": 1},
            {"name": "out2", "center": [3, 3, 1], "half_extents": [0.5, 0.5, 1], "threshold": 1},
        ],
        "cases": [
            {"bits": [0, 0], "target": [0, 0]},
            {"bits": [1, 0], "target": [1, 0]},
        ],
    }


def test_vector_target_config_parses() -> None:
    config = parse_config(_multi_config())

    assert config.cases[0].target == (0, 0)
    assert len(config.readout) == 2


def test_multi_readout_bits_returns_one_output_per_zone() -> None:
    config = parse_config(_multi_config())
    positions = np.array([[1, 1, 1], [3, 3, 1], [0, 0, 0]], dtype=np.float64)

    assert readout_counts(positions, config) == (1, 1)
    assert readout_bits(positions, config) == (1, 1)


def test_run_cases_uses_full_target_tuple() -> None:
    config = parse_config(_multi_config())

    result = run_cases(config)

    assert len(result.cases[0].output) == 2
    assert result.cases[0].correct is not None
