import numpy as np

from tele_neuron.config import load_config
from tele_neuron.readout import count_in_zone, readout_bits, readout_counts


def test_readout_count_and_threshold() -> None:
    config = load_config("configs/xor_first_model.json")
    zone = config.readout[0]
    positions = np.array(
        [
            zone.center,
            [zone.center[0] + 0.1, zone.center[1], zone.center[2]],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert count_in_zone(positions, zone) == 2
    assert readout_counts(positions, config) == (2,)
    assert readout_bits(positions, config) == (0,)
