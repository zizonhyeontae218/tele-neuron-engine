import numpy as np

from tele_neuron.config import load_config
from tele_neuron.physics import initialize_state, step_state


def test_initialize_state_is_deterministic() -> None:
    config = load_config("configs/and_tiny.json")

    first = initialize_state(config)
    second = initialize_state(config)

    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.velocities, second.velocities)


def test_step_state_keeps_balls_inside_space() -> None:
    config = load_config("configs/and_tiny.json")
    state = initialize_state(config)

    for _ in range(8):
        step_state(state, config, (1, 1))

    upper = np.array(config.space.size) - config.balls.radius
    assert np.all(state.positions >= config.balls.radius)
    assert np.all(state.positions <= upper)
