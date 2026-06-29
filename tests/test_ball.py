import numpy as np
import pytest

from tele_neuron.ball import BallState


def test_ball_state_from_vectors_round_trips_arrays() -> None:
    ball = BallState.from_vectors(
        id=1,
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([0.1, 0.2, 0.3]),
        mass=2.0,
        radius=0.5,
    )

    np.testing.assert_array_equal(ball.position, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ball.velocity, np.array([0.1, 0.2, 0.3]))


def test_ball_state_rejects_invalid_physical_values() -> None:
    with pytest.raises(ValueError, match="mass"):
        BallState(1, 0, 0, 0, 0, 0, 0, 0, 1)

    with pytest.raises(ValueError, match="radius"):
        BallState(1, 0, 0, 0, 0, 0, 0, 1, 0)
