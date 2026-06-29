import numpy as np

from tele_neuron.collision import resolve_collisions, spatial_hash_candidates


def test_spatial_hash_candidates_finds_nearby_pair() -> None:
    positions = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [3.0, 3.0, 3.0]])

    assert (0, 1) in spatial_hash_candidates(positions, cell_size=0.5)


def test_collision_response_separates_and_changes_velocity() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=np.float64)
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([1.0, 1.0], dtype=np.float64)

    collisions = resolve_collisions(
        positions=positions,
        velocities=velocities,
        masses=masses,
        radius=0.1,
        cell_size=0.5,
        restitution=1.0,
    )

    assert collisions == 1
    assert positions[1, 0] - positions[0, 0] >= 0.2
    assert velocities[0, 0] < 0
    assert velocities[1, 0] > 0
