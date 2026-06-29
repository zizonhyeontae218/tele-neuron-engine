from pathlib import Path

from tele_neuron.config import load_config
from tele_neuron.learner import list_algorithm_files, train_model_payload


ROOT = Path(__file__).resolve().parents[1]


def test_algorithm_files_are_discoverable() -> None:
    algorithms = {item["name"] for item in list_algorithm_files(ROOT)}

    assert "random_search" in algorithms
    assert "genetic_algorithm" in algorithms
    assert "differentiable_physics_approximation" in algorithms


def test_train_model_payload_creates_model_with_masses(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "and_tiny.json")
    algorithm_dir = tmp_path / "Vanila_learner" / "Algorithm"
    algorithm_dir.mkdir(parents=True)
    source_algorithm = ROOT / "Vanila_learner" / "Algorithm" / "random_search.py"
    (algorithm_dir / "random_search.py").write_text(source_algorithm.read_text(encoding="utf-8"), encoding="utf-8")
    config_payload = {
        "name": config.name,
        "task": config.task,
        "space": {"size": list(config.space.size)},
        "simulation": {
            "steps": config.simulation.steps,
            "dt": config.simulation.dt,
            "damping": config.simulation.damping,
            "restitution": config.simulation.restitution,
            "cell_size": config.simulation.cell_size,
            "seed": config.simulation.seed,
        },
        "balls": {
            "count": config.balls.count,
            "mass": config.balls.mass,
            "radius": config.balls.radius,
            "initial_speed": config.balls.initial_speed,
        },
        "input": {
            "points": [list(point) for point in config.input.points],
            "strength": config.input.strength,
            "epsilon": config.input.epsilon,
        },
        "readout": [
            {
                "name": zone.name,
                "center": list(zone.center),
                "half_extents": list(zone.half_extents),
                "threshold": zone.threshold,
            }
            for zone in config.readout
        ],
        "cases": [
            {"bits": list(case.bits), "target": list(case.target)}
            for case in config.cases
        ],
    }

    result = train_model_payload(
        tmp_path,
        config_payload,
        algorithm="random_search",
        settings_payload={
            "iterations": 1,
            "population": 2,
            "seed": 3,
            "min_mass": 0.7,
            "max_mass": 1.2,
            "mutation_scale": 0.05,
        },
    )

    assert result["result"]["total"] == 4
    assert len(result["model"]["masses"]) == config.balls.count
    assert (tmp_path / result["path"]).is_file()
