from tele_neuron.config import load_config
from tele_neuron.trainer import run_cases, search_best_seed


def test_run_cases_returns_all_config_cases() -> None:
    config = load_config("configs/and_tiny.json")

    result = run_cases(config)

    assert result.total == 4
    assert len(result.cases) == 4


def test_search_best_seed_returns_scored_result() -> None:
    config = load_config("configs/or_tiny.json")

    result = search_best_seed(config, seed_start=0, seed_end=2)

    assert result.total == 4
    assert 0 <= result.score <= 4
