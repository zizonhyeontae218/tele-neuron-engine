from tele_neuron.config import load_config
from tele_neuron.trainer import run_cases


def test_same_seed_produces_same_case_results() -> None:
    config = load_config("configs/xor_first_model.json")

    first = run_cases(config)
    second = run_cases(config)

    assert first == second
