"""Invariants of cyclic address dynamics, events and exact checkpoint resume."""
import pytest

torch = pytest.importorskip("torch")

from tele_neuron.dynamath import Config, DynaMath, Session
from tele_neuron.dynamath.training import train_text


def model():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    return DynaMath(Config("ab", size=12, rank=3))


def test_additive_signed_duplicate_addresses_and_phase_order():
    m = model()
    initial = torch.ones(1, 12)
    added = m.inject(initial, [[0, 0, 1]], -0.25)
    assert added[0, 0] == 0.5 and added[0, 1] == 0.75
    assert torch.equal(initial, torch.ones_like(initial))
    s = Session(m)
    s.enqueue("aab")
    expected = m.inject(s.state, [[0, 0, 1]])
    for phase in range(10):
        expected = m.transition(expected, phase)
        s.tick(1, emit=False)
        assert torch.equal(expected, s.state)
    assert s.ticks == 10


def test_event_does_not_mutate_state_tie_and_refractory():
    s = Session(model(), threshold=0.1, refractory=5, io_ticks=5)
    s.ticks = 5
    s.state[0, 2:4] = 0.5
    before = s.state.clone()
    assert s.poll() == ["a"]  # lowest address resolves equal winners
    assert torch.equal(s.state, before)
    assert s.poll() == []
    s.ticks = 9
    assert s.poll() == []
    s.ticks = 10
    assert s.poll() == ["a"]  # repeated letters allowed after refractory


def test_idle_progress_and_snapshot_pending_queue_phase(tmp_path):
    s = Session(model(), threshold=-1, refractory=5, io_ticks=5)
    s.consume("ab")
    before = s.state.clone()
    s.tick(2)
    assert not torch.equal(s.state, before)
    s.enqueue("ab")
    path = tmp_path / "session.pt"
    s.save(path)
    restored = Session.load(path)
    assert restored.ticks == 12 and restored.pending == s.pending
    assert s.tick(23) == restored.tick(23)
    assert torch.equal(s.state, restored.state)
    assert s.last_event == restored.last_event


def test_generation_uses_learned_state_and_additive_feedback():
    m = model()
    actual = Session(m, threshold=-1, refractory=5, io_ticks=5)
    manual = Session(m, threshold=-1, refractory=5, io_ticks=5)
    result = actual.generate("ab", max_chars=5, max_ticks=100, stop="")
    manual.consume("ab")
    expected = manual.poll()
    while len(expected) < 5:
        manual.enqueue(expected[-1])
        expected.extend(manual.tick(5))
    assert result == "".join(expected)
    assert torch.equal(actual.state, manual.state)
    assert actual.ticks == 30
    saved = actual.state.clone()
    assert actual.generate("", max_chars=0) == ""
    assert torch.equal(actual.state, saved)


def test_silence_limits_and_input_validation():
    s = Session(model(), threshold=1)
    assert s.generate("a", max_ticks=17) == ""
    assert s.ticks == 17
    assert s.pending == []
    before = s.state.clone()
    with pytest.raises(ValueError):
        s.generate("az")
    assert torch.equal(before, s.state)
    with pytest.raises(ValueError):
        s.tick(-1)
    with pytest.raises(ValueError):
        Config("aa")


def test_bounded_idle_after_large_signed_input():
    m = model()
    state = m.inject(m.zero(), [[0]], 1000)
    for phase in range(500):
        state = m.transition(state, phase)
    assert torch.isfinite(state).all()
    assert float(state.detach().abs().max()) <= 1.00001


def test_character_training_resume_is_exact(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("ab a ba ab a ba\n" * 3)
    val.write_text("ba ab a\n")
    partial = tmp_path / "partial.pt"
    resumed = tmp_path / "resumed.pt"
    full = tmp_path / "full.pt"
    train_text(train, val, partial, steps=2)
    train_text(train, val, resumed, steps=4, resume=partial)
    train_text(train, val, full, steps=4)
    a, pa = DynaMath.load(resumed)
    b, pb = DynaMath.load(full)
    for name in a.state_dict():
        assert torch.equal(a.state_dict()[name], b.state_dict()[name])
    assert torch.equal(pa["state"], pb["state"])
    assert torch.equal(pa["offsets"], pb["offsets"])
    assert pa["ticks"] == pb["ticks"] and pa["phase"] == 0
