"""Focused word/address, session, FIFO and exact restart invariants."""
import copy
import json
from pathlib import Path
import pytest
import torch
from dynamath.v003a import Model as Parent
from dynamath.v003b import Config, Model, WordTokenizer
from dynamath.v003b.data import LaneStream, remove_leakage, overlap
from dynamath.v003b.evaluation import generate, teacher_forced, corpus_generation
from dynamath.v003b.model import block
from dynamath.v003b.training import lr_at_update, run_training, seed_all


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def fixture():
    ds = [{"id": i, "turns": ["Hello " * (i+1), "I'm fine.", "How are you?", "Good!"]} for i in range(9)]
    tok = WordTokenizer.build([s for d in ds for s in d["turns"]])
    cfg = json.loads(Path("configs/dynamath/experiment-003b.json").read_text())
    cfg.update(batch=2, tbptt_slots=3, steps=8, warmup=2, checkpoint_interval=2,
               milestones=[2, 8], latent=4, rank=2)
    return ds, tok, cfg


def test_word_rules_and_metadata(tmp_path):
    tok = WordTokenizer.build(["I'm fine, don't worry!", "cafe\u0301 café."])
    assert tok.split("I'm fine, don't worry!") == ["I'm", "fine", ",", "don't", "worry", "!"]
    assert tok.decode(tok.encode("I'm fine, don't worry!")) == "I'm fine, don't worry!"
    assert tok.split("cafe\u0301") == ["café"]
    assert tok.encode("unseenword") == [4]
    assert tok.vocabulary[:5] == ["USER", "BOT", "EOS", "SILENCE", "UNK"]
    assert tok.vocabulary[5] == "café"
    assert tok.vocabulary[6:] == sorted(tok.vocabulary[6:])
    tok.save(tmp_path/"tokenizer.json")
    assert WordTokenizer.load(tmp_path/"tokenizer.json").sha256 == tok.sha256
    small = WordTokenizer.build(["b a a c"], limit=7)
    assert small.vocabulary[5:] == ["a", "b"]


def test_exact_003a_equations_and_addresses():
    _, tok, _ = fixture()
    config = Config(tok.vocabulary, memories=1, latent=4, rank=2)
    seed_all(43)
    parent = Parent(config).eval()
    seed_all(43)
    model = Model(config).eval()
    s, m = model.zero(2)
    ids = torch.tensor([5, 6])
    s = model.inject(s, ids)
    assert s[0, 5] == 1 and s[1, 6] == 1
    assert int(s.count_nonzero()) == 2
    w = len(tok.vocabulary)
    probe = s.clone()
    probe[:, w+7] = .5
    assert torch.equal(model.logits(probe)[:, 7], torch.tensor([3., 3.]))
    ps, pm = s.clone(), m.clone()
    for t in range(18):
        s, m, g = model.step(s, m, t)
        ps, pm, pg = parent.step(ps, pm, t)
        assert torch.equal(s, ps) and torch.equal(m, pm) and torch.equal(g, pg)


def test_clocks_reset_per_dialogue_and_keep_within():
    _, tok, _ = fixture()
    seed_all(43)
    model = Model(Config(tok.vocabulary, memories=1, latent=4, rank=2)).eval()
    s, m = model.zero(2)
    s += .2
    m += .1
    clocks = torch.tensor([12, 21])
    x = torch.tensor([5, 6])
    original_s, original_m = s.clone(), m.clone()
    ns, nm, clock, out = block(model, s, m, clocks, x, torch.tensor([True, False]))
    for lane, t in ((0, 0), (1, 21)):
        a, b = model.zero() if lane == 0 else (original_s[1:2], original_m[1:2])
        a = model.inject(a, x[lane:lane+1])
        for p in range(3):
            a, b, _ = model.step(a, b, t+p)
        assert torch.equal(ns[lane:lane+1], a)
        assert torch.equal(nm[lane:lane+1], b)
    assert clock.tolist() == [3, 24]
    called = []
    original = model.step
    def recording(s, m, t):
        called.append(t % 9)
        return original(s, m, t)
    model.step = recording
    s, m = model.zero()
    clocks = torch.zeros(1, dtype=torch.long)
    for slot in range(3):
        s, m, clocks, _ = block(model, s, m, clocks, torch.tensor([5]), torch.tensor([slot == 0]))
    assert called == list(range(9))


def test_lanes_no_shortest_truncation_and_leakage_priority():
    ds, tok, _ = fixture()
    stream = LaneStream(ds, tok, batch=2)
    count = 0
    for _ in range(int(stream.lengths.max())):
        _, _, _, active = stream.next(cycle=False)
        count += int(active.sum())
    assert count == stream.total_slots
    assert count == sum(len(tok.pairs({"Q": d["turns"][i], "A": d["turns"][i+1]})[0]) for d in ds for i in (0, 2))
    raw = {"test": [{"id": 0, "turns": ["a", "b"]}],
           "validation": [{"id": 1, "turns": ["a", "b", "c", "d"]}, {"id": 2, "turns": ["e", "f"]}],
           "train": [{"id": 3, "turns": ["e", "f"]}, {"id": 4, "turns": ["g", "h"]}]}
    clean, removed = remove_leakage(raw)
    assert [clean[k][0]["id"] for k in ("test", "validation", "train")] == [0, 2, 4]
    assert removed["validation"][0]["id"] == 1
    assert all(not any(v.values()) for v in overlap(clean).values())


def test_exact_restart_weights_optimizer_schedule_rng_sessions(tmp_path):
    ds, tok, cfg = fixture()
    def fresh():
        seed_all(43)
        return Model(Config(tok.vocabulary, memories=1, latent=4, rank=2))
    a = fresh()
    args = (cfg, tok.sha256, "data")
    full, fa = run_training(a, LaneStream(ds, tok, 2, shuffle=True), *args, tmp_path/"full")
    b = fresh()
    run_training(b, LaneStream(ds, tok, 2, shuffle=True), *args, tmp_path/"split", stop_at=4)
    b, p = Model.load(tmp_path/"split/latest.pt")
    resumed, fb = run_training(b, LaneStream(ds, tok, 2, shuffle=True), *args,
                               tmp_path/"split", resume=p["restart"])
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name]), name
    for key in ("s", "m", "clocks", "torch_rng"):
        assert torch.equal(fa[key], fb[key]), key
    assert fa["scheduler"] == fb["scheduler"]
    for i, state in fa["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, fb["optimizer"]["state"][i][key])
    assert torch.equal(fa["stream"]["positions"], fb["stream"]["positions"])
    assert fa["python_rng"] == fb["python_rng"]
    assert (fa["numpy_rng"][1] == fb["numpy_rng"][1]).all()
    assert full["completed_steps"] == resumed["completed_steps"] == 8
    bad = copy.copy(p["restart"])
    bad["tokenizer_sha256"] = "bad"
    with pytest.raises(ValueError, match="tokenizer"):
        run_training(b, LaneStream(ds, tok, 2, shuffle=True), *args, tmp_path/"bad", resume=bad)
    contract = json.loads(Path("configs/dynamath/experiment-003b.json").read_text())
    assert lr_at_update(1000, contract) == 2e-4
    assert lr_at_update(50000, contract) == 2e-5


def test_actual_fifo_word_output_and_long_prompt_budget():
    ds, tok, _ = fixture()
    class Probe(Model):
        def step(self, s, m, t):
            ns = s.clone()
            w = len(self.config.vocabulary)
            ns[:, w:2*w] = 0
            # Actual output address, observed by Session and fed into its FIFO.
            ns[:, w+tok.ids["Hello"]] = .5
            return ns, m, m[:, 0]*0
    model = Probe(Config(tok.vocabulary, memories=1, latent=4, rank=2)).eval()
    sample, session = generate(model, tok, "Hello " * 100, response_slots=4)
    assert sample["input_tokens_admitted"] == sample["input_token_count"] == 102
    assert sample["input_processing_ticks"] > 180
    assert sample["response_generation_ticks"] == 11
    assert sample["text"].startswith("Hello")
    assert any(r["admitted"] and r["admitted"][1] == "feedback" for r in sample["trace"])
    assert sample["question_events"]
    assert any(source == "feedback" for _, source in session.queue)


def test_full_finite_evaluation_counts_all_unequal_lanes():
    ds, tok, _ = fixture()
    model = Model(Config(tok.vocabulary, memories=1, latent=4, rank=2)).eval()
    metrics = teacher_forced(model, ds, tok, batch=2)
    assert metrics["input_slots"] == LaneStream(ds, tok, 2).total_slots
    assert metrics["response_tokens"] > 0
    actual = corpus_generation(model, tok, ds[:2], slots=2)
    assert actual["summary"]["prompts"] == 4
    assert actual["summary"]["dialogues"] == 2
    assert [s["pair_index"] for s in actual["samples"]] == [0, 1, 0, 1]
    assert all(s["input_tokens_admitted"] == s["input_token_count"] for s in actual["samples"])
    assert "trace" not in actual["samples"][0]


def test_old_character_checkpoint_still_loads():
    path = Path("models/dynamath/003a/english/control-final.pt")
    model, payload = Parent.load(path)
    assert isinstance(model.config.vocabulary, str)
    assert payload["schema"] == "dynamath-003a/v1"
