"""Per-tick semantics, Korean protocol and exact persistent TBPTT continuation."""
import pytest

torch = pytest.importorskip('torch')

from dynamath import Config, DynaMath, Session
from dynamath.korean import KoreanSession, advance, train
from dynamath.korean_data import BOT, EOS, SILENCE, USER, UNK, Tokenizer, grouped_split


def constant_model():
    torch.set_num_threads(1)
    m = DynaMath(Config('ab', size=12, rank=3))
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.bias[:, 2:4] = 2
    return m


@pytest.mark.parametrize('start', range(5))
def test_before_after_every_phase_first_eligible_tick(start):
    m = constant_model()
    new, old = Session(m), Session(m, refractory=5, io_ticks=5)
    new.ticks = old.ticks = start
    assert new.tick() == ['a']
    assert old.tick() == (['a'] if (start + 1) % 5 == 0 else [])
    assert torch.equal(new.state, old.state)  # timing change never changes law
    assert new.poll() == []  # duplicate polls cannot repeat an event


def test_constant_suprathreshold_repeated_events_and_crossing():
    s = Session(constant_model())
    assert s.tick(5) == ['a'] * 5
    s = Session(constant_model(), threshold=.9)
    assert s.tick() == []
    assert s.poll() == []
    s.threshold = .1
    before = s.state.clone()
    assert s.poll() == ['a']
    assert torch.equal(s.state, before)


def test_real_threshold_crossing_and_silence():
    s = Session(constant_model(), threshold=.6)
    assert s.tick() == []
    assert s.tick() == ['a']
    silent = Session(constant_model(), threshold=1.)
    assert silent.tick(20) == []


def test_checkpoint_nonzero_phase_idle_pending_and_event_clock(tmp_path):
    s = Session(constant_model())
    assert s.tick() == ['a']
    s.enqueue('aab')
    s.save(tmp_path / 'state.pt')
    restored = Session.load(tmp_path / 'state.pt')
    assert restored.ticks == 1 and restored.io_ticks == restored.refractory == 1
    assert restored.pending == [0, 0, 1]
    assert restored.poll() == []
    assert s.tick(13) == restored.tick(13)
    assert torch.equal(s.state, restored.state)


def test_tokenizer_roundtrip_coverage_and_pair_boundaries():
    rows = [{'Q': '한글 값! ㅋㅋ', 'A': '안녕, 세계?'}]
    tok = Tokenizer.build(rows)
    for text in ('한글 값! ㅋㅋ', '안녕, 세계?'):
        assert tok.decode(tok.encode(text)) == text
    assert tok.encode('🦄') == [tok.ids[UNK]]
    x, y = tok.pairs(rows[0])
    boundary = x.index(tok.ids[BOT])
    assert x[0] == tok.ids[USER]
    assert all(t == tok.ids[SILENCE] for t in y[:boundary])
    assert y[boundary] == tok.encode(rows[0]['A'])[0]
    assert x[-1] == tok.ids[EOS] and y[-2:] == [tok.ids[EOS], tok.ids[SILENCE]]


def test_question_groups_cannot_cross_splits():
    rows = [{'id': i, 'Q': q, 'A': '답', 'label': '0'} for i, q in enumerate(['안 녕?', '안녕!', '다른 질문'])]
    splits, _ = grouped_split(rows)
    assert next(s for s, ids in splits.items() if 0 in ids) == next(s for s, ids in splits.items() if 1 in ids)


def test_silence_is_learned_address_competition_not_external_mask():
    tok = Tokenizer.build([{'Q': '가', 'A': '나'}])
    m = DynaMath(Config(tok.vocabulary, size=2 * len(tok.vocabulary) + 12, rank=3))
    s = KoreanSession(m)
    s.ticks = 1
    s.state[0, len(tok.vocabulary) + tok.ids[SILENCE]] = .5
    before = s.state.clone()
    assert s.poll() == []
    assert torch.equal(s.state, before)
    # A real symbol wins on the next tick even if no BOT was injected.
    s.ticks = 2
    s.state[0, len(tok.vocabulary) + tok.ids[SILENCE]] = 0
    s.state[0, len(tok.vocabulary) + tok.encode('가')[0]] = .8
    assert s.poll() == [tok.vocabulary[tok.encode('가')[0]]]


def test_teacher_forcing_and_runtime_share_exact_tick_protocol():
    tok = Tokenizer.build([{'Q': '가', 'A': '나'}])
    m = DynaMath(Config(tok.vocabulary, size=2 * len(tok.vocabulary) + 12, rank=3))
    s = KoreanSession(m)
    state = m.zero()
    x, _ = tok.pairs({'Q': '가', 'A': '나'})
    for tick, token in enumerate(x * 2):
        state = advance(m, state, torch.tensor([token]), tick, 1)
        s.advance(token)
        assert torch.equal(state, s.state)
    assert s.ticks == 2 * len(x)


def test_free_feedback_adds_prediction_on_subsequent_tick_and_keeps_phase():
    tok = Tokenizer.build([{'Q': 'a', 'A': 'a'}])
    m = DynaMath(Config(tok.vocabulary, size=2 * len(tok.vocabulary) + 12, rank=3))
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.bias[:, len(tok.vocabulary) + tok.ids['a']] = 2.
    s = KoreanSession(m)
    result = s.reply('a', max_tokens=3)
    assert result['text'] == 'aaa' and result['ticks'] == 5
    expected = m.zero()
    for t, token in enumerate([tok.ids[c] for c in [USER, 'a', BOT, 'a', 'a']]):
        expected = advance(m, expected, torch.tensor([token]), t, 1)
    assert torch.equal(s.state, expected)
    assert s.pending == [tok.ids['a']]  # next feedback is queued, never assigned
    assert len(result['question_events']) == 2  # inappropriate events remain visible


def test_korean_resume_restores_optimizer_state_offsets_and_absolute_phase(tmp_path):
    rows = [{'id': 0, 'Q': '가', 'A': '나'}]
    tok = Tokenizer.build(rows)
    cfg = dict(seed=43, batch=2, tbptt=3, rank=3, latent=12, gain=6., lr=.002,
               clip=1., steps=2, seconds_cap_per_run=1000)
    train(rows, tok, tmp_path / 'partial.pt', cfg)
    cfg['steps'] = 4
    a, pa = train(rows, tok, tmp_path / 'resumed.pt', cfg, resume=tmp_path / 'partial.pt')
    b, pb = train(rows, tok, tmp_path / 'full.pt', cfg)
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    _, sa = DynaMath.load(tmp_path / 'resumed.pt')
    _, sb = DynaMath.load(tmp_path / 'full.pt')
    assert torch.equal(sa['state'], sb['state'])
    assert torch.equal(sa['offsets'], sb['offsets'])
    assert sa['phase'] == sb['phase'] == 2
    assert sa['ticks'] == sb['ticks'] == 12
    with pytest.raises(ValueError):
        train([dict(rows[0], A='가')], tok, tmp_path / 'bad.pt', cfg, resume=tmp_path / 'partial.pt')
