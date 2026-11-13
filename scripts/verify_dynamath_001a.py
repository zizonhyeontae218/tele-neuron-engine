"""Replay saved-checkpoint samples without corpus access and measure decoded Korean."""
import json
from pathlib import Path
import time
import tempfile

import torch

from tele_neuron.dynamath.core import DynaMath
import tele_neuron.dynamath.korean as korean
from tele_neuron.dynamath.training import dump, setup


def main():
    setup(43)
    def forbidden(*args, **kwargs):
        raise AssertionError('generation must not load corpus/answers')
    korean.load = forbidden
    results = {}
    evidence = Path('docs/dynamath/experiment-001a')
    for name in ('five-1tick', 'five-5tick', 'one-1tick', 'one-matched-1tick'):
        model, payload = DynaMath.load(f'models/dynamath/001a/{name}.pt')
        saved = json.loads((evidence / f'{name}.json').read_text())
        io_ticks = payload['io_ticks']
        replayed = 0
        for sample in saved['free_generation']['samples']:
            actual = korean.KoreanSession(model, io_ticks=io_ticks).reply(sample['question'])
            assert actual == {k: sample[k] for k in actual}, (name, sample['row_id'])
            replayed += 1
        session = korean.KoreanSession(model, io_ticks=io_ticks)
        session.reply('안녕하세요')
        session.tick(2)
        session.pending.append(session.tokenizer.encode('가')[0])
        with tempfile.TemporaryDirectory(prefix='dynamath-001a-') as directory:
            path = Path(directory) / (name + '-runtime.pt')
            session.save(path)
            restored = korean.KoreanSession.load(path)
        assert restored.ticks == session.ticks and restored.pending == session.pending
        assert restored.tick(11) == session.tick(11)
        assert torch.equal(restored.state, session.state)
        start = time.perf_counter()
        samples = [korean.KoreanSession(model, io_ticks=io_ticks).reply('안녕하세요') for _ in range(20)]
        seconds = time.perf_counter() - start
        syllables = sum(sum('\uac00' <= c <= '\ud7a3' for c in s['text']) for s in samples)
        results[name] = {'replayed_samples_identical': replayed, 'snapshot_resume_exact': True,
          'dataset_access_disabled': True, 'generation_wall_seconds': seconds,
          'generation_count': len(samples), 'decoded_Hangul_syllables': syllables,
          'decoded_Hangul_syllables_per_second': syllables / seconds,
          'decoded_NFC_characters': sum(len(s['text']) for s in samples),
          'decoded_NFC_characters_per_second': sum(len(s['text']) for s in samples) / seconds,
          'generated_tokens': sum(len(s['token_ids']) for s in samples),
          'uncomposed_Jamo': sum(s['uncomposed_jamo'] for s in samples),
          'benchmark_question': '안녕하세요', 'first_output': samples[0]}
    dump(evidence / 'saved-checkpoint-verification.json', results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
