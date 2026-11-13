"""DynaMath 001a: persistent Korean address dynamics, bounded TBPTT and CLI."""
import argparse
import collections
import datetime
import json
import math
import platform
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from .core import Config, DynaMath
from .runtime import Session
from .training import dump, prediction_loss, setup
from .korean_data import BOT, EOS, SILENCE, USER, UNK, CONTROLS, Tokenizer, load, normalize, sha


class KoreanSession(Session):
    def __init__(self, model, threshold=.1, refractory=1, io_ticks=1):
        super().__init__(model, threshold, refractory, io_ticks)
        self.tokenizer = Tokenizer(model.config.vocabulary)

    def poll(self):
        # SILENCE competes with every token at a fixed ordinary output address.
        events = super().poll()
        return [e for e in events if e != SILENCE]

    def advance(self, token=None):
        if token is not None:
            self.pending.append(int(token))
        return self.tick(self.io_ticks)

    def reply(self, question, max_tokens=120, max_ticks=600):
        if max_tokens < 0 or max_ticks < 0:
            raise ValueError("negative generation budget")
        started, output, question_events = self.ticks, [], []
        eos, timed_out = False, False
        for token in [self.ids[USER]] + self.tokenizer.encode(question):
            if self.ticks - started + self.io_ticks > max_ticks:
                timed_out = True
                break
            question_events.extend(self.advance(token))
        pending = self.ids[BOT]
        if not timed_out:
            while len(output) < max_tokens and self.ticks - started + self.io_ticks <= max_ticks:
                events = self.advance(pending)
                pending = None
                if not events:
                    continue  # genuine idle laws after silence; no truth fallback
                token = self.ids[events[0]]
                if events[0] == EOS:
                    eos = True
                    # The emitted EOS is additive input at a subsequent tick.
                    # Queue it if the generation budget cannot include that tick.
                    if self.ticks - started + self.io_ticks <= max_ticks:
                        self.advance(token)
                    else:
                        self.pending.append(token)
                    break
                output.append(token)
                pending = token
            timed_out = not eos and self.ticks - started + self.io_ticks > max_ticks
            # Preserve feedback at a token cap for the next caller/tick.
            if pending is not None and output:
                self.pending.append(pending)
        decoded = self.tokenizer.decode(output)
        return {"question": question, "text": decoded, "token_ids": output,
                "raw_symbols": "".join(self.model.config.vocabulary[i] for i in output),
                "eos": eos, "timeout": timed_out, "token_limit": len(output) >= max_tokens,
                "ticks": self.ticks - started, "question_events": question_events,
                "unexpected_controls": sum(self.model.config.vocabulary[i] in CONTROLS for i in output),
                "uncomposed_jamo": sum('\u1100' <= c <= '\u11ff' for c in decoded),
                "replacement_characters": decoded.count('�')}


def stream(rows, tok):
    inputs, targets = [], []
    for row in rows:
        x, y = tok.pairs(row)
        inputs.extend(x)
        targets.extend(y)
    return torch.tensor(inputs), torch.tensor(targets)


def advance(model, state, token, ticks, io_ticks):
    state = model.inject(state, token)
    for i in range(io_ticks):
        state = model.transition(state, ticks + i)
    return state


@torch.no_grad()
def evaluate(model, rows, tok, io_ticks=1, batch=16):
    # Partition only at whole-pair boundaries. Each lane has a persistent state.
    lanes = [rows[i::batch] for i in range(batch)]
    streams = [stream(lane, tok) for lane in lanes if lane]
    length = max(len(x) for x, _ in streams)
    width = len(streams)
    x = torch.full((length, width), tok.ids[EOS], dtype=torch.long)
    y = torch.full_like(x, tok.ids[SILENCE])
    mask = torch.zeros_like(x, dtype=torch.bool)
    for j, (a, b) in enumerate(streams):
        x[:len(a), j], y[:len(b), j], mask[:len(a), j] = a, b, True
    state = model.zero(width)
    totals = collections.Counter()
    response_loss = total_loss = 0.
    for pos in range(length):
        state = advance(model, state, x[pos], pos * io_ticks, io_ticks)
        logits = model.logits(state)
        ce = F.cross_entropy(logits, y[pos], reduction='none')
        active = mask[pos]
        response = active & (y[pos] != tok.ids[SILENCE])
        question = active & ((x[pos] == tok.ids[USER]) | ((y[pos] == tok.ids[SILENCE]) & (x[pos] != tok.ids[EOS])))
        winners = logits.argmax(1)
        emits = (model.output(state).max(1).values >= .1) & (winners != tok.ids[SILENCE])
        total_loss += float(ce[active].sum())
        response_loss += float(ce[response].sum())
        totals.update(positions=int(active.sum()), response_tokens=int(response.sum()),
                      correct=int(((winners == y[pos]) & active).sum()),
                      response_correct=int(((winners == y[pos]) & response).sum()),
                      question_ticks=int(question.sum()),
                      question_false_emits=int((emits & question).sum()))
    nats = response_loss / totals['response_tokens']
    chars = sum(len(normalize(r['A'])) for r in rows)
    bytes_ = sum(len(normalize(r['A']).encode()) for r in rows)
    return {**dict(totals), 'response_CE': nats, 'response_perplexity': math.exp(nats),
            'all_tick_CE': total_loss / totals['positions'],
            'response_accuracy': totals['response_correct'] / totals['response_tokens'],
            'silence_false_positive_rate': totals['question_false_emits'] / totals['question_ticks'],
            'answer_bits_per_NFC_character_including_EOS': response_loss / math.log(2) / chars,
            'answer_bits_per_UTF8_byte_including_EOS': response_loss / math.log(2) / bytes_,
            'normalized_answer_characters': chars, 'evaluation_lanes': width}


def train(rows, tok, out, config, laws=5, io_ticks=1, rank=None, resume=None):
    setup(config['seed'])
    x, y = stream(rows, tok)
    batch, length = config['batch'], config['tbptt']
    identity = sha((tok.vocabulary + ':' + str([(r['id'], r['Q'], r['A']) for r in rows])).encode())
    rank = config['rank'] if rank is None else rank
    cfg = Config(tok.vocabulary, size=2 * len(tok.vocabulary) + config['latent'], rank=rank, laws=laws, gain=config['gain'])
    first, elapsed, history = 0, 0., []
    if resume:
        model, p = DynaMath.load(resume)
        if p.get('task') != 'korean-001a' or p['data_identity'] != identity or p['io_ticks'] != io_ticks or model.config != cfg:
            raise ValueError('incompatible resume data/protocol/geometry')
        for key in ('seed', 'batch', 'tbptt', 'lr', 'clip', 'seconds_cap_per_run'):
            if p['run_config'][key] != config[key]:
                raise ValueError('resume configuration changed: ' + key)
        state, offsets = p['state'], p['offsets']
        first, history, elapsed = p['step'], p['history'], p['training_seconds']
        torch.set_rng_state(p['torch_rng'])
    else:
        model = DynaMath(cfg)
        state = model.zero(batch)
        offsets = torch.arange(batch) * len(x) // batch
    if config['steps'] < first:
        raise ValueError('total steps precede checkpoint')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if resume:
        optimizer.load_state_dict(p['optimizer'])
    model.train()
    start = time.perf_counter()
    step = first
    for step_index in range(first, config['steps']):
        if elapsed + time.perf_counter() - start >= config['seconds_cap_per_run']:
            break
        state = state.detach()
        loss = 0.
        for pos in range(length):
            indices = (offsets + pos) % len(x)
            ticks = (step_index * length + pos) * io_ticks
            state = advance(model, state, x[indices], ticks, io_ticks)
            loss = loss + prediction_loss(model, state, y[indices]) / length
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()
        offsets = (offsets + length) % len(x)
        step = step_index + 1
        if step % 100 == 0 or step == config['steps']:
            row = {'step': step, 'loss': float(loss.detach()), 'seconds': elapsed + time.perf_counter() - start}
            history.append(row)
            print(json.dumps(row), flush=True)
    elapsed += time.perf_counter() - start
    model.eval()
    positions = step * batch * length
    visited = torch.zeros(len(x), dtype=torch.bool)
    for offset in torch.arange(batch) * len(x) // batch:
        visited[(offset + torch.arange(min(step * length, len(x)))) % len(x)] = True
    complete = unique_pairs = 0
    for r in rows:
        count = len(tok.pairs(r)[0])
        unique_pairs += int(visited[complete:complete + count].all())
        complete += count
    metadata = {'step': step, 'task': 'korean-001a', 'data_identity': identity,
      'ticks': step * length * io_ticks, 'phase': step * length * io_ticks % laws,
      'io_ticks': io_ticks, 'run_config': config, 'training_seconds': elapsed,
      'history': history, 'training_positions': positions, 'training_state_ticks': positions * io_ticks,
      'unique_stream_positions': int(visited.sum()), 'total_stream_positions': len(x),
      'fully_covered_train_pairs': unique_pairs, 'eligible_train_pairs': len(rows),
      'parameters': sum(p.numel() for p in model.parameters()), 'N': cfg.size, 'rank': rank,
      'laws': laws, 'active_coefficients_per_tick': 2 * cfg.size * rank + rank + 3 * cfg.size,
      'approx_flops_per_tick': 4 * cfg.size * rank + 12 * cfg.size + 2 * rank,
      'peak_training_process_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'ended_by': 'step_cap' if step == config['steps'] else 'time_cap',
      'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    model.save(out, **metadata, state=state.detach(), offsets=offsets,
               optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state())
    return model, metadata


def char_f1(a, b):
    ca, cb = collections.Counter(normalize(a)), collections.Counter(normalize(b))
    hits = sum((ca & cb).values())
    return 2 * hits / (len(a) + len(b)) if a or b else 1.


def frozen_plan(splits, directory):
    # Chosen by row IDs before any trained output is observed.
    plan = {'samples': {s: [r['id'] for r in splits[s][:8]] for s in ('train', 'test')},
      'semantic_rubric': [
        {'Q': '안녕하세요', 'accept_any': ['안녕', '반가'], 'meaning': 'returns a greeting'},
        {'Q': '고마워', 'accept_any': ['천만', '고마', '감사', '괜찮'], 'meaning': 'acknowledges thanks'},
        {'Q': '오늘 너무 슬퍼', 'accept_any': ['힘내', '괜찮', '위로', '슬프', '슬퍼'], 'meaning': 'acknowledges sadness or offers comfort'},
        {'Q': '시험 때문에 걱정돼', 'accept_any': ['시험', '공부', '준비', '잘할', '힘내'], 'meaning': 'responds to exam worry'},
        {'Q': '잠이 안 와', 'accept_any': ['잠', '쉬', '휴식', '눈'], 'meaning': 'responds to sleeplessness'},
        {'Q': '밥을 못 먹어서 배고파', 'accept_any': ['먹', '밥', '식사', '음식'], 'meaning': 'responds to hunger'}],
      'rubric_limits': 'keyword proxy only; fragments can false-pass; not a human semantic rating',
      'memory': {'bits': [0, 1], 'idle_gaps': [0, 5, 50],
                 'names': ['민수', '지영'], 'question': '내 이름이 뭐야?'},
      'max_tokens': 120, 'max_ticks': 600,
      'test_policy': 'full frozen test evaluated once per final model; no test-based selection',
      'quality_metrics': ['exact_NFC_answer', 'character_multiset_F1', 'keyword_rubric_proxy', 'EOS', 'unexpected_controls', 'uncomposed_Jamo', 'timeouts']}
    path = Path(directory) / 'evaluation-plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('frozen evaluation plan changed')
    dump(path, plan)
    return plan


@torch.no_grad()
def generate_evaluation(model, splits, tok, plan, io_ticks):
    common = collections.Counter(r['A'] for r in splits['train']).most_common(1)[0]
    frequent = common[0]
    samples, semantics = [], []
    for split, ids in plan['samples'].items():
        by_id = {r['id']: r for r in splits[split]}
        for row_id in ids:
            r = by_id[row_id]
            sample = KoreanSession(model, io_ticks=io_ticks).reply(r['Q'], plan['max_tokens'], plan['max_ticks'])
            samples.append({**sample, 'split': split, 'row_id': row_id, 'target': r['A'],
              'exact_answer': normalize(sample['text']) == r['A'], 'character_F1': char_f1(sample['text'], r['A']),
              'frequent_answer': frequent, 'frequent_exact': frequent == r['A'], 'frequent_character_F1': char_f1(frequent, r['A'])})
    for rule in plan['semantic_rubric']:
        sample = KoreanSession(model, io_ticks=io_ticks).reply(rule['Q'], plan['max_tokens'], plan['max_ticks'])
        semantics.append({**sample, 'rule': rule, 'keyword_proxy_pass': any(k in sample['text'] for k in rule['accept_any']),
                          'frequent_answer': frequent, 'frequent_proxy_pass': any(k in frequent for k in rule['accept_any'])})
    return {'samples': samples, 'semantics': semantics,
            'frequent_answer_baseline': {'answer': frequent, 'train_count': common[1],
              'full_test_exact': sum(r['A'] == frequent for r in splits['test']) / len(splits['test']),
              'full_test_character_F1': sum(char_f1(frequent, r['A']) for r in splits['test']) / len(splits['test']),
              'kind': 'deterministic constant answer; no neural perplexity'},
            'summary': {'EOS_rate': sum(r['eos'] for r in samples) / len(samples),
              'timeout_rate': sum(r['timeout'] for r in samples) / len(samples),
              'mean_decoded_characters': sum(len(r['text']) for r in samples) / len(samples),
              'valid_response_format_rate': sum(r['eos'] and not r['unexpected_controls'] and not r['uncomposed_jamo'] and not r['replacement_characters'] for r in samples) / len(samples),
              'heldout_sample_exact': sum(r['exact_answer'] for r in samples if r['split'] == 'test') / len(plan['samples']['test']),
              'keyword_proxy_pass_rate': sum(r['keyword_proxy_pass'] for r in semantics) / len(semantics)}}


@torch.no_grad()
def memory_probe(model, tok, plan, io_ticks):
    rows = []
    for kind, values in [('bit', plan['memory']['bits']), ('name', plan['memory']['names'])]:
        for value in values:
            for gap in plan['memory']['idle_gaps']:
                first = f'기억할 숫자는 {value}야.' if kind == 'bit' else f'내 이름은 {value}야.'
                second = '내가 기억하라고 한 숫자는 뭐야?' if kind == 'bit' else plan['memory']['question']
                for reset in (False, True):
                    session = KoreanSession(model, io_ticks=io_ticks)
                    first_output = session.reply(first)
                    idle_events = session.tick(gap)
                    if reset:
                        # Reset information only; preserve absolute law phase/clock.
                        session.state = model.zero()
                        session.pending = []
                    result = session.reply(second)
                    rows.append({'kind': kind, 'value': value, 'gap': gap, 'reset_at_turn': reset,
                      'first': first_output, 'idle_events': idle_events, 'second': result,
                      'recall_contains_target': str(value) in result['text']})
    return {'protocol': 'zero-shot two-turn contrast, current question identical across target values; reset preserves absolute clock; no task-specific training',
            'rows': rows, 'persistent_accuracy': sum(r['recall_contains_target'] for r in rows if not r['reset_at_turn']) / (len(rows) / 2),
            'reset_accuracy': sum(r['recall_contains_target'] for r in rows if r['reset_at_turn']) / (len(rows) / 2)}


@torch.no_grad()
def benchmark(model, io_ticks):
    session = KoreanSession(model, io_ticks=io_ticks)
    start = time.perf_counter()
    session.tick(3000)
    elapsed = time.perf_counter() - start
    start = time.perf_counter()
    outputs = [KoreanSession(model, io_ticks=io_ticks).reply('안녕하세요', max_tokens=120, max_ticks=600) for _ in range(5)]
    seconds = time.perf_counter() - start
    # Impulse latency includes fixed input injection, one transition, and detection.
    latency = []
    for _ in range(50):
        start = time.perf_counter()
        session.advance(session.ids[BOT])
        latency.append(time.perf_counter() - start)
    n, r = model.config.size, model.config.rank
    chars = sum(len(o['text']) for o in outputs)
    return {'batch1_ticks_per_second': 3000 / elapsed, 'tick_wall_seconds': elapsed,
            'input_step_detection_latency_seconds_mean': sum(latency) / len(latency),
            'decoded_characters': chars, 'complete_generation_seconds': seconds,
            'decoded_NFC_characters_per_second': chars / seconds,
            'generated_tokens': sum(len(o['token_ids']) for o in outputs),
            'generated_tokens_per_second': sum(len(o['token_ids']) for o in outputs) / seconds,
            'uncomposed_jamo': sum(o['uncomposed_jamo'] for o in outputs),
            'approx_flops_per_tick': 4 * n * r + 12 * n + 2 * r,
            'flops_exclusions': 'nonlinear functions, parameter transforms, indexing, allocation, event and Python overhead',
            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'state_bytes': n * 4, 'peak_process_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cpu': next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
            'platform': platform.platform(), 'python': platform.python_version(), 'torch': str(torch.__version__),
            'threads': torch.get_num_threads(), 'source_environment': 'managed Codex Cloud, CPU only'}


def experiment(root='.'):
    root = Path(root)
    config = json.loads((root / 'configs/dynamath/experiment-001a.json').read_text())
    splits, tok = load(root / 'data/dynamath/001a')
    evidence = root / 'docs/dynamath/experiment-001a'
    plan = frozen_plan(splits, evidence)
    setup(config['seed'])
    n, r = 2 * len(tok.vocabulary) + config['latent'], config['rank']
    matched_rank = round((5 * (2 * n * r + r + 3 * n) - 3 * n) / (2 * n + 1))
    results = {}
    for name, laws, io_ticks, rank in [('five-1tick', 5, 1, r), ('five-5tick', 5, 5, r), ('one-1tick', 1, 1, r), ('one-matched-1tick', 1, 1, matched_rank)]:
        out = root / 'models/dynamath/001a' / (name + '.pt')
        result_path = evidence / (name + '.json')
        if out.exists() or result_path.exists():
            raise ValueError('refusing to rerun/overwrite existing experiment: ' + name)
        print('START ' + name, flush=True)
        setup(config['seed'])
        before_model = DynaMath(Config(tok.vocabulary, size=n, rank=rank, laws=laws, gain=config['gain']))
        before = evaluate(before_model, splits['validation'], tok, io_ticks)
        # train() resets RNG to the identical initialization used above.
        model, training = train(splits['train'], tok, out, config, laws, io_ticks, rank)
        saved, payload = DynaMath.load(out)
        result = {'designation': 'DynaMath 001a', 'variant': name, 'training': training,
          'validation_before': before,
          'teacher_forced': {s: evaluate(saved, rows, tok, io_ticks) for s, rows in splits.items()},
          'free_generation': generate_evaluation(saved, splits, tok, plan, io_ticks),
          'memory': memory_probe(saved, tok, plan, io_ticks), 'benchmark': benchmark(saved, io_ticks),
          'checkpoint_sha256': sha(out.read_bytes())}
        dump(result_path, result)
        results[name] = result
        print('DONE ' + name + ' ' + json.dumps(result['teacher_forced']['test']), flush=True)
    dump(evidence / 'results.json', {n: {'training': r['training'], 'teacher_forced': r['teacher_forced'],
         'generation_summary': r['free_generation']['summary'], 'memory_persistent': r['memory']['persistent_accuracy'],
         'memory_reset': r['memory']['reset_accuracy'], 'benchmark': r['benchmark']} for n, r in results.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('experiment')
    exp.add_argument('--root', default='.')
    fit = sub.add_parser('train', help='explicit bounded run/resume; --steps is total')
    fit.add_argument('--config', default='configs/dynamath/experiment-001a.json')
    fit.add_argument('--data', default='data/dynamath/001a')
    fit.add_argument('--output', required=True)
    fit.add_argument('--resume')
    fit.add_argument('--steps', type=int)
    fit.add_argument('--laws', type=int, choices=[1, 5], default=5)
    fit.add_argument('--io-ticks', type=int, choices=[1, 5], default=1)
    fit.add_argument('--rank', type=int)
    for name in ('generate', 'chat', 'tick'):
        p = sub.add_parser(name)
        p.add_argument('--model', default='models/dynamath/001a/five-1tick.pt')
        p.add_argument('--state-in')
        p.add_argument('--state-out')
        p.add_argument('--max-tokens', type=int, default=120)
        p.add_argument('--max-ticks', type=int, default=600)
        if name == 'generate':
            p.add_argument('--question', default='안녕하세요')
        if name == 'tick':
            p.add_argument('--count', type=int, default=1)
            p.add_argument('--input', default='', help='text queued additively on next tick')
    args = ap.parse_args()
    setup(43)
    if args.command == 'experiment':
        experiment(args.root)
        return
    if args.command == 'train':
        config = json.loads(Path(args.config).read_text())
        if args.steps is not None:
            config['steps'] = args.steps
        if Path(args.output).exists():
            raise ValueError('output exists; resume into a new path')
        splits, tok = load(args.data)
        _, metadata = train(splits['train'], tok, args.output, config, args.laws,
                            args.io_ticks, args.rank, args.resume)
        print(json.dumps(metadata, ensure_ascii=False))
        return
    if args.state_in:
        session = KoreanSession.load(args.state_in)
    else:
        model, payload = DynaMath.load(args.model)
        session = KoreanSession(model, io_ticks=payload.get('io_ticks', 1))
    try:
        if args.command == 'generate':
            print(json.dumps(session.reply(args.question, args.max_tokens, args.max_ticks), ensure_ascii=False))
        elif args.command == 'tick':
            session.pending.extend(session.tokenizer.encode(args.input))
            print(json.dumps({'ticks_before': session.ticks, 'events': session.tick(args.count), 'ticks_after': session.ticks,
                              'phase': session.ticks % session.model.config.laws}, ensure_ascii=False))
        else:
            print('DynaMath 001a experimental Korean demo; /quit exits.')
            while True:
                try:
                    question = input('you> ')
                except EOFError:
                    break
                if question == '/quit':
                    break
                result = session.reply(question, args.max_tokens, args.max_ticks)
                print(json.dumps(result, ensure_ascii=False))
    except KeyboardInterrupt:
        print('stopped')
    finally:
        if args.state_out:
            session.save(args.state_out)


if __name__ == '__main__':
    main()
