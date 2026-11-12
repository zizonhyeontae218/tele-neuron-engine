"""Bounded CPU experiments and resumable TBPTT character learning."""
import hashlib
import json
import math
import platform
import resource
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .core import Config, DynaMath
from .runtime import Session


def dump(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")


def setup(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def prediction_loss(model, state, targets):
    logits = model.logits(state)
    selected = model.output(state).gather(1, targets[:, None]).squeeze(1)
    # A fixed 0.25 margin teaches positive output-address threshold eligibility.
    return F.cross_entropy(logits, targets) + 0.1 * F.relu(0.25 - selected).mean()


def logic_batch(model, rng, batch=64, phase=0, max_gap=2):
    a = torch.randint(2, (batch,), generator=rng)
    b = torch.randint(2, (batch,), generator=rng)
    op = torch.randint(2, (batch,), generator=rng)  # 0 xor, 1 recall first bit
    amp = 0.9 + 0.2 * torch.rand(batch, 1, generator=rng)
    state = model.zero(batch)
    state = model.cycle(model.inject(state, a, amp), phase=phase)
    for _ in range(int(torch.randint(max_gap + 1, (), generator=rng))):
        state = model.cycle(state, phase=phase)
    state = model.cycle(model.inject(state, 2 + b, amp), phase=phase)
    for _ in range(int(torch.randint(max_gap + 1, (), generator=rng))):
        state = model.cycle(state, phase=phase)
    state = model.cycle(state, 4 + op, phase=phase)
    target = 6 + torch.where(op.bool(), a, a ^ b)
    return state, target, op


@torch.no_grad()
def logic_eval(model, seed=202, phase=0, max_gap=2):
    rng = torch.Generator().manual_seed(seed)
    totals, correct = [0, 0], [0, 0]
    loss = eligible = 0.0
    for _ in range(8):
        state, target, op = logic_batch(model, rng, phase=phase, max_gap=max_gap)
        pred = model.logits(state).argmax(1)
        for task in (0, 1):
            mask = op == task
            totals[task] += int(mask.sum())
            correct[task] += int(((pred == target) & mask).sum())
        loss += float(F.cross_entropy(model.logits(state), target)) / 8
        eligible += float((model.output(state).max(1).values >= 0.1).float().mean()) / 8
    return {"xor_accuracy": correct[0] / totals[0],
            "recall_accuracy": correct[1] / totals[1], "counts": totals,
            "cross_entropy": loss, "threshold_coverage": eligible}


def train_logic(out, laws=5, steps=600, seed=41):
    setup(seed)
    model = DynaMath(Config("aAbBxr01", size=64, rank=16, laws=laws))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = torch.Generator().manual_seed(seed + 1)
    before = logic_eval(model)
    history = []
    start = time.perf_counter()
    for step in range(steps):
        state, target, _ = logic_batch(model, rng, phase=step % 5)
        loss = prediction_loss(model, state, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            row = {"step": step + 1, "loss": float(loss.detach())}
            history.append(row)
            print(f"logic laws={laws} {row}", flush=True)
    seconds = time.perf_counter() - start
    after = logic_eval(model)
    phase_results = [logic_eval(model, phase=i) for i in range(5)]
    extrapolation = logic_eval(model, seed=303, max_gap=3)
    model.save(out, optimizer=optimizer.state_dict(), rng=rng.get_state(),
               torch_rng=torch.get_rng_state(), step=steps,
               task="logic", seed=seed, history=history)
    return {"before": before, "after": after, "phase_results": phase_results,
            "longer_gap_probe": extrapolation, "history": history,
            "steps": steps, "seconds": seconds,
            "parameters": sum(p.numel() for p in model.parameters())}


def encode(model, text):
    ids = {char: i for i, char in enumerate(model.config.vocabulary)}
    return torch.tensor([ids[c] for c in text], dtype=torch.long)


@torch.no_grad()
def text_eval(model, text):
    data = encode(model, text)
    state = model.zero()
    loss, correct, covered = 0.0, 0, 0
    for i in range(len(data) - 1):
        state = model.cycle(state, data[i:i+1])
        logits = model.logits(state)
        target = data[i+1:i+2]
        loss += float(F.cross_entropy(logits, target))
        correct += int(logits.argmax(1) == target)
        covered += int(float(model.output(state).max()) >= 0.1)
    count = len(data) - 1
    ce = loss / count
    return {"characters": count, "cross_entropy": ce,
            "perplexity": math.exp(ce), "accuracy": correct / count,
            "threshold_coverage": covered / count}


def train_text(train_path, val_path, out, steps=800, seed=43, resume=None):
    if steps < 0:
        raise ValueError("negative total training steps")
    setup(seed)
    train_text = Path(train_path).read_text()
    val_text = Path(val_path).read_text()
    # Vocabulary is fixed from training only. Validation may not expand it.
    vocabulary = "".join(sorted(set(train_text)))
    batch, length, lr = 16, 24, 0.002
    history, first = [], 0
    if resume:
        model, payload = DynaMath.load(resume)
        if payload.get("task") != "characters":
            raise ValueError("resume requires a character training checkpoint")
        if payload["train_sha256"] != hashlib.sha256(train_text.encode()).hexdigest():
            raise ValueError("training corpus changed")
        if payload["validation_sha256"] != hashlib.sha256(val_text.encode()).hexdigest():
            raise ValueError("validation corpus changed")
        if model.config.vocabulary != vocabulary:
            raise ValueError("vocabulary mismatch")
        state, offsets, first = payload["state"], payload["offsets"], payload["step"]
        history, before = payload["history"], payload["before"]
        if steps < first:
            raise ValueError("total steps precede the resume checkpoint")
        torch.set_rng_state(payload["torch_rng"])
    else:
        model = DynaMath(Config(vocabulary))
        state = model.zero(batch)
        offsets = torch.randint(len(train_text) - 1, (batch,))
        before = text_eval(model, val_text)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if resume:
        optimizer.load_state_dict(payload["optimizer"])
    data = encode(model, train_text)
    start = time.perf_counter()
    model.train()
    for step in range(first, steps):
        state = state.detach()
        loss = 0.0
        for i in range(length):
            indices = (offsets + i) % (len(data) - 1)
            state = model.cycle(state, data[indices])
            loss = loss + prediction_loss(model, state, data[indices + 1]) / length
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        offsets = (offsets + length) % (len(data) - 1)
        if (step + 1) % 100 == 0:
            row = {"step": step + 1, "loss": float(loss.detach())}
            history.append(row)
            print(f"characters {row}", flush=True)
    seconds = time.perf_counter() - start
    model.eval()
    model.save(out, optimizer=optimizer.state_dict(), step=steps, task="characters",
               state=state.detach(), ticks=steps * length * 5, phase=0,
               offsets=offsets, torch_rng=torch.get_rng_state(), seed=seed,
               history=history, before=before, batch=batch, tbptt=length,
               train_sha256=hashlib.sha256(train_text.encode()).hexdigest(),
               validation_sha256=hashlib.sha256(val_text.encode()).hexdigest())
    prompts = [prompt for prompt in
               ("user: hi\nbot:", "user: what are you?\nbot:", "user: hello\nbot:")
               if set(prompt) <= set(vocabulary)]
    if not prompts:
        prompts = [train_text[:16]]
    samples = {prompt: Session(model).generate(prompt) for prompt in prompts}
    return model, {"before": before, "after": text_eval(model, val_text),
                   "train": text_eval(model, train_text), "history": history,
                   "samples": samples, "seconds": seconds, "steps": steps,
                   "train_characters": len(train_text), "validation_characters": len(val_text),
                   "parameters": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def benchmark(model, repeats=3, ticks=5000):
    session = Session(model)
    session.consume("user: hi\nbot:")
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.tick(ticks)
        times.append(time.perf_counter() - start)
    start = time.perf_counter()
    generated = 0
    used_ticks = 0
    for _ in range(10):
        sample = Session(model)
        output = sample.generate("user: hi\nbot:", max_chars=80, max_ticks=1000, stop="")
        generated += len(output)
        used_ticks += sample.ticks
    elapsed = time.perf_counter() - start
    impulse = Session(model)
    impulse.consume("user: hi\nbot:")
    earlier = impulse.state.clone()
    norms = []
    for _ in range(10):
        impulse.tick(1000, emit=False)
        norms.append(float(impulse.state.abs().max()))
    other = Session(model)
    other.consume("user: hello\nbot:")
    other.tick(10000, emit=False)
    k, n, r = model.config.laws, model.config.size, model.config.rank
    return {"tick_seconds": times, "ticks_per_second": [ticks / t for t in times],
            "generation_characters": generated, "generation_seconds": elapsed,
            "generation_characters_per_second": generated / elapsed,
            "generation_ticks": used_ticks, "max_abs_per_1000_idle_ticks": norms,
            "idle_distance_between_prompts": float((other.state - impulse.state).norm()),
            "idle_distance_from_conditioned_state": float((earlier - impulse.state).norm()),
            "approx_flops_per_tick": 4 * n * r + 12 * n + 2 * r,
            "nonlinear_evaluations_per_tick": n + r,
            "parameter_bytes_fp32": sum(p.numel() * p.element_size() for p in model.parameters()),
            "state_bytes_fp32": n * 4,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu": next((line.split(":", 1)[1].strip() for line in
                         Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), platform.processor()),
            "platform": platform.platform(),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "threads": torch.get_num_threads()}


def experiment(root, logic_steps=600, text_steps=800):
    root = Path(root)
    out = root / "models/dynamath"
    out.mkdir(parents=True, exist_ok=True)
    results = {"logic_five": train_logic(out / "logic-five.pt", steps=logic_steps),
               "logic_single": train_logic(out / "logic-single.pt", laws=1, steps=logic_steps)}
    dump(root / "docs/dynamath/logic-results.json", results)
    model, text = train_text(root / "data/dynamath/train.txt", root / "data/dynamath/validation.txt",
                             out / "characters.pt", steps=text_steps)
    results["characters"] = text
    results["benchmark"] = benchmark(model)
    dump(root / "docs/dynamath/results.json", results)
    print(json.dumps(results, indent=2), flush=True)
