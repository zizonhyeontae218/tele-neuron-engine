"""Pinned full DailyDialog, dialogue-level leakage filtering and lossless lanes."""
import random
import unicodedata
from pathlib import Path
import torch
from dynamath.v003a.english_data import load, pairs, URL, REVISION, SHA256
from .tokenizer import WordTokenizer, digest


def all_qa(dialogue):
    turns = dialogue["turns"]
    return set(zip(turns, turns[1:]))


def overlap(splits):
    docs = {k: {tuple(d["turns"]) for d in ds} for k, ds in splits.items()}
    qa = {k: {p for d in ds for p in all_qa(d)} for k, ds in splits.items()}
    return {a+":"+b: {"exact_dialogues": len(docs[a] & docs[b]),
                       "exact_adjacent_qa": len(qa[a] & qa[b])}
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}


def remove_leakage(raw):
    clean, removed, seen_docs, higher_qa = {}, {}, set(), set()
    for name in ("test", "validation", "train"):
        kept, excluded = [], []
        for original in raw[name]:
            d = {**original, "turns": [unicodedata.normalize("NFC", t) for t in original["turns"]]}
            key = tuple(d["turns"])
            reasons = []
            if key in seen_docs:
                reasons.append("exact_dialogue")
            if all_qa(d) & higher_qa:
                reasons.append("cross_split_exact_adjacent_qa")
            if reasons:
                excluded.append({"id": d["id"], "reasons": reasons})
            else:
                kept.append(d)
                seen_docs.add(key)
        clean[name], removed[name] = kept, excluded
        higher_qa.update(p for d in kept for p in all_qa(d))
    return clean, removed


def prepare(cache="work/dailydialog/data.zip", limit=8192):
    raw = load(cache, raw=True)
    splits, removed = remove_leakage(raw)
    train_texts = [t for d in splits["train"] for t in d["turns"]]
    tok = WordTokenizer.build(train_texts, limit)
    audit = {"source": URL, "revision": REVISION, "archive_sha256": SHA256,
             "license": "CC BY-NC-SA 4.0", "seed": 43,
             "priority": ["test", "validation", "train"],
             "matching": "NFC exact full dialogues and ALL adjacent utterance Q/A pairs; remove whole lower-priority dialogue; no fuzzy matching",
             "before_overlap": overlap(raw), "after_overlap": overlap(splits),
             "removed": removed, "splits": {}, "tokenizer_sha256": tok.sha256}
    for name, ds in splits.items():
        sequences = [tok.encode(t) for d in ds for t in d["turns"]]
        n = sum(map(len, sequences))
        audit["splits"][name] = {"raw_dialogues": len(raw[name]), "dialogues": len(ds),
            "pairs": sum(len(pairs(d)) for d in ds), "text_tokens": n,
            "unknown_tokens": sum(s.count(4) for s in sequences),
            "oov_rate": sum(s.count(4) for s in sequences) / max(n, 1),
            "sha256": digest(ds), "ids": [d["id"] for d in ds]}
    audit["data_sha256"] = digest(splits)
    return splits, tok, audit


class LaneStream:
    """Each lane consumes every assigned slot, then wraps independently in train.

    Evaluation pads only inactive lanes and masks them out. No min-lane cut.
    Positions and epoch counts are restart state, not a random replay sampler.
    """
    def __init__(self, dialogues, tokenizer, batch=8, seed=43, shuffle=False, device="cpu"):
        dialogues = list(dialogues)
        if shuffle:
            random.Random(seed).shuffle(dialogues)
        lanes = []
        for lane in range(batch):
            xs, ys, resets = [], [], []
            for d in dialogues[lane::batch]:
                dx, dy = [], []
                for p in pairs(d):
                    a, b = tokenizer.pairs(p)
                    dx.extend(a)
                    dy.extend(b)
                if dx:
                    xs.extend(dx)
                    ys.extend(dy)
                    resets.extend([True] + [False] * (len(dx) - 1))
            lanes.append((xs, ys, resets))
        self.lengths = torch.tensor([len(l[0]) for l in lanes], device=device)
        self.total_slots = int(self.lengths.sum())
        if not self.total_slots:
            raise ValueError("empty data")
        width = int(self.lengths.max())
        self.x = torch.full((batch, width), 3, device=device, dtype=torch.long)
        self.y = self.x.clone()
        self.reset = torch.zeros((batch, width), device=device, dtype=torch.bool)
        for i, (x, y, r) in enumerate(lanes):
            n = len(x)
            self.x[i, :n] = torch.tensor(x, device=device, dtype=torch.long)
            self.y[i, :n] = torch.tensor(y, device=device, dtype=torch.long)
            self.reset[i, :n] = torch.tensor(r, device=device, dtype=torch.bool)
        self.positions = torch.zeros(batch, dtype=torch.long, device=device)
        self.epochs = torch.zeros_like(self.positions)
        self.rows = torch.arange(batch, device=device)
        self.sha256 = digest({"dialogues": dialogues, "tokenizer": tokenizer.sha256,
                              "batch": batch, "shuffle": shuffle, "seed": seed})

    def next(self, cycle=True):
        if cycle and bool((self.lengths == 0).any()):
            raise ValueError("training requires a nonempty lane for each batch row")
        active = self.positions < self.lengths
        pos = self.positions.clamp(max=self.x.shape[1]-1)
        x, y, reset = (a[self.rows, pos] for a in (self.x, self.y, self.reset))
        self.positions += active.long()
        if cycle:
            wrapped = self.positions == self.lengths
            self.epochs += wrapped.long()
            self.positions = torch.where(wrapped, 0, self.positions)
        return x, y, reset & active, active

    def state_dict(self):
        return {"positions": self.positions.clone(), "epochs": self.epochs.clone(), "sha256": self.sha256}

    def load_state_dict(self, state):
        if state["sha256"] != self.sha256:
            raise ValueError("stream identity mismatch")
        for key in ("positions", "epochs"):
            value = state[key].to(self.positions.device)
            if value.shape != self.positions.shape or bool((value < 0).any()):
                raise ValueError("invalid lane cursor")
            setattr(self, key, value)
        if bool((self.positions >= self.lengths).any()):
            raise ValueError("invalid training position")
