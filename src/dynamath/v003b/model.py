"""003a equations, split parameter storage and independently clocked sessions."""
import copy
from dataclasses import asdict
import math
import os
from pathlib import Path
import torch
from torch import nn
from dynamath.v002 import Config, Session
from dynamath.v003a import Model as Parent, BASE
from dynamath.v003a.ast import interpret, validate


class Formula(nn.Module):
    """One independent 003a law. Storage split avoids nine-law slice gradients."""
    def __init__(self, parent, index, ast):
        super().__init__()
        self.ast = copy.deepcopy(ast)
        validate(self.ast)
        for name, value in parent.named_parameters():
            setattr(self, name, nn.Parameter(value[index].detach().clone()))

    def forward(self, s, m):
        h = (s @ self.v + self.b + m @ self.q).tanh()
        z = 1.5 * self.d.tanh() * s + h @ self.u + self.c + m @ self.r
        candidate = interpret(self.ast, z, s, m, self.alpha)
        a = .05 + .9 * self.leak.sigmoid()
        ns = (1-a) * s + a * candidate
        probability = (ns @ self.gate + self.gb).sigmoid()
        hard = (probability >= .5).to(s.dtype)
        gate = hard + (probability-probability.detach()) if self.training else hard
        value = math.e * (ns @ self.writer + self.wb).tanh()
        return ns, ((1-gate) * m[:, 0] + gate * value)[:, None], hard


class Model(nn.Module):
    def __init__(self, config, asts=None):
        super().__init__()
        parent = Parent(config, asts)
        self.config, self.n = config, parent.n
        self.asts = copy.deepcopy(parent.asts)
        self.laws = nn.ModuleList(Formula(parent, i, ast) for i, ast in enumerate(self.asts))

    def zero(self, batch=1):
        ref = self.laws[0].v
        return ref.new_zeros(batch, self.n), ref.new_zeros(batch, 1)

    def inject(self, s, tokens):
        if tokens is None:
            return s
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=s.device)
        if tokens.ndim == 1:
            tokens = tokens[:, None]
        if tokens.shape[0] != len(s) or bool(((tokens < 0) | (tokens >= len(self.config.vocabulary))).any()):
            raise ValueError("invalid input address")
        return s + torch.zeros_like(s).scatter_add(1, tokens, torch.ones_like(tokens, dtype=s.dtype))

    def step(self, s, m, t):
        return self.laws[int(t) % 9](s, m)

    def logits(self, s):
        w = len(self.config.vocabulary)
        return self.config.gain * s[:, w:2*w]

    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"schema": "dynamath-003b/v1", "config": asdict(self.config),
                    "asts": self.asts, "weights": self.state_dict(), **extra}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, device="cpu"):
        p = torch.load(path, map_location="cpu", weights_only=False)
        if p["schema"] != "dynamath-003b/v1":
            raise ValueError("schema")
        # Construction consumes RNG; caller restores the checkpoint RNG AFTER load.
        obj = cls(Config(**p["config"]), p["asts"]).to(device)
        obj.load_state_dict(p["weights"])
        return obj.eval(), p


def block(model, s, m, clocks, x, reset, active=None):
    """One token slot; groups rows only by their own law phase (0/3/6)."""
    if active is None:
        active = torch.ones_like(reset)
    zs, zm = model.zero(len(s))
    s = torch.where(reset[:, None], zs, s)
    m = torch.where(reset[:, None], zm, m)
    clocks = torch.where(reset, 0, clocks)
    ns, nm = s.clone(), m.clone()
    w = len(model.config.vocabulary)
    outputs = [s.new_zeros(len(s), w), s.new_zeros(len(s), w)]
    phases = (clocks % 9).tolist()
    enabled = active.tolist()
    for phase in (0, 3, 6):
        indices = [i for i, p in enumerate(phases) if p == phase and enabled[i]]
        if not indices:
            continue
        idx = torch.tensor(indices, device=s.device)
        a, b = model.inject(s[idx], x[idx]), m[idx]
        for p in range(3):
            a, b, _ = model.step(a, b, phase+p)
            if p:
                outputs[p-1] = outputs[p-1].index_copy(0, idx, model.logits(a))
        ns, nm = ns.index_copy(0, idx, a), nm.index_copy(0, idx, b)
    return ns, nm, clocks + 3*active.long(), outputs
