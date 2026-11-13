"""Five low-rank nonlinear transition laws; no learned encoder or readout."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

SCHEMA = "dynamath-1/v1"


@dataclass(frozen=True)
class Config:
    vocabulary: str
    size: int = 160
    rank: int = 24
    laws: int = 5
    gain: float = 6.0

    def __post_init__(self):
        if not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary must contain unique characters")
        if self.size < 2 * len(self.vocabulary) or self.rank < 1:
            raise ValueError("need disjoint input/output addresses and positive rank")
        if self.laws not in (1, 5):
            raise ValueError("five laws required; one is reserved for the baseline")


class DynaMath(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        k, n, r = config.laws, config.size, config.rank
        self.u = nn.Parameter(torch.randn(k, r, n) * 0.15)
        self.v = nn.Parameter(torch.randn(k, n, r) / n**0.5)
        self.inner_bias = nn.Parameter(torch.zeros(k, r))
        self.bias = nn.Parameter(torch.zeros(k, n))
        self.diagonal = nn.Parameter(torch.full((k, n), 0.55))
        # Different fixed starting time scales; all subsequently learn.
        self.leak = nn.Parameter(torch.linspace(-1.0, 1.0, n).repeat(k, 1))

    def zero(self, batch=1):
        return self.u.new_zeros(batch, self.config.size)

    def inject(self, state, tokens, amplitude=1.0):
        """Functional additive injection, including duplicate-address accumulation."""
        if tokens is None:
            return state
        tokens = torch.as_tensor(tokens, device=state.device, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens[:, None]
        if tokens.shape[0] != state.shape[0]:
            raise ValueError("one token row per state row")
        if (tokens < 0).any() or (tokens >= len(self.config.vocabulary)).any():
            raise ValueError("unknown input address")
        delta = torch.zeros_like(state)
        values = torch.ones_like(tokens, dtype=state.dtype) * amplitude
        return state + delta.scatter_add(1, tokens, values)

    def transition(self, state, phase):
        k = phase % self.config.laws
        alpha = 0.05 + 0.90 * self.leak[k].sigmoid()
        d = 1.5 * self.diagonal[k].tanh()
        inner = (state @ self.v[k] + self.inner_bias[k]).tanh()
        candidate = (d * state + inner @ self.u[k] + self.bias[k]).tanh()
        return (1 - alpha) * state + alpha * candidate

    def cycle(self, state, tokens=None, phase=0):
        state = self.inject(state, tokens)
        # Five ticks even for the single-law reference: equal timing and I/O.
        for tick in range(5):
            state = self.transition(state, phase + tick)
        return state

    def output(self, state):
        width = len(self.config.vocabulary)
        return state[:, width:2 * width]

    def logits(self, state):
        return self.config.gain * self.output(state)

    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": SCHEMA, "config": asdict(self.config),
                    "weights": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema"] != SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        model = cls(Config(**payload["config"]))
        model.load_state_dict(payload["weights"])
        return model.eval(), payload
