"""Deterministic tick/event runtime with additive feedback and full snapshots."""
import time

import torch

from .core import DynaMath


class Session:
    def __init__(self, model: DynaMath, threshold=0.10, refractory=5):
        if not -1 <= threshold <= 1 or refractory < 1:
            raise ValueError("invalid event threshold/refractory")
        self.model = model.eval()
        self.state = model.zero()
        self.ticks = 0
        self.last_event = -refractory
        self.threshold = threshold
        self.refractory = refractory
        self.pending = []
        self.ids = {char: i for i, char in enumerate(model.config.vocabulary)}

    def enqueue(self, text):
        # Validate before mutation so unknown input cannot leave a partial queue.
        if any(char not in self.ids for char in text):
            raise ValueError("input contains characters outside the model vocabulary")
        self.pending.extend(self.ids[char] for char in text)

    @torch.no_grad()
    def tick(self, n=1, emit=True):
        if n < 0:
            raise ValueError("negative tick count")
        events = []
        for _ in range(n):
            tokens = None
            if self.pending:
                tokens, self.pending = [self.pending], []
            self.state = self.model.transition(
                self.model.inject(self.state, tokens), self.ticks)
            self.ticks += 1
            if emit:
                events.extend(self.poll())
        return events

    @torch.no_grad()
    def poll(self):
        """Read output addresses without changing the information state."""
        if self.ticks == 0 or self.ticks % 5 or self.ticks - self.last_event < self.refractory:
            return []
        values = self.model.output(self.state)[0]
        winner = int(values.argmax())  # deterministic lowest-index tie
        if float(values[winner]) < self.threshold:
            return []
        self.last_event = self.ticks
        return [self.model.config.vocabulary[winner]]

    def consume(self, text):
        """Conditioning masks outward events, but advances the same persistent state."""
        for char in text:
            self.enqueue(char)
            self.tick(5, emit=False)

    def generate(self, prompt, max_chars=120, max_ticks=2000, stop="\n"):
        if max_chars < 0 or max_ticks < 0:
            raise ValueError("negative generation budget")
        if any(char not in self.ids for char in prompt):
            raise ValueError("input contains characters outside the model vocabulary")
        # Align a partially ticked session using genuine idle transitions.
        remaining = max_ticks
        alignment = (-self.ticks) % 5
        if alignment > remaining:
            return ""
        self.tick(alignment, emit=False)
        remaining -= alignment
        output = []
        # The last prompt character produces the first prediction, as in training.
        for char in prompt:
            if remaining < 5:
                return ""
            self.enqueue(char)
            self.tick(5, emit=False)
            remaining -= 5
        # Read the current output once at this boundary, then continue ticking.
        if prompt and max_chars:
            output.extend(self.poll())
        while len(output) < max_chars and remaining > 0:
            if output:
                if output[-1] in stop:
                    break
                # Exactly once per emitted token, via the normal additive queue.
                self.enqueue(output[-1])
            events = self.tick(min(5, remaining))
            remaining -= min(5, remaining)
            if events:
                output.extend(events)
            else:
                # Avoid refeeding the last token when a later idle cycle is silent.
                while remaining > 0 and not events:
                    events = self.tick(min(5, remaining))
                    remaining -= min(5, remaining)
                output.extend(events)
        return "".join(output)

    def realtime(self, seconds, hz=100.0):
        """Optional wall-clock loop; stopped processes perform no computation."""
        if seconds < 0 or hz <= 0:
            raise ValueError("invalid realtime budget")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            start = time.monotonic()
            yield from self.tick()
            time.sleep(max(0, 1 / hz - (time.monotonic() - start)))

    def save(self, path):
        self.model.save(path, session={"state": self.state, "ticks": self.ticks,
                        "last_event": self.last_event, "pending": self.pending,
                        "threshold": self.threshold, "refractory": self.refractory})

    @classmethod
    def load(cls, path):
        model, payload = DynaMath.load(path)
        values = payload["session"]
        session = cls(model, values["threshold"], values["refractory"])
        session.state = values["state"]
        session.ticks = values["ticks"]
        session.last_event = values["last_event"]
        session.pending = values["pending"]
        return session
