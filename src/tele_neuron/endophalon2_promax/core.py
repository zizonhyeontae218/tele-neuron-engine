"""Collection, objective, bilevel training and evaluation for ProMax.

The model is one spring network (Endophalon-2-Pro engine) with five stimulus
channels (a, b, and one per opcode) and **one** ridge readout over the soft
readout zones. Each event's target is that event's own operation applied to
the bit history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon2_pro.core import drift
from tele_neuron.endophalon2_pro.engine import EventRow, run_events
from tele_neuron.endophalon2_pro.theta import SpringTheta
from tele_neuron.endophalon2_promax.config import ProMaxConfig
from tele_neuron.endophalon2_promax.tasks import OPS, channels, fixed_op_targets, op_stream, op_targets


FAILED_J = 10.0


@dataclass(frozen=True, slots=True)
class OpData:
    """Recorded features of one theta, laid out per stream and event."""

    features: np.ndarray  # (streams, events, window, zones)
    bits: np.ndarray  # (streams, events, 2)
    ops: np.ndarray  # (streams, events)
    energy: np.ndarray  # (streams, events)
    wall: np.ndarray  # (streams, events) mean relative spring stretch
    streams: tuple[int, ...]
    finite: bool
    nodes: np.ndarray | None = None  # (streams, events, N*3) node displacements at the readout delay

    def index(self, streams: range) -> np.ndarray:
        position = {stream: i for i, stream in enumerate(self.streams)}
        return np.array([position[s] for s in streams], dtype=np.int64)


def collect(config: ProMaxConfig, thetas: list[SpringTheta], streams: range, *, nodes: bool = False) -> list[OpData]:
    spec, seed = config.stream, config.physics.seed
    data = {s: op_stream(seed, s, spec.pulses) for s in streams}
    rows = [
        EventRow(index, channels(*data[s]), spec.onsets(seed, s), s, s)
        for index in range(len(thetas))
        for s in streams
    ]
    record = run_events(
        config.network, config.physics, thetas, rows, spec,
        workers=config.workers, chunk_size=config.chunk_size,
        record_nodes=config.readout_delay if nodes else None,
    )
    per = len(streams)
    shape = (per, spec.pulses)
    out = []
    for index in range(len(thetas)):
        part = slice(index * per, (index + 1) * per)
        out.append(
            OpData(
                features=record.zones[part].reshape(*shape, spec.interval, -1),
                bits=np.stack([data[s][0] for s in streams]),
                ops=np.stack([data[s][1] for s in streams]),
                energy=record.energy[part].reshape(shape),
                wall=record.stretch[part].reshape(shape),
                streams=tuple(streams),
                finite=bool(record.finite[part].all()),
                nodes=record.nodes[part].reshape(*shape, -1) if nodes and record.nodes is not None else None,
            )
        )
    return out


def dataset(
    config: ProMaxConfig,
    data: OpData,
    streams: range,
    *,
    features: str = "zones",
    only_op: int | None = None,
    as_op: int | None = None,
    events: slice | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, targets and opcodes for the scored events of ``streams``.

    features: "zones" (the model), "zones+op" (readout also given the opcode),
    "history" (physics-free: a_k, b_k, a_{k-1}, b_{k-1} and the opcode), or
    "nodes" (linear node displacements, a diagnostic).
    only_op keeps events whose opcode is ``only_op``. as_op scores every event as
    if its opcode were ``as_op``.
    """
    window = events or slice(config.washout, config.stream.pulses)
    xs, ys, os = [], [], []
    for row in data.index(streams):
        bits, ops = data.bits[row], data.ops[row]
        target, valid = op_targets(bits, ops)
        if as_op is not None:
            target = fixed_op_targets(bits, as_op)
        keep = np.zeros(config.stream.pulses, dtype=bool)
        keep[window] = True
        keep &= valid
        if only_op is not None:
            keep &= ops == only_op
        onehot = np.eye(len(OPS))[ops]
        if features == "zones":
            x = data.features[row, :, config.readout_delay - 1]
        elif features == "zones+op":
            x = np.hstack([data.features[row, :, config.readout_delay - 1], onehot])
        elif features == "history":
            previous = np.vstack([[0, 0], bits[:-1]])
            x = np.hstack([bits, previous, onehot]).astype(np.float64)
        elif features == "nodes":
            assert data.nodes is not None, "collect(..., nodes=True) is required"
            x = data.nodes[row]
        else:
            raise ValueError(f"unknown feature set {features!r}")
        xs.append(x[keep])
        ys.append(target[keep])
        os.append(ops[keep])
    return np.vstack(xs), np.concatenate(ys)[:, None].astype(np.float64), np.concatenate(os)


def single_readout(config: ProMaxConfig, data: OpData, fit: range, score: range, *, features: str = "zones", events: slice | None = None) -> dict[str, Any]:
    """One readout for all operations: fit on ``fit`` streams, score on ``score`` streams."""
    x, y, _ = dataset(config, data, fit, features=features)
    readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
    sx, sy, sops = dataset(config, data, score, features=features, events=events)
    raw = readout.scores(sx)
    hit = ((raw >= 0.5) == sy)[:, 0]
    return {
        "accuracy": float(hit.mean()),
        "per_op": {name: float(hit[sops == index].mean()) for index, name in enumerate(OPS)},
        "mse": float(np.mean((raw - sy) ** 2)),
        "readout": readout,
        "raw": raw[:, 0],
        "target": sy[:, 0],
    }


def separate_readouts(config: ProMaxConfig, data: OpData, fit: range, score: range) -> dict[str, Any]:
    """Reference: one readout per operation, each fit and scored only on its own events."""
    per_op, hits = {}, []
    for index, name in enumerate(OPS):
        x, y, _ = dataset(config, data, fit, only_op=index)
        readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
        sx, sy, _ = dataset(config, data, score, only_op=index)
        hit = (readout.predict(sx) == sy)[:, 0]
        per_op[name] = float(hit.mean())
        hits.append(hit)
    return {"accuracy": float(np.concatenate(hits).mean()), "per_op": per_op}


@dataclass(frozen=True, slots=True)
class Fitness:
    j: float
    l_val: float
    r_noise: float
    r_drift: float
    r_complexity: float
    val_accuracy: float
    val_per_op: dict[str, float]
    finite: bool


def fitness(config: ProMaxConfig, data: OpData, train: range, val: range) -> Fitness:
    if not data.finite:
        return Fitness(FAILED_J, FAILED_J, 0.0, 0.0, 0.0, 0.0, {name: 0.0 for name in OPS}, False)
    result = single_readout(config, data, train, val)
    raw, target = result["raw"], result["target"]
    ones, zeros = raw[target == 1], raw[target == 0]
    r_noise = float(0.5 * (ones.std() + zeros.std()) / (abs(ones.mean() - zeros.mean()) + 1e-6))
    r_drift = drift(config, data, range(train.start, val.stop))  # type: ignore[arg-type]
    weights = result["readout"].weights
    r_complexity = float(np.mean(weights[:-1] ** 2))
    spec = config.objective
    j = (
        result["mse"]
        + spec.get("lambda_noise", 0.0) * r_noise
        + spec.get("lambda_drift", 0.0) * r_drift
        + spec.get("lambda_complexity", 0.0) * r_complexity
    )
    return Fitness(j, result["mse"], r_noise, r_drift, r_complexity, result["accuracy"], result["per_op"], True)


@dataclass(frozen=True, slots=True)
class Report:
    theta: SpringTheta
    fitness: Fitness
    fitness_init: Fitness
    evaluations: int
    history: tuple[dict[str, Any], ...]
    test: dict[str, Any]
    random_search: dict[str, Any] | None = None


def train(config: ProMaxConfig, *, random_baseline: bool = False, log: Callable[[str], None] | None = None) -> Report:
    train_range, val_range, _ = config.stream_ranges()
    outer = config.outer
    generations, population = int(outer.get("generations", 30)), int(outer.get("population", 16))
    elite, min_sigma = int(outer.get("elite", 4)), float(outer.get("min_sigma", 0.05))
    space = config.space
    rng = np.random.default_rng(int(outer.get("seed", 1)))
    mean = space.encode(config.theta_init)
    sigma = np.full(space.dim, float(outer.get("init_sigma", 0.5)))
    weights = np.log(elite + 0.5) - np.log(np.arange(1, elite + 1))
    weights /= weights.sum()
    best_u, best_fit, init_fit = mean.copy(), None, None
    history: list[dict[str, Any]] = []
    evaluations = 0

    for generation in range(generations + 1):
        candidates = mean[None, :] if generation == 0 else np.vstack(
            [mean, mean + rng.standard_normal((population - 1, space.dim)) * sigma]
        )
        datas = collect(config, [space.decode(u) for u in candidates], range(0, val_range.stop))
        scored = [fitness(config, data, train_range, val_range) for data in datas]
        evaluations += len(candidates)
        js = np.array([fit.j for fit in scored])
        if generation == 0:
            init_fit = scored[0]
        top = int(np.argmin(js))
        if best_fit is None or scored[top].j < best_fit.j:
            best_u, best_fit = candidates[top].copy(), scored[top]
        history.append(
            {"generation": generation, "best_j": best_fit.j, "best_val_accuracy": best_fit.val_accuracy,
             "best_val_per_op": best_fit.val_per_op, "center_j": scored[0].j, "sigma": float(sigma.mean()),
             "failed": int(sum(not fit.finite for fit in scored))}
        )
        if log:
            per_op = " ".join(f"{name}={value:.2f}" for name, value in best_fit.val_per_op.items())
            log(f"  gen {generation:>3} best J={best_fit.j:.4f} val={best_fit.val_accuracy:.3f} [{per_op}] sigma={sigma.mean():.3f}")
        if generation == 0:
            continue
        chosen = candidates[np.argsort(js)[:elite]]
        spread = np.sqrt(weights @ (chosen - mean) ** 2)
        mean = weights @ chosen
        sigma = np.maximum(0.7 * spread + 0.3 * sigma, min_sigma)

    assert best_fit is not None and init_fit is not None
    theta = space.decode(best_u)
    random = random_search(config, budget=evaluations, seed=int(outer.get("seed", 1)) + 1000) if random_baseline else None
    return Report(theta, best_fit, init_fit, evaluations, tuple(history), test_evaluation(config, theta, config.theta_init), random)


def test_evaluation(config: ProMaxConfig, theta: SpringTheta, theta_init: SpringTheta) -> dict[str, Any]:
    """Held-out test streams: the model and every control, for trained and initial theta."""
    train_range, _, test_range = config.stream_ranges()
    streams = range(0, test_range.stop)
    half = config.washout + (config.stream.pulses - config.washout) // 2
    trained, initial = collect(config, [theta, theta_init], streams, nodes=True)
    out: dict[str, Any] = {}
    for name, data in (("trained", trained), ("initial", initial)):
        model = single_readout(config, data, train_range, test_range)
        early = single_readout(config, data, train_range, test_range, events=slice(config.washout, half))
        late = single_readout(config, data, train_range, test_range, events=slice(half, None))
        out[name] = {
            "single_readout": {"accuracy": model["accuracy"], "per_op": model["per_op"], "early": early["accuracy"], "late": late["accuracy"]},
            "single_readout_plus_opcode": _strip(single_readout(config, data, train_range, test_range, features="zones+op")),
            "separate_readouts": separate_readouts(config, data, train_range, test_range),
            "linear_nodes": _strip(single_readout(config, data, train_range, test_range, features="nodes")),
            "drift": drift(config, data, test_range),  # type: ignore[arg-type]
            "finite": data.finite,
        }
    out["history_baseline"] = _strip(single_readout(config, trained, train_range, test_range, features="history"))
    return out


def random_search(config: ProMaxConfig, *, budget: int, seed: int) -> dict[str, Any]:
    """Control: the same number of theta evaluations around theta_init, without updates."""
    train_range, val_range, test_range = config.stream_ranges()
    space = config.space
    rng = np.random.default_rng(seed)
    center = space.encode(config.theta_init)
    sigma = float(config.outer.get("init_sigma", 0.5))
    candidates = np.vstack([center, center + rng.standard_normal((max(1, budget - 1), space.dim)) * sigma])
    population = int(config.outer.get("population", 16))
    best_u, best_fit = None, None
    for start in range(0, len(candidates), population):
        block = candidates[start : start + population]
        for u, data in zip(block, collect(config, [space.decode(u) for u in block], range(0, val_range.stop))):
            fit = fitness(config, data, train_range, val_range)
            if best_fit is None or fit.j < best_fit.j:
                best_u, best_fit = u, fit
    theta = space.decode(best_u)
    [data] = collect(config, [theta], range(0, test_range.stop))
    return {"theta": theta.to_payload(), "j": best_fit.j, "test": _strip(single_readout(config, data, train_range, test_range))}


def _strip(result: dict[str, Any]) -> dict[str, Any]:
    return {"accuracy": result["accuracy"], "per_op": result["per_op"]}
