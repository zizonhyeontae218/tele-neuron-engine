"""Endophalon-3 collection, single-readout scoring, controls and bilevel training.

The pattern follows Endophalon-2-ProMax (one physics, one readout, the input
picks the operation), generalized to the level -> operation -> data ISA.
Nothing here has been trained yet: see docs/endophalon3_handoff.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon2_pro.core import drift
from tele_neuron.endophalon2_pro.engine import EventRow
from tele_neuron.endophalon3.config import E3Config
from tele_neuron.endophalon3.engine import run
from tele_neuron.endophalon3.isa import OPS, random_stream
from tele_neuron.endophalon3.theta import Theta3


FAILED_J = 10.0


@dataclass(frozen=True, slots=True)
class E3Data:
    features: np.ndarray  # (streams, events, zones) at the readout delay
    channels: np.ndarray  # (streams, events, C) encoded instructions
    ops: np.ndarray  # (streams, events) opcode indices into OPS
    levels: np.ndarray  # (streams, events) 0 = A, 1 = B
    a: np.ndarray
    b: np.ndarray
    energy: np.ndarray
    wall: np.ndarray  # mean relative spring stretch (named for Endophalon-2-Pro's drift())
    streams: tuple[int, ...]
    finite: bool
    nodes: np.ndarray | None = None  # (streams, events, N*3)

    def index(self, streams: range) -> np.ndarray:
        position = {s: i for i, s in enumerate(self.streams)}
        return np.array([position[s] for s in streams], dtype=np.int64)


def collect(config: E3Config, thetas: list[Theta3], streams: range, *, nodes: bool = False) -> list[E3Data]:
    spec, seed = config.stream, config.physics.seed
    made = {s: random_stream(config.level, seed, s, spec.pulses) for s in streams}
    encoded = {s: made[s].channels(config.encoding) for s in streams}
    rows = [EventRow(t, encoded[s], spec.onsets(seed, s), s, s) for t in range(len(thetas)) for s in streams]
    rec = run(
        config.network, config.physics, thetas, rows, spec, slow=config.slow, delays=(config.readout_delay,),
        record_nodes=nodes, workers=config.workers, chunk_size=config.chunk_size,
    )
    per, shape = len(streams), (len(streams), spec.pulses)
    out = []
    for t in range(len(thetas)):
        part = slice(t * per, (t + 1) * per)
        out.append(
            E3Data(
                features=rec.zones[part, :, 0],
                channels=np.stack([encoded[s] for s in streams]),
                ops=np.stack([made[s].ops for s in streams]),
                levels=np.stack([made[s].levels for s in streams]),
                a=np.stack([made[s].a for s in streams]),
                b=np.stack([made[s].b for s in streams]),
                energy=rec.energy[part].reshape(shape),
                wall=rec.stretch[part].reshape(shape),
                streams=tuple(streams),
                finite=bool(rec.finite[part].all()),
                nodes=rec.nodes[part].reshape(*shape, -1) if nodes and rec.nodes is not None else None,
            )
        )
    return out


def dataset(config: E3Config, data: E3Data, streams: range, *, features: str = "zones", only_op: int | None = None, events: slice | None = None):
    """Features, targets and opcodes of the scored events.

    features: "zones" (the model), "zones+instr" (readout also sees the encoded
    instruction), "history" (physics-free: instruction + a_k, b_k, a_{k-1},
    a_{k-2}), "nodes" (linear readout of all node displacements, a diagnostic).
    """
    from tele_neuron.endophalon3.isa import evaluate

    window = events or slice(config.washout, config.stream.pulses)
    xs, ys, os = [], [], []
    for row in data.index(streams):
        target, valid = evaluate(data.ops[row], data.a[row], data.b[row])
        keep = np.zeros(config.stream.pulses, dtype=bool)
        keep[window] = True
        keep &= valid
        if only_op is not None:
            keep &= data.ops[row] == only_op
        if features == "zones":
            x = data.features[row]
        elif features == "zones+instr":
            x = np.hstack([data.features[row], data.channels[row]])
        elif features == "history":
            a = data.a[row]
            x = np.column_stack([data.channels[row], np.concatenate([[0], a[:-1]]), np.concatenate([[0, 0], a[:-2]])]).astype(np.float64)
        elif features == "nodes":
            assert data.nodes is not None, "collect(..., nodes=True) is required"
            x = data.nodes[row]
        else:
            raise ValueError(f"unknown feature set {features!r}")
        xs.append(x[keep])
        ys.append(target[keep])
        os.append(data.ops[row][keep])
    return np.vstack(xs), np.concatenate(ys)[:, None].astype(np.float64), np.concatenate(os)


def single_readout(config: E3Config, data: E3Data, fit: range, score: range, *, features: str = "zones", events: slice | None = None) -> dict[str, Any]:
    x, y, _ = dataset(config, data, fit, features=features)
    readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
    sx, sy, sops = dataset(config, data, score, features=features, events=events)
    raw = readout.scores(sx)[:, 0]
    hit = (raw >= 0.5) == sy[:, 0]
    per_op = {OPS[i]: float(hit[sops == i].mean()) for i in sorted(set(sops.tolist()))}
    return {"accuracy": float(hit.mean()), "per_op": per_op, "mse": float(np.mean((raw - sy[:, 0]) ** 2)),
            "readout": readout, "raw": raw, "target": sy[:, 0], "ops": sops}


def separate_readouts(config: E3Config, data: E3Data, fit: range, score: range) -> dict[str, Any]:
    """Upper reference: one readout per operation, fit and scored on that op's events only."""
    per_op, hits = {}, []
    for i, name in enumerate(OPS):
        if name not in config.ops:
            continue
        x, y, _ = dataset(config, data, fit, only_op=i)
        readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
        sx, sy, _ = dataset(config, data, score, only_op=i)
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
    l_val_mean: float | None = None
    l_val_worst: float | None = None


def operation_balanced_loss(raw: np.ndarray, target: np.ndarray, ops: np.ndarray,
                            worst_op_weight: float = 0.0) -> tuple[float, float, float]:
    """Blend pooled and worst-operation validation MSE; zero preserves the base loss.

    This changes training selection only. It adds no features, routing head,
    physics parameter, or information to the production readout.
    """
    if not 0.0 <= worst_op_weight <= 1.0:
        raise ValueError("objective.worst_op_weight must be within [0, 1]")
    error = (np.asarray(raw).reshape(-1) - np.asarray(target).reshape(-1)) ** 2
    mean = float(np.mean(error))
    worst = max(float(error[ops == i].mean()) for i in np.unique(ops))
    combined = mean if worst_op_weight == 0.0 else (1.0 - worst_op_weight) * mean + worst_op_weight * worst
    return float(combined), mean, worst


def fitness(config: E3Config, data: E3Data, train: range, val: range) -> Fitness:
    """J = L_val + l_noise*R_noise + l_drift*R_drift + l_complexity*R_complexity, on the single readout."""
    if not data.finite:
        return Fitness(FAILED_J, FAILED_J, 0.0, 0.0, 0.0, 0.0, {name: 0.0 for name in config.ops}, False)
    result = single_readout(config, data, train, val)
    raw, target = result["raw"], result["target"]
    ones, zeros = raw[target == 1], raw[target == 0]
    r_noise = float(0.5 * (ones.std() + zeros.std()) / (abs(ones.mean() - zeros.mean()) + 1e-6))
    r_drift = drift(config, data, range(train.start, val.stop))  # type: ignore[arg-type]
    r_complexity = float(np.mean(result["readout"].weights[:-1] ** 2))
    spec = config.objective
    loss, mean, worst = operation_balanced_loss(raw, target, result["ops"], spec.get("worst_op_weight", 0.0))
    j = loss + spec.get("lambda_noise", 0.0) * r_noise + spec.get("lambda_drift", 0.0) * r_drift + spec.get("lambda_complexity", 0.0) * r_complexity
    return Fitness(j, loss, r_noise, r_drift, r_complexity, result["accuracy"], result["per_op"], True, mean, worst)


@dataclass(frozen=True, slots=True)
class Report:
    theta: Theta3
    fitness: Fitness
    fitness_init: Fitness
    evaluations: int
    history: tuple[dict[str, Any], ...]
    test: dict[str, Any]
    random_search: dict[str, Any] | None = None


def train(config: E3Config, *, random_baseline: bool = False, log: Callable[[str], None] | None = None) -> Report:
    """Outer cross-entropy search over theta; inner exact ridge readout (same loop as ProMax)."""
    train_r, val_r, _ = config.stream_ranges()
    outer = config.outer
    generations, population = int(outer.get("generations", 40)), int(outer.get("population", 24))
    elite, min_sigma = int(outer.get("elite", 6)), float(outer.get("min_sigma", 0.05))
    space, rng = config.space, np.random.default_rng(int(outer.get("seed", 1)))
    mean = space.encode(config.theta_init)
    sigma = np.full(space.dim, float(outer.get("init_sigma", 0.5)))
    weights = np.log(elite + 0.5) - np.log(np.arange(1, elite + 1))
    weights /= weights.sum()
    best_u, best, first = mean.copy(), None, None
    history: list[dict[str, Any]] = []
    evaluations = 0
    for gen in range(generations + 1):
        cands = mean[None] if gen == 0 else np.vstack([mean, mean + rng.standard_normal((population - 1, space.dim)) * sigma])
        scored = [fitness(config, d, train_r, val_r) for d in collect(config, [space.decode(u) for u in cands], range(0, val_r.stop))]
        evaluations += len(cands)
        js = np.array([f.j for f in scored])
        first = first or scored[0]
        top = int(np.argmin(js))
        if best is None or scored[top].j < best.j:
            best_u, best = cands[top].copy(), scored[top]
        history.append({"generation": gen, "best_j": best.j, "best_val_accuracy": best.val_accuracy,
                        "best_val_per_op": best.val_per_op, "sigma": float(sigma.mean()),
                        "failed": int(sum(not f.finite for f in scored))})
        if log:
            log(f"  gen {gen:>3} best J={best.j:.4f} val={best.val_accuracy:.3f} " + " ".join(f"{k}={v:.2f}" for k, v in best.val_per_op.items()))
        if gen == 0:
            continue
        chosen = cands[np.argsort(js)[:elite]]
        spread = np.sqrt(weights @ (chosen - mean) ** 2)
        mean = weights @ chosen
        sigma = np.maximum(0.7 * spread + 0.3 * sigma, min_sigma)
    assert best is not None and first is not None
    theta = space.decode(best_u)
    random = random_search(config, budget=evaluations, seed=int(outer.get("seed", 1)) + 1000) if random_baseline else None
    return Report(theta, best, first, evaluations, tuple(history), test_evaluation(config, theta, config.theta_init), random)


def test_evaluation(config: E3Config, theta: Theta3, theta_init: Theta3) -> dict[str, Any]:
    """Held-out test streams: the model and every control, trained vs initial theta."""
    train_r, _, test_r = config.stream_ranges()
    half = config.washout + (config.stream.pulses - config.washout) // 2
    trained, initial = collect(config, [theta, theta_init], range(0, test_r.stop), nodes=True)
    out: dict[str, Any] = {}
    for name, data in (("trained", trained), ("initial", initial)):
        model = single_readout(config, data, train_r, test_r)
        out[name] = {
            "single_readout": {"accuracy": model["accuracy"], "per_op": model["per_op"],
                               "early": single_readout(config, data, train_r, test_r, events=slice(config.washout, half))["accuracy"],
                               "late": single_readout(config, data, train_r, test_r, events=slice(half, None))["accuracy"]},
            "single_readout_plus_instruction": _strip(single_readout(config, data, train_r, test_r, features="zones+instr")),
            "separate_readouts": separate_readouts(config, data, train_r, test_r),
            "linear_nodes": _strip(single_readout(config, data, train_r, test_r, features="nodes")),
            "drift": drift(config, data, test_r),  # type: ignore[arg-type]
            "finite": data.finite,
        }
    out["history_baseline"] = _strip(single_readout(config, trained, train_r, test_r, features="history"))
    return out


def random_search(config: E3Config, *, budget: int, seed: int) -> dict[str, Any]:
    """Control: same number of theta evaluations around theta_init, without updates."""
    train_r, val_r, test_r = config.stream_ranges()
    space, rng = config.space, np.random.default_rng(seed)
    center = space.encode(config.theta_init)
    cands = np.vstack([center, center + rng.standard_normal((max(1, budget - 1), space.dim)) * float(config.outer.get("init_sigma", 0.5))])
    pop = int(config.outer.get("population", 24))
    best_u, best = None, None
    for start in range(0, len(cands), pop):
        block = cands[start : start + pop]
        for u, d in zip(block, collect(config, [space.decode(u) for u in block], range(0, val_r.stop))):
            f = fitness(config, d, train_r, val_r)
            if best is None or f.j < best.j:
                best_u, best = u, f
    theta = space.decode(best_u)
    [data] = collect(config, [theta], range(0, test_r.stop))
    return {"theta": theta.to_payload(), "j": best.j, "test": _strip(single_readout(config, data, train_r, test_r))}


def capacity_probe(config: E3Config, theta: Theta3) -> dict[str, Any]:
    """Stage 1 (no training): what can the physics state hold for this level?

    Linear readout of node displacements per op (separate readouts on nodes)
    and the single zone readout, on held-out streams.
    """
    train_r, _, test_r = config.stream_ranges()
    [data] = collect(config, [theta], range(0, test_r.stop), nodes=True)
    nodes_per_op = {}
    for i, name in enumerate(OPS):
        if name in config.ops:
            x, y, _ = dataset(config, data, train_r, features="nodes", only_op=i)
            readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
            sx, sy, _ = dataset(config, data, test_r, features="nodes", only_op=i)
            nodes_per_op[name] = float((readout.predict(sx) == sy).mean())
    return {"finite": data.finite, "zones_single": _strip(single_readout(config, data, train_r, test_r)),
            "nodes_single": _strip(single_readout(config, data, train_r, test_r, features="nodes")),
            "nodes_separate_per_op": nodes_per_op}


def _strip(result: dict[str, Any]) -> dict[str, Any]:
    return {"accuracy": result["accuracy"], "per_op": result["per_op"]}


def report_payload(config: E3Config, report: Report) -> dict[str, Any]:
    return {"level": config.level, "encoding": config.encoding, "regions": config.regions,
            "objective": dict(config.objective),
            "readout_zones": config.zones, "readout_delay": config.readout_delay, "events": asdict(config.stream),
            "evaluations": report.evaluations, "fitness_initial": asdict(report.fitness_init),
            "fitness_trained": asdict(report.fitness), "test": report.test,
            "random_search": report.random_search, "history": list(report.history)}
