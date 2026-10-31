"""Diagnose frozen Endophalon-3 states on train/validation only; no theta search."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon3.config import load_config
from tele_neuron.endophalon3.core import collect, dataset
from tele_neuron.endophalon3.isa import OPS
from tele_neuron.endophalon3.theta import Theta3

NAMES = ['a_binary_s1', 'a_binary_s2', 'a_no_regions_s1', 'a_no_regions_s2',
         'a_onehot_s1', 'b_binary_s1', 'b_binary_s2', 'b_onehot_s1', 'b_onehot_s2']


def metrics(raw, target, ops):
    raw = np.asarray(raw).reshape(-1)
    target = np.asarray(target).reshape(-1)
    hit = (raw >= .5) == target
    mse = (raw - target) ** 2
    per = {OPS[i]: {'accuracy': float(hit[ops == i].mean()),
                    'mse': float(mse[ops == i].mean()), 'n': int((ops == i).sum())}
           for i in sorted(set(ops.tolist()))}
    return {'accuracy': float(hit.mean()), 'mse': float(mse.mean()),
            'worst_op_mse': max(v['mse'] for v in per.values()), 'per_op': per}


def diagnose(config, theta, *, saved=None):
    train, val, test = config.stream_ranges()
    streams = range(0, val.stop)
    assert not set(streams).intersection(test)
    started = time.monotonic()
    [data] = collect(config, [theta], streams, nodes=True)
    assert data.finite
    split_data = {}
    for label, split in [('train', train), ('validation', val)]:
        split_data[label] = {key: dataset(config, data, split, features=key)
                             for key in ['zones', 'zones+instr', 'nodes', 'history']}
    result = {label: {} for label in split_data}
    for feature, control in [('zones', 'shared_zones'), ('zones+instr', 'zones_plus_instruction'),
                             ('nodes', 'shared_nodes'), ('history', 'shared_history')]:
        x, y, _ = split_data['train'][feature]
        readout = RidgeReadout(config.ridge_alpha).fit(x, y)
        if feature == 'zones' and saved is not None:
            assert np.allclose(readout.weights, saved.weights, rtol=1e-10, atol=1e-10)
            readout = saved  # Replay the original serialized model, not a replacement.
        for label, tables in split_data.items():
            sx, sy, ops = tables[feature]
            result[label][control] = metrics(readout.scores(sx), sy, ops)
    for feature in ['zones', 'nodes', 'history']:
        x, y, ops = split_data['train'][feature]
        heads = {i: RidgeReadout(config.ridge_alpha).fit(x[ops == i], y[ops == i])
                 for i in range(len(config.ops))}
        for label, tables in split_data.items():
            sx, sy, so = tables[feature]
            raw = np.empty(len(sy))
            for i, head in heads.items():
                raw[so == i] = head.scores(sx[so == i])[:, 0]
            result[label]['separate_' + feature] = metrics(raw, sy, so)
    # Explicit opcode-dependent interactions are a diagnostic external router.
    # Their extra features are never used in the production single-zone head.
    gated = {}
    for label, tables in split_data.items():
        x, y, ops = tables['zones']
        indicator = np.eye(len(config.ops))[ops]
        features = np.column_stack([(indicator[:, :, None] * x[:, None, :]).reshape(len(x), -1), indicator])
        gated[label] = (features, y, ops)
    head = RidgeReadout(config.ridge_alpha).fit(gated['train'][0], gated['train'][1])
    for label, (x, y, ops) in gated.items():
        result[label]['opcode_gated_zones_control'] = metrics(head.scores(x), y, ops)
    decoding = {}
    for feature in ['zones', 'nodes']:
        x, _, ops = split_data['train'][feature]
        head = RidgeReadout(config.ridge_alpha).fit(x, np.eye(len(config.ops))[ops])
        decoding[feature] = {}
        for label, tables in split_data.items():
            sx, _, so = tables[feature]
            pred = head.scores(sx).argmax(axis=1)
            confusion = np.zeros((len(config.ops), len(config.ops)), dtype=int)
            np.add.at(confusion, (so, pred), 1)
            decoding[feature][label] = {'accuracy': float((pred == so).mean()),
                                       'confusion': confusion.tolist()}
    return {'metrics': result, 'opcode_decoding': decoding,
            'streams': {'train': list(train), 'validation': list(val)},
            'test_collected': False, 'finite': data.finite,
            'seconds': time.monotonic() - started,
            'diagnostic_only': ['separate heads', '432 node features', 'opcode-gated feature products']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='docs/endophalon3_routing_runs/diagnosis')
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for name in NAMES:
        path = ROOT / 'models/endophalon3' / ('20261031_' + name + '.json')
        card = json.loads(path.read_text())
        config = replace(load_config(ROOT / card['config']), workers=1, regions=card['metrics']['regions'])
        cases.append((name, config, Theta3.from_payload(card['theta']), card, path))
    for config_name in ['level_a', 'level_a_onehot', 'level_b', 'level_b_onehot']:
        config = replace(load_config(ROOT / f'configs/endophalon3/{config_name}.json'), workers=1)
        cases.append(('initial_' + config_name, config, config.theta_init, None, None))
    all_results = {}
    for name, config, theta, card, path in cases:
        target = output / (name + '.json')
        if target.exists():
            result = json.loads(target.read_text())
        else:
            result = diagnose(config, theta, saved=RidgeReadout.from_payload(card['readout']) if card else None)
            result.update(name=name, level=config.level, encoding=config.encoding,
                          nodes=config.network.nodes, substeps=config.physics.substeps, zones=config.zones,
                          card=str(path.relative_to(ROOT)) if path else None,
                          card_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else None)
            if card:
                actual = result['metrics']['validation']['shared_zones']
                expected = card['metrics']['fitness_trained']
                assert actual['accuracy'] == expected['val_accuracy']
                assert {k: v['accuracy'] for k, v in actual['per_op'].items()} == expected['val_per_op']
            target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        all_results[name] = result
        val = result['metrics']['validation']
        print(name, {k: round(val[k]['accuracy'], 4) for k in
                     ['shared_zones', 'separate_zones', 'shared_nodes', 'separate_nodes', 'opcode_gated_zones_control']},
              'opcode zones/nodes', round(result['opcode_decoding']['zones']['validation']['accuracy'], 4),
              round(result['opcode_decoding']['nodes']['validation']['accuracy'], 4), flush=True)
    summary = output / 'summary.json'
    if not summary.exists():
        compact = {n: {'file': str((output / (n + '.json')).relative_to(ROOT)), 'card_sha256': d['card_sha256'], 'validation': {k: {'accuracy': v['accuracy'], 'mse': v['mse'], 'worst_op_mse': v['worst_op_mse']} for k, v in d['metrics']['validation'].items()}, 'opcode_decoding_validation': {k: v['validation']['accuracy'] for k, v in d['opcode_decoding'].items()}} for n, d in all_results.items()}
        summary.write_text(json.dumps(compact, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
