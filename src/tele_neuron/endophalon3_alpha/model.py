"""Two parameter/readout banks on the same network and event clock."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon3.config import parse_config
from . import FAMILY, SCHEMA


@dataclass(frozen=True)
class Bank:
    raw_config: dict
    head: RidgeReadout | None

    @property
    def config(self):
        return parse_config(self.raw_config)


class Model:
    def __init__(self, banks: dict[str, Bank], provenance=None):
        if set(banks) != {'A', 'B'}:
            raise ValueError('Exactly A and B banks are required')
        self.banks = banks
        self.provenance = provenance or {}
        self.configs = {name: bank.config for name, bank in banks.items()}
        a, b = self.configs['A'], self.configs['B']
        for name, cfg in self.configs.items():
            if cfg.level != name or (cfg.network.nodes, cfg.physics.substeps, cfg.zones) != (144, 2, 16):
                raise ValueError('Each bank requires its own level and 144 nodes / 2 substeps / 16 zones')
            if cfg.stream.gap_min != cfg.stream.gap_max:
                raise ValueError('Alpha requires regular timing')
            if cfg.stream.interval != 16 or cfg.readout_delay != 12:
                raise ValueError('Alpha requires gap 16 and delay 12')
            head = banks[name].head
            if head is not None:
                if head.mean is None or head.scale is None or head.weights is None or head.mean.shape != (16,) or head.scale.shape != (16,) or head.weights.shape != (17,1):
                    raise ValueError('Each head must read exactly 16 zones')
                if not all(np.isfinite(v).all() for v in [head.mean,head.scale,head.weights]) or np.any(head.scale <= 0):
                    raise ValueError('Readout must be finite with positive scales')
        for key in ('space_size','lattice','neighbors','lattice_jitter','physics','stream','readout_delay','encoding'):
            if getattr(a,key) != getattr(b,key):
                raise ValueError(f'A/B must share {key}')
        for cfg in self.configs.values():
            if not np.isfinite(cfg.space.encode(cfg.theta_init)).all():
                raise ValueError('Non-finite theta')
        self.config = a
        self.network = a.network

    @classmethod
    def from_cards(cls, a_path, b_path, root='.'):
        banks = {}; provenance = {}
        for level, path in [('A',a_path),('B',b_path)]:
            path = Path(path); raw_bytes=path.read_bytes();card=json.loads(raw_bytes)
            raw=json.loads((Path(root)/card['config']).read_text())
            raw['theta_init']=card['theta'];raw['regions']=card['metrics']['regions']
            banks[level]=Bank(raw,RidgeReadout.from_payload(card['readout']))
            provenance[level]={'legacy_card':str(path),'sha256':hashlib.sha256(raw_bytes).hexdigest()}
        return cls(banks,provenance)

    def to_payload(self):
        return {'schema':SCHEMA,'family':FAMILY,'architecture':'two banks, one physical state',
                'parameter_switch':'preserve positions and velocities; record mechanical energy jump',
                'provenance':self.provenance,'banks':{name:{'config':bank.raw_config,'head':bank.head.to_payload() if bank.head else None} for name,bank in self.banks.items()}}

    @classmethod
    def from_payload(cls, payload):
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unsupported model schema')
        return cls({name:Bank(item['config'],RidgeReadout.from_payload(item['head']) if item['head'] else None) for name,item in payload['banks'].items()},payload.get('provenance'))

    @property
    def fingerprint(self):
        raw=json.dumps(self.to_payload(),sort_keys=True,allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def save(self, path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(self.to_payload(),f,indent=2,allow_nan=False);f.write('\n')

    @classmethod
    def load(cls,path):
        return cls.from_payload(json.loads(Path(path).read_text()))
