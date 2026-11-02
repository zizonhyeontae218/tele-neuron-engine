"""Current native A/B model format, independent of retired 3 implementations."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
from tele_neuron.endophalon.readout import RidgeReadout
from .config import parse_config


class Module:
    def __init__(self,raw,head=None,provenance=None):
        self.raw=raw;self.config=parse_config(raw);self.level=self.config.level;self.head=head;self.provenance=provenance or {}
        c=self.config
        if (c.network.nodes,c.physics.substeps,c.zones,c.readout_delay,c.stream.gap_min,c.stream.gap_max)!=(144,2,16,12,16,16):raise ValueError('Current model requires 144/2/16 zones, delay12, gap16')
        if not np.isfinite(c.space.encode(c.theta_init)).all():raise ValueError('Non-finite theta')
        if head is not None:
            if head.mean.shape!=(16,) or head.scale.shape!=(16,) or head.weights.shape!=(17,1):raise ValueError('Invalid readout shapes')
            if not all(np.isfinite(x).all() for x in [head.mean,head.scale,head.weights]) or np.any(head.scale<=0):raise ValueError('Invalid readout')
        self.network=c.network
    def to_payload(self):return {'schema':'endophalon3-module/v1','family':'Teleneuron-Endophalon-3-'+self.level,'config':self.raw,'readout':self.head.to_payload() if self.head else None,'provenance':self.provenance}
    @classmethod
    def from_payload(cls,p):
        if p.get('schema')!='endophalon3-module/v1':raise ValueError('Unsupported native module')
        return cls(p['config'],RidgeReadout.from_payload(p['readout']) if p['readout'] else None,p.get('provenance'))
    @property
    def fingerprint(self):return hashlib.sha256(json.dumps(self.to_payload(),sort_keys=True,allow_nan=False).encode()).hexdigest()
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(self.to_payload(),f,indent=2,allow_nan=False);f.write('\n')
    @classmethod
    def load(cls,path):return cls.from_payload(json.loads(Path(path).read_text()))
