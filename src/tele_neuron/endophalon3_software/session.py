"""One fixed-profile reservoir, with local clock/RNG and no other model state."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from tele_neuron.endophalon2.engine import soft_zones
from tele_neuron.endophalon2_pro.engine import SpringScatter,masses_for
from .physics import _spring_forces
from .isa import Instruction

@dataclass(frozen=True)
class Result:
    instruction:Instruction
    event:int
    score:float|None
    output:int|None
    features:np.ndarray
    kinetic_before:float

class Session:
    def __init__(self,module,streams=(0,)):
        self.module=module;self.streams=tuple(streams);self.count=len(self.streams);c=module.config
        if not self.count or len(set(self.streams))!=self.count or any(type(s) is not int or s<0 for s in self.streams):raise ValueError('Invalid stream IDs')
        self.network=module.network;self.physics=c.physics;self.spec=c.stream;self.clock=0;self.events=0;self.failed=False
        self.positions=np.empty((self.count,self.network.nodes,3));self.velocities=np.empty_like(self.positions)
        for i,s in enumerate(self.streams):
            rng=np.random.default_rng([self.physics.seed,0x4A49,s]);self.positions[i]=self.network.home+rng.normal(0,self.physics.position_jitter,(self.network.nodes,3));self.velocities[i]=rng.normal(0,self.physics.velocity_jitter,(self.network.nodes,3))
        self.rngs=[np.random.default_rng([self.physics.seed,0x42,s]) for s in self.streams];self.scatter=SpringScatter(self.network,self.count)
        t=c.theta_init;repeat=lambda v:np.repeat(np.asarray(v)[None],self.count,axis=0)
        scales=np.where(c.slow,t.stiffness_slow,1.);edges=.5*(scales[self.network.left]+scales[self.network.right])
        self.p={'k1':repeat(t.k_linear*edges),'k3':repeat(t.k_cubic*edges),'damp':repeat(np.where(c.slow,t.damping_slow,t.damping)**(1./self.physics.substeps)),'masses':repeat(masses_for(self.network,t.mass_spread,self.physics.seed)),**{n:repeat(getattr(t,n)) for n in ['anchor','bath','pulse_strength','stimulus_width','readout_width','stimulus_points','readout_centers']}}
    def _tick(self,channels):
        p=self.p;h=self.physics.dt/self.physics.substeps;kicks=np.stack([r.standard_normal((self.network.nodes,3)) for r in self.rngs])
        for sub in range(self.physics.substeps):
            forces=_spring_forces(self.positions,self.network,p['k1'],p['k3'],self.scatter);forces-=p['anchor'][:,None,None]*(self.positions-self.network.home[None])
            if channels is not None and np.any(channels):
                for channel in range(channels.shape[1]):
                    gain=p['pulse_strength']*channels[:,channel]
                    if not np.any(gain):continue
                    away=self.positions-p['stimulus_points'][:,channel][:,None,:];r2=np.einsum('bnd,bnd->bn',away,away);unit=away/(np.sqrt(r2)[:,:,None]+1e-9)
                    forces+=unit*(gain[:,None]*np.exp(-r2/(2.*p['stimulus_width'][:,None]**2)))[:,:,None]
            self.velocities*=p['damp'][:,:,None];self.velocities+=forces/p['masses'][:,:,None]*h
            if sub==0:self.velocities+=p['bath'][:,None,None]*kicks
            self.positions+=self.velocities*h
        self.clock+=1
        if not np.isfinite(self.positions).all() or not np.isfinite(self.velocities).all():self.failed=True;raise FloatingPointError('Non-finite local session')
    def step(self,instructions):
        if self.failed:raise RuntimeError('Restore a valid session before continuing')
        if len(instructions)!=self.count or any(not isinstance(i,Instruction) or i.level!=self.module.level for i in instructions):raise ValueError('Instructions must match this local model')
        channels=np.stack([i.encode(self.module.config.encoding) for i in instructions])
        if not self.events:
            for _ in range(self.spec.warmup):self._tick(None)
        kinetic=.5*np.mean(self.p['masses']*np.sum(self.velocities**2,axis=2),axis=1)
        for d in range(self.spec.interval):
            self._tick(channels if d<self.spec.pulse_steps else None)
            if d+1==self.module.config.readout_delay:features=soft_zones(self.positions,self.p['readout_centers'],self.p['readout_width']).copy()
        if not np.isfinite(features).all():self.failed=True;raise FloatingPointError('Non-finite observations')
        raw=self.module.head.scores(features)[:,0] if self.module.head else [None]*self.count
        if self.module.head and not np.isfinite(raw).all():self.failed=True;raise FloatingPointError('Non-finite output')
        results=[Result(ins,self.events,float(raw[i]) if raw[i] is not None else None,int(raw[i]>=.5) if raw[i] is not None else None,features[i],float(kinetic[i])) for i,ins in enumerate(instructions)];self.events+=1;return results
    def execute(self,line):return self.step([Instruction.parse(line)])[0]
    def snapshot(self):
        if self.failed:raise RuntimeError('Failed session cannot be checkpointed')
        return {'schema':'endophalon3-local-session/v1','model':self.module.fingerprint,'streams':list(self.streams),'positions':self.positions.tolist(),'velocities':self.velocities.tolist(),'events':self.events,'clock':self.clock,'rng':[deepcopy(r.bit_generator.state) for r in self.rngs]}
    @classmethod
    def restore(cls,module,p):
        if p.get('schema')!='endophalon3-local-session/v1' or p.get('model')!=module.fingerprint:raise ValueError('Wrong module checkpoint')
        s=cls(module,tuple(p['streams']));events=p['events'];clock=p['clock']
        if type(events) is not int or events<0 or clock!=(s.spec.warmup+events*s.spec.interval if events else 0) or len(p['rng'])!=s.count:raise ValueError('Invalid local clock/RNG')
        for key in ['positions','velocities']:
            v=np.asarray(p[key],dtype=float)
            if v.shape!=s.positions.shape or not np.isfinite(v).all():raise ValueError('Invalid local arrays')
            setattr(s,key,v)
        s.events=events;s.clock=clock
        for r,state in zip(s.rngs,p['rng']):r.bit_generator.state=state
        return s
