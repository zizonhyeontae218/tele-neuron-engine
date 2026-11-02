"""Persistent batched physical session. Runtime carries no input/target history."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from tele_neuron.endophalon2.engine import soft_zones
from tele_neuron.endophalon2_pro.engine import SpringScatter, masses_for
from tele_neuron.endophalon3.engine import _spring_forces
from tele_neuron.endophalon3.isa import Instruction


@dataclass(frozen=True)
class Result:
    instruction: Instruction
    event: int
    score: float | None
    output: int | None
    features: np.ndarray
    kinetic_before: float
    switching_energy: float
    switched: bool


class Session:
    def __init__(self, model, streams=(0,)):
        self.model=model;self.streams=tuple(streams)
        if not self.streams or len(set(self.streams))!=len(self.streams) or any(not isinstance(s,int) or s<0 for s in self.streams):
            raise ValueError('Nonempty distinct nonnegative stream IDs required')
        self.count=len(self.streams);self.network=model.network;self.spec=model.config.stream;self.physics=model.config.physics
        self.positions=np.empty((self.count,self.network.nodes,3));self.velocities=np.empty_like(self.positions)
        for i,s in enumerate(self.streams):
            rng=np.random.default_rng([self.physics.seed,0x4A49,s])
            self.positions[i]=self.network.home+rng.normal(0,self.physics.position_jitter,(self.network.nodes,3))
            self.velocities[i]=rng.normal(0,self.physics.velocity_jitter,(self.network.nodes,3))
        self.rngs=[np.random.default_rng([self.physics.seed,0x42,s]) for s in self.streams]
        self.scatter=SpringScatter(self.network,self.count)
        self.clock=0;self.events=0;self.active=None;self.selected='A';self.failed=False

    def _parameters(self, ids):
        cfgs=[self.model.configs['AB'[i]] for i in ids];thetas=[c.theta_init for c in cfgs]
        array=lambda name:np.array([getattr(t,name) for t in thetas])
        masks=np.array([c.slow for c in cfgs])
        scale=np.where(masks,array('stiffness_slow')[:,None],1.)
        edge=.5*(scale[:,self.network.left]+scale[:,self.network.right])
        return {'k1':array('k_linear')[:,None]*edge,'k3':array('k_cubic')[:,None]*edge,
                'damp':np.where(masks,array('damping_slow')[:,None],array('damping')[:,None])**(1./self.physics.substeps),
                'masses':np.array([masses_for(self.network,t.mass_spread,self.physics.seed) for t in thetas]),
                **{name:array(name) for name in ['anchor','bath','pulse_strength','stimulus_width','readout_width','stimulus_points','readout_centers']}}

    def _energy(self,p):
        # Stored mechanical energy; excludes drive/bath work and dissipation.
        kinetic=.5*np.sum(p['masses']*np.sum(self.velocities**2,axis=2),axis=1)
        delta=self.positions[:,self.network.right]-self.positions[:,self.network.left]
        extension=np.linalg.norm(delta,axis=2)-self.network.rest[None]
        springs=np.sum(.5*p['k1']*extension**2+.25*p['k3']*extension**4,axis=1)
        anchors=.5*p['anchor']*np.sum((self.positions-self.network.home[None])**2,axis=(1,2))
        return kinetic+springs+anchors

    def _tick(self,p,channels):
        nodes=self.network.nodes;h=self.physics.dt/self.physics.substeps
        kicks=np.stack([rng.standard_normal((nodes,3)) for rng in self.rngs])
        for sub in range(self.physics.substeps):
            forces=_spring_forces(self.positions,self.network,p['k1'],p['k3'],self.scatter)
            forces-=p['anchor'][:,None,None]*(self.positions-self.network.home[None])
            if channels is not None and np.any(channels):
                for channel in range(channels.shape[1]):
                    gain=p['pulse_strength']*channels[:,channel]
                    if not np.any(gain):continue
                    away=self.positions-p['stimulus_points'][:,channel][:,None,:]
                    r2=np.einsum('bnd,bnd->bn',away,away)
                    unit=away/(np.sqrt(r2)[:,:,None]+1e-9)
                    forces+=unit*(gain[:,None]*np.exp(-r2/(2.*p['stimulus_width'][:,None]**2)))[:,:,None]
            self.velocities*=p['damp'][:,:,None]
            self.velocities+=forces/p['masses'][:,:,None]*h
            if sub==0:self.velocities+=p['bath'][:,None,None]*kicks
            self.positions+=self.velocities*h
        self.clock+=1
        if not np.isfinite(self.positions).all() or not np.isfinite(self.velocities).all():
            self.failed=True
            raise FloatingPointError('Non-finite session; discard or restore a valid checkpoint')

    def select(self,level):
        if level not in ('A','B'):raise ValueError('Unknown level')
        if self.count!=1:raise ValueError('Text level selection requires a single stream')
        self.selected=level

    def execute(self,line):
        if self.count!=1:raise ValueError('Text commands require a single stream')
        tokens=line.split()
        if len(tokens)==2 and tokens[0].lower()=='level':
            self.select(tokens[1].upper());return None
        if len(tokens)==3:line=self.selected+' '+line
        ins=Instruction.parse(line)
        result=self.step([ins])[0];self.selected=ins.level
        return result

    def step(self,instructions):
        # Validate every row before state, clock or RNG changes.
        if self.failed:raise RuntimeError('Session has failed; restore a checkpoint')
        if len(instructions)!=self.count or any(not isinstance(i,Instruction) for i in instructions):
            raise ValueError('One valid Instruction per stream is required')
        ids=np.array([int(i.level=='B') for i in instructions]);p=self._parameters(ids)
        channels=np.stack([i.encode(self.model.config.encoding) for i in instructions])
        if self.active is None:
            for _ in range(self.spec.warmup):self._tick(p,None)
            switched=np.zeros(self.count,bool);jump=np.zeros(self.count)
        else:
            switched=ids!=self.active
            jump=self._energy(p)-self._energy(self._parameters(self.active))
        # Switching itself never modifies positions, velocities, RNG or time.
        self.active=ids.copy()
        kinetic=.5*np.mean(p['masses']*np.sum(self.velocities**2,axis=2),axis=1)
        features=None
        for delay in range(self.spec.interval):
            self._tick(p,channels if delay<self.spec.pulse_steps else None)
            if delay+1==self.model.config.readout_delay:
                features=soft_zones(self.positions,p['readout_centers'],p['readout_width']).copy()
        if not np.isfinite(features).all():
            self.failed=True;raise FloatingPointError('Non-finite observations')
        results=[]
        for row,ins in enumerate(instructions):
            head=self.model.banks[ins.level].head
            score=float(head.scores(features[row:row+1])[0,0]) if head else None
            if score is not None and not np.isfinite(score):
                self.failed=True;raise FloatingPointError('Non-finite readout')
            results.append(Result(ins,self.events,score,int(score>=.5) if score is not None else None,features[row],float(kinetic[row]),float(jump[row]),bool(switched[row])))
        self.events+=1
        return results

    def snapshot(self):
        if self.failed:raise RuntimeError('Cannot checkpoint a failed session')
        return {'schema':'endophalon3-session/v1','model':self.model.fingerprint,'streams':list(self.streams),
                'positions':self.positions.tolist(),'velocities':self.velocities.tolist(),'clock':self.clock,
                'events':self.events,'active':self.active.tolist() if self.active is not None else None,
                'selected':self.selected,'rng':[deepcopy(r.bit_generator.state) for r in self.rngs]}

    @classmethod
    def restore(cls,model,payload):
        if payload.get('schema')!='endophalon3-session/v1' or payload.get('model')!=model.fingerprint:
            raise ValueError('Session/model checkpoint mismatch')
        result=cls(model,tuple(payload['streams']))
        for name in ['positions','velocities']:
            value=np.array(payload[name],dtype=np.float64)
            if value.shape!=result.positions.shape or not np.isfinite(value).all():raise ValueError('Invalid state arrays')
            setattr(result,name,value)
        events,clock=payload['events'],payload['clock'];active=payload['active']
        if not isinstance(events,int) or events<0 or not isinstance(clock,int) or clock!=(result.spec.warmup+events*result.spec.interval if events else 0):raise ValueError('Invalid session clock')
        if (events==0 and active is not None) or (events>0 and (not isinstance(active,list) or len(active)!=result.count or any(v not in (0,1) for v in active))):raise ValueError('Invalid active profiles')
        if payload['selected'] not in ('A','B') or len(payload['rng'])!=result.count:raise ValueError('Invalid checkpoint metadata')
        result.events=events;result.clock=clock;result.active=np.array(active) if active is not None else None;result.selected=payload['selected']
        for rng,state in zip(result.rngs,payload['rng']):rng.bit_generator.state=state
        return result
