"""Synchronized local physics; only external spatial force crosses the boundary."""
import numpy as np
from tele_neuron.endophalon2.engine import soft_zones
from tele_neuron.endophalon3_software.session import Session
from tele_neuron.endophalon3_software.physics import _spring_forces


def zones(s):
    return soft_zones(s.positions, s.p['readout_centers'], s.p['readout_width'])


def reference(s):
    return soft_zones(np.repeat(s.network.home[None], s.count, axis=0),
                      s.p['readout_centers'], s.p['readout_width'])


def coupling_force(s, observation, home, u, enabled=True):
    """No labels, head outputs, or other nonphysical information in this interface."""
    if not enabled:
        return np.zeros_like(s.positions)
    u = np.asarray(u, dtype=float)
    if u.shape != (21,) or not np.isfinite(u).all():
        raise ValueError('Invalid coupling parameters')
    q = np.tanh(((observation-home)/0.02) @ np.tanh(u[:16])/4)
    center = np.asarray(s.module.config.space_size)*(1+np.tanh(u[16:19]))/2
    width = .3+1.7*(1+np.tanh(u[19]))/2
    gain = .08*(1+np.tanh(u[20]))/2
    away = s.positions-center
    r2 = np.einsum('bnd,bnd->bn', away, away)
    force = away/(np.sqrt(r2)[...,None]+1e-9)
    return force*(gain*q[:,None]*np.exp(-r2/(2*width**2)))[...,None]


def tick(s, channels, external):
    """Original integrator equations, with an additional bounded external force."""
    p=s.p; h=s.physics.dt/s.physics.substeps
    kicks=np.stack([r.standard_normal((s.network.nodes,3)) for r in s.rngs])
    for sub in range(s.physics.substeps):
        forces=_spring_forces(s.positions,s.network,p['k1'],p['k3'],s.scatter)
        forces-=p['anchor'][:,None,None]*(s.positions-s.network.home[None])
        if channels is not None and np.any(channels):
            for channel in range(channels.shape[1]):
                gain=p['pulse_strength']*channels[:,channel]
                if not np.any(gain):continue
                away=s.positions-p['stimulus_points'][:,channel][:,None,:]
                r2=np.einsum('bnd,bnd->bn',away,away)
                unit=away/(np.sqrt(r2)[:,:,None]+1e-9)
                forces+=unit*(gain[:,None]*np.exp(-r2/(2*p['stimulus_width'][:,None]**2)))[:,:,None]
        forces+=external
        s.velocities*=p['damp'][:,:,None]
        s.velocities+=forces/p['masses'][:,:,None]*h
        if sub==0:s.velocities+=p['bath'][:,None,None]*kicks
        s.positions+=s.velocities*h
    s.clock+=1
    if not np.isfinite(s.positions).all() or not np.isfinite(s.velocities).all():
        s.failed=True
        raise FloatingPointError('Non-finite physical state')


class Coupled:
    """Live runtime: each step samples actual sender state before either advances."""
    def __init__(self, sender, receiver, sender_ids, receiver_ids, u, enabled=True):
        self.sender=Session(sender,sender_ids);self.receiver=Session(receiver,receiver_ids)
        self.home=reference(self.sender);self.u=np.asarray(u);self.enabled=enabled

    def step(self, sender_channels, receiver_channels):
        force=coupling_force(self.receiver,zones(self.sender),self.home,self.u,self.enabled)
        self.sender._tick(sender_channels)
        tick(self.receiver,receiver_channels,force)
        assert self.sender.clock==self.receiver.clock
