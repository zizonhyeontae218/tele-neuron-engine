"""Reproduce primary-run arithmetic (including original floating-point ordering).

The supported runtime uses exact hard replacement. This archival training runner
preserves primary arithmetic so the published experiment remains reproducible.
Writes the normal 002 paths: use a separate checkout when preserving evidence.
"""
import math
from dynamath.v002 import Model
from dynamath.v002 import __main__ as runner

class PrimaryModel(Model):
    def step(self,s,m,t):
        i=t%9;owner=self.owner(i)
        mask=m.new_zeros(self.config.memories)
        if self.config.cross:mask[:]=1
        else:mask[owner]=1
        read=m*mask
        h=(s@self.v[i]+self.b[i]+read@self.q[i]).tanh()
        candidate=(1.5*self.d[i].tanh()*s+h@self.u[i]+self.c[i]+read@self.r[i]).tanh()
        a=.05+.9*self.leak[i].sigmoid();ns=(1-a)*s+a*candidate
        probability=(ns@self.gate[i]+self.gb[i]).sigmoid();hard=(probability>=.5).to(s.dtype)
        gate=hard+probability-probability.detach() if self.training else hard
        value=math.e*(ns@self.writer[i]+self.wb[i]).tanh()
        new=m[:,owner] if self.config.frozen else m[:,owner]+gate*(value-m[:,owner])
        nm=m.clone();nm[:,owner]=new
        return ns,nm,hard

if __name__=='__main__':
    runner.Model=PrimaryModel
    runner.run()
