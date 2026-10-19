"""Teleneuron-Endophalon-2: continuous nonlinear computation in a running physics system.

Earlier versions reset the physics for every input. Endophalon-2 never resets:
the physics keeps running, input pulses arrive at stimulus points every
``interval`` steps, and a linear readout reads soft readout zones ``n`` steps
after each pulse. The question is whether nonlinear computation can continue
indefinitely on top of whatever the previous pulses left behind.

Rest dynamics keep a long-running system alive: an anchor spring pulls each
ball weakly toward its home position (so pulses cannot jam everything against
the walls), and an optional heat bath keeps damping from freezing the system.
"""

FAMILY = "Teleneuron-Endophalon-2"
