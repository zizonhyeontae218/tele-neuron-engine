"""Teleneuron-Endophalon-2-Pro: continuous computation on a nonlinear spring network.

Endophalon-2 ran colliding balls without reset. 2-Pro changes the physics
itself: 144 nodes joined to their nearest neighbors by springs with a cubic
(stiffening) term. The network keeps running. Whenever an input event
arrives, at irregular times, a Gaussian push near the stimulus points shapes
it, and a linear readout reads soft readout zones a fixed number of steps
later.

The springs give the restoring force that Endophalon-2 needed an anchor for,
and there is no collision detection, so the physics is cheaper per node.
"""

FAMILY = "Teleneuron-Endophalon-2-Pro"
