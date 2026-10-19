"""Teleneuron-Endophalon-Pro: one trainable physics system under a linear readout.

Endophalon fixes the physics and learns only a readout. Plus picks among three
fixed presets. Pro *trains the physics itself* with a bilevel loop:

1. outer loop proposes physics parameters theta
2. the physics generates features for every input case
3. inner loop: the readout W*(theta) is solved exactly (closed-form ridge)
4. theta is scored on validation realizations with
   J(theta) = L_val(theta, W*(theta)) + l1*R_noise + l2*R_stability + l3*R_complexity
5. theta is updated, and the loop repeats

The best readout a physics system allows is used as that system's fitness. The
experiment asks whether training the physics lets a smaller readout do the job.
"""

FAMILY = "Teleneuron-Endophalon-Pro"
