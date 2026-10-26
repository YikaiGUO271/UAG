"""One-pair secant proposals with transported metric and momentum.

The stage anchor defines the required objective decrease. A private orbit uses
one FG per transition and survives intervening ordinary UAG steps.
"""

import math
import numpy as np
from . import _runtime as API


class PrivateState:

    def __init__(self, dimension, diagnostic_capacity=0):
        self.point = (np.zeros(dimension), 0.0, np.zeros(dimension))
        self.previous_gradient = np.zeros(dimension)
        self.previous_step = np.zeros(dimension)
        self.previous_auxiliary = np.zeros(dimension)
        self.work = np.zeros(dimension)
        self.valid = False
        self.have_pair = False
        self.clock = 0
        self.damping = 1.0
        self.scale = 1.0
        self.diag = np.full((diagnostic_capacity, 10), np.nan)
        self.diag_count = 0
        self.diag_dropped = 0
        self.diag_started = False
        self.diag_events = np.zeros(80, dtype=np.int64)


def reset(private, anchor, scale):
    private.point = anchor
    private.previous_gradient[:] = anchor[2]
    private.previous_step[:] = 0.0
    private.previous_auxiliary[:] = anchor[0]
    private.clock = 0
    private.damping = 1.0
    private.scale = scale
    private.have_pair = False
    private.valid = True


def resume(anchor, run, private):
    """One paid proposal; calibration information seeds an absent history."""
    private.diag_events = run.events
    run.events[40] += 1
    if not private.valid:
        if private.diag_started:
            run.events[42] += 1
        else:
            run.events[41] += 1
        private.diag_started = True
        reset(private,run.best,run.L)
        displacement=run.best[0]-anchor[0]
        if np.dot(displacement,displacement)>0:
            private.previous_step[:]=displacement
            private.previous_gradient[:]=anchor[2]
            private.have_pair=True
    if run.calls >= run.max_queries:
        run.status = 2
        return 0
    p = private.point
    g = p[2]
    sy = yy = ss = sg = 0.0
    if private.have_pair:
        for i in range(len(g)):
            difference = g[i] - private.previous_gradient[i]
            sy += private.previous_step[i] * difference
            yy += difference * difference
            ss += private.previous_step[i] * private.previous_step[i]
            sg += private.previous_step[i] * g[i]
    reliable = (
        private.have_pair
        and yy > 0.0
        and (sy > 1.4210854715202004e-14 * math.sqrt(ss * yy))
    )
    if reliable:
        run.events[56] += 1
        gamma = sy / yy
        coefficient = sg / sy
        correction = 0.0
        for i in range(len(g)):
            difference = g[i] - private.previous_gradient[i]
            intermediate = gamma * (g[i] - coefficient * difference)
            private.work[i] = -intermediate
            correction += difference * intermediate
        correction /= sy
        for i in range(len(g)):
            private.work[i] -= (coefficient - correction) * private.previous_step[i]
    else:
        run.events[57] += 1
        for i in range(len(g)):
            private.work[i] = -g[i] / private.scale
    diag_clock_before = private.clock
    diag_damping_before = private.damping
    diag_alignment_trigger = 0.0
    alignment = 0.0
    for i in range(len(g)):
        auxiliary = p[0][i] + private.damping * private.work[i]
        alignment += g[i] * (auxiliary - private.previous_auxiliary[i])
    if alignment > 0.0:
        run.events[43] += 1
        if private.clock > 0:
            run.events[44] += 1
        diag_alignment_trigger = 1.0
        private.clock = 0
    diag_clock_after_alignment = private.clock
    momentum = private.clock / (private.clock + 3.0)
    if private.clock > 0:
        run.events[51] += 1
    else:
        run.events[52] += 1
    length2 = 0.0
    for i in range(len(g)):
        auxiliary = p[0][i] + private.damping * private.work[i]
        displacement = (
            auxiliary - p[0][i] + momentum * (auxiliary - private.previous_auxiliary[i])
        )
        private.previous_auxiliary[i] = auxiliary
        private.work[i] = displacement
        length2 += displacement * displacement
    if not math.isfinite(length2):
        run.events[47] += 1
        private.valid = False
        return 0
    anchor_displacement = p[0]-anchor[0]
    cap2=max(ss, float(np.dot(anchor_displacement,anchor_displacement)))
    fraction = math.sqrt(cap2/length2) if cap2>0 and length2>cap2 else 1.0
    if fraction < 1.0:
        run.events[55] += 1
    actual2 = slope = 0.0
    for i in range(len(g)):
        location = p[0][i] + fraction * private.work[i]
        displacement = location - p[0][i]
        private.work[i] = location
        private.previous_step[i] = displacement
        private.previous_gradient[i] = g[i]
        actual2 += displacement * displacement
        slope += g[i] * displacement
    if actual2 <= 0.0 or not math.isfinite(actual2):
        run.events[48] += 1
        private.valid = False
        return 0
    diag_calls_before = run.calls
    diag_index = -1
    q = API.query(private.work, run)
    private.point = q
    private.have_pair = True
    if run.calls > diag_calls_before:
        run.events[53] += 1
        if private.diag_count < private.diag.shape[0]:
            diag_index = private.diag_count
            private.diag[diag_index, 0] = run.calls
            private.diag[diag_index, 1] = diag_clock_before
            private.diag[diag_index, 2] = diag_clock_after_alignment
            private.diag[diag_index, 3] = momentum
            private.diag[diag_index, 4] = diag_damping_before
            private.diag[diag_index, 5] = np.nan
            private.diag[diag_index, 6] = fraction
            private.diag[diag_index, 7] = diag_alignment_trigger
            private.diag[diag_index, 8] = -1.0
            private.diag[diag_index, 9] = private.clock
            private.diag_count += 1
        elif private.diag.shape[0] > 0:
            private.diag_dropped += 1
    if run.status != 0:
        return 0
    change = q[1] - p[1]
    if diag_index >= 0:
        private.diag[diag_index, 5] = change
    difference2 = 0.0
    for i in range(len(g)):
        difference = q[2][i] - g[i]
        difference2 += difference * difference
    observed = max(
        2.0 * math.sqrt(difference2 / actual2), 4.0 * max(0.0, change - slope) / actual2
    )
    if math.isfinite(observed):
        private.scale = max(private.scale, observed)
    if change > 0.0:
        run.events[45] += 1
        if private.clock > 0:
            run.events[46] += 1
        if diag_index >= 0:
            private.diag[diag_index, 8] = 1.0
        private.damping *= 0.5
        private.clock = 0
        private.previous_auxiliary[:] = q[0]
    else:
        run.events[54] += 1
        if diag_index >= 0:
            private.diag[diag_index, 8] = 0.0
        private.damping = min(1.0, 2.0 * private.damping)
        private.clock += 1
    if diag_index >= 0:
        private.diag[diag_index, 9] = private.clock
    if private.damping <= 0.0 or not math.isfinite(private.damping):
        run.events[49] += 1
        private.valid = False
        return 0
    return 0


def transport(private, point):
    """Move a proposal base while retaining its secant metric and momentum.

    previous_gradient stores g(base)-y after transport, not a queried gradient.
    Thus the same observed (s,y) pair defines the one-pair inverse BFGS map.
    Translating the auxiliary position preserves base-auxiliary momentum.
    """
    if not private.valid or point[1]>=private.point[1]:return False
    for i in range(len(point[0])):
        private.previous_gradient[i] += point[2][i]-private.point[2][i]
        private.previous_auxiliary[i] += point[0][i]-private.point[0][i]
    private.point=point
    private.diag_events[50] += 1
    return True
