"""Regularized NAG, a conditional residual probe, and two-point correction.

The forward core and paid replay use the same advance function. The estimate
quadratic and aggregate gap are not evaluated. log_mass is a stable encoding
of the retained mass A; negative infinity denotes the exact empty entry.
"""
import math
import numpy as np
from numba import njit, types
from numba.experimental import jitclass
from . import runtime as API, secant as MS


@jitclass([
    ('log_mass', types.float64), ('z', API.ARRAY), ('point', API.POINT_TYPE),
    ('entry_ratio', types.float64), ('backward', types.float64),
    ('previous_beta', types.float64), ('postcommit', types.boolean),
])
class State:
    def __init__(self, point):
        self.log_mass = -np.inf
        self.z = point[0].copy()
        self.point = point
        self.entry_ratio = 0.0
        self.backward = 0.0
        self.previous_beta = 0.0
        self.postcommit = False


@jitclass([('rows', types.float64[:, ::1]), ('count', types.int64)])
class Journal:
    def __init__(self, capacity):
        self.rows = np.full((capacity, 18), np.nan)
        self.count = 0


@njit(cache=False)
def norm(x):
    return math.sqrt(float(np.dot(x, x)))


@njit(cache=False)
def inverse_mass(log_mass):
    return np.inf if log_mass == -np.inf else math.exp(-log_mass)


@njit(cache=False)
def record(journal, code, state, run, lam, a=0., b=0., c=0., d=0., e=0., f=0., g=0., h=0.):
    if journal.count < journal.rows.shape[0]:
        row = journal.rows[journal.count]
        row[0] = run.calls
        row[1] = code
        row[2] = run.phase
        row[3] = lam
        row[4] = run.ell
        row[5] = state.log_mass
        row[6] = norm(state.z-state.point[0])
        row[7] = norm(state.point[2])
        row[8] = state.point[1]
        row[9] = state.entry_ratio
        row[10], row[11], row[12], row[13] = a, b, c, d
        row[14], row[15], row[16], row[17] = e, f, g, h
        journal.count += 1


@njit(cache=False)
def clone(state):
    p = state.point
    result = State((p[0].copy(), p[1], p[2].copy()))
    result.log_mass = state.log_mass
    result.z = state.z.copy()
    result.entry_ratio = state.entry_ratio
    result.backward = state.backward
    result.previous_beta = state.previous_beta
    result.postcommit = state.postcommit
    return result


@njit(cache=False)
def prepare(state, outer, lam, local, run, journal):
    sigma = lam/2.
    budget = run.eps**2/(128.*lam) if local else float(outer[2]@outer[2])/(2.*sigma)
    displacement = state.z-outer[0]
    raw = norm(displacement)
    radius = math.sqrt(2.*budget/(inverse_mass(state.log_mass)+sigma))
    fraction = min(1., radius/raw) if raw > 0. else 1.
    state.z = outer[0]+fraction*displacement
    state.point = outer
    state.entry_ratio = 0. if state.log_mass == -np.inf else 1.
    state.backward, state.previous_beta = 0., 0.
    state.postcommit = False
    run.events[100+int(local)] += 1
    run.events[102] += int(state.log_mass != -np.inf)
    run.events[103] += int(fraction < 1.)
    record(journal, 1, state, run, lam, raw, fraction, budget, radius)


@njit(cache=False)
def parameters(log_mass, lam, ell):
    """Return eta, theta, A/a, and the new log mass without forming A."""
    sigma = lam/2.
    eta = 1./(lam+ell)
    denominator = ell+sigma
    if log_mass == -np.inf:
        return eta, 1., 0., -math.log(denominator), 0.
    inverse = inverse_mass(log_mass)
    # Scale the positive quadratic root before taking its discriminant.
    scale = max(inverse, sigma, denominator)
    r, s, d = inverse/scale, sigma/scale, denominator/scale
    beta = 2.*d/(r+2.*s+math.sqrt((r+2.*s)**2+4.*d*(r+s)))
    theta = 1./(beta+1.+s/(r+s))
    return eta, theta, beta, log_mass+math.log1p(1./beta), beta/(1.+beta)


@njit(cache=False)
def remainder(p, q, coefficient):
    d = p[0]-q[0]
    return p[1]-q[1]-float(q[2]@d)+coefficient*float(d@d)


@njit(cache=False)
def objective(p, outer, lam):
    s = p[0]-outer[0]
    return p[1]+lam*float(s@s)/2.


@njit(cache=False)
def actual_output(run, outer, lam, local):
    return local and run.best[1] <= outer[1]-run.eps**2/(128.*lam)


@njit(cache=False)
def advance(state, outer, lam, run, journal, local, replay=False, target=None):
    """0=commit, 1=outer output, 2=pair, 3=stop, 4=replay, 5=history Q.

    A replay checks its fixed target only at regenerated committed supports.
    It reproduces all core smoothness decisions and conditional probes.
    """
    p = state.point
    sigma = lam/2.
    s = p[0]-outer[0]
    s2 = float(s@s)
    residual = p[2]+lam*s
    residual2 = float(residual@residual)
    if residual2 <= lam*lam*s2/4.:
        if residual2 == 0.:
            corrected = p
        else:
            run.kind = 6
            corrected = API.query(p[0]-residual/lam, run)
        if run.status:
            return 3, run.last, run.last
        if actual_output(run, outer, lam, local):
            return 1, run.best, run.best
        decrease = outer[1]-corrected[1]
        gradient2 = float(corrected[2]@corrected[2])
        if decrease >= lam*s2/4. and gradient2 <= 4.*lam*lam*s2:
            run.events[104] += 1
            record(journal, 10, state, run, lam, residual2, s2, decrease, gradient2)
            return 1, corrected, corrected
        run.events[105] += 1
        record(journal, 9, state, run, lam, residual2, s2, decrease, gradient2)
    reverse = remainder(outer, p, lam/4.)
    if reverse < 0.:
        run.events[106] += 1
        record(journal, 5, state, run, lam, reverse)
        return 2, outer, p

    if state.postcommit:
        factor = state.entry_ratio+inverse_mass(state.log_mass)/sigma
        g02 = float(outer[2]@outer[2])
        while residual2 > 22.*factor*g02*(lam+run.ell)/sigma:
            run.events[107] += 1
            record(journal, 6, state, run, lam, residual2, 22.*factor*g02*(lam+run.ell)/sigma)
            run.kind = 4
            u = API.query(p[0]-residual/(lam+run.ell), run)
            if run.status:
                return 3, run.last, run.last
            if actual_output(run, outer, lam, local):
                return 1, run.best, run.best
            slack = objective(p, outer, lam)-objective(u, outer, lam)-residual2/(2.*(lam+run.ell))
            if slack < 0.:
                run.ell *= 2.
                run.events[108] += 1
                record(journal, 7, state, run, lam, slack)
                if not math.isfinite(run.ell):
                    raise ArithmeticError('probe scale overflow')
                continue
            run.events[109] += 1
            reverse = remainder(u, outer, lam/4.)
            record(journal, 8, state, run, lam, reverse, slack)
            if reverse < 0.:
                return 2, u, outer
            return 4, u, u

    # Only the start of a new NAG operation lowers the common search scale.
    run.ell /= 2.
    run.events[110] += 1
    while True:
        if not (0. < run.ell < np.inf):
            raise ArithmeticError('NAG search scale is invalid')
        eta, theta, beta, log_mass_new, retained = parameters(state.log_mass, lam, run.ell)
        y_location = p[0]+theta*(state.z-p[0])
        run.kind = 2
        y = p if np.array_equal(y_location, p[0]) else API.query(y_location, run)
        if run.status:
            return 3, run.last, run.last
        if actual_output(run, outer, lam, local):
            return 1, run.best, run.best
        direction = y[0]-p[0]
        length2 = float(direction@direction)
        tau = theta*state.previous_beta
        if tau > 0. and length2 > 0.:
            curvature = 2.*(remainder(y, p, 0.)+tau**3*state.backward)/((1.+tau)*length2)
            run.events[111] += 1
            if curvature < -lam/2.:
                run.events[112] += 1
                record(journal, 11, state, run, lam, curvature, tau, state.backward, length2)
                return 5, p, y
        current_comparison = remainder(p, y, lam/2.)
        if current_comparison < 0.:
            run.events[113] += 1
            record(journal, 3, state, run, lam, current_comparison)
            return 2, p, y
        outer_comparison = remainder(outer, y, lam/4.)
        if outer_comparison < 0.:
            run.events[114] += 1
            record(journal, 4, state, run, lam, outer_comparison)
            return 2, outer, y
        gradient = y[2]+lam*(y[0]-outer[0])
        run.kind = 3
        q = API.query(y[0]-eta*gradient, run)
        if run.status:
            return 3, run.last, run.last
        if actual_output(run, outer, lam, local):
            return 1, run.best, run.best
        slack = objective(y, outer, lam)-objective(q, outer, lam)-eta*float(gradient@gradient)/2.
        if slack < 0.:
            run.ell *= 2.
            run.events[115] += 1
            continue
        displacement = q[0]-p[0]
        old_z_distance2 = 0.
        if journal.count < journal.rows.shape[0]:
            z_difference = state.z-y[0]
            old_z_distance2 = float(z_difference@z_difference)
        state.log_mass = log_mass_new
        state.entry_ratio *= retained
        state.z = q[0]+beta*displacement
        state.backward = p[1]-q[1]+float(q[2]@displacement)
        state.previous_beta = beta
        state.point = q
        state.postcommit = True
        run.events[116] += 1
        if journal.count < journal.rows.shape[0]:
            record(journal, 2, state, run, lam, theta, beta, retained, slack,
                   current_comparison, outer_comparison, old_z_distance2, norm(displacement))
        if replay and target is not None and remainder(target, y, lam/4.) < 0.:
            return 2, target, y
        return 0, q, y


@njit(cache=False)
def calibrate(outer, run):
    scale = run.L
    direction = 0
    last_pass = scale
    g2 = float(outer[2]@outer[2])
    while True:
        if not 0. < scale < np.inf:
            raise ArithmeticError('initial scale invalid')
        run.kind = 1
        p = API.query(outer[0]-outer[2]/scale, run)
        if run.status:
            return scale
        passed = p[1] <= outer[1]-g2/(2.*scale)
        if direction == 0:
            direction = -1 if passed else 1
        elif direction == 1 and passed:
            return scale
        elif direction == -1 and not passed:
            return last_pass
        if passed:
            last_pass = scale
        scale = math.ldexp(scale, direction)


@njit(cache=False)
def replay_pair(entry, entry_ell, target, outer, lam, run, local, forward_cost):
    shadow = API.Runtime(run.L, run.eps, min(run.max_queries, run.calls+forward_cost), len(outer[0]), run.meter)
    shadow.calls = run.calls
    shadow.ell = entry_ell
    shadow.events = np.zeros_like(run.events)
    shadow.phase = 3
    shadow.last, shadow.best, shadow.minimum = outer, outer, outer
    replay_state = clone(entry)
    journal = Journal(0)
    before = run.calls
    found = False
    u, w = target, target
    while not shadow.status:
        action, u, w = advance(replay_state, outer, lam, shadow, journal, local, True, target)
        if action == 2 and np.array_equal(u[0], target[0]):
            found = True
            break
        if action != 0:
            if shadow.status:
                break
            raise ArithmeticError('replay reached an earlier boundary without its negative support')
    run.calls = shadow.calls
    run.counts += shadow.counts
    run.last = shadow.last
    if shadow.best[1] < run.best[1]:
        run.best = shadow.best
    if norm(shadow.minimum[2]) < norm(run.minimum[2]):
        run.minimum = shadow.minimum
    run.events[117] += 1
    run.events[118] += run.calls-before
    if shadow.status:
        run.status = shadow.status
        if shadow.status == 2 and run.calls < run.max_queries:
            raise ArithmeticError('replay exceeded its charged forward prefix')
    if not found and not run.status:
        raise ArithmeticError('negative support was not recovered')
    return u, w


@njit(cache=False)
def private_step(outer, state, run, private):
    run.events[18] += int(MS.transport(private, state.point))
    run.kind = 5
    before = run.calls
    MS.resume(outer, run, private)
    run.events[119] += run.calls-before


@njit(cache=False)
def solve(x0, run, private, journal, private_enabled=True):
    if run.events.shape[0] < 128:
        expanded = np.zeros(128, dtype=np.int64)
        expanded[:run.events.shape[0]] = run.events
        run.events = expanded
    run.kind, run.phase = 0, 0
    outer = API.query(x0, run)
    if run.status:
        return run.last
    b_cal = calibrate(outer, run)
    if run.status:
        return run.last
    run.L, run.ell = b_cal, b_cal
    lam = 2.*b_cal
    state = State(outer)
    local = False
    while not run.status:
        run.phase = 2 if local else 1
        prepare(state, outer, lam, local, run, journal)
        entry, entry_ell = clone(state), run.ell
        forward_cost = 0
        delta = run.eps**2/(128.*lam)
        if not local and private_enabled:
            run.events[18] += int(MS.transport(private, outer))
            observed = private.scale if private.valid else b_cal
            ratio = (max(run.ell, observed)+lam)/(lam/2.)
            quota = 2+int(math.ceil(4.*math.sqrt(ratio)*math.asinh(math.sqrt(ratio))))
            run.events[120] += quota
            for _ in range(quota):
                # Preserve the uninterrupted private orbit inside its entry
                # quota. Only entry and postcommit stages transport its base.
                run.kind = 5
                before_private = run.calls
                MS.resume(outer, run, private)
                run.events[119] += run.calls-before_private
                if run.status:
                    return run.last
        while not run.status:
            before = run.calls
            action, u, w = advance(state, outer, lam, run, journal, local)
            forward_cost += run.calls-before
            if run.status:
                return run.last
            if action == 1:
                chosen = u
                break
            if action == 0:
                if private_enabled:
                    private_step(outer, state, run, private)
                    if run.status:
                        return run.last
                if actual_output(run, outer, lam, local):
                    chosen = run.best
                    break
                continue
            if action not in (2, 4, 5):
                raise ArithmeticError('invalid inner output')
            if run.best[1] <= outer[1]-delta:
                chosen = run.best
                break
            if not local:
                # An ordinary failure needs no direction yet. Support recovery
                # can wait for the local call, where an escape will use it.
                local = True
                chosen = outer
                break
            if action == 4:
                u, w = replay_pair(entry, entry_ell, u, outer, lam, run, local, forward_cost)
                if run.status:
                    return run.last
            if action == 5:
                center = u
                direction = w[0]-u[0]
            else:
                center = w
                direction = u[0]-w[0]
            distance = norm(direction)
            if distance <= 0. or distance > run.eps/(2.*lam)*(1.+1e-12):
                raise ArithmeticError('directional comparison is not local')
            direction = direction/distance
            radius = 2.*run.eps/lam
            run.kind = 7
            run.events[121] += 1
            API.query(center[0]+radius*direction, run)
            if run.status:
                return run.last
            if run.best[1] > outer[1]-delta:
                API.query(center[0]-radius*direction, run)
                if run.status:
                    return run.last
            if run.best[1] <= outer[1]-delta:
                chosen = run.best
                break
            run.events[122] += 1
            record(journal, 13, state, run, lam)
            lam *= 2.
            local = False
            chosen = outer
            break
        if np.array_equal(chosen[0], outer[0]):
            continue
        old_norm, new_norm = norm(outer[2]), norm(chosen[2])
        if journal.count < journal.rows.shape[0]:
            record(journal, 12, state, run, lam, old_norm, new_norm,
                   outer[1]-chosen[1], norm(chosen[0]-outer[0]))
        run.events[123] += 1
        if new_norm > old_norm/2.:
            lam /= 2.
            run.events[124] += 1
        outer = chosen
        local = False
    return run.last
