# Generated from frozen official S2MPJ analytic callbacks; do not hand edit.
import numpy as np
from pathlib import Path
from numba import njit
data = np.load(Path(__file__).with_suffix('.npz'))
globals().update({key: data[key] for key in data.files})
@njit(cache=True, error_model="numpy")
def to_scalar(v):
    return v.item()


@njit(cache=True, error_model='numpy')
def eTG_2(EV_, iel_):
    U_ = np.zeros((2, 3))
    IV_ = np.zeros(2)
    U_[0, 0] = U_[0, 0] + 1
    U_[0, 1] = U_[0, 1] - 1
    U_[1, 2] = U_[1, 2] + 1
    IV_[0] = to_scalar(U_[0:1, :].dot(EV_))
    IV_[1] = to_scalar(U_[1:2, :].dot(EV_))
    ALPHA = 0.1
    U1SQ = IV_[0] * IV_[0]
    U2SQ = IV_[1] * IV_[1]
    T = ALPHA + U2SQ
    T2 = T * T
    AU2SQ = elpar[iel_][0] + U2SQ
    EXPA = np.exp(-U1SQ / T)
    EXPA1 = -2.0 * IV_[0] * EXPA / T
    EXPA2 = 2.0 * U1SQ * IV_[1] * EXPA / T2
    EXPA11 = -2.0 * (EXPA + IV_[0] * EXPA1) / T
    EXPA12 = 2.0 * IV_[0] * (2.0 * IV_[1] * EXPA / T - EXPA2) / T
    EXPA22 = 2.0 * U1SQ * (IV_[1] * EXPA2 + EXPA * (1.0 - 4.0 * U2SQ / T)) / T2
    TMEXPA = 2.0 - EXPA
    f_ = AU2SQ * TMEXPA
    dim = len(IV_)
    g_ = np.zeros(dim)
    g_[0] = -AU2SQ * EXPA1
    g_[1] = -AU2SQ * EXPA2 + 2.0 * IV_[1] * TMEXPA
    g_ = U_.T.dot(g_)
    return (f_, g_)

@njit(cache=True, error_model="numpy")
def element(t, ev, iel):
    if t == 0:
        return eTG_2(ev, iel)
    return 0., np.empty(0)


@njit(cache=True, error_model="numpy")
def group(t, fin, ig):
    return fin, 1.


@njit(cache=True, error_model="numpy")
def fg(x):
    n = len(x)
    ne = len(eltype)
    ef = np.empty(ne)
    eg = np.empty(len(elidx))
    # Shared nonlinear elements are evaluated once, then reused in each group.
    for ie in range(ne):
        lo, hi = elptr[ie], elptr[ie+1]
        ev = np.empty((hi-lo, 1))
        for k in range(lo, hi):
            ev[k-lo, 0] = x[elidx[k]]
        f, g = element(eltype[ie], ev, ie)
        ef[ie] = f
        for k in range(lo, hi):
            eg[k] = g[k-lo]
    ftotal = 0.
    gradient = np.zeros(n)
    for i in range(n):
        hx = 0.
        for k in range(hptr[i], hptr[i+1]):
            hx += hval[k] * x[hidx[k]]
        gradient[i] = hx
        ftotal += 0.5*x[i]*hx
    # Accumulate only the sparse support of each group's exact analytic gradient.
    for ig in objgrps:
        fin = -gconst[ig]
        for k in range(aptr[ig], aptr[ig+1]):
            fin += aval[k]*x[aidx[k]]
        for k in range(grptr[ig], grptr[ig+1]):
            fin += grw[k]*ef[grelt[k]]
        fa, ga = group(grtype[ig], fin, ig)
        ftotal += fa/gscale[ig]
        factor = ga/gscale[ig]
        for k in range(aptr[ig], aptr[ig+1]):
            gradient[aidx[k]] += factor*aval[k]
        for k in range(grptr[ig], grptr[ig+1]):
            ie = grelt[k]
            weight = factor*grw[k]
            for j in range(elptr[ie], elptr[ie+1]):
                gradient[elidx[j]] += weight*eg[j]
    return ftotal, gradient

@njit(cache=True, error_model="numpy")
def value(x):
    return fg(x)[0]
