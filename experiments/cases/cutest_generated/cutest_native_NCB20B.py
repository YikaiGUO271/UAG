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
def eBP_2(EV_, iel_):
    S = 1.0
    D1 = S + EV_[0, 0] * EV_[0, 0]
    Y1 = EV_[0, 0] / D1
    DY1 = (1.0 - 2.0 * EV_[0, 0] * EV_[0, 0] / D1) / D1
    D2Y1 = (8.0 * EV_[0, 0] ** 3 / D1 - 6.0 * EV_[0, 0]) / D1 ** 2
    D2 = S + EV_[1, 0] * EV_[1, 0]
    Y2 = EV_[1, 0] / D2
    DY2 = (1.0 - 2.0 * EV_[1, 0] * EV_[1, 0] / D2) / D2
    D2Y2 = (8.0 * EV_[1, 0] ** 3 / D2 - 6.0 * EV_[1, 0]) / D2 ** 2
    D3 = S + EV_[2, 0] * EV_[2, 0]
    Y3 = EV_[2, 0] / D3
    DY3 = (1.0 - 2.0 * EV_[2, 0] * EV_[2, 0] / D3) / D3
    D2Y3 = (8.0 * EV_[2, 0] ** 3 / D3 - 6.0 * EV_[2, 0]) / D3 ** 2
    D4 = S + EV_[3, 0] * EV_[3, 0]
    Y4 = EV_[3, 0] / D4
    DY4 = (1.0 - 2.0 * EV_[3, 0] * EV_[3, 0] / D4) / D4
    D2Y4 = (8.0 * EV_[3, 0] ** 3 / D4 - 6.0 * EV_[3, 0]) / D4 ** 2
    D5 = S + EV_[4, 0] * EV_[4, 0]
    Y5 = EV_[4, 0] / D5
    DY5 = (1.0 - 2.0 * EV_[4, 0] * EV_[4, 0] / D5) / D5
    D2Y5 = (8.0 * EV_[4, 0] ** 3 / D5 - 6.0 * EV_[4, 0]) / D5 ** 2
    D6 = S + EV_[5, 0] * EV_[5, 0]
    Y6 = EV_[5, 0] / D6
    DY6 = (1.0 - 2.0 * EV_[5, 0] * EV_[5, 0] / D6) / D6
    D2Y6 = (8.0 * EV_[5, 0] ** 3 / D6 - 6.0 * EV_[5, 0]) / D6 ** 2
    D7 = S + EV_[6, 0] * EV_[6, 0]
    Y7 = EV_[6, 0] / D7
    DY7 = (1.0 - 2.0 * EV_[6, 0] * EV_[6, 0] / D7) / D7
    D2Y7 = (8.0 * EV_[6, 0] ** 3 / D7 - 6.0 * EV_[6, 0]) / D7 ** 2
    D8 = S + EV_[7, 0] * EV_[7, 0]
    Y8 = EV_[7, 0] / D8
    DY8 = (1.0 - 2.0 * EV_[7, 0] * EV_[7, 0] / D8) / D8
    D2Y8 = (8.0 * EV_[7, 0] ** 3 / D8 - 6.0 * EV_[7, 0]) / D8 ** 2
    D9 = S + EV_[8, 0] * EV_[8, 0]
    Y9 = EV_[8, 0] / D9
    DY9 = (1.0 - 2.0 * EV_[8, 0] * EV_[8, 0] / D9) / D9
    D2Y9 = (8.0 * EV_[8, 0] ** 3 / D9 - 6.0 * EV_[8, 0]) / D9 ** 2
    D10 = S + EV_[9, 0] * EV_[9, 0]
    Y10 = EV_[9, 0] / D10
    DY10 = (1.0 - 2.0 * EV_[9, 0] * EV_[9, 0] / D10) / D10
    D2Y10 = (8.0 * EV_[9, 0] ** 3 / D10 - 6.0 * EV_[9, 0]) / D10 ** 2
    D11 = S + EV_[10, 0] * EV_[10, 0]
    Y11 = EV_[10, 0] / D11
    DY11 = (1.0 - 2.0 * EV_[10, 0] * EV_[10, 0] / D11) / D11
    D2Y11 = (8.0 * EV_[10, 0] ** 3 / D11 - 6.0 * EV_[10, 0]) / D11 ** 2
    D12 = S + EV_[11, 0] * EV_[11, 0]
    Y12 = EV_[11, 0] / D12
    DY12 = (1.0 - 2.0 * EV_[11, 0] * EV_[11, 0] / D12) / D12
    D2Y12 = (8.0 * EV_[11, 0] ** 3 / D12 - 6.0 * EV_[11, 0]) / D12 ** 2
    D13 = S + EV_[12, 0] * EV_[12, 0]
    Y13 = EV_[12, 0] / D13
    DY13 = (1.0 - 2.0 * EV_[12, 0] * EV_[12, 0] / D13) / D13
    D2Y13 = (8.0 * EV_[12, 0] ** 3 / D13 - 6.0 * EV_[12, 0]) / D13 ** 2
    D14 = S + EV_[13, 0] * EV_[13, 0]
    Y14 = EV_[13, 0] / D14
    DY14 = (1.0 - 2.0 * EV_[13, 0] * EV_[13, 0] / D14) / D14
    D2Y14 = (8.0 * EV_[13, 0] ** 3 / D14 - 6.0 * EV_[13, 0]) / D14 ** 2
    D15 = S + EV_[14, 0] * EV_[14, 0]
    Y15 = EV_[14, 0] / D15
    DY15 = (1.0 - 2.0 * EV_[14, 0] * EV_[14, 0] / D15) / D15
    D2Y15 = (8.0 * EV_[14, 0] ** 3 / D15 - 6.0 * EV_[14, 0]) / D15 ** 2
    D16 = S + EV_[15, 0] * EV_[15, 0]
    Y16 = EV_[15, 0] / D16
    DY16 = (1.0 - 2.0 * EV_[15, 0] * EV_[15, 0] / D16) / D16
    D2Y16 = (8.0 * EV_[15, 0] ** 3 / D16 - 6.0 * EV_[15, 0]) / D16 ** 2
    D17 = S + EV_[16, 0] * EV_[16, 0]
    Y17 = EV_[16, 0] / D17
    DY17 = (1.0 - 2.0 * EV_[16, 0] * EV_[16, 0] / D17) / D17
    D2Y17 = (8.0 * EV_[16, 0] ** 3 / D17 - 6.0 * EV_[16, 0]) / D17 ** 2
    D18 = S + EV_[17, 0] * EV_[17, 0]
    Y18 = EV_[17, 0] / D18
    DY18 = (1.0 - 2.0 * EV_[17, 0] * EV_[17, 0] / D18) / D18
    D2Y18 = (8.0 * EV_[17, 0] ** 3 / D18 - 6.0 * EV_[17, 0]) / D18 ** 2
    D19 = S + EV_[18, 0] * EV_[18, 0]
    Y19 = EV_[18, 0] / D19
    DY19 = (1.0 - 2.0 * EV_[18, 0] * EV_[18, 0] / D19) / D19
    D2Y19 = (8.0 * EV_[18, 0] ** 3 / D19 - 6.0 * EV_[18, 0]) / D19 ** 2
    D20 = S + EV_[19, 0] * EV_[19, 0]
    Y20 = EV_[19, 0] / D20
    DY20 = (1.0 - 2.0 * EV_[19, 0] * EV_[19, 0] / D20) / D20
    D2Y20 = (8.0 * EV_[19, 0] ** 3 / D20 - 6.0 * EV_[19, 0]) / D20 ** 2
    SUM = Y1 + Y2 + Y3 + Y4 + Y5 + Y19 + Y20 + Y6 + Y7 + Y8 + Y9 + Y10 + Y11 + Y12 + Y13 + Y14 + Y15 + Y16 + Y17 + Y18
    TWOSUM = SUM + SUM
    f_ = SUM * SUM
    dim = len(EV_)
    g_ = np.zeros(dim)
    g_[0] = TWOSUM * DY1
    g_[1] = TWOSUM * DY2
    g_[2] = TWOSUM * DY3
    g_[3] = TWOSUM * DY4
    g_[4] = TWOSUM * DY5
    g_[5] = TWOSUM * DY6
    g_[6] = TWOSUM * DY7
    g_[7] = TWOSUM * DY8
    g_[8] = TWOSUM * DY9
    g_[9] = TWOSUM * DY10
    g_[10] = TWOSUM * DY11
    g_[11] = TWOSUM * DY12
    g_[12] = TWOSUM * DY13
    g_[13] = TWOSUM * DY14
    g_[14] = TWOSUM * DY15
    g_[15] = TWOSUM * DY16
    g_[16] = TWOSUM * DY17
    g_[17] = TWOSUM * DY18
    g_[18] = TWOSUM * DY19
    g_[19] = TWOSUM * DY20
    return (f_, g_)

@njit(cache=True, error_model='numpy')
def eQR_2(EV_, iel_):
    f_ = EV_[0, 0] ** 4
    dim = len(EV_)
    g_ = np.zeros(dim)
    g_[0] = 4.0 * EV_[0, 0] ** 3
    return (f_, g_)

@njit(cache=True, error_model="numpy")
def element(t, ev, iel):
    if t == 0:
        return eBP_2(ev, iel)
    if t == 1:
        return eQR_2(ev, iel)
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
