"""Data-independent price path, fractional program and fixed-sequence test functions.
These functions are shared with the RouterBench paper replay.
"""
import numpy as np
from scipy.optimize import linprog
TOL_MASS = 1e-8
REL_TOL = 1e-12
ALPHA_CERT = 0.05

def primitives(s, X, V, D, w=None):
    """pi(x), v(x,u), c(x,u) from rows with states s (optional weights); empty states get pi = 0 and the pooled v, c."""
    n, U_ = V.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    nx = np.bincount(s, weights=w, minlength=X)
    v = np.stack([np.bincount(s, weights=w * V[:, u], minlength=X) for u in range(U_)], 1)
    c = np.stack([np.bincount(s, weights=w * D[:, u], minlength=X) for u in range(U_)], 1)
    pv = (w[:, None] * V).sum(0) / w.sum(); pc = (w[:, None] * D).sum(0) / w.sum()
    empty = nx <= 0
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.where(empty[:, None], pv[None, :], v / np.maximum(nx, 1e-300)[:, None])
        c = np.where(empty[:, None], pc[None, :], c / np.maximum(nx, 1e-300)[:, None])
    return nx / nx.sum(), v, c

def phi(lam, pi, v, c):
    return float(pi @ np.max(v - lam * c, axis=1))

def ratio_opt(pi, v, c):
    """eta* = unique root of Phi (bisection, as X9); deterministic argmax rule at eta* as one-hot g."""
    lo, hi = 0.0, 1.0
    while phi(hi, pi, v, c) > 0:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid, pi, v, c) > 0:
            lo = mid
        else:
            hi = mid
    root = 0.5 * (lo + hi)
    return root, np.eye(v.shape[1])[np.argmax(v - root * c, axis=1)]

def cc_lp(pi, v, c, floor_value):
    """Charnes-Cooper LP of X9 (eq:app-cc-lp): max E[V]/E[D] s.t. E[V] >= floor_value over randomized state-wise routers."""
    X, U_ = v.shape; nY = X * U_
    keep = pi > 0
    cobj = np.concatenate([-v.ravel(), [0.0, 0.0]])
    A = np.zeros((X + 2, nY + 2)); A[0, :nY] = c.ravel()
    for x in range(X):
        A[1 + x, x * U_:(x + 1) * U_] = 1.0; A[1 + x, nY] = -pi[x]
    A[X + 1, :nY] = v.ravel(); A[X + 1, nY] = -floor_value; A[X + 1, nY + 1] = -1.0
    b = np.zeros(X + 2); b[0] = 1.0
    bounds = [(0, None) if keep[x] else (0, 0) for x in range(X) for _ in range(U_)] + [(0, None), (0, None)]
    res = linprog(cobj, A_eq=A, b_eq=b, bounds=bounds, method='highs-ds')
    if res.status != 0:
        return None
    y = res.x[:nY].reshape(X, U_); t = res.x[nY]
    g = np.zeros((X, U_))
    g[keep] = y[keep] / (pi[keep, None] * t)
    g = np.where(g < TOL_MASS, 0.0, g)
    g[~keep] = 0.0
    eta = float(-res.fun)
    for x in np.where(~keep)[0]:
        g[x, np.argmax(v[x] - eta * c[x])] = 1.0
    g = g / g.sum(1, keepdims=True)
    kappa = max(float(res.eqlin.marginals[X + 1]), 0.0)
    return dict(eta=eta, g=g, kappa=kappa, binding=bool(res.x[nY + 1] < 1e-9), price=eta / (1 + kappa),
                n_rand=int(((g > 0).sum(1) >= 2).sum()))

def state_envelope(vx, cx, lam_max):
    """Upper envelope of u -> v_u - lam c_u on lam in [0, lam_max] (R8): start at argmax v (ties -> cheaper, lower index);
    at each breakpoint move to the cheaper action (ties -> cheapest, lower index)."""
    vmax = vx.max()
    cand = np.where(vx >= vmax - REL_TOL)[0]
    a = int(cand[np.lexsort((cand, cx[cand]))][0])
    a0, lam, ev = a, 0.0, []
    while True:
        cheaper = np.where(cx < cx[a])[0]
        if len(cheaper) == 0:
            break
        lams = np.maximum((vx[a] - vx[cheaper]) / (cx[a] - cx[cheaper]), lam)
        lmin = float(lams.min())
        if lmin > lam_max:
            break
        cand = cheaper[lams <= lmin + REL_TOL * max(1.0, lmin)]
        a = int(cand[np.lexsort((cand, cx[cand]))][0])
        lam = lmin; ev.append((lam, a))
    return a0, ev

def price_path(pi, v, c, lam_max):
    X = v.shape[0]
    g = np.empty(X, int); events = []
    for x in range(X):
        a0, ev = state_envelope(v[x], c[x], lam_max)
        g[x] = a0; events += [(lam, x, a) for lam, a in ev]
    events.sort(key=lambda e: (e[0], e[1]))
    pols, lams = [g.copy()], [0.0]
    i = 0
    while i < len(events):
        lam = events[i][0]; j = i
        while j < len(events) and events[j][0] <= lam + REL_TOL * max(1.0, lam):
            g[events[j][1]] = events[j][2]; j += 1
        pols.append(g.copy()); lams.append(lam); i = j
    P = np.array(pols); rows = np.arange(X)
    EV = np.array([pi @ v[rows, p] for p in P]); ED = np.array([pi @ c[rows, p] for p in P])
    return dict(P=P, lam=np.array(lams), EV=EV, ED=ED, K=len(P))

def pvalues(Zk, eps):
    """One-sided p-values for H_k: Q(g_k) < -eps from K x n paired differences in [-1, 1] (R8): Hoeffding and empirical
    Bernstein (Maurer & Pontil 2009, Thm 4 on W = (Z+1)/2, closed-form inversion)."""
    K, n = Zk.shape
    Q = Zk.mean(1)
    out = dict(Qhat=Q, hoeffding=np.exp(-n * np.maximum(Q + eps, 0.0) ** 2 / 2.0))
    W = (Zk + 1.0) / 2.0; wbar = W.mean(1); wvar = W.var(1, ddof=1); w0 = (1.0 - eps) / 2.0
    a = np.sqrt(2.0 * np.maximum(wvar, 0.0) / n); b = 7.0 / (3.0 * (n - 1)); d = wbar - w0
    s = (-a + np.sqrt(a * a + 4 * b * np.maximum(d, 0.0))) / (2 * b)
    out['emp_bernstein'] = np.where(d > 0, np.minimum(1.0, 2.0 * np.exp(-s * s)), 1.0)
    return out

def fixed_sequence(p, alpha=ALPHA_CERT):
    fail = np.where(p > alpha)[0]
    return int(fail[0]) if len(fail) else len(p)
