"""Numerical functions extracted from R7_policy and rb_policy; no dataset I/O."""
from concurrent.futures import ThreadPoolExecutor
import ast
import numpy as np
import scipy.linalg as sla

def kridge_fit_predict(Ktr, Ytr, Kte, alpha):
    mu = Ytr.mean(0)
    A = sla.solve(Ktr + alpha * np.eye(len(Ktr)), Ytr - mu, assume_a='pos')
    return Kte @ A + mu

def route(Vh, Ch, lam, allowed):
    sc = Vh[:, allowed] - lam * Ch[:, allowed]
    return np.asarray(allowed)[np.argmax(sc, axis=1)]

def budget_route(Vh, Ch, B, allowed, bounds=(-4.0, 7.0)):
    """Max predicted quality s.t. mean predicted cost <= B on the routed batch (Lagrangian; bisection on log lambda)."""
    lo, hi = bounds
    if Ch[np.arange(len(Ch)), route(Vh, Ch, 10 ** lo, allowed)].mean() <= B:
        return route(Vh, Ch, 10 ** lo, allowed), 10 ** lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if Ch[np.arange(len(Ch)), route(Vh, Ch, 10 ** mid, allowed)].mean() <= B: hi = mid
        else: lo = mid
    return route(Vh, Ch, 10 ** hi, allowed), 10 ** hi

def pick_points(recs):
    """(i) max probing efficiency; (ii) closest to eps=0 on probing: best probing efficiency among points with probing dV >= 0
    (if none, the point with the largest probing dV)."""
    i1 = int(np.argmax([r['p_r'] for r in recs]))
    ok = [j for j, r in enumerate(recs) if r['p_dV'] >= 0]
    i2 = max(ok, key=lambda j: recs[j]['p_r']) if ok else int(np.argmax([r['p_dV'] for r in recs]))
    return i1, i2

def kernel_rows(A, BT, chunk=1000, workers=8):
    """Dense float32 block A @ B^T of unit-norm TF-IDF rows (cosine similarities), computed in threaded row chunks."""
    out = np.empty((A.shape[0], BT.shape[1]), np.float32)

    def job(i):
        j = min(i + chunk, A.shape[0])
        out[i:j] = (A[i:j] @ BT).toarray()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, range(0, A.shape[0], chunk)))
    return out

def kr_solve(K, idx, Y, alpha, n_refine=1, chunk=2048):
    """(K[idx, idx] + alpha I) A = Y by float32 Cholesky + float64 iterative refinement (residual streamed from K in row
    chunks, so only one m x m float32 matrix is held).  Returns A and the final relative residual."""
    m = len(idx)
    full = m == K.shape[0] and bool(np.all(idx == np.arange(m)))
    M = K.copy() if full else K[np.ix_(idx, idx)]
    ar_ = np.arange(m); M[ar_, ar_] += np.float32(alpha)
    cf = sla.cho_factor(M.T, lower=False, overwrite_a=True, check_finite=False)     # M symmetric; M.T is F-contiguous

    def resid(A):
        R = Y - alpha * A
        for i in range(0, m, chunk):
            Kc = K[i:i + chunk] if full else K[np.ix_(idx[i:i + chunk], idx)]
            R[i:i + chunk] -= Kc.astype(np.float64) @ A
        return R
    A = sla.cho_solve(cf, Y.astype(np.float32), check_finite=False).astype(np.float64)
    for _ in range(n_refine):
        A += sla.cho_solve(cf, resid(A).astype(np.float32), check_finite=False)
    rel = float(np.linalg.norm(resid(A)) / np.linalg.norm(Y))
    del M, cf
    return A, rel

def paper_split(names, reference_score, seed):   # verbatim logic of router_selection_recheck.split
    strata = np.array([f"{name}|{int(value >= .5)}" for name, value in zip(names, reference_score)])
    fit = np.zeros(len(strata), bool)
    generator = np.random.default_rng(seed)
    for name in sorted(set(strata)):
        ids = np.flatnonzero(strata == name)
        fit[generator.permutation(ids)[:int(round(.7 * len(ids)))]] = True
    return fit

def parse_prompt(p):
    try:
        L = ast.literal_eval(p)
        if isinstance(L, (list, tuple)):
            return '\n'.join(str(x) for x in L), True
    except Exception:
        pass
    return str(p), False

def strat_weights(strata, nboot, seed):
    rng = np.random.default_rng(seed)
    n = len(strata); W = np.zeros((nboot, n))
    for s in sorted(set(strata)):
        ids = np.flatnonzero(strata == s)
        W[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=nboot)
    return np.vstack([np.ones((1, n)), W])
