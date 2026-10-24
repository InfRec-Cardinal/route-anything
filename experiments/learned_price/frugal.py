"""FrugalGPT-style prompt/response scorer and exhaustive chain search.

Extracted/adapted from R8_baselines; this is a local baseline adaptation,
not the original FrugalGPT package. No model calls are made.
"""
import itertools
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from features import fit_block

N_TAU = 50
QGRID = np.arange(N_TAU) / N_TAU
C_GRID = [0.3, 1.0, 3.0, 10.0]
RESP_CHARS = 4000


def fit_scorers(train_rows, test_rows, models, scores, folds):
    """Original prompt+response+empty-flag features and per-model OOF AUC tuning."""
    train_prompts = [row["prompt"] for row in train_rows]
    test_prompts = [row["prompt"] for row in test_rows]
    prompt_features = fit_block(train_prompts)
    Pfit, Ptest = prompt_features(train_prompts), prompt_features(test_prompts)
    response_features = fit_block([(row["responses"].get(model) or "")[:RESP_CHARS]
                                   for model in models for row in train_rows])
    held, test, metadata = [], [], {}
    for index, model in enumerate(models):
        def features(rows, prompt_block):
            responses = [(row["responses"].get(model) or "") for row in rows]
            response_block = response_features([response[:RESP_CHARS] for response in responses])
            empty = sparse.csr_matrix(np.array([not response.strip() for response in responses],
                                               dtype=np.float32)[:, None])
            return sparse.hstack([prompt_block, response_block, empty]).tocsr()
        Xfit, Xtest = features(train_rows, Pfit), features(test_rows, Ptest)
        labels = (scores[:, index] >= 1 - 1e-12).astype(int)
        best = None
        for regularization in C_GRID:
            prediction = np.empty(len(train_rows))
            for train, valid in folds:
                classifier = LogisticRegression(C=regularization, solver="liblinear", max_iter=1000).fit(Xfit[train], labels[train])
                prediction[valid] = classifier.predict_proba(Xfit[valid])[:, 1]
            auc = roc_auc_score(labels, prediction)
            if best is None or auc > best[1] + 1e-12:
                best = (regularization, auc, prediction)
        classifier = LogisticRegression(C=best[0], solver="liblinear", max_iter=1000).fit(Xfit, labels)
        held.append(best[2]); test.append(classifier.predict_proba(Xtest)[:, 1])
        metadata[model] = {"C": best[0], "oof_auc": float(best[1])}
    return np.column_stack(held), np.column_stack(test), metadata


def fit_chain(G, scores, costs, models, reference_index):
    # The original search routines use one immutable menu per CLI run.
    global CHAINS, MODELS, ETA_B_PROBING
    MODELS = list(models)
    CHAINS = [chain for length in (1, 2, 3) for chain in itertools.permutations(range(len(models)), length)]
    ETA_B_PROBING = scores[:, reference_index].sum() / costs[:, reference_index].sum()
    search = search_chains(G, scores, costs)
    result = pick(search, "eps0", len(scores), scores[:, reference_index].mean())
    result["searched_chains"] = len(CHAINS)
    result["searched_configurations"] = len(search["SV"])
    return result

def search_chains(G, Y, Cm):
    """Aggregate value/cost sums of every (chain, thresholds) on the fitting rows. G = out-of-fold scores."""
    TAU = np.stack([np.quantile(G[:, m], QGRID) for m in range(G.shape[1])])     # (8, N_TAU)
    ACC = [(G[:, m][None, :] >= TAU[m][:, None] - 1e-15).astype(float) for m in range(G.shape[1])]
    recs_c, recs_i, recs_j, SVl, SDl = [], [], [], [], []
    for ci_, ch in enumerate(CHAINS):
        if len(ch) == 1:
            a, = ch; recs_c.append([ci_]); recs_i.append([-1]); recs_j.append([-1])
            SVl.append([Y[:, a].sum()]); SDl.append([Cm[:, a].sum()])
        elif len(ch) == 2:
            a, b = ch; A = ACC[a]
            SVl.append(A @ Y[:, a] + (1 - A) @ Y[:, b]); SDl.append(Cm[:, a].sum() + (1 - A) @ Cm[:, b])
            recs_c.append(np.full(N_TAU, ci_)); recs_i.append(np.arange(N_TAU)); recs_j.append(np.full(N_TAU, -1))
        else:
            a, b, c = ch; A, B = ACC[a], ACC[b]
            sv = (A @ Y[:, a])[:, None] + (1 - A) @ (B * Y[:, b]).T + (1 - A) @ ((1 - B) * Y[:, c]).T
            sd = Cm[:, a].sum() + ((1 - A) @ Cm[:, b])[:, None] + (1 - A) @ ((1 - B) * Cm[:, c]).T
            SVl.append(sv.ravel()); SDl.append(sd.ravel())
            ii, jj = np.meshgrid(np.arange(N_TAU), np.arange(N_TAU), indexing='ij')
            recs_c.append(np.full(N_TAU * N_TAU, ci_)); recs_i.append(ii.ravel()); recs_j.append(jj.ravel())
    return dict(TAU=TAU, chain=np.concatenate(recs_c).astype(int), i=np.concatenate(recs_i).astype(int),
                j=np.concatenate(recs_j).astype(int), SV=np.concatenate(SVl), SD=np.concatenate(SDl))

def chain_apply(ch, taus, G, Y, Cm):
    """Per-task returned score and total charged cost of chain ch with stage thresholds taus (last stage always returns)."""
    n = len(Y); V = np.zeros(n); D = np.zeros(n); alive = np.ones(n, bool); stage_share = []
    for k, m in enumerate(ch):
        D[alive] += Cm[alive, m]
        acc = alive & ((G[:, m] >= taus[k] - 1e-15) if k < len(ch) - 1 else np.ones(n, bool))
        V[acc] = Y[acc, m]; stage_share.append(float(acc.mean())); alive = alive & ~acc
    return V, D, stage_share

def pick(sr, crit, n, yB_mean, budget=None):
    ev, ed = sr['SV'] / n, sr['SD'] / n
    r = (sr['SV'] / sr['SD']) / ETA_B_PROBING
    if crit == 'eps0':
        ok = ev - yB_mean >= -1e-12; key = np.where(ok, r, -np.inf)
    elif crit == 'maxeff':
        key = r
    else:                                  # budget: max quality s.t. mean cost <= budget
        ok = ed <= budget + 1e-15; key = np.where(ok, ev, -np.inf)
    feasible = bool(np.isfinite(key).any())
    if not feasible:                        # e.g. a budget below the cheapest model: report the cheapest configuration, flagged
        key = -ed
    best = np.flatnonzero(key == key.max())
    k = best[np.argmin(ed[best])]           # tie-break: cheaper
    ch = CHAINS[sr['chain'][k]]
    taus = [sr['TAU'][ch[0], sr['i'][k]] if len(ch) > 1 else None, sr['TAU'][ch[1], sr['j'][k]] if len(ch) > 2 else None][:len(ch) - 1]
    qs = [float(QGRID[sr['i'][k]]) if len(ch) > 1 else None, float(QGRID[sr['j'][k]]) if len(ch) > 2 else None][:len(ch) - 1]
    return dict(k=int(k), chain=[MODELS[m] for m in ch], chain_idx=list(ch), taus=[float(t) for t in taus], tau_quantiles=qs,
                probing=dict(r=float(r[k]), dV=float(ev[k] - yB_mean), cost_mUSD=float(1000 * ed[k]), EV=float(ev[k])),
                n_feasible=int(np.isfinite(key).sum()) if feasible else 0, feasible=feasible)
