#!/usr/bin/env python3
"""
X8_operating_point -- "Route Anything: A Theory of Inference Routers": the verifier's operating point.

Paper objects tested (labels from main_body.tex / appendix_core.tex, app:operating-point):
  H = 1{Sigma >= theta}; p = pi0 TPR + (1-pi0) FPR; p v_h = pi0 TPR;
  floor eq:floor in operating-point form: pi0 TPR(theta) b >= vbar a   (no FPR);
  all-accept efficiency-optimal iff a beta0 <= pi0 b (general, constant b) / a/b <= pi0/vbar (homogeneous);
  Q = pi0 (1-beta1) TPR - (1-pi0) beta0 FPR (eq:app-operating-Q), floor Q >= -eps bounds FPR from above and,
  for a score with a monotone likelihood ratio (MLR), theta from below (theta_min(eps));
  premise Sigma _||_ Y_B | correctness;  eq:app-operating-gain: Q + eta_B S = pi0(1-beta1+vbar)TPR + (1-pi0)(vbar-beta0)FPR - vbar a/b.

Verifiers Sigma on the RECORDED responses (response_matrix.json), truth Y = 1{released score >= 1} (variant >= 0.5):
  V1 visible tests (prompt doctest/assert examples, as ACRouter's run_visible_tests; 0.5 when no example)  [free, imperfect]
  V2 hidden tests (common.sandbox.verify_response pass fraction)                                            [near-oracle]
  V3 LLM judge p_correct (config chosen by the Addendum's 60-item calibration)                              [LIVE]
  oracle: Sigma = released score (X1 control)
Primaries Qwen3-Max, kimi-k2.5, qwen3.5-plus (V3: first two); reference claude-opus-4-6 (recorded).
Primitives on probing, realized values on id_test (OUT-OF-SAMPLE), paired bootstrap 1000, seed 20260917.

Stages (``--stage``): sample | sandbox | calib | judge | analysis | all (default).  Everything is cached
(sandbox results in cache/, relay calls in common/cache through common.packy) so re-runs re-spend nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
sys.path.insert(0, EX)
from common import judge as judge_mod  # noqa: E402
from common import packy, sandbox, sources  # noqa: E402

T0 = time.time()
EXP = "X8"
SEED = 20260917
B_BOOT = 1000
N_PROBING = 500
N_IDTEST = 500          # requested; the runnable EXEC-CLEAN pool of id_test has fewer tasks -> all are used (deviation)
BUDGET_PATH = os.path.join(EX, "common", "budget.json")


def _allocation(default: float = 80000.0) -> float:
    """X8 allocation from common/budget.json (raised by the coordinator on 2026-09-17 from 80,000 to 400,000 for the full grok-4.6 judge run)."""
    try:
        with open(BUDGET_PATH) as f:
            return float(json.load(f).get("allocations_units", {}).get("X8_operating_point", default))
    except (OSError, ValueError):
        return default


ALLOC_UNITS = _allocation()
BUDGET_MARGIN = 0.97    # never plan beyond 97 % of the allocation
ACR = os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")
CSV = os.path.join(ACR, "coderouterbench", "id_results_long.csv")
RESP = os.path.join(ACR, "matrices", "phase1_acrouter_v2", "response_matrix.json")
OUT = HERE
FIG = os.path.join(OUT, "figures")
CACHE = os.path.join(OUT, "cache")
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

REF = "claude-opus-4-6"
PRIMARIES = ["Qwen3-Max", "kimi-k2.5", "qwen3.5-plus"]
JUDGED_PRIMARIES = ["Qwen3-Max", "kimi-k2.5"]
EXEC_CLEAN = ["humaneval", "he", "mbpp", "hex", "bcb"]
THETAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
THETA_REJECT = 1.05     # 22nd point: reject everything (H = 0, pays a + b); used for theta_min(eps) when no grid point is admissible
EPS_GRID = [0.0, 0.01, 0.02, 0.05]
VERIFIERS = ["V1", "V2", "V3", "oracle"]
VERIFIER_LABEL = {"V1": "V1 visible tests", "V2": "V2 hidden tests", "V3": "V3 LLM judge", "oracle": "oracle (released score)"}
BASES = ["usd", "sec"]
JUDGE_CONFIGS = [
    # id, model, thinking, max_tokens
    ("dsv4pro_think", "deepseek-v4-pro", True, 1024),
    ("dsv4pro_nothink", "deepseek-v4-pro", False, 320),
    ("dsflash", "deepseek-flash", True, 1024),
    ("qwen3max", "qwen3-max", False, 320),
    ("grok46", "grok-4.6", False, 1024),
]
JUDGE_WORKERS = 10   # raised from 4 by the coordinator (18:30) after the grok-4.6 cap in common/packy.py was raised to 10
# Judge plan (the defaults reproduce the executed run -- coordinator decision 2026-09-17: full designed sample, both primaries):
#   X8_JUDGE_PLAN = "full" (default): every sampled task; "allocation": calls capped so that the LEDGER estimate stays inside the X8
#                   allocation at the stored per-model rate.
#   X8_JUDGE_PRIMARIES: comma-separated primaries judged by V3 (default Qwen3-Max,kimi-k2.5 as designed).
JUDGE_PLAN_MODE = os.environ.get("X8_JUDGE_PLAN", "full")
JUDGE_PLAN_PRIMARIES = [m for m in os.environ.get("X8_JUDGE_PRIMARIES", "Qwen3-Max,kimi-k2.5").split(",") if m]
SANDBOX_WORKERS = 8
N_CALIB = 60
LEDGER = os.path.join(EX, "common", "cache", "usage_ledger.jsonl")


def log(*a: Any) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def jdump(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars for json."""
    if isinstance(obj, dict):
        return {str(k): jdump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jdump(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jdump(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jdump(obj), f, indent=1)
    os.replace(tmp, path)


# ============================================================================================ 1. data


def load_matrix() -> Dict[str, Any]:
    df = pd.read_csv(CSV)
    assert df.shape[0] == 79992, df.shape
    S = df.pivot(index="task_id", columns="model", values="score")
    C = df.pivot(index="task_id", columns="model", values="cost_usd")
    L = df.pivot(index="task_id", columns="model", values="latency_ms") / 1000.0
    TT = df.pivot(index="task_id", columns="model", values="total_tokens")
    meta = df.drop_duplicates("task_id").set_index("task_id").loc[S.index, ["split", "dimension", "source_split"]].copy()
    meta["source"] = meta.index.str.split("_").str[1]
    meta["dim"] = meta.index.str.split("_").str[0]
    meta["allzero"] = S.max(axis=1).eq(0)
    assert (meta.split == "id_test").sum() == 2919 and (meta.split == "probing").sum() == 7080
    return dict(df=df, S=S, C=C, L=L, TT=TT, meta=meta)


def exec_clean_pool(M: Dict[str, Any]) -> pd.DataFrame:
    """Runnable EXEC-CLEAN tasks (Addendum): humaneval; he dim4 completion; mbpp dim1/dim4 (not testgen); hex python; bcb not all-zero & runnable."""
    meta = M["meta"]
    idx = meta.index
    sel = (meta.source == "humaneval") | ((meta.source == "he") & idx.str.contains("_completion_")) | \
          ((meta.source == "mbpp") & ~idx.str.contains("_testgen_")) | ((meta.source == "hex") & idx.str.contains("_python_")) | \
          ((meta.source == "bcb") & ~meta.allzero)
    pool = meta[sel].copy()
    runnable = {}
    for tid in pool.index:
        t = sources.load_task(tid)
        runnable[tid] = bool(t["runnable"]) and bool(t["found"])
    pool["runnable"] = pd.Series(runnable)
    return pool[pool.runnable].copy()


def largest_remainder(counts: Dict[str, int], n: int) -> Dict[str, int]:
    tot = sum(counts.values())
    if n >= tot:
        return dict(counts)
    raw = {k: n * v / tot for k, v in counts.items()}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = n - sum(base.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rem]:
        base[k] += 1
    return base


def stratified_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    """Stratified (by source) sample of ``n`` task ids and a stratified priority order (any prefix is ~stratified)."""
    counts = pool.groupby("source").size().to_dict()
    alloc = largest_remainder(counts, n)
    chosen: Dict[str, List[str]] = {}
    for s in sorted(counts):
        ids = sorted(pool.index[pool.source == s].tolist())
        k = alloc[s]
        pick = sorted(rng.choice(ids, size=k, replace=False).tolist()) if k < len(ids) else ids
        chosen[s] = pick
    # priority order: interleave sources by fractional rank so that every prefix keeps the source mix
    order = []
    for s, ids in chosen.items():
        perm = rng.permutation(len(ids))
        for r, j in enumerate(perm):
            order.append(((r + 0.5) / len(ids), s, ids[j]))
    order.sort()
    prio = [tid for _, _, tid in order]
    return sorted(prio), prio


def build_samples(M: Dict[str, Any]) -> Dict[str, Any]:
    path = os.path.join(CACHE, "samples.json")
    cached = load_json(path, None)
    if cached is not None:
        return cached
    pool = exec_clean_pool(M)
    rng = np.random.default_rng(SEED)
    pool_p = pool[pool.split == "probing"]
    pool_t = pool[pool.split == "id_test"]
    ids_p, prio_p = stratified_sample(pool_p, N_PROBING, rng)
    ids_t, prio_t = stratified_sample(pool_t, N_IDTEST, rng)
    rest_p = sorted(set(pool_p.index) - set(ids_p))
    out = dict(
        pool_counts={sp: pool[pool.split == sp].groupby("source").size().to_dict() for sp in ["probing", "id_test"]},
        probing=ids_p, probing_priority=prio_p, id_test=ids_t, id_test_priority=prio_t, probing_rest=rest_p,
        n_probing=len(ids_p), n_id_test=len(ids_t),
        note=("id_test: the runnable EXEC-CLEAN pool has %d tasks < %d requested -> all used" % (len(ids_t), N_IDTEST)) if len(ids_t) < N_IDTEST else "",
    )
    save_json(path, out)
    return out


_RESP: Optional[Dict[str, Dict[str, str]]] = None


def responses() -> Dict[str, Dict[str, str]]:
    global _RESP
    if _RESP is None:
        log("loading response_matrix.json ...")
        with open(RESP) as f:
            _RESP = json.load(f)
    return _RESP


def response_of(tid: str, model: str) -> str:
    d = responses().get(tid, {})
    r = d.get(model)
    if r is None and model == "Qwen3-Max":
        r = d.get("通义千问Max")
    return r or ""


# ============================================================================================ 2. verifiers (sandbox)

_VIS_DOCTEST = re.compile(r">>>\s*(.+?)\n\s*(.+)$", re.MULTILINE)
_VIS_ASSERT = re.compile(r"assert\s+(.+?)\s*==\s*(.+)$", re.MULTILINE)
_VIS_MARK = re.compile(r"__X8_VISIBLE__ (\{.*\})")


def visible_examples(prompt: str) -> List[Tuple[str, str]]:
    """ACRouter run_visible_tests extraction: doctest '>>> call\\n expected' first, else 'assert call == expected'; cap 5."""
    ex = _VIS_DOCTEST.findall(prompt or "")
    if not ex:
        ex = _VIS_ASSERT.findall(prompt or "")
    return [(c, e.strip()) for c, e in ex[:5]]


def visible_harness(examples: Sequence[Tuple[str, str]]) -> str:
    """One sandbox process; each example is exec'd as ACRouter writes it (``_result = call; _expected = expected; assert``)."""
    return "\n".join([
        "import json as __json",
        "__passed = 0",
        "__ex = " + repr(list(examples)),
        "for __call, __exp in __ex:",
        "    try:",
        "        exec(compile('_result = ' + __call + '\\n_expected = ' + __exp + '\\nassert _result == _expected\\n', '<visible>', 'exec'), globals())",
        "        __passed += 1",
        "    except BaseException:",
        "        pass",
        "print('__X8_VISIBLE__ ' + __json.dumps({'passed': __passed, 'total': len(__ex)}))",
        "",
    ])


def visible_test_verifier(task: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    """V1: fraction of prompt-visible examples passed; 0.5 when the prompt has no example (ACRouter semantics)."""
    prompt = task.get("raw_prompt") or task.get("prompt") or ""
    examples = visible_examples(prompt)
    if not examples:
        return dict(sigma=0.5, n_examples=0, n_passed=0, elapsed_s=0.0, kind="no_visible_examples")
    code = sandbox.extract_code(response_text or "", "python", entry_point=task.get("entry_point"))
    if task.get("source") in ("bcb", "humaneval", "he", "hex"):
        wpi = getattr(sandbox, "_with_prompt_imports", None)
        if wpi is not None:
            code = wpi(code, task.get("raw_prompt"))
    t_s = sandbox.DEFAULT_TIMEOUTS.get(task.get("source"), 10)
    res = sandbox.run_python_tests(code, visible_harness(examples), entry_point=None, timeout_s=t_s)
    m = _VIS_MARK.search(res.get("stdout") or "")
    n_pass = 0
    kind = "harness_failed"
    if m:
        try:
            st = json.loads(m.group(1))
            n_pass = int(st.get("passed", 0))
            kind = "ran"
        except (ValueError, TypeError):
            pass
    elif res.get("timed_out"):
        kind = "timeout"
    return dict(sigma=n_pass / len(examples), n_examples=len(examples), n_passed=n_pass, elapsed_s=float(res.get("elapsed_s") or 0.0), kind=kind,
                error=(res.get("error") or "")[:200])


def hidden_test_verifier(task: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    """V2: common.sandbox.verify_response pass fraction (the released score's own mechanism)."""
    res = sandbox.verify_response(task, response_text)
    frac = res.get("frac")
    return dict(sigma=float(frac) if frac is not None else 0.0, passed=bool(res.get("passed")), elapsed_s=float(res.get("elapsed_s") or 0.0),
                failure_kind=res.get("failure_kind"), n_cases=res.get("n_cases"), n_passed=res.get("n_passed"), error=(res.get("error") or "")[:200])


def run_sandbox_stage(samples: Dict[str, Any]) -> Dict[str, Any]:
    """V1 and V2 for every primary x sampled task (both splits); cached in cache/sandbox.json."""
    path = os.path.join(CACHE, "sandbox.json")
    cache = load_json(path, {})
    tids = list(samples["probing"]) + list(samples["id_test"])
    jobs = []
    for m in PRIMARIES:
        for tid in tids:
            for v in ("V1", "V2"):
                key = f"{v}|{m}|{tid}"
                if key not in cache:
                    jobs.append((key, v, m, tid))
    log(f"sandbox: {len(cache)} cached, {len(jobs)} to run")
    if jobs:
        tasks = {tid: sources.load_task(tid) for tid in tids}
        responses()

        def work(job):
            key, v, m, tid = job
            task = tasks[tid]
            resp = response_of(tid, m)
            try:
                r = visible_test_verifier(task, resp) if v == "V1" else hidden_test_verifier(task, resp)
            except Exception as e:  # never let one task kill the run
                r = dict(sigma=0.0, elapsed_s=0.0, kind="exception", error=repr(e)[:200])
            r["empty_response"] = not bool((resp or "").strip())
            return key, r

        done = 0
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=SANDBOX_WORKERS) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for fut in as_completed(futs):
                key, r = fut.result()
                cache[key] = r
                done += 1
                if done % 200 == 0 or done == len(jobs):
                    save_json(path, cache)
                    log(f"sandbox {done}/{len(jobs)} ({(time.time() - t_start) / done:.2f} s/job)")
        save_json(path, cache)
    return cache


# ============================================================================================ 3. judge (LIVE)


def x8_units_spent() -> Tuple[float, int, int]:
    """(units, live calls, tokens) charged to tag X8 in the shared ledger."""
    units = 0.0
    calls = 0
    toks = 0
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("tag") != EXP or r.get("cached"):
                    continue
                units += float(r.get("units_est") or 0.0)
                calls += 1
                toks += int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
    return units, calls, toks


def make_client() -> packy.PackyClient:
    return packy.PackyClient(quiet=True)


def calibration_items(M: Dict[str, Any], samples: Dict[str, Any]) -> List[Dict[str, Any]]:
    """60 recorded responses (30 released-correct / 30 incorrect) from EXEC-CLEAN probing tasks outside the 500-sample, any of the 8 models."""
    path = os.path.join(CACHE, "calibration_items.json")
    cached = load_json(path, None)
    if cached is not None:
        return cached
    rng = np.random.default_rng(SEED + 1)
    S = M["S"]
    models = list(S.columns)
    pool = list(samples["probing_rest"])
    pairs = [(tid, m) for tid in pool for m in models]
    rng.shuffle(pairs)
    pos, neg = [], []
    for tid, m in pairs:
        resp = response_of(tid, m)
        if not resp.strip():
            continue
        y = int(S.loc[tid, m] >= 1.0 - 1e-12)
        (pos if y else neg).append(dict(task_id=tid, model=m, truth=y, score=float(S.loc[tid, m])))
        if len(pos) >= N_CALIB // 2 and len(neg) >= N_CALIB // 2:
            break
    items = pos[: N_CALIB // 2] + neg[: N_CALIB // 2]
    save_json(path, items)
    return items


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), ties count 1/2; NaN if one class is missing."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def run_calibration(client: packy.PackyClient, M: Dict[str, Any], samples: Dict[str, Any], dry: bool = False) -> Dict[str, Any]:
    path = os.path.join(CACHE, "calibration.json")
    items = calibration_items(M, samples)
    tasks = {it["task_id"]: sources.load_task(it["task_id"]) for it in items}
    pairs = [(tasks[it["task_id"]], response_of(it["task_id"], it["model"])) for it in items]
    y = np.array([it["truth"] for it in items], int)
    out = load_json(path, {"configs": {}})
    rng = np.random.default_rng(SEED + 2)
    for cid, model, thinking, max_tokens in JUDGE_CONFIGS:
        if cid in out["configs"] and out["configs"][cid].get("n_error", 1) == 0 and len(out["configs"][cid].get("p", [])) == len(items):
            continue
        if dry:
            continue
        if cid in out["configs"] and out["configs"][cid].get("p"):
            out.setdefault("configs_history", {}).setdefault(cid, []).append({k: v for k, v in out["configs"][cid].items() if k not in ("p", "verdicts")} | {"note": "first pass; error rows (p_correct 0.5) retried in the next pass"})
        spent0, _, _ = x8_units_spent()
        t0 = time.time()
        try:
            res = judge_mod.judge_many(client, pairs, model=model, workers=JUDGE_WORKERS, tag=EXP, max_tokens=max_tokens, thinking=thinking)
        except (packy.BudgetExceeded, packy.ModelUnavailable) as e:
            log(f"calibration {cid}: {type(e).__name__}: {e}")
            out["configs"][cid] = dict(model=model, thinking=thinking, max_tokens=max_tokens, fatal=str(e)[:300], auroc=float("nan"))
            save_json(path, out)
            continue
        wall = time.time() - t0
        p = np.array([r["p_correct"] for r in res], float)
        parse_ok = np.array([bool(r["parse_ok"]) for r in res])
        err = np.array([bool(r.get("error")) for r in res])
        toks = np.array([int((r.get("usage") or {}).get("total_tokens") or 0) for r in res], float)
        ptoks = np.array([int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in res], float)
        ctoks = np.array([int((r.get("usage") or {}).get("completion_tokens") or 0) for r in res], float)
        lat = np.array([float(r.get("latency_s") or 0.0) for r in res], float)
        cached = np.array([bool(r.get("cached")) for r in res])
        spent1, _, _ = x8_units_spent()
        rate = client.units_per_token(model)
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, len(y), len(y))
            boots.append(auroc(y[idx], p[idx]))
        boots = np.array([b for b in boots if math.isfinite(b)])
        acc = float(((p >= 0.5).astype(int) == y).mean())
        out["configs"][cid] = dict(
            model=model, thinking=thinking, max_tokens=max_tokens, n=len(items), auroc=auroc(y, p),
            auroc_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else [None, None],
            accuracy=acc, brier=float(np.mean((p - y) ** 2)), tpr_at_half=float(((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1)),
            fpr_at_half=float(((p >= 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            n_parse_fail=int((~parse_ok).sum()), n_error=int(err.sum()), n_cached=int(cached.sum()),
            mean_total_tokens=float(toks.mean()), mean_prompt_tokens=float(ptoks.mean()), mean_completion_tokens=float(ctoks.mean()),
            mean_latency_s=float(lat.mean()), wall_s=wall, units_per_token=rate, units_est=float(toks.sum() * rate),
            units_ledger_delta=spent1 - spent0, p=p.tolist(), verdicts=[r["verdict"] for r in res],
            finish_reasons=dict(Counter(str(r.get("finish_reason")) for r in res)),
        )
        save_json(path, out)
        log(f"calibration {cid}: AUROC {out['configs'][cid]['auroc']:.3f} acc {acc:.3f} tokens/call {toks.mean():.0f} lat {lat.mean():.1f}s units {toks.sum() * rate:.0f} (ledger +{spent1 - spent0:.0f})")
    # choose the best AUROC (tie -> fewer tokens)
    ok = {k: v for k, v in out["configs"].items() if v.get("auroc") is not None and math.isfinite(v.get("auroc", float("nan")))}
    if ok:
        best = sorted(ok, key=lambda k: (-ok[k]["auroc"], ok[k].get("mean_total_tokens", 1e9)))[0]
        out["best"] = best
        out["best_auroc"] = ok[best]["auroc"]
        out["flag_low_auroc"] = bool(ok[best]["auroc"] < 0.75)
    out["items"] = items
    out["truth"] = y.tolist()
    save_json(path, out)
    return out


def plan_judge_sample(calib: Dict[str, Any], samples: Dict[str, Any], mode: str = JUDGE_PLAN_MODE, primaries: Sequence[str] = tuple(JUDGE_PLAN_PRIMARIES)) -> Dict[str, Any]:
    """Judged task sets.  ``mode='full'``: every sampled task for every primary in ``primaries`` (design: 2 x (500 + 334) = 1,668 calls).
    ``mode='allocation'``: the number of calls is capped so that the LEDGER estimate (mean calibration tokens x the stored per-model rate,
    +10 %) stays inside the X8 allocation; the calls are split equally over the primaries and, per primary, equally over id_test
    (realization) and probing (primitives), taking prefixes of the stratified priority orders (every prefix keeps the source mix).
    Reason for the cap (2026-09-17): the selected judge grok-4.6 is billed at the uncalibrated default 0.05 units/token, so the full
    sample would be ~260k ledger units (3.25x the allocation, ~87 % of the shared cap) and ~3.8 h at its concurrency cap of 4."""
    best = calib.get("best")
    cfg = calib["configs"][best]
    spent, _, _ = x8_units_spent()
    per_call = max(float(cfg["mean_total_tokens"]) * float(cfg["units_per_token"]), 1.0) * 1.10  # 10 % safety on the per-call estimate
    n_p, n_t = len(samples["probing"]), len(samples["id_test"])
    primaries = list(primaries)
    full = len(primaries) * (n_p + n_t)
    affordable = int(max((ALLOC_UNITS * BUDGET_MARGIN - spent) // per_call, 0))
    if mode == "full" or affordable >= full:
        k_p, k_t = n_p, n_t
        constrained = False
    else:
        per_primary = affordable // max(len(primaries), 1)
        k_t = min(n_t, per_primary // 2)
        k_p = min(n_p, per_primary - k_t)
        constrained = True
    planned = len(primaries) * (k_p + k_t)
    return dict(best=best, mode=mode, primaries=primaries, per_call_units_est=per_call, spent_before=spent, remaining_units=ALLOC_UNITS - spent,
                n_calls_affordable_at_allocation=affordable, n_calls_full=full, n_calls_planned=planned, k_probing=k_p, k_id_test=k_t,
                probing_tasks=list(samples["probing_priority"][:k_p]), id_test_tasks=list(samples["id_test_priority"][:k_t]),
                constrained=constrained, ledger_units_est_planned=planned * per_call, units_per_token_used=float(cfg["units_per_token"]),
                over_allocation_at_ledger_rate=bool(spent + planned * per_call > ALLOC_UNITS),
                note=("full sample" if not constrained else "capped to the X8 allocation at the ledger rate (uncalibrated judge model billed at the default 0.05 units/token)"))


def run_judge_stage(client: packy.PackyClient, calib: Dict[str, Any], samples: Dict[str, Any], dry: bool = False) -> Dict[str, Any]:
    path = os.path.join(CACHE, "judge.json")
    out = load_json(path, {"rows": {}})
    if "plan" not in out:
        out["plan"] = plan_judge_sample(calib, samples)
        save_json(path, out)
    plan = out["plan"]
    cfg = calib["configs"][plan["best"]]
    tids = list(plan["id_test_tasks"]) + list(plan["probing_tasks"])   # id_test (realization) first
    prims = list(plan.get("primaries", JUDGED_PRIMARIES))
    todo = [(m, tid) for tid in tids for m in prims if f"{m}|{tid}" not in out["rows"]]
    # one retry pass for rows whose relay call failed (packy does not cache errors, so a re-call is live)
    retry = [tuple(k.split("|", 1)) for k, r in out["rows"].items() if r.get("error")]
    todo += [(m, tid) for m, tid in retry if (m, tid) not in todo]
    log(f"judge: config {plan['best']} ({cfg['model']}, thinking={cfg['thinking']}, max_tokens={cfg['max_tokens']}); {len(out['rows'])} cached rows ({len(retry)} with errors to retry), {len(todo)} to run; plan {plan['n_calls_planned']} calls ({plan['note']}), primaries {prims}")
    if dry or not todo:
        return out
    tasks = {tid: sources.load_task(tid) for tid in tids}
    rate = client.units_per_token(cfg["model"])
    batch = 40
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        pairs = [(tasks[tid], response_of(tid, m)) for m, tid in chunk]
        try:
            res = judge_mod.judge_many(client, pairs, model=cfg["model"], workers=JUDGE_WORKERS, tag=EXP, max_tokens=cfg["max_tokens"], thinking=cfg["thinking"])
        except (packy.BudgetExceeded, packy.ModelUnavailable) as e:
            log(f"judge: fatal {type(e).__name__}: {e}")
            out["fatal"] = str(e)[:300]
            break
        for (m, tid), r in zip(chunk, res):
            u = r.get("usage") or {}
            out["rows"][f"{m}|{tid}"] = dict(
                p_correct=float(r["p_correct"]), verdict=r["verdict"], confidence=r.get("confidence"), parse_ok=bool(r["parse_ok"]),
                error=(r.get("error") or None), total_tokens=int(u.get("total_tokens") or 0), prompt_tokens=int(u.get("prompt_tokens") or 0),
                completion_tokens=int(u.get("completion_tokens") or 0), latency_s=float(r.get("latency_s") or 0.0), cached=bool(r.get("cached")),
                units_est=float(int(u.get("total_tokens") or 0) * rate), finish_reason=r.get("finish_reason"),
            )
        save_json(path, out)
        spent, _, _ = x8_units_spent()
        log(f"judge {min(i + batch, len(todo))}/{len(todo)}; X8 ledger units {spent:.0f}")
    return out


# ============================================================================================ 4. analysis helpers


def boot_weights(n: int, rng: np.random.Generator, B: int = B_BOOT) -> np.ndarray:
    """(B+1) x n; row 0 = ones (point estimate), rows 1..B multinomial task-resampling counts (paired across policies)."""
    Wb = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    return np.vstack([np.ones((1, n)), Wb])


def ci(v: np.ndarray) -> List[float]:
    v = np.asarray(v, float)
    w = v[1:][np.isfinite(v[1:])]
    if len(w) == 0:
        return [float(v[0]) if np.isfinite(v[0]) else float("nan"), float("nan"), float("nan")]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]


def excl0(c: Sequence[float], x: float = 0.0) -> bool:
    return bool(np.isfinite(c[1]) and np.isfinite(c[2]) and (c[1] > x or c[2] < x))


def sgn(x: float) -> int:
    return int(np.sign(x)) if np.isfinite(x) else 0


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def cascade_stats(W: np.ndarray, H: np.ndarray, yp: np.ndarray, yb: np.ndarray, cp: np.ndarray, cb: np.ndarray, tv: np.ndarray) -> Dict[str, np.ndarray]:
    """Verify-then-escalate cascade (eq:accounting): V = H Y_P + (1-H) Y_B, D = A + (1-H) C_B with A = primary cost + verifier cost tv.
    All quantities are weighted sums under W (row 0 point estimate, rows 1..B bootstrap) -> paired CIs."""
    Hf = H.astype(float)
    miss = 1.0 - Hf
    A = cp + tv
    V = Hf * yp + miss * yb
    D = A + miss * cb
    X = np.column_stack([V, D, Hf, Hf * yp, Hf * yb, Hf * cb, yb, cb, Hf * (yp < yb), A, miss * cb, Hf * (yp > yb), cp, tv])
    Mx = W @ X
    n = W.sum(1)
    EV = Mx[:, 0] / n
    ED = Mx[:, 1] / n
    eta = _div(Mx[:, 0], Mx[:, 1])
    vbar = Mx[:, 6] / n
    cbar = Mx[:, 7] / n
    eta_B = _div(Mx[:, 6], Mx[:, 7])
    p = Mx[:, 2] / n
    Q = EV - vbar
    S = cbar - ED
    return dict(EV=EV, ED=ED, eta=eta, eta_B=eta_B, r=_div(eta, eta_B), d_eta=eta - eta_B, vbar=vbar, cbar=cbar, p=p,
                v_h=_div(Mx[:, 3], Mx[:, 2]), vbar_h=_div(Mx[:, 4], Mx[:, 2]), cbar_h=_div(Mx[:, 5], Mx[:, 2]), Q=Q, S=S,
                a=Mx[:, 9] / n, a_primary=Mx[:, 12] / n, a_verifier=Mx[:, 13] / n, e=Mx[:, 8] / n, e_plus=Mx[:, 11] / n,
                gain=Q + eta_B * S, avoided=Mx[:, 5] / n, attempt=Mx[:, 9] / n)


def H_of(sig: np.ndarray, theta: float) -> np.ndarray:
    if theta > 1.0:  # reject everything
        return np.zeros(len(sig), bool)
    return sig >= theta - 1e-12


def primitives(Wp: np.ndarray, sig: np.ndarray, yp: np.ndarray, yb: np.ndarray, cp: np.ndarray, cb: np.ndarray, tv: np.ndarray, thetas: Sequence[float]) -> Dict[str, Any]:
    """Operating-point primitives on the probing sample (rows = bootstrap resamples) and the paper's predictions per theta."""
    n = Wp.sum(1)
    pi0 = Wp @ yp / n
    vbar = Wp @ yb / n
    b = Wp @ cb / n
    a = Wp @ (cp + tv) / n
    beta1 = _div(Wp @ (yp * yb), Wp @ yp)
    beta0 = _div(Wp @ ((1 - yp) * yb), Wp @ (1 - yp))
    T = len(thetas)
    R = Wp.shape[0]
    TPR = np.zeros((R, T))
    FPR = np.zeros((R, T))
    plug = defaultdict(lambda: np.zeros((R, T)))
    for j, th in enumerate(thetas):
        H = H_of(sig, th).astype(float)
        TPR[:, j] = _div(Wp @ (H * yp), Wp @ yp)
        FPR[:, j] = _div(Wp @ (H * (1 - yp)), Wp @ (1 - yp))
        st = cascade_stats(Wp, H.astype(bool), yp, yb, cp, cb, tv)
        for k in ("eta", "r", "d_eta", "Q", "e", "p", "v_h", "vbar_h", "cbar_h", "EV", "ED", "gain"):
            plug[k][:, j] = st[k]
    P0, V0, B0, A0 = pi0[:, None], vbar[:, None], b[:, None], a[:, None]
    B1, B0b = beta1[:, None], beta0[:, None]
    p = P0 * TPR + (1 - P0) * FPR
    floor_hom = P0 * TPR * B0 - V0 * A0                       # eq:floor in operating-point form (no FPR)
    floor_naive = p * B0 - V0 * A0                            # FPR added with the same weight as TPR (p b >= vbar a)
    floor_penal = (P0 * TPR - (1 - P0) * FPR) * B0 - V0 * A0  # FPR penalized (false accepts counted as lost reference value)
    gain_gen = (P0 * (1 - B1 + V0) * TPR + (1 - P0) * (V0 - B0b) * FPR) * B0 - V0 * A0   # eq:app-operating-gain x b
    Q_pred = P0 * (1 - B1) * TPR - (1 - P0) * B0b * FPR       # eq:app-operating-Q
    EV_pred = P0 * TPR + P0 * (1 - TPR) * B1 + (1 - P0) * (1 - FPR) * B0b
    ED_pred = A0 + (1 - p) * B0
    eta_pred = EV_pred / ED_pred
    eta_B_pred = vbar / b
    return dict(n=n, pi0=pi0, vbar=vbar, b=b, a=a, beta1=beta1, beta0=beta0, TPR=TPR, FPR=FPR, p=p, floor_hom=floor_hom, floor_naive=floor_naive, floor_penal=floor_penal,
                gain_gen=gain_gen, Q_pred=Q_pred, EV_pred=EV_pred, ED_pred=ED_pred, eta_pred=eta_pred, r_pred=eta_pred / eta_B_pred[:, None],
                eta_B_pred=eta_B_pred, allacc_general=a * beta0 - pi0 * b, allacc_hom=a * vbar - pi0 * b, allacc_uninformative=a * vbar - pi0 * (a + b),
                plug={k: v for k, v in plug.items()})


def theta_min_from(Q: np.ndarray, eps: float, thetas: Sequence[float]) -> np.ndarray:
    """Per row: the smallest theta with Q >= -eps (the reject-all point at the end always qualifies)."""
    ok = Q >= -eps - 1e-12
    ok[:, -1] = True
    idx = np.argmax(ok, axis=1)
    return np.asarray(thetas, float)[idx]


def argmax_theta(eta: np.ndarray, thetas: Sequence[float], n_grid: int) -> np.ndarray:
    """Per row: the smallest grid theta attaining the maximum efficiency (grid points only, not reject-all); all-NaN rows (degenerate resamples) -> theta 0."""
    e = eta[:, :n_grid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmax(e, axis=1, keepdims=True)
    ok = e >= m - 1e-12
    return np.asarray(thetas[:n_grid], float)[np.argmax(ok, axis=1)]


def is_upset(mask: np.ndarray) -> bool:
    """Admissible set an interval [theta_min, ...) on the grid?"""
    idx = np.where(mask)[0]
    return bool(len(idx) == 0 or (mask[idx[0]:].all()))


def bins_for(sig: np.ndarray, max_distinct: int = 8) -> Tuple[np.ndarray, List[str]]:
    """Bins of a score in [0, 1]: its distinct values when there are at most ``max_distinct``; else the fixed bins {0}, (0,.25], (.25,.5], (.5,.75], (.75,1), {1}."""
    vals = np.unique(sig[np.isfinite(sig)])
    if len(vals) <= max_distinct:
        b = np.searchsorted(vals, sig)
        return b, [f"{v:.3g}" for v in vals]
    labels = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"]
    b = np.full(len(sig), 5, int)
    b[sig <= 1e-12] = 0
    b[(sig > 1e-12) & (sig <= 0.25)] = 1
    b[(sig > 0.25) & (sig <= 0.5)] = 2
    b[(sig > 0.5) & (sig <= 0.75)] = 3
    b[(sig > 0.75) & (sig < 1 - 1e-12)] = 4
    used = sorted(set(b.tolist()))
    remap = {u: i for i, u in enumerate(used)}
    return np.array([remap[x] for x in b], int), [labels[u] for u in used]


def mlr_check(sig: np.ndarray, y: np.ndarray, min_n: int = 15) -> Dict[str, Any]:
    """Likelihood ratio P(Sigma in bin | Y=1)/P(Sigma in bin | Y=0) per bin; monotone (nondecreasing) in the score?  Also ROC concavity."""
    ok = np.isfinite(sig)
    sig, y = sig[ok], y[ok]
    b, labels = bins_for(sig)
    N1, N0 = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    rows = []
    for k, lab in enumerate(labels):
        n1 = int(((b == k) & (y == 1)).sum())
        n0 = int(((b == k) & (y == 0)).sum())
        lr = (n1 / N1) / (n0 / N0) if n0 > 0 else (float("inf") if n1 > 0 else float("nan"))
        rows.append(dict(bin=lab, n1=n1, n0=n0, n=n1 + n0, LR=lr))
    kept = [r for r in rows if r["n"] >= min_n and np.isfinite(r["LR"]) or (r["n"] >= min_n and r["LR"] == float("inf"))]
    viol = 0
    for r1, r2 in zip(kept, kept[1:]):
        if r2["LR"] < r1["LR"] * (1 - 1e-9):
            viol += 1
    # ROC concavity over distinct thresholds
    vals = np.unique(sig)
    pts = [(1.0, 1.0)]
    for v in vals:
        H = sig >= v - 1e-12
        pts.append((float((H & (y == 0)).sum() / N0), float((H & (y == 1)).sum() / N1)))
    pts.append((0.0, 0.0))
    pts = sorted(set(pts))
    slopes = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        slopes.append((y2 - y1) / (x2 - x1) if x2 > x1 else float("inf"))
    concave_viol = sum(1 for s1, s2 in zip(slopes, slopes[1:]) if s2 > s1 + 1e-9)
    return dict(bins=rows, n_bins_kept=len(kept), lr_violations=viol, mlr_holds=bool(viol == 0 and len(kept) >= 2), roc_points=pts, roc_concavity_violations=concave_viol)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (ph, c - h, c + h)


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged)."""
    from scipy.stats import rankdata
    return rankdata(x)


def _spearman(x: np.ndarray, z: np.ndarray) -> float:
    rx, rz = _rank(x), _rank(z)
    if rx.std() == 0 or rz.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, rz)[0, 1])


def _strat_support(x: np.ndarray, z: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    """Strata that can contribute to the within-stratum statistic (>= 10 points, non-constant x and z) and the number of
    'informative' points among them (points in the minority x-value class summed over strata: a stratum with one deviating
    score value carries a single informative point)."""
    strata = []
    n_pts = 0
    n_inf = 0
    for s_ in np.unique(g):
        m = g == s_
        if m.sum() < 10 or len(np.unique(x[m])) < 2 or len(np.unique(z[m])) < 2:
            continue
        strata.append(str(s_))
        n_pts += int(m.sum())
        vals, cnt = np.unique(x[m], return_counts=True)
        n_inf += int(m.sum() - cnt.max())
    return dict(strata=strata, n_points=n_pts, n_informative=n_inf)


def _strat_stat(x: np.ndarray, z: np.ndarray, g: np.ndarray) -> float:
    """n-weighted mean of within-stratum Spearman correlations (strata with >= 10 points and non-constant x, z)."""
    num = den = 0.0
    for s_ in np.unique(g):
        m = g == s_
        if m.sum() < 10:
            continue
        r = _spearman(x[m], z[m])
        if np.isfinite(r):
            num += m.sum() * r
            den += m.sum()
    return num / den if den > 0 else float("nan")


def cond_indep(sig: np.ndarray, y: np.ndarray, yb: np.ndarray, rng: np.random.Generator, thetas: Sequence[float], n_perm: int = 2000, src: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Sigma _||_ Y_B | Y: within each correctness stratum, Spearman trend of Y_B in Sigma with a permutation p-value (pooled), the same test
    stratified by source (Y_B permuted within source; statistic = n-weighted mean of within-source Spearman rho), a binned table and beta(theta) ranges."""
    out: Dict[str, Any] = {}
    ok = np.isfinite(sig)
    sig, y, yb = sig[ok], y[ok], yb[ok]
    src = src[ok] if src is not None else None
    for stratum, name in ((1, "given_correct"), (0, "given_incorrect")):
        idx = y == stratum
        x, z = sig[idx], yb[idx]
        res: Dict[str, Any] = dict(n=int(idx.sum()))
        if len(np.unique(x)) < 2 or len(np.unique(z)) < 2 or idx.sum() < 10:
            res.update(rho=float("nan"), p_perm=float("nan"), note="degenerate (constant Sigma or Y_B in stratum)")
        else:
            rho = _spearman(x, z)
            perm = np.empty(n_perm)
            for i in range(n_perm):
                perm[i] = _spearman(x, rng.permutation(z))
            res.update(rho=rho, p_perm=float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (n_perm + 1)))
        if src is not None and idx.sum() >= 10:
            g = src[idx]
            stat = _strat_stat(x, z, g)
            sup = _strat_support(x, z, g)
            res["within_source_support"] = sup
            res["within_source_degenerate"] = bool(sup["n_informative"] < 10)
            if np.isfinite(stat):
                perm = np.empty(n_perm)
                for i in range(n_perm):
                    zp = z.copy()
                    for s_ in np.unique(g):
                        m = np.where(g == s_)[0]
                        zp[m] = z[rng.permutation(m)]
                    perm[i] = _strat_stat(x, zp, g)
                perm = perm[np.isfinite(perm)]
                res.update(rho_within_source=stat, p_perm_within_source=float((np.sum(np.abs(perm) >= abs(stat) - 1e-12) + 1) / (len(perm) + 1)),
                           within_source_rho={str(s_): _spearman(x[g == s_], z[g == s_]) for s_ in np.unique(g) if (g == s_).sum() >= 10})
            else:
                res.update(rho_within_source=float("nan"), p_perm_within_source=float("nan"))
        b, labels = bins_for(x)
        res["table"] = [dict(bin=lab, n=int((b == k).sum()), p_yb=wilson(int(((b == k) & (z == 1)).sum()), int((b == k).sum()))) for k, lab in enumerate(labels)]
        bt = []
        for th in thetas:
            H = H_of(x, th)
            if H.sum() >= 20:
                bt.append(float(z[H].mean()))
        res["beta_theta_range"] = [min(bt), max(bt)] if bt else [float("nan"), float("nan")]
        out[name] = res
    return out


# ============================================================================================ 5. assemble arrays


def assemble(M: Dict[str, Any], samples: Dict[str, Any], sb: Dict[str, Any], jd: Optional[Dict[str, Any]], truth_level: float = 1.0, include_empty: bool = False) -> Dict[str, Any]:
    """Arrays per split.  Empty recorded responses (data gap in response_matrix.json) are UNVERIFIABLE: their Sigma is set to NaN for every
    verifier (the task is dropped for that primary) unless ``include_empty`` (sensitivity: counted as failed by V1/V2 and judged as '(empty response)')."""
    S, C, L, TT = M["S"], M["C"], M["L"], M["TT"]
    pool_ids = list(samples["probing"]) + list(samples["id_test"])
    # blended $/token of each model on the sampled tasks (converts judge tokens to dollars; stated convention)
    price_tok = {m: float(C.loc[pool_ids, m].sum() / max(TT.loc[pool_ids, m].sum(), 1)) for m in PRIMARIES + [REF]}
    judge_model = None
    judge_rate = 0.05
    if jd and jd.get("rows"):
        judge_model = jd.get("model")
        judge_rate = float(jd.get("units_per_token") or 0.05)
    D: Dict[str, Any] = dict(price_per_token=price_tok, truth_level=truth_level, judge_model=judge_model, judge_units_per_token=judge_rate)
    for split in ("probing", "id_test"):
        tids = list(samples[split])
        n = len(tids)
        A: Dict[str, Any] = dict(tids=tids, n=n, src=np.array([t.split("_")[1] for t in tids]))
        A["YB"] = (S.loc[tids, REF].to_numpy() >= truth_level - 1e-12).astype(float)
        A["SB_raw"] = S.loc[tids, REF].to_numpy(float)
        A["CB"] = {"usd": C.loc[tids, REF].to_numpy(float), "sec": L.loc[tids, REF].to_numpy(float)}
        A["YP"], A["SP_raw"], A["CP"], A["SIG"], A["TV"], A["JTOK"], A["JUNITS"], A["EMPTY"] = {}, {}, {}, defaultdict(dict), defaultdict(dict), {}, {}, {}
        for m in PRIMARIES:
            A["YP"][m] = (S.loc[tids, m].to_numpy() >= truth_level - 1e-12).astype(float)
            A["SP_raw"][m] = S.loc[tids, m].to_numpy(float)
            A["CP"][m] = {"usd": C.loc[tids, m].to_numpy(float), "sec": L.loc[tids, m].to_numpy(float)}
            v1 = [sb.get(f"V1|{m}|{t}") for t in tids]
            v2 = [sb.get(f"V2|{m}|{t}") for t in tids]
            A["SIG"]["V1"][m] = np.array([r["sigma"] if r else np.nan for r in v1], float)
            A["TV"]["V1"][m] = np.array([float(r.get("elapsed_s") or 0.0) if r else 0.0 for r in v1], float)
            A["SIG"]["V2"][m] = np.array([r["sigma"] if r else np.nan for r in v2], float)
            A["TV"]["V2"][m] = np.array([float(r.get("elapsed_s") or 0.0) if r else 0.0 for r in v2], float)
            A["SIG"]["oracle"][m] = A["SP_raw"][m].copy()
            A["TV"]["oracle"][m] = np.zeros(n)
            A["EMPTY"][m] = np.array([bool(r.get("empty_response")) if r else False for r in v2])
            if jd and jd.get("rows") and m in (jd.get("plan") or {}).get("primaries", JUDGED_PRIMARIES):
                rows = [jd["rows"].get(f"{m}|{t}") for t in tids]
                # relay errors (HTTP 400/5xx, no verdict) are UNVERIFIABLE -> NaN; judge parse failures keep the judge's own 0.5
                A["SIG"]["V3"][m] = np.array([r["p_correct"] if (r and not r.get("error")) else np.nan for r in rows], float)
                A["V3_ERR"] = A.get("V3_ERR", {})
                A["V3_ERR"][m] = np.array([bool(r and r.get("error")) for r in rows])
                A["TV"]["V3"][m] = np.array([float(r.get("latency_s") or 0.0) if r else 0.0 for r in rows], float)
                A["JTOK"][m] = np.array([float(r.get("total_tokens") or 0) if r else 0.0 for r in rows], float)
                A["JUNITS"][m] = A["JTOK"][m] * judge_rate
            if not include_empty:
                for v in list(A["SIG"].keys()):
                    if m in A["SIG"][v]:
                        A["SIG"][v][m] = np.where(A["EMPTY"][m], np.nan, A["SIG"][v][m])
        D[split] = A
    return D


def verifier_cost(A: Dict[str, Any], v: str, m: str, basis: str, D: Dict[str, Any], judge_price: str = "primary") -> np.ndarray:
    """Verifier cost added to the attempt: V1/V2 sandbox seconds (seconds basis only), V3 judge latency (sec) or judge tokens x $/token (usd)."""
    n = A["n"]
    if v in ("V1", "V2"):
        return A["TV"][v][m] if basis == "sec" else np.zeros(n)
    if v == "V3":
        if basis == "sec":
            return A["TV"]["V3"][m]
        conv = {"primary": D["price_per_token"][m], "opus": D["price_per_token"][REF], "free": 0.0}[judge_price]
        return A["JTOK"][m] * conv
    return np.zeros(n)


# ============================================================================================ 6. analysis


def analyze(D: Dict[str, Any], rng: np.random.Generator, judge_price: str = "primary", quick: bool = False) -> Dict[str, Any]:
    thetas = THETAS + [THETA_REJECT]
    NG = len(THETAS)
    P, Tt = D["probing"], D["id_test"]
    Wt = boot_weights(Tt["n"], rng)
    Wp = boot_weights(P["n"], rng)
    R: Dict[str, Any] = dict(thetas=thetas, n_grid=NG, judge_price=judge_price, cells={}, per_source={}, sign_cells=[], allacc_cells=[],
                             thetamin_cells=[], argmax_cells=[], mlr={}, cond_indep={}, roc={}, c6={}, eta_B={})
    verifiers = [v for v in VERIFIERS if v in Tt["SIG"] and any(m in Tt["SIG"][v] for m in PRIMARIES)]
    for basis in BASES:
        R["eta_B"][basis] = ci(_div(Wt @ Tt["YB"], Wt @ Tt["CB"][basis]))
    sources_list = sorted(set(P["src"]))

    def run_cell(v: str, m: str, basis: str, mask_p: np.ndarray, mask_t: np.ndarray, Wp_: np.ndarray, Wt_: np.ndarray) -> Dict[str, Any]:
        sig_p, sig_t = P["SIG"][v][m][mask_p], Tt["SIG"][v][m][mask_t]
        prim = primitives(Wp_, sig_p, P["YP"][m][mask_p], P["YB"][mask_p], P["CP"][m][basis][mask_p], P["CB"][basis][mask_p], verifier_cost(P, v, m, basis, D, judge_price)[mask_p], thetas)
        real: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros((Wt_.shape[0], len(thetas))))
        Hs = []
        for j, th in enumerate(thetas):
            H = H_of(sig_t, th)
            Hs.append(H)
            st = cascade_stats(Wt_, H, Tt["YP"][m][mask_t], Tt["YB"][mask_t], Tt["CP"][m][basis][mask_t], Tt["CB"][basis][mask_t], verifier_cost(Tt, v, m, basis, D, judge_price)[mask_t])
            for k in ("eta", "eta_B", "r", "d_eta", "Q", "e", "e_plus", "p", "v_h", "vbar_h", "cbar_h", "EV", "ED", "gain", "avoided", "attempt", "a_verifier", "S", "vbar"):
                real[k][:, j] = st[k]
        yt = Tt["YP"][m][mask_t]
        ybt = Tt["YB"][mask_t]
        TPR_t = np.array([float(H[yt == 1].mean()) if (yt == 1).any() else np.nan for H in Hs])
        FPR_t = np.array([float(H[yt == 0].mean()) if (yt == 0).any() else np.nan for H in Hs])
        prim_t = dict(pi0=float(yt.mean()), vbar=float(ybt.mean()), beta1=float(ybt[yt == 1].mean()) if (yt == 1).any() else float("nan"),
                      beta0=float(ybt[yt == 0].mean()) if (yt == 0).any() else float("nan"),
                      a=float((Tt["CP"][m][basis][mask_t] + verifier_cost(Tt, v, m, basis, D, judge_price)[mask_t]).mean()), b=float(Tt["CB"][basis][mask_t].mean()))
        return dict(prim=prim, real=dict(real), Hs=Hs, TPR_t=TPR_t, FPR_t=FPR_t, prim_t=prim_t)

    def summarize_cell(v: str, m: str, basis: str, src: str, cell: Dict[str, Any]) -> Dict[str, Any]:
        prim, real = cell["prim"], cell["real"]
        out: Dict[str, Any] = dict(verifier=v, primary=m, basis=basis, source=src, n_probing=int(prim["n"][0]), n_id_test=int(len(cell["Hs"][0])))
        for k in ("pi0", "vbar", "b", "a", "beta1", "beta0", "eta_B_pred", "allacc_general", "allacc_hom", "allacc_uninformative"):
            out[k] = ci(prim[k])
        out["theta"] = thetas
        out["TPR_probing"] = prim["TPR"][0].tolist()
        out["FPR_probing"] = prim["FPR"][0].tolist()
        out["TPR_id_test"] = cell["TPR_t"].tolist()
        out["FPR_id_test"] = cell["FPR_t"].tolist()
        for k in ("floor_hom", "floor_naive", "floor_penal", "gain_gen", "Q_pred", "eta_pred", "r_pred"):
            out[k] = [ci(prim[k][:, j]) for j in range(len(thetas))]
        for k in ("r", "d_eta", "Q", "e", "p", "v_h", "vbar_h", "cbar_h", "EV", "ED", "gain", "avoided", "attempt", "a_verifier", "S"):
            out["real_" + k] = [ci(real[k][:, j]) for j in range(len(thetas))]
        out["plug_r"] = [ci(prim["plug"]["r"][:, j]) for j in range(len(thetas))]
        out["plug_Q"] = [ci(prim["plug"]["Q"][:, j]) for j in range(len(thetas))]
        # predictions (point estimates from probing) and realizations (id_test)
        pred_argmax_model = float(argmax_theta(prim["eta_pred"], thetas, NG)[0])
        pred_argmax_plug = float(argmax_theta(prim["plug"]["eta"], thetas, NG)[0])
        real_argmax = argmax_theta(real["eta"], thetas, NG)
        out["argmax"] = dict(pred_model=pred_argmax_model, pred_plug=pred_argmax_plug, real=float(real_argmax[0]),
                             real_ci=[float(np.percentile(real_argmax[1:], 2.5)), float(np.percentile(real_argmax[1:], 97.5))],
                             real_freq_pred_model=float(np.mean(real_argmax[1:] == pred_argmax_model)),
                             pred_model_fit_ci=[float(np.percentile(argmax_theta(prim["eta_pred"], thetas, NG)[1:], 2.5)), float(np.percentile(argmax_theta(prim["eta_pred"], thetas, NG)[1:], 97.5))])
        out["theta_min"] = {}
        for eps in EPS_GRID:
            tm_pred = theta_min_from(prim["Q_pred"], eps, thetas)
            tm_plug = theta_min_from(prim["plug"]["Q"], eps, thetas)
            tm_real = theta_min_from(real["Q"], eps, thetas)
            adm = real["Q"][0] >= -eps - 1e-12
            adm[-1] = True
            out["theta_min"][str(eps)] = dict(pred_model=float(tm_pred[0]), pred_model_fit_ci=[float(np.percentile(tm_pred[1:], 2.5)), float(np.percentile(tm_pred[1:], 97.5))],
                                              pred_plug=float(tm_plug[0]), real=float(tm_real[0]),
                                              real_ci=[float(np.percentile(tm_real[1:], 2.5)), float(np.percentile(tm_real[1:], 97.5))],
                                              real_freq_pred_model=float(np.mean(tm_real[1:] == tm_pred[0])), admissible_is_interval=is_upset(adm),
                                              admissible_pred_is_interval=is_upset(np.array((prim["Q_pred"][0] >= -eps - 1e-12)[:-1].tolist() + [True])))
        # all-accept optimality
        e0 = real["eta"][:, 0]
        j_best = 1 + int(np.nanargmax(real["eta"][0, 1:NG]))          # point-estimate argmax over theta > 0 (fixed across resamples)
        diff = e0 - real["eta"][:, j_best]
        out["allaccept"] = dict(pred_general=bool(prim["allacc_general"][0] <= 0), pred_hom=bool(prim["allacc_hom"][0] <= 0), pred_uninformative=bool(prim["allacc_uninformative"][0] <= 0),
                                real_opt=bool(diff[0] >= -1e-12), diff_ci=ci(diff), decided=excl0(ci(diff)), best_other_theta=float(thetas[j_best]),
                                id_test=dict(pi0=cell["prim_t"]["pi0"], beta0=cell["prim_t"]["beta0"], beta1=cell["prim_t"]["beta1"], a_over_b=cell["prim_t"]["a"] / cell["prim_t"]["b"] if cell["prim_t"]["b"] > 0 else float("nan"),
                                             pred_general_id_test=bool(cell["prim_t"]["a"] * cell["prim_t"]["beta0"] <= cell["prim_t"]["pi0"] * cell["prim_t"]["b"]) if np.isfinite(cell["prim_t"]["beta0"]) else None),
                                pred_general_fit_ci=ci(prim["allacc_general"]), a_over_b=ci(_div(prim["a"], prim["b"])), pi0_over_beta0=ci(_div(prim["pi0"], prim["beta0"])), pi0_over_vbar=ci(_div(prim["pi0"], prim["vbar"])))
        return out

    # ---- global cells (all sources pooled) and per-source cells
    cells_raw: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for v in verifiers:
        for m in PRIMARIES:
            if m not in Tt["SIG"][v] or m not in P["SIG"][v]:
                continue
            mask_p_all = np.isfinite(P["SIG"][v][m])
            mask_t_all = np.isfinite(Tt["SIG"][v][m])
            if mask_p_all.sum() < 20 or mask_t_all.sum() < 20:
                continue
            for basis in BASES:
                Wp_ = Wp[:, mask_p_all]
                Wt_ = Wt[:, mask_t_all]
                cell = run_cell(v, m, basis, mask_p_all, mask_t_all, Wp_, Wt_)
                cells_raw[(v, m, basis, "all")] = cell
                R["cells"][f"{v}|{m}|{basis}"] = summarize_cell(v, m, basis, "all", cell)
                if quick:
                    continue
                for s in sources_list:
                    mp = mask_p_all & (P["src"] == s)
                    mt = mask_t_all & (Tt["src"] == s)
                    if mp.sum() < 20 or mt.sum() < 15 or P["YP"][m][mp].sum() < 3 or (1 - P["YP"][m][mp]).sum() < 3:
                        continue
                    cell_s = run_cell(v, m, basis, mp, mt, Wp[:, mp], Wt[:, mt])
                    cells_raw[(v, m, basis, s)] = cell_s
                    R["per_source"][f"{v}|{m}|{basis}|{s}"] = summarize_cell(v, m, basis, s, cell_s)
    # ---- C1 sign cells (distinct operating points; reject-all excluded)
    for (v, m, basis, s), cell in cells_raw.items():
        prim, real = cell["prim"], cell["real"]
        seen = set()
        for j, th in enumerate(THETAS):
            key = cell["Hs"][j].tobytes()
            if key in seen:
                continue
            seen.add(key)
            d = ci(real["d_eta"][:, j])
            R["sign_cells"].append(dict(verifier=v, primary=m, basis=basis, source=s, theta=th, TPR=float(prim["TPR"][0, j]), FPR=float(prim["FPR"][0, j]),
                                        sign_real=sgn(d[0]), decided_real=excl0(d), d_eta_ci=d, r_real=float(real["r"][0, j]),
                                        pred_hom=sgn(prim["floor_hom"][0, j]), decided_hom=excl0(ci(prim["floor_hom"][:, j])),
                                        pred_naive=sgn(prim["floor_naive"][0, j]), pred_penal=sgn(prim["floor_penal"][0, j]), pred_gen=sgn(prim["gain_gen"][0, j]), decided_gen=excl0(ci(prim["gain_gen"][:, j])),
                                        pred_plug=sgn(prim["plug"]["d_eta"][0, j])))
        R["allacc_cells"].append(dict(verifier=v, primary=m, basis=basis, source=s, **{k: val for k, val in R[("per_source" if s != "all" else "cells")][f"{v}|{m}|{basis}" + (f"|{s}" if s != "all" else "")]["allaccept"].items()}))
        summ = R[("per_source" if s != "all" else "cells")][f"{v}|{m}|{basis}" + (f"|{s}" if s != "all" else "")]
        for eps in EPS_GRID:
            tm = summ["theta_min"][str(eps)]
            R["thetamin_cells"].append(dict(verifier=v, primary=m, basis=basis, source=s, eps=eps, **tm,
                                            inside=bool(tm["real_ci"][0] - 1e-9 <= tm["pred_model"] <= tm["real_ci"][1] + 1e-9),
                                            inside_plug=bool(tm["real_ci"][0] - 1e-9 <= tm["pred_plug"] <= tm["real_ci"][1] + 1e-9),
                                            naive0_inside=bool(tm["real_ci"][0] - 1e-9 <= 0.0 <= tm["real_ci"][1] + 1e-9)))
        am = summ["argmax"]
        R["argmax_cells"].append(dict(verifier=v, primary=m, basis=basis, source=s, **am,
                                      inside=bool(am["real_ci"][0] - 1e-9 <= am["pred_model"] <= am["real_ci"][1] + 1e-9),
                                      inside_plug=bool(am["real_ci"][0] - 1e-9 <= am["pred_plug"] <= am["real_ci"][1] + 1e-9),
                                      naive0_inside=bool(am["real_ci"][0] - 1e-9 <= 0.0 <= am["real_ci"][1] + 1e-9)))

    def sign_summary(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        dec = [c for c in cells if c["decided_real"]]
        out: Dict[str, Any] = dict(n_cells=len(cells), n_decided=len(dec))
        if not dec:
            return out
        sr = np.array([c["sign_real"] for c in dec])
        for k in ("pred_hom", "pred_naive", "pred_penal", "pred_gen", "pred_plug"):
            out["agree_" + k] = float(np.mean(np.array([c[k] for c in dec]) == sr))
            out["n_agree_" + k] = int(np.sum(np.array([c[k] for c in dec]) == sr))
        out["baseline_always_plus"] = float(np.mean(sr >= 0))
        out["baseline_always_minus"] = float(np.mean(sr < 0))
        out["baseline_majority"] = max(out["baseline_always_plus"], out["baseline_always_minus"])
        # per-basis majority (dollars are almost always +, seconds almost always -): the relevant naive baseline for a pooled set
        agree_pb = 0
        for b_ in BASES:
            sub = [c for c in dec if c["basis"] == b_]
            if sub:
                s_ = np.array([c["sign_real"] for c in sub])
                agree_pb += max((s_ >= 0).sum(), (s_ < 0).sum())
        out["baseline_per_basis_majority"] = float(agree_pb / len(dec))
        # cells where the naive / penalized variants disagree with the FPR-free floor (the only cells that can discriminate)
        for k in ("pred_naive", "pred_penal", "pred_gen"):
            dis = [c for c in dec if c[k] != c["pred_hom"]]
            out["n_disagree_" + k] = len(dis)
            out["hom_right_when_disagree_" + k] = float(np.mean([c["pred_hom"] == c["sign_real"] for c in dis])) if dis else float("nan")
        both = [c for c in dec if c["decided_hom"]]
        out["n_both_decided_hom"] = len(both)
        if both:
            out["agree_hom_both_decided"] = float(np.mean(np.array([c["pred_hom"] for c in both]) == np.array([c["sign_real"] for c in both])))
        return out

    R["sign_summary"] = dict(all=sign_summary(R["sign_cells"]),
                             by_basis={b: sign_summary([c for c in R["sign_cells"] if c["basis"] == b]) for b in BASES},
                             by_verifier={v: sign_summary([c for c in R["sign_cells"] if c["verifier"] == v]) for v in verifiers},
                             pooled_only=sign_summary([c for c in R["sign_cells"] if c["source"] == "all"]),
                             per_source_only=sign_summary([c for c in R["sign_cells"] if c["source"] != "all"]))

    def allacc_summary(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(n_cells=len(cells))
        dec = [c for c in cells if c["decided"]]
        out["n_decided"] = len(dec)
        for pool_name, pool in (("all", cells), ("decided", dec)):
            if not pool:
                continue
            ro = np.array([c["real_opt"] for c in pool])
            for k in ("pred_general", "pred_hom", "pred_uninformative"):
                pr = np.array([c[k] for c in pool])
                out[f"{pool_name}_agree_{k}"] = float(np.mean(pr == ro))
                out[f"{pool_name}_n_pred_true_{k}"] = int(pr.sum())
                # sufficiency: predicted all-accept -> realized all-accept optimal
                out[f"{pool_name}_sufficiency_{k}"] = float(np.mean(ro[pr])) if pr.any() else float("nan")
                out[f"{pool_name}_n_violations_{k}"] = int(np.sum(pr & ~ro))
            out[f"{pool_name}_share_real_opt"] = float(ro.mean())
            out[f"{pool_name}_baseline_majority"] = float(max(ro.mean(), 1 - ro.mean()))
        return out

    R["allacc_summary"] = dict(all=allacc_summary(R["allacc_cells"]), by_basis={b: allacc_summary([c for c in R["allacc_cells"] if c["basis"] == b]) for b in BASES},
                               pooled_only=allacc_summary([c for c in R["allacc_cells"] if c["source"] == "all"]))

    def inside_summary(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not cells:
            return dict(n=0)
        return dict(n=len(cells), inside_model=float(np.mean([c["inside"] for c in cells])), inside_plug=float(np.mean([c["inside_plug"] for c in cells])),
                    naive0_inside=float(np.mean([c["naive0_inside"] for c in cells])),
                    exact_match_model=float(np.mean([abs(c["real"] - c["pred_model"]) < 1e-9 for c in cells])),
                    mean_abs_dev_model=float(np.mean([abs(c["real"] - c["pred_model"]) for c in cells])),
                    mean_freq_pred_model=float(np.mean([c["real_freq_pred_model"] for c in cells])))

    R["thetamin_summary"] = dict(all=inside_summary(R["thetamin_cells"]), by_eps={str(e): inside_summary([c for c in R["thetamin_cells"] if c["eps"] == e]) for e in EPS_GRID},
                                 by_verifier={v: inside_summary([c for c in R["thetamin_cells"] if c["verifier"] == v]) for v in verifiers},
                                 pooled_only=inside_summary([c for c in R["thetamin_cells"] if c["source"] == "all"]),
                                 interval_share_real=float(np.mean([c["admissible_is_interval"] for c in R["thetamin_cells"]])) if R["thetamin_cells"] else float("nan"),
                                 interval_share_real_by_verifier={v: float(np.mean([c["admissible_is_interval"] for c in R["thetamin_cells"] if c["verifier"] == v])) for v in verifiers})
    R["argmax_summary"] = dict(all=inside_summary(R["argmax_cells"]), by_verifier={v: inside_summary([c for c in R["argmax_cells"] if c["verifier"] == v]) for v in verifiers},
                               by_basis={b: inside_summary([c for c in R["argmax_cells"] if c["basis"] == b]) for b in BASES},
                               pooled_only=inside_summary([c for c in R["argmax_cells"] if c["source"] == "all"]))

    # ---- C4 MLR, C5 conditional independence, ROC
    for v in verifiers:
        for m in PRIMARIES:
            if m not in P["SIG"][v]:
                continue
            for split, A in (("probing", P), ("id_test", Tt)):
                sig = A["SIG"][v][m]
                ok = np.isfinite(sig)
                if ok.sum() < 20:
                    continue
                R["mlr"][f"{v}|{m}|{split}"] = mlr_check(sig[ok], A["YP"][m][ok])
                R["roc"][f"{v}|{m}|{split}"] = dict(auroc=auroc(A["YP"][m][ok], sig[ok]), n=int(ok.sum()), pi0=float(A["YP"][m][ok].mean()),
                                                    n_distinct=int(len(np.unique(sig[ok]))), share_neutral=float(np.mean(sig[ok] == 0.5)) if v == "V1" else None,
                                                    empty_responses=int(A["EMPTY"][m][ok].sum()), empty_but_truth1=int((A["EMPTY"][m][ok] & (A["YP"][m][ok] == 1)).sum()))
                if not quick:
                    R["cond_indep"][f"{v}|{m}|{split}"] = cond_indep(sig[ok], A["YP"][m][ok], A["YB"][ok], rng, THETAS, src=A["src"][ok])
    # ---- C6: cheap imperfect verifier at its floor-respecting theta vs oracle cascade vs no verifier (IN-SAMPLE choice on id_test; OOS choice on probing)
    for m in PRIMARIES:
        for basis in BASES:
            key = f"{m}|{basis}"
            avail = [v for v in verifiers if (v, m, basis, "all") in cells_raw]
            if "V1" not in avail or "oracle" not in avail:
                continue
            # common task subset (judged subset when V3 is available for this primary)
            if "V3" in avail:
                mask_t = np.isfinite(Tt["SIG"]["V3"][m])
                mask_p = np.isfinite(P["SIG"]["V3"][m])
            else:
                mask_t = np.ones(Tt["n"], bool)
                mask_p = np.ones(P["n"], bool)
            Wt_ = Wt[:, mask_t]
            Wp_ = Wp[:, mask_p]
            sub = {v: run_cell(v, m, basis, mask_p, mask_t, Wp_, Wt_) for v in avail}
            entry: Dict[str, Any] = dict(n_id_test=int(mask_t.sum()), n_probing=int(mask_p.sum()), verifiers=avail, eps={})
            oracle_eta1 = sub["oracle"]["real"]["eta"][:, NG - 1]     # theta = 1 on the released score (X1 control)
            accept_all = sub["V1"]["real"]["eta"][:, 0]               # theta = 0: no verifier, primary only (identical for every verifier up to verifier cost)
            eta_B_sub = sub["V1"]["real"]["eta_B"][:, 0]
            entry["oracle_theta1_r"] = ci(oracle_eta1 / eta_B_sub)
            entry["accept_all_r"] = ci(accept_all / eta_B_sub)
            for eps in EPS_GRID:
                row: Dict[str, Any] = {}
                for v in avail:
                    real, prim = sub[v]["real"], sub[v]["prim"]
                    adm = real["Q"][0] >= -eps - 1e-12
                    etas = np.where(adm, real["eta"][0], -np.inf)
                    j_is = int(np.argmax(etas))                        # IN-SAMPLE: theta chosen on id_test itself
                    adm_p = prim["Q_pred"][0] >= -eps - 1e-12
                    etas_p = np.where(adm_p, prim["eta_pred"][0], -np.inf)
                    j_oos = int(np.argmax(etas_p))                     # OUT-OF-SAMPLE: theta chosen from probing primitives (model-based)
                    row[v] = dict(theta_is=thetas[j_is], r_is=ci(real["eta"][:, j_is] / eta_B_sub), Q_is=ci(real["Q"][:, j_is]), e_is=ci(real["e"][:, j_is]),
                                  TPR_is=float(sub[v]["TPR_t"][j_is]), FPR_is=float(sub[v]["FPR_t"][j_is]),
                                  d_vs_oracle_is=ci(real["eta"][:, j_is] - oracle_eta1), d_vs_acceptall_is=ci(real["eta"][:, j_is] - accept_all), d_vs_ref_is=ci(real["eta"][:, j_is] - eta_B_sub),
                                  theta_oos=thetas[j_oos], r_oos=ci(real["eta"][:, j_oos] / eta_B_sub), Q_oos=ci(real["Q"][:, j_oos]),
                                  d_vs_oracle_oos=ci(real["eta"][:, j_oos] - oracle_eta1), d_vs_acceptall_oos=ci(real["eta"][:, j_oos] - accept_all), d_vs_ref_oos=ci(real["eta"][:, j_oos] - eta_B_sub),
                                  floor_ok_oos=ci(real["Q"][:, j_oos] + eps),
                                  r_pred_oos=ci(prim["r_pred"][:, j_oos]), floor_sign_oos=sgn(prim["floor_hom"][0, j_oos]), sign_real_oos=sgn(float(real["eta"][0, j_oos] - eta_B_sub[0])),
                                  decided_real_oos=excl0(ci(real["eta"][:, j_oos] - eta_B_sub)))
                    row[v]["r_pred_covered_oos"] = bool(row[v]["r_oos"][1] - 1e-12 <= row[v]["r_pred_oos"][0] <= row[v]["r_oos"][2] + 1e-12)
                entry["eps"][str(eps)] = row
            R["c6"][key] = entry
    return R


# ============================================================================================ 7. figures

COL = {"V1": "#23639b", "V2": "#3a7c5e", "V3": "#ba5934", "oracle": "#6f6f6f"}
PRIM_SHORT = {"Qwen3-Max": "Qwen3-Max", "kimi-k2.5": "kimi-k2.5", "qwen3.5-plus": "qwen3.5-plus"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "text.usetex": False, "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.linewidth": 0.65, "lines.linewidth": 1.4})
    return plt


def fig_eta_theta(R: Dict[str, Any], path: str) -> None:
    plt = setup_mpl()
    NG = R["n_grid"]
    th = np.array(R["thetas"][:NG])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for i, basis in enumerate(BASES):
        for j, m in enumerate(PRIMARIES):
            ax = axes[i, j]
            for v in VERIFIERS:
                c = R["cells"].get(f"{v}|{m}|{basis}")
                if not c:
                    continue
                r = np.array(c["real_r"])[:NG]
                ax.plot(th, r[:, 0], color=COL[v], label=VERIFIER_LABEL[v], marker="o", ms=2.2)
                ax.fill_between(th, r[:, 1], r[:, 2], color=COL[v], alpha=0.15, lw=0)
                pr = np.array(c["r_pred"])[:NG, 0]
                ax.plot(th, pr, color=COL[v], ls=":", lw=0.9)
            ax.axhline(1.0, color="k", lw=0.6, ls="--")
            if i == 1:
                ax.set_xlabel(r"threshold $\theta$")
            if j == 0:
                ax.set_ylabel(r"$\eta(\theta)/\eta_B$ " + ("(dollars)" if basis == "usd" else "(seconds)"))
            ax.text(0.02, 0.97, PRIM_SHORT[m], transform=ax.transAxes, va="top", fontsize=8)
            if basis == "usd":
                ax.set_yscale("log")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_Q_theta(R: Dict[str, Any], path: str) -> None:
    plt = setup_mpl()
    NG = R["n_grid"]
    th = np.array(R["thetas"][:NG])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for j, m in enumerate(PRIMARIES):
        ax = axes[j]
        for v in VERIFIERS:
            c = R["cells"].get(f"{v}|{m}|usd")
            if not c:
                continue
            q = np.array(c["real_Q"])[:NG]
            ax.plot(th, q[:, 0], color=COL[v], marker="o", ms=2.2, label=VERIFIER_LABEL[v])
            ax.fill_between(th, q[:, 1], q[:, 2], color=COL[v], alpha=0.15, lw=0)
            ax.plot(th, np.array(c["Q_pred"])[:NG, 0], color=COL[v], ls=":", lw=0.9)
            for eps, mk in ((0.01, "v"), (0.02, "^"), (0.05, "s")):
                tm = c["theta_min"][str(eps)]["pred_model"]
                if tm <= 1.0:
                    ax.plot([tm], [-eps], marker=mk, color=COL[v], ms=4.5, mfc="white", mew=1.0, ls="none")
        for eps in (0.01, 0.02, 0.05):
            ax.axhline(-eps, color="k", lw=0.5, ls="--", alpha=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"threshold $\theta$")
        if j == 0:
            ax.set_ylabel(r"$Q(\theta)=\mathbb{E}[H(Y_P-Y_B)]$")
        ax.text(0.02, 0.05, PRIM_SHORT[m], transform=ax.transAxes, va="bottom", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_roc(R: Dict[str, Any], path: str, basis: str = "usd") -> None:
    plt = setup_mpl()
    NG = R["n_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for j, m in enumerate(PRIMARIES):
        ax = axes[j]
        c_any = None
        for v in VERIFIERS:
            c = R["cells"].get(f"{v}|{m}|{basis}")
            if not c:
                continue
            c_any = c_any or c
            fpr = np.array(c["FPR_id_test"])[:NG]
            tpr = np.array(c["TPR_id_test"])[:NG]
            o = np.argsort(fpr)
            ax.plot(fpr[o], tpr[o], color=COL[v], marker="o", ms=2.5, label=VERIFIER_LABEL[v])
        if c_any is not None:
            pi0, vbar, b, a = c_any["pi0"][0], c_any["vbar"][0], c_any["b"][0], c_any["a"][0]
            b1, b0 = c_any["beta1"][0], c_any["beta0"][0]
            eta_B = vbar / b
            F = np.linspace(0, 1, 101)
            # iso-efficiency lines eta(TPR, FPR) = r * eta_B, model-based (constant b, conditional independence)
            for rr, ls in ((1.0, "--"), (2.0, "-."), (4.0, ":")):
                lam = rr * eta_B
                # EV - lam ED = 0 with EV = pi0 TPR + pi0 (1-TPR) b1 + (1-pi0)(1-FPR) b0, ED = a + (1 - pi0 TPR - (1-pi0) FPR) b
                # coefficients: TPR: pi0(1 - b1) + lam b pi0 ; FPR: -(1-pi0) b0 + lam b (1-pi0); const: pi0 b1 + (1-pi0) b0 - lam (a + b)
                cT = pi0 * (1 - b1) + lam * b * pi0
                cF = -(1 - pi0) * b0 + lam * b * (1 - pi0)
                c0 = pi0 * b1 + (1 - pi0) * b0 - lam * (a + b)
                if abs(cT) > 1e-15:
                    T = -(c0 + cF * F) / cT
                    ok = (T >= 0) & (T <= 1)
                    if ok.any():
                        ax.plot(F[ok], T[ok], color="#999999", ls=ls, lw=0.8)
                        ax.text(F[ok][-1], min(T[ok][-1], 0.98), f"{rr:g}", fontsize=6.5, color="#777777", ha="right", va="bottom")
            # floor line Q = -eps (eps = 0.02): FPR = (eps + pi0 (1-b1) TPR)/((1-pi0) b0)
            eps = 0.02
            T = np.linspace(0, 1, 101)
            Fl = (eps + pi0 * (1 - b1) * T) / max((1 - pi0) * b0, 1e-12)
            ok = Fl <= 1
            ax.plot(Fl[ok], T[ok], color="#ba5934", lw=0.8, ls="--", alpha=0.8)
            ax.text(0.98, 0.04, r"floor $Q\geq-0.02$", fontsize=6.5, color="#ba5934", ha="right", transform=ax.transAxes)
        ax.plot([0, 1], [0, 1], color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR")
        if j == 0:
            ax.set_ylabel("TPR")
        ax.text(0.02, 0.97, PRIM_SHORT[m], transform=ax.transAxes, va="top", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_judge_calibration(calib: Dict[str, Any], path: str) -> None:
    plt = setup_mpl()
    y = np.array(calib["truth"], int)
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    cols = ["#23639b", "#3a7c5e", "#ba5934", "#7a5195", "#c9a227"]
    for (cid, model, thinking, mt), col in zip(JUDGE_CONFIGS, cols):
        cfg = calib["configs"].get(cid)
        if not cfg or not cfg.get("p"):
            continue
        p = np.array(cfg["p"], float)
        ths = np.unique(np.r_[p, 1.1])[::-1]
        pts = [(0.0, 0.0)] + [(float(((p >= t) & (y == 0)).mean() / max((y == 0).mean(), 1e-9)), float(((p >= t) & (y == 1)).mean() / max((y == 1).mean(), 1e-9))) for t in ths]
        pts = sorted(set(pts))
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=col, marker="o", ms=2.5, label=f"{model}{' (thinking)' if thinking and model.startswith('deepseek-v4') else ''}: AUROC {cfg['auroc']:.2f}")
    ax.plot([0, 1], [0, 1], color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cond_indep(R: Dict[str, Any], path: str, m: str = "Qwen3-Max") -> None:
    plt = setup_mpl()
    vs = [v for v in ("V1", "V3", "V2") if f"{v}|{m}|probing" in R["cond_indep"]]
    if not vs:
        return
    fig, axes = plt.subplots(1, len(vs), figsize=(2.4 * len(vs), 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, vs):
        ci_ = R["cond_indep"][f"{v}|{m}|probing"]
        for name, col, mk in (("given_correct", "#3a7c5e", "o"), ("given_incorrect", "#ba5934", "s")):
            tab = [t for t in ci_[name]["table"] if t["n"] >= 8]
            x = np.arange(len(tab))
            ph = np.array([t["p_yb"][0] for t in tab])
            lo = np.array([t["p_yb"][1] for t in tab])
            hi = np.array([t["p_yb"][2] for t in tab])
            ax.errorbar(x + (0.08 if name == "given_incorrect" else -0.08), ph, yerr=[ph - lo, hi - ph], color=col, marker=mk, ms=3, lw=0.8, capsize=1.5, ls="none",
                        label=(r"$Y=1$ (correct)" if name == "given_correct" else r"$Y=0$"))
            ax.set_xticks(x)
            ax.set_xticklabels([t["bin"] for t in tab], rotation=45, fontsize=6.5)
        ax.set_xlabel(r"$\Sigma$ bin (" + v + ")")
        ax.text(0.02, 0.05, f"p={ci_['given_correct']['p_perm']:.2f} / {ci_['given_incorrect']['p_perm']:.2f}", transform=ax.transAxes, fontsize=6.5)
    axes[0].set_ylabel(r"$\mathbb{P}(Y_B=1\mid Y,\Sigma)$")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_likelihood_ratio(R: Dict[str, Any], path: str) -> None:
    """C4: likelihood ratio P(bin | Y=1)/P(bin | Y=0) by score bin (probing); one panel per verifier, one series per primary.
    Open markers: bins with n < 15 (not used for the monotonicity count); up/down triangles: infinite / zero ratio (drawn at the axis caps)."""
    import re as _re
    from matplotlib.lines import Line2D
    plt = setup_mpl()
    vs = [v for v in VERIFIERS if any(f"{v}|{m}|probing" in R["mlr"] for m in PRIMARIES)]
    if not vs:
        return
    fig, axes = plt.subplots(1, len(vs), figsize=(1.85 * len(vs) + 0.7, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    PCOL = {"Qwen3-Max": "#23639b", "kimi-k2.5": "#ba5934", "qwen3.5-plus": "#3a7c5e"}
    PMK = {"Qwen3-Max": "o", "kimi-k2.5": "s", "qwen3.5-plus": "D"}

    def edge(lbl: str) -> float:
        try:
            return float(lbl)
        except ValueError:
            mt = _re.match(r"[\(\[]\s*([-.0-9]+)", lbl)
            return float(mt.group(1)) + 1e-6 if mt else 0.0

    for ax, v in zip(axes, vs):
        allb = [b for m in PRIMARIES for b in R["mlr"].get(f"{v}|{m}|probing", {}).get("bins", [])]
        labels = sorted({b["bin"] for b in allb}, key=edge)
        pos = {lab: i for i, lab in enumerate(labels)}
        fin = [b["LR"] for b in allb if np.isfinite(b["LR"]) and b["LR"] > 0]
        cap_hi = (max(fin) if fin else 1.0) * 4
        cap_lo = (min(fin) if fin else 1.0) / 4
        for k, m in enumerate(PRIMARIES):
            mm = R["mlr"].get(f"{v}|{m}|probing")
            if not mm:
                continue
            off = (k - 1) * 0.14
            for b in mm["bins"]:
                x = pos[b["bin"]] + off
                lr, small = b["LR"], b["n"] < 15
                if not np.isfinite(lr):
                    y, mk = cap_hi, "^"
                elif lr <= 0:
                    y, mk = cap_lo, "v"
                else:
                    y, mk = lr, PMK[m]
                ax.plot([x], [y], marker=mk, color=PCOL[m], ms=3.6 if mk in "^v" else 3.2, mfc=("white" if small else PCOL[m]), ls="none")
            pts = [(pos[b["bin"]] + off, b["LR"]) for b in mm["bins"] if np.isfinite(b["LR"]) and b["LR"] > 0 and b["n"] >= 15]
            if len(pts) > 1:
                ax.plot([q[0] for q in pts], [q[1] for q in pts], color=PCOL[m], lw=0.9)
        ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=6.5)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_xlabel(VERIFIER_LABEL[v], fontsize=8)
    axes[0].set_ylabel(r"$\mathbb{P}(\Sigma\in\mathrm{bin}\mid Y{=}1)\,/\,\mathbb{P}(\Sigma\in\mathrm{bin}\mid Y{=}0)$", fontsize=7.5)
    handles = [Line2D([], [], color=PCOL[m], marker=PMK[m], ms=3.2, lw=0.9, label=PRIM_SHORT[m]) for m in PRIMARIES]
    handles += [Line2D([], [], color="#555555", marker="o", mfc="white", ms=3.2, lw=0, label="bin with n < 15"), Line2D([], [], color="#555555", marker="^", ms=3.6, lw=0, label="infinite ratio (cap)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ============================================================================================ 8. claims, audit, reports


def claims_from(R: Dict[str, Any], calib: Optional[Dict[str, Any]], D: Dict[str, Any]) -> List[Dict[str, Any]]:
    cl: List[Dict[str, Any]] = []
    ss = R["sign_summary"]["all"]
    if ss.get("n_decided", 0) >= 10:
        hom, naive, penal, gen = ss["agree_pred_hom"], ss["agree_pred_naive"], ss["agree_pred_penal"], ss["agree_pred_gen"]
        base = ss["baseline_majority"]
        ok = hom >= 0.9 and hom >= base and naive <= hom + 0.02 and penal <= hom + 0.02 and gen <= hom + 0.02
        verdict = "PASS" if ok else ("INCONCLUSIVE" if hom >= base and (naive > hom + 0.02 or penal > hom + 0.02 or gen > hom + 0.02) else "FAIL")
        res = (f"{ss['n_decided']} CI-decided distinct operating points (of {ss['n_cells']}): floor without FPR {hom:.3f}; with FPR added as p b {naive:.3f} (disagrees with the FPR-free floor in {ss['n_disagree_pred_naive']} cells, floor right in {ss['hom_right_when_disagree_pred_naive']:.2f} of them); "
               f"with FPR penalized {penal:.3f} (disagrees in {ss['n_disagree_pred_penal']}, floor right in {ss['hom_right_when_disagree_pred_penal']:.2f}); general affine form with beta's {gen:.3f} (disagrees in {ss['n_disagree_pred_gen']}); plug-in probing sign {ss['agree_pred_plug']:.3f}; "
               f"naive baselines always+ {ss['baseline_always_plus']:.3f} / majority {base:.3f} / per-basis majority {ss['baseline_per_basis_majority']:.3f}; "
               f"by basis usd {R['sign_summary']['by_basis']['usd'].get('agree_pred_hom', float('nan')):.3f} (n={R['sign_summary']['by_basis']['usd'].get('n_decided', 0)}), "
               f"sec {R['sign_summary']['by_basis']['sec'].get('agree_pred_hom', float('nan')):.3f} (n={R['sign_summary']['by_basis']['sec'].get('n_decided', 0)})")
    else:
        verdict, res = "INCONCLUSIVE", f"only {ss.get('n_decided', 0)} CI-decided cells"
    cl.append(dict(id="C1", kind="OUT-OF-SAMPLE", statement="The sign of eta(theta) - eta_B is predicted by the floor pi0 TPR(theta) b >= vbar a without FPR; adding FPR does not improve the sign prediction",
                   result=res, verdict=verdict))
    aa = R["allacc_summary"]["all"]
    if aa.get("n_cells", 0) >= 4:
        suff, agree, viol = aa.get("all_sufficiency_pred_general", float("nan")), aa.get("all_agree_pred_general", float("nan")), aa.get("all_n_violations_pred_general", 0)
        verdict = "PASS" if (np.isfinite(suff) and suff >= 0.9 and agree >= aa.get("all_baseline_majority", 0)) else ("INCONCLUSIVE" if aa.get("all_n_pred_true_pred_general", 0) < 3 else "FAIL")
        viol_cells = [c for c in R["allacc_cells"] if c["pred_general"] and not c["real_opt"]]
        viol_txt = "; ".join(f"{c['verifier']}/{c['primary']}/{c['basis']}/{c['source']}: probing a/b {c['a_over_b'][0]:.3f} vs pi0/beta0 {c['pi0_over_beta0'][0]:.3f}, id_test a/b {c['id_test']['a_over_b']:.3f} vs pi0/beta0 {(c['id_test']['pi0'] / c['id_test']['beta0']) if c['id_test']['beta0'] else float('nan'):.3f} (condition on id_test itself: {c['id_test']['pred_general_id_test']}), eta(0)-eta(best) CI {fmt(c['diff_ci'], 3)}" for c in viol_cells)
        res = (f"{aa['n_cells']} cells (verifier x primary x basis x [source]): general condition a beta0 <= pi0 b predicts theta=0 optimal in {aa['all_n_pred_true_pred_general']} cells, "
               f"realized in {suff:.3f} of them ({viol} violations); agreement {agree:.3f} vs majority baseline {aa['all_baseline_majority']:.3f}; homogeneous form agreement {aa['all_agree_pred_hom']:.3f} "
               f"(sufficiency {aa.get('all_sufficiency_pred_hom', float('nan')):.3f}); CI-decided cells {aa['n_decided']}: agreement {aa.get('decided_agree_pred_general', float('nan')):.3f}, sufficiency {aa.get('decided_sufficiency_pred_general', float('nan')):.3f}; "
               f"dollars: theta=0 realized-optimal share {R['allacc_summary']['by_basis']['usd'].get('all_share_real_opt', float('nan')):.2f}, seconds: {R['allacc_summary']['by_basis']['sec'].get('all_share_real_opt', float('nan')):.2f}; violations: {viol_txt or 'none'}")
    else:
        verdict, res = "INCONCLUSIVE", "too few cells"
    cl.append(dict(id="C2", kind="OUT-OF-SAMPLE", statement="The general all-accept condition a beta0 <= pi0 b (probing) predicts whether theta = 0 is realized-optimal on id_test (sufficient condition)", result=res, verdict=verdict))
    tm, am = R["thetamin_summary"]["all"], R["argmax_summary"]["all"]
    if tm.get("n", 0) and am.get("n", 0):
        verdict = "PASS" if (tm["inside_model"] >= 0.8 and am["inside_model"] >= 0.8) else ("INCONCLUSIVE" if min(tm["inside_model"], am["inside_model"]) >= 0.6 else "FAIL")
        res = (f"theta_min(eps): predicted (model-based, probing) inside the realized 95% bootstrap interval in {tm['inside_model']:.3f} of {tm['n']} cells (plug-in {tm['inside_plug']:.3f}; naive theta_min=0 {tm['naive0_inside']:.3f}; exact match {tm['exact_match_model']:.3f}); "
               f"by eps: " + ", ".join(f"{e}: {R['thetamin_summary']['by_eps'][str(e)].get('inside_model', float('nan')):.2f}" for e in EPS_GRID) +
               f"; argmax theta: inside {am['inside_model']:.3f} of {am['n']} cells (plug-in {am['inside_plug']:.3f}; naive 0 {am['naive0_inside']:.3f}; exact {am['exact_match_model']:.3f}); pooled-only cells: theta_min {R['thetamin_summary']['pooled_only'].get('inside_model', float('nan')):.2f}, argmax {R['argmax_summary']['pooled_only'].get('inside_model', float('nan')):.2f}")
    else:
        verdict, res = "INCONCLUSIVE", "no cells"
    cl.append(dict(id="C3", kind="OUT-OF-SAMPLE", statement="theta_min(eps) from Q >= -eps and argmax_theta eta(theta), both predicted from probing primitives, lie inside the realized id_test bootstrap intervals", result=res, verdict=verdict))
    mlr_rows = {k: v for k, v in R["mlr"].items() if k.endswith("|probing")}
    by_v = {}
    for k, v in mlr_rows.items():
        by_v.setdefault(k.split("|")[0], []).append(v)
    parts = []
    for v, lst in by_v.items():
        parts.append(f"{v}: MLR holds in {sum(x['mlr_holds'] for x in lst)}/{len(lst)} primaries (LR violations {[x['lr_violations'] for x in lst]}, ROC concavity violations {[x['roc_concavity_violations'] for x in lst]})")
    ints = R["thetamin_summary"]["interval_share_real_by_verifier"]
    res = "; ".join(parts) + "; realized admissible set {theta: Q >= -eps} is an interval in " + ", ".join(f"{v} {s:.2f}" for v, s in ints.items()) + " of (primary x basis x eps x source) cells"
    # implication tested: MLR holds (probing) => realized admissible set is an interval; cells whose verifier/primary has MLR
    mlr_ok_keys = {k.split("|")[0] + "|" + k.split("|")[1] for k, v in mlr_rows.items() if v["mlr_holds"]}
    impl_cells = [c for c in R["thetamin_cells"] if f"{c['verifier']}|{c['primary']}" in mlr_ok_keys]
    impl_share = float(np.mean([c["admissible_is_interval"] for c in impl_cells])) if impl_cells else float("nan")
    res += f"; implication MLR => interval: realized admissible set is an interval in {impl_share:.3f} of the {len(impl_cells)} cells whose (verifier, primary) has MLR on probing"
    verdict4 = "PASS" if (impl_cells and impl_share >= 0.9) else ("INCONCLUSIVE" if not impl_cells else "FAIL")
    cl.append(dict(id="C4", kind="DIAGNOSTIC", statement="Monotone likelihood ratio of Sigma per verifier (diagnostic) and its implication: under MLR the admissible thresholds {Q >= -eps} form an interval [theta_min, inf)", result=res, verdict=verdict4))
    ci_rows = {k: v for k, v in R["cond_indep"].items() if k.endswith("|probing")}
    n_rej = sum(1 for v in ci_rows.values() for s in ("given_correct", "given_incorrect") if np.isfinite(v[s].get("p_perm", float("nan"))) and v[s]["p_perm"] < 0.05)
    n_tests = sum(1 for v in ci_rows.values() for s in ("given_correct", "given_incorrect") if np.isfinite(v[s].get("p_perm", float("nan"))))
    def _w_ok(vv: Dict[str, Any]) -> bool:
        return np.isfinite(vv.get("p_perm_within_source", float("nan"))) and not vv.get("within_source_degenerate", False)
    n_rej_w = sum(1 for v in ci_rows.values() for s in ("given_correct", "given_incorrect") if _w_ok(v[s]) and v[s]["p_perm_within_source"] < 0.05)
    n_tests_w = sum(1 for v in ci_rows.values() for s in ("given_correct", "given_incorrect") if _w_ok(v[s]))
    n_degen_w = sum(1 for v in ci_rows.values() for s in ("given_correct", "given_incorrect") if np.isfinite(v[s].get("p_perm_within_source", float("nan"))) and v[s].get("within_source_degenerate", False))
    rng_ = [f"{k.split('|')[0]}/{k.split('|')[1]}: Y=1 rho {v['given_correct']['rho']:+.2f} (p {v['given_correct']['p_perm']:.3f}; within-source {v['given_correct'].get('rho_within_source', float('nan')):+.2f}, p {v['given_correct'].get('p_perm_within_source', float('nan')):.3f}) | Y=0 rho {v['given_incorrect']['rho']:+.2f} (p {v['given_incorrect']['p_perm']:.3f}; within-source {v['given_incorrect'].get('rho_within_source', float('nan')):+.2f}, p {v['given_incorrect'].get('p_perm_within_source', float('nan')):.3f}); beta1(theta) {v['given_correct']['beta_theta_range'][0]:.2f}-{v['given_correct']['beta_theta_range'][1]:.2f}, beta0(theta) {v['given_incorrect']['beta_theta_range'][0]:.2f}-{v['given_incorrect']['beta_theta_range'][1]:.2f}" for k, v in ci_rows.items() if np.isfinite(v['given_correct'].get('rho', float('nan'))) or np.isfinite(v['given_incorrect'].get('rho', float('nan')))]
    cl.append(dict(id="C5", kind="DIAGNOSTIC", statement="Sigma is conditionally independent of Y_B given the primary's correctness (no trend of P(Y_B=1 | Y, Sigma-bin)); also tested within source (task difficulty confounder)",
                   result=f"pooled: {n_rej}/{n_tests} stratum tests reject independence at 5% (permutation Spearman trend, probing); within-source (Y_B permuted within source): {n_rej_w}/{n_tests_w} reject ({n_degen_w} further tests degenerate: fewer than 10 informative points, i.e. the score hardly varies within a source in that stratum); " + "; ".join(rng_),
                   verdict=("INCONCLUSIVE" if n_tests == 0 else ("PASS" if n_rej == 0 else "FAIL"))))
    c6 = R["c6"]
    parts = []
    wins_or, wins_ref, wins_ref_ci, wins_aa, n6 = 0, 0, 0, 0, 0
    sign_ok, sign_dec, cov = 0, 0, 0
    for key, e in c6.items():
        row = e["eps"].get("0.02", {}).get("V1")
        if not row:
            continue
        n6 += 1
        cov += row["r_pred_covered_oos"]
        if row["decided_real_oos"]:
            sign_dec += 1
            sign_ok += row["floor_sign_oos"] == row["sign_real_oos"]
        wins_or += row["d_vs_oracle_is"][0] >= 0
        wins_ref += row["d_vs_ref_is"][0] >= 0
        wins_ref_ci += row["d_vs_ref_is"][1] > 0
        wins_aa += row["d_vs_acceptall_is"][0] >= 0
        eb = R["eta_B"][key.split("|")[1]][0]
        v3 = e["eps"].get("0.02", {}).get("V3")
        parts.append(f"{key}: V1 theta*={row['theta_is']:g} (TPR {row['TPR_is']:.2f}, FPR {row['FPR_is']:.2f}) r={row['r_is'][0]:.2f} [{row['r_is'][1]:.2f},{row['r_is'][2]:.2f}] vs oracle theta=1 {e['oracle_theta1_r'][0]:.2f}, "
                     f"accept-all {e['accept_all_r'][0]:.2f} (Q {e['eps']['0.02']['oracle']['Q_is'][0] if 'oracle' in e['eps']['0.02'] else float('nan'):+.3f} / floor-violating), reference 1; V1 - oracle {row['d_vs_oracle_is'][0] / eb:+.2f} eta_B [{row['d_vs_oracle_is'][1] / eb:+.2f},{row['d_vs_oracle_is'][2] / eb:+.2f}], "
                     f"V1 - reference {row['d_vs_ref_is'][0] / eb:+.2f} eta_B [{row['d_vs_ref_is'][1] / eb:+.2f},{row['d_vs_ref_is'][2] / eb:+.2f}]" + (f"; V3 theta*={v3['theta_is']:g} r={v3['r_is'][0]:.2f} [{v3['r_is'][1]:.2f},{v3['r_is'][2]:.2f}]" if v3 else ""))
    verdict6 = "NOT-INSTANTIABLE" if not n6 else ("PASS" if (sign_dec and sign_ok >= 0.9 * sign_dec) else ("INCONCLUSIVE" if not sign_dec else "FAIL"))
    cl.append(dict(id="C6", kind="IN-SAMPLE",
                   statement="A cheap imperfect verifier (V1 visible tests) at its best floor-respecting theta (eps = 0.02, chosen on id_test: IN-SAMPLE) vs the oracle cascade (theta = 1 on the released score), vs the reference (always-Opus, no verifier) and vs accept-all (primary alone, floor-violating); in addition the operating-point floor predicts V1's sign vs the reference at the theta chosen from probing (OUT-OF-SAMPLE sub-test, verdict criterion)",
                   result=f"over {n6} (primary x basis) cells, V1 at its in-sample floor-respecting theta beats the oracle cascade in {wins_or}, the reference in {wins_ref} ({wins_ref_ci} with CI excluding 0) and accept-all in {wins_aa}; at the theta chosen from probing the floor pi0 TPR b >= vbar a predicts the realized sign vs the reference in {sign_ok}/{sign_dec} CI-decided cells (verdict criterion); the constant-b model-based eta/eta_B lies inside the realized CI in {cov}/{n6} (reported only: the model ignores the selection effect cbar_h != b, vbar_h != vbar and is expected to over-predict, cf. the paper's instance and X1); " + "; ".join(parts), verdict=verdict6))
    if calib:
        best = calib.get("best")
        cfg = calib["configs"].get(best, {}) if best else {}
        cl.append(dict(id="J", kind="LIVE", statement="Judge calibration (Addendum): best AUROC over the 5 configurations on 60 items; AUROC >= 0.75 required to use the judge without a flag",
                       result=f"best {best} ({cfg.get('model')}, thinking={cfg.get('thinking')}): AUROC {cfg.get('auroc', float('nan')):.3f} {cfg.get('auroc_ci')} accuracy {cfg.get('accuracy', float('nan')):.3f}; " + ", ".join(f"{k}: {v.get('auroc', float('nan')):.3f}" for k, v in calib['configs'].items()),
                       verdict="PASS" if not calib.get("flag_low_auroc", True) else "FAIL"))  # FAIL = flag: best AUROC < 0.75, judge used anyway as instructed
    return cl


def audit(D: Dict[str, Any], R: Dict[str, Any], calib: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Independent recomputation of three headline numbers with plain Python (no numpy)."""
    out: Dict[str, Any] = {}
    Tt, P = D["id_test"], D["probing"]
    # (1) eta/eta_B for V2, Qwen3-Max, theta = 1.0, dollars, id_test point estimate
    m, v, th = "Qwen3-Max", "V2", 1.0
    sig = Tt["SIG"][v][m].tolist()
    yp, yb, cp, cb = Tt["YP"][m].tolist(), Tt["YB"].tolist(), Tt["CP"][m]["usd"].tolist(), Tt["CB"]["usd"].tolist()
    V = D_ = VB = CB_ = 0.0
    for s, a_, b_, c_, d_ in zip(sig, yp, yb, cp, cb):
        h = 1.0 if s >= th - 1e-12 else 0.0
        V += h * a_ + (1 - h) * b_
        D_ += c_ + (1 - h) * d_
        VB += b_
        CB_ += d_
    r_audit = (V / D_) / (VB / CB_)
    r_main = R["cells"][f"{v}|{m}|usd"]["real_r"][THETAS.index(th)][0]
    out["eta_ratio_V2_Qwen3Max_theta1_usd"] = dict(audit=r_audit, main=r_main, match=bool(abs(r_audit - r_main) < 1e-9))
    # (2) TPR/FPR of V1, Qwen3-Max, theta = 0.5 on probing
    sig = P["SIG"]["V1"][m].tolist()
    yp = P["YP"][m].tolist()
    n1 = sum(1 for y in yp if y == 1)
    n0 = len(yp) - n1
    tpr = sum(1 for s, y in zip(sig, yp) if y == 1 and s >= 0.5 - 1e-12) / n1
    fpr = sum(1 for s, y in zip(sig, yp) if y == 0 and s >= 0.5 - 1e-12) / n0
    c = R["cells"][f"V1|{m}|usd"]
    j = THETAS.index(0.5)
    out["TPR_FPR_V1_Qwen3Max_theta05_probing"] = dict(audit=[tpr, fpr], main=[c["TPR_probing"][j], c["FPR_probing"][j]], match=bool(abs(tpr - c["TPR_probing"][j]) < 1e-9 and abs(fpr - c["FPR_probing"][j]) < 1e-9))
    # (3) best judge AUROC by O(n^2) pairwise comparison
    if calib and calib.get("best"):
        p = calib["configs"][calib["best"]]["p"]
        y = calib["truth"]
        pos = [pp for pp, yy in zip(p, y) if yy == 1]
        neg = [pp for pp, yy in zip(p, y) if yy == 0]
        s = 0.0
        for a_ in pos:
            for b_ in neg:
                s += 1.0 if a_ > b_ else (0.5 if a_ == b_ else 0.0)
        au = s / (len(pos) * len(neg))
        out["judge_best_auroc"] = dict(audit=au, main=calib["configs"][calib["best"]]["auroc"], match=bool(abs(au - calib["configs"][calib["best"]]["auroc"]) < 1e-9))
    # (4) accounting identity eta - eta_B = (Q + eta_B S)/(cbar - S) on every realized cell
    resid = 0.0
    for c in R["cells"].values():
        for j in range(len(c["theta"])):
            d = c["real_d_eta"][j][0]
            Q, S, cbar = c["real_Q"][j][0], c["real_S"][j][0], c["real_ED"][j][0] + c["real_S"][j][0]
            eta_B = (c["real_EV"][j][0] - Q) / cbar
            resid = max(resid, abs(d - (Q + eta_B * S) / (cbar - S)) / max(abs(d), 1e-12))
    out["identity_max_rel_resid"] = resid
    return out


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]" if all(isinstance(v, (int, float)) and np.isfinite(v) for v in x) else (f"{x[0]:.{nd}f}" if isinstance(x[0], (int, float)) and np.isfinite(x[0]) else "-")
    if isinstance(x, float):
        return f"{x:.{nd}f}" if np.isfinite(x) else "-"
    return str(x)


def write_reports(res: Dict[str, Any], D: Dict[str, Any], R: Dict[str, Any], calib: Optional[Dict[str, Any]], jd: Optional[Dict[str, Any]], samples: Dict[str, Any]) -> None:
    NG = R["n_grid"]
    L: List[str] = []
    L.append("# X8_operating_point -- the verifier's operating point (app:operating-point) on CodeRouterBench EXEC-CLEAN tasks, recorded responses, LIVE judge")
    L.append("")
    if res.get("interim"):
        L.append(f"**{res['interim_note']}** (judge rows available: {res.get('judge_rows_n', 0)}). This file is overwritten by the final run.")
        L.append("")
    L.append(f"Generated by `run.py` (seed {SEED}; paired task-resampling bootstrap {B_BOOT} on id_test and {B_BOOT} on probing; runtime {res['runtime_seconds']:.0f} s for this run, "
             f"sandbox and relay results cached in `cache/` and `common/cache/`). Files: `run.py`, `results.json`, `README.md`, `summary_for_paper.md`, `figures/fig1_eta_theta.pdf`, `figures/fig2_Q_theta.pdf`, "
             f"`figures/fig3_roc_iso_efficiency.pdf`, `figures/fig4_judge_calibration.pdf`, `figures/fig5_conditional_independence.pdf`, `figures/fig6_likelihood_ratio.pdf`, `cache/samples.json`, `cache/sandbox.json`, `cache/calibration.json`, `cache/judge.json`, `rows_id_test.csv`, `rows_probing.csv`.")
    L.append("")
    L.append("## Figures")
    L.append("* `fig1_eta_theta.pdf` (C1, C2): realized eta(theta)/eta_B on id_test per verifier and primary (solid, shaded 95% paired-bootstrap CI; dollars on a log axis, seconds below) with the model-based prediction from probing primitives (dotted; constant b, conditional independence). The dashed line is the reference (eta_B); theta = 0 is accept-all.")
    L.append("* `fig2_Q_theta.pdf` (C3): realized Q(theta) with CIs and the model-based prediction (dotted); horizontal dashed lines at -eps for eps in {0.01, 0.02, 0.05}; the open markers (v, ^, s) sit at the predicted theta_min(eps) for those eps.")
    L.append("* `fig3_roc_iso_efficiency.pdf` (C6, ROC comparison): id_test ROC points of the 21 thresholds per verifier; grey lines are model-based iso-efficiency lines eta = {1, 2, 4} x eta_B in the (FPR, TPR) plane (dollars), the red dashed line is the floor Q >= -0.02 (eq:app-operating-fpr): admissible operating points lie above/left of it.")
    L.append("* `fig4_judge_calibration.pdf` (J): ROC of the five judge configurations on the 60 calibration items.")
    L.append("* `fig5_conditional_independence.pdf` (C5): P(Y_B = 1 | Y, Sigma-bin) with Wilson intervals for V1, V3 (if judged) and V2, Qwen3-Max, probing; the text gives the pooled permutation p-values (given correct / given incorrect).")
    L.append("* `fig6_likelihood_ratio.pdf` (C4): likelihood ratio by score bin (probing), one panel per verifier and one series per primary (open markers = bins with n < 15, excluded from the monotonicity count; up/down triangles = infinite / zero ratio at the axis caps); a nondecreasing sequence is the MLR property under which the admissible thresholds form an interval.")
    L.append("")
    L.append("## Paper objects tested")
    L.append("* `H = 1{Sigma >= theta}`, `p = pi0 TPR + (1-pi0) FPR`, `p v_h = pi0 TPR` (main_body 'The verifier's operating point', app:operating-point).")
    L.append("* Floor eq:floor in operating-point form: `pi0 TPR(theta) b >= vbar a` -- false accepts do not enter (homogeneous case beta1 = beta0 = vbar). General affine form eq:app-operating-gain: `Q + eta_B S = pi0(1-beta1+vbar) TPR + (1-pi0)(vbar-beta0) FPR - vbar a/b`.")
    L.append("* Accepting everything (theta = 0) is efficiency-optimal over all operating points iff `a beta0 <= pi0 b` (general, constant b, Sigma _||_ Y_B | Y); homogeneous form `a/b <= pi0/vbar`; sufficient not necessary (uninformative score: `a/(a+b) <= pi0/vbar`).")
    L.append("* Quality floor `Q = pi0 (1-beta1) TPR - (1-pi0) beta0 FPR >= -eps` (eq:app-operating-Q) bounds FPR from above (eq:app-operating-fpr) and, for a score with a monotone likelihood ratio, theta from below: theta_min(eps).")
    L.append("* Premise `Sigma _||_ Y_B | Y` (conditional independence) tested directly; `e = (1-pi0) beta0 FPR`.")
    L.append("")
    L.append("## Setup")
    pc = samples["pool_counts"]
    L.append(f"**Data.** `$ACROUTER_DATA/coderouterbench/id_results_long.csv` (scores, `cost_usd`, `latency_ms`) and the recorded responses `matrices/phase1_acrouter_v2/response_matrix.json`. "
             f"EXEC-CLEAN sources (Addendum): humaneval (dim1), he dim4 completion, mbpp dim1 + dim4 (testgen excluded), hex python (dim7), bcb with released scores not all-zero and all libs importable (runnable). "
             f"Runnable pool: probing {pc['probing']} = {sum(pc['probing'].values())} tasks, id_test {pc['id_test']} = {sum(pc['id_test'].values())} tasks. "
             f"Samples (seed {SEED}, stratified by source, largest-remainder allocation): probing {samples['n_probing']} tasks {dict(Counter(t.split('_')[1] for t in samples['probing']))}; id_test {samples['n_id_test']} tasks {dict(Counter(t.split('_')[1] for t in samples['id_test']))}. "
             f"**Deviation:** {samples['note'] or 'none'}.")
    L.append(f"**Released-grader caveat (coordinator, Addendum 2).** The released CodeRouterBench grader lacked plotting/ML libraries: bcb tasks importing matplotlib/seaborn/sklearn/scipy are scored 0 for all 8 models. All-zero bcb tasks are treated as UNGRADED and excluded from the EXEC-CLEAN pool here "
             f"({res['data_checks']['bcb_allzero_excluded']} bcb tasks excluded: probing {res['data_checks']['bcb_allzero_excluded_by_split'].get('probing', 0)}, id_test {res['data_checks']['bcb_allzero_excluded_by_split'].get('id_test', 0)}; {res['data_checks']['bcb_kept']} bcb tasks kept), so pi0, TPR and FPR are computed on graded tasks only. "
             "No lcb task is in the EXEC-CLEAN pool, so the 28 empty-but-scored lcb responses do not enter; empty recorded responses of the primaries in the sample are treated as unverifiable (dropped), see Sensitivities.")
    L.append(f"**Truth.** `Y = 1{{released score >= 1}}` for the primary and `Y_B = 1{{Opus score >= 1}}` for the reference (values of a correct/incorrect output 1/0 as in app:operating-point); the `>= 0.5` variant is reported in `results.json:truth_ge_05` (fractional scores are {res['data_checks']['fractional_score_share']:.2%} of the sampled rows, so it is numerically indistinguishable). "
             f"Reference = always claude-opus-4-6 (recorded). Primaries Qwen3-Max, kimi-k2.5, qwen3.5-plus; V3 on Qwen3-Max and kimi-k2.5 only.")
    L.append("**Verifiers (Sigma) on the recorded responses.** V1 = fraction of prompt-visible examples passed, extracted exactly as ACRouter's `run_visible_tests` (doctest `>>> call` / next line, else `assert call == expected`; at most 5; `_result = call; _expected = expected; assert _result == _expected` per example; 0.5 when the prompt has no example; one sandbox process per response through `common.sandbox.run_python_tests`, extraction by `common.sandbox.extract_code` plus the prompt's imports). "
             "V2 = `common.sandbox.verify_response` pass fraction (hidden tests; the released score's own mechanism). V3 = LLM judge `p_correct` (`common.judge`, configuration selected by the calibration below). Oracle control = the released score itself (X1's `theta` on the score). "
             f"Threshold grid: 21 points theta in {{0, 0.05, ..., 1}}; a 22nd 'reject-all' point (H = 0, pays a + b) is used only for theta_min(eps).")
    L.append("**Costs.** Dollars: csv `cost_usd` (a = primary, b = Opus); V1/V2 add no dollars; V3 adds judge tokens x the primary's blended $/token on the sampled tasks (stated convention; `judge_price` sensitivities 'free' and 'opus' in `results.json:sensitivity_judge_price`). "
             f"Seconds: `latency_ms` sequential (primary + verifier + Opus on a miss); V1/V2 add their sandbox wall time measured here (Apple M4), V3 adds the judge call latency. Judge tokens are also reported in ledger units ({D['judge_units_per_token']:.5g} units/token, the rate stored in `common/budget.json` for the judge model).")
    L.append(f"**Statistics.** eta = sum V / sum D (ratio of sums). Primitives (pi0, TPR, FPR, beta1, beta0, vbar, b, a, cbar_h, vbar_h) on the probing sample; realized eta(theta)/eta_B, Q(theta), e(theta) on id_test with {B_BOOT} paired multinomial task resamples (same resamples for every verifier, primary and theta; 95% percentile CIs). "
             "Predictions use probing point estimates; 'fit CIs' are probing resamples. Sign agreement is computed only on cells whose realized CI excludes 0 and is reported next to naive baselines. Distinct operating points: theta values giving the same acceptance vector on id_test are counted once. "
             "Labels: OUT-OF-SAMPLE = primitives from probing, realization on id_test; IN-SAMPLE = theta chosen on id_test; DIAGNOSTIC = hypothesis check; LIVE = relay calls.")
    L.append("")
    # judge calibration
    L.append("## Judge calibration (Addendum; LIVE)")
    if calib:
        L.append(f"60 recorded responses (30 released-correct / 30 incorrect) from EXEC-CLEAN probing tasks outside the 500-task sample, all 8 models {dict(Counter(i['model'] for i in calib['items']))}, sources {dict(Counter(i['task_id'].split('_')[1] for i in calib['items']))}, seed {SEED}+1. "
                 f"AUROC of `p_correct` against the released truth (rank-based, ties 1/2; bootstrap CI over items).")
        L.append("")
        L.append("| config | model | thinking | max_tokens | AUROC [CI] | accuracy@0.5 | Brier | TPR/FPR@0.5 | parse fail | errors | tokens/call (prompt+completion) | latency s | ledger units (rate) | finish reasons |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cid, model, thinking, mt in JUDGE_CONFIGS:
            c = calib["configs"].get(cid)
            if not c:
                L.append(f"| {cid} | {model} | {thinking} | {mt} | not run | | | | | | | | | |")
                continue
            if c.get("fatal"):
                L.append(f"| {cid} | {model} | {thinking} | {mt} | FATAL: {c['fatal'][:80]} | | | | | | | | | |")
                continue
            L.append(f"| {cid}{' **(best)**' if cid == calib.get('best') else ''} | {model} | {thinking} | {mt} | {c['auroc']:.3f} [{c['auroc_ci'][0]:.2f}, {c['auroc_ci'][1]:.2f}] | {c['accuracy']:.3f} | {c['brier']:.3f} | {c['tpr_at_half']:.2f}/{c['fpr_at_half']:.2f} | {c['n_parse_fail']} | {c['n_error']} | "
                     f"{c['mean_total_tokens']:.0f} ({c['mean_prompt_tokens']:.0f}+{c['mean_completion_tokens']:.0f}) | {c['mean_latency_s']:.1f} | {c['units_est']:.0f} ({c['units_per_token']:.4g}/tok) | {c['finish_reasons']} |")
        L.append("")
        L.append(f"Selected: **{calib.get('best')}** (AUROC {calib.get('best_auroc', float('nan')):.3f}); flag AUROC < 0.75: **{calib.get('flag_low_auroc')}**.")
        hist = calib.get("configs_history") or {}
        if hist:
            L.append("Selection-time values (first pass, before the relay's intermittent HTTP 400 'Incorrect API key' flake was made retryable in common/packy.py; error items were imputed p_correct = 0.5 in that pass and re-judged afterwards): " +
                     "; ".join(f"{cid}: AUROC {h[-1]['auroc']:.3f}, accuracy {h[-1]['accuracy']:.3f}, {h[-1]['n_error']} relay-error items" for cid, h in hist.items()) + ". The selection (best AUROC) is unchanged by the re-judging.")
        if jd:
            pl = jd["plan"]
            L.append(f"Main judging plan (mode `{pl.get('mode')}`, primaries {pl.get('primaries')}): per-call ledger estimate {pl['per_call_units_est']:.1f} units ({pl['units_per_token_used']:.4g} units/token x mean calibration tokens x 1.1), X8 spent before planning {pl['spent_before']:.0f} units, "
                     f"calls affordable within the 80,000 allocation {pl['n_calls_affordable_at_allocation']} vs full design {pl.get('n_calls_full')}; planned {pl['n_calls_planned']} calls = {len(pl.get('primaries', []))} primaries x (id_test {pl['k_id_test']} + probing {pl['k_probing']} tasks; stratified priority prefixes); "
                     f"{pl['note']}; ledger estimate {pl['ledger_units_est_planned']:.0f} units; rows judged {len(jd['rows'])}: {sum(1 for r in jd['rows'].values() if r.get('error'))} with relay errors after the retry passes (intermittent HTTP 400 'Incorrect API key' flake of the relay) -> SKIPPED as unverifiable (Sigma = NaN, task dropped for that primary in V3 cells), {sum(1 for r in jd['rows'].values() if not r.get('parse_ok') and not r.get('error'))} unparseable judge replies -> IMPUTED p_correct = 0.5 (the judge module's convention); fatal: {jd.get('fatal')}.")
    else:
        L.append("Not run (dry mode).")
    L.append("")
    # primitives table
    L.append("## Results")
    L.append("### (1) Primitives on probing (point [fit CI]) and realized efficiency on id_test at three thresholds (dollars; seconds in results.json)")
    L.append("")
    L.append("| verifier | primary | n_p / n_t | pi0 | vbar | beta1 | beta0 | a/b | AUROC(Sigma;Y) probing / id_test | a beta0 <= pi0 b | a/b <= pi0/vbar | r(0) | r(0.5) | r(1) | Q(0) | Q(0.5) | Q(1) | e(0.5) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in VERIFIERS:
        for m in PRIMARIES:
            c = R["cells"].get(f"{v}|{m}|usd")
            if not c:
                continue
            j0, j5, j1 = 0, THETAS.index(0.5), THETAS.index(1.0)
            ro = R["roc"].get(f"{v}|{m}|probing", {})
            rt = R["roc"].get(f"{v}|{m}|id_test", {})
            L.append(f"| {v} | {m} | {c['n_probing']} / {c['n_id_test']} | {fmt(c['pi0'])} | {fmt(c['vbar'])} | {fmt(c['beta1'])} | {fmt(c['beta0'])} | {c['allaccept']['a_over_b'][0]:.4f} | {ro.get('auroc', float('nan')):.3f} / {rt.get('auroc', float('nan')):.3f} | "
                     f"{'yes' if c['allaccept']['pred_general'] else 'no'} | {'yes' if c['allaccept']['pred_hom'] else 'no'} | {fmt(c['real_r'][j0], 2)} | {fmt(c['real_r'][j5], 2)} | {fmt(c['real_r'][j1], 2)} | {fmt(c['real_Q'][j0])} | {fmt(c['real_Q'][j5])} | {fmt(c['real_Q'][j1])} | {fmt(c['real_e'][j5])} |")
    L.append("")
    L.append("### (2) Operating-point curves per verifier (dollars, pooled sources; TPR/FPR from probing, realized r = eta/eta_B and Q from id_test with CIs; predicted r and Q model-based from probing)")
    for v in VERIFIERS:
        for m in PRIMARIES:
            c = R["cells"].get(f"{v}|{m}|usd")
            if not c:
                continue
            L.append("")
            L.append(f"**{VERIFIER_LABEL[v]} / {m}** (n_probing {c['n_probing']}, n_id_test {c['n_id_test']}; eta_B id_test {fmt(R['eta_B']['usd'], 1)} per $)")
            L.append("")
            L.append("| theta | TPR_p | FPR_p | TPR_t | FPR_t | p_t | floor_hom sign | gain_gen sign | r pred | r realized [CI] | Q pred | Q realized [CI] | e realized | net gain $/task (avoided, attempt) |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for j, th in enumerate(c["theta"]):
                if j >= NG and th != THETA_REJECT:
                    continue
                L.append(f"| {'reject-all' if th > 1 else f'{th:.2f}'} | {c['TPR_probing'][j]:.3f} | {c['FPR_probing'][j]:.3f} | {c['TPR_id_test'][j]:.3f} | {c['FPR_id_test'][j]:.3f} | {c['real_p'][j][0]:.3f} | {'+' if c['floor_hom'][j][0] >= 0 else '-'} | {'+' if c['gain_gen'][j][0] >= 0 else '-'} | "
                         f"{c['r_pred'][j][0]:.2f} | {fmt(c['real_r'][j], 2)} | {c['Q_pred'][j][0]:+.3f} | {fmt(c['real_Q'][j])} | {c['real_e'][j][0]:.3f} | {c['real_gain'][j][0] / max(R['eta_B']['usd'][0], 1e-12) * 1000:+.2f} m$ ({c['real_avoided'][j][0] * 1000:.2f}, {c['real_attempt'][j][0] * 1000:.2f}) |")
    L.append("")
    L.append("### (3) C1: sign of eta(theta) - eta_B, predicted from probing (distinct operating points, reject-all excluded; realized CI-decided cells only)")
    L.append("")
    L.append("| cell set | cells | CI-decided | floor without FPR | with FPR as p b (naive) | with FPR penalized | general affine form (beta's) | plug-in probing sign | always + | majority | per-basis majority | both-decided (hom) | disagreements naive / penal / gen (floor right) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in [("all", R["sign_summary"]["all"]), ("dollars", R["sign_summary"]["by_basis"]["usd"]), ("seconds", R["sign_summary"]["by_basis"]["sec"]), ("pooled sources only", R["sign_summary"]["pooled_only"]), ("per-source cells only", R["sign_summary"]["per_source_only"])] + [(v, R["sign_summary"]["by_verifier"].get(v, {})) for v in VERIFIERS]:
        if not s or "n_cells" not in s:
            continue
        L.append(f"| {name} | {s['n_cells']} | {s.get('n_decided', 0)} | {fmt(s.get('agree_pred_hom'))} | {fmt(s.get('agree_pred_naive'))} | {fmt(s.get('agree_pred_penal'))} | {fmt(s.get('agree_pred_gen'))} | {fmt(s.get('agree_pred_plug'))} | {fmt(s.get('baseline_always_plus'))} | {fmt(s.get('baseline_majority'))} | {fmt(s.get('baseline_per_basis_majority'))} | {fmt(s.get('agree_hom_both_decided'))} (n={s.get('n_both_decided_hom', 0)}) | "
                 f"{s.get('n_disagree_pred_naive', 0)} ({fmt(s.get('hom_right_when_disagree_pred_naive'), 2)}) / {s.get('n_disagree_pred_penal', 0)} ({fmt(s.get('hom_right_when_disagree_pred_penal'), 2)}) / {s.get('n_disagree_pred_gen', 0)} ({fmt(s.get('hom_right_when_disagree_pred_gen'), 2)}) |")
    L.append("")
    L.append("### (4) C2: accepting everything (theta = 0) -- predicted by a beta0 <= pi0 b (general) / a vbar <= pi0 b (homogeneous), realized = theta 0 attains max_theta eta on id_test")
    L.append("")
    L.append("| cell set | cells | pred true (general) | sufficiency: realized-optimal given predicted | violations | agreement general | agreement homogeneous | share realized-optimal | majority baseline | CI-decided cells: agreement / sufficiency |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in [("all", R["allacc_summary"]["all"]), ("dollars", R["allacc_summary"]["by_basis"]["usd"]), ("seconds", R["allacc_summary"]["by_basis"]["sec"]), ("pooled only", R["allacc_summary"]["pooled_only"])]:
        L.append(f"| {name} | {s['n_cells']} | {s.get('all_n_pred_true_pred_general', 0)} | {fmt(s.get('all_sufficiency_pred_general'))} | {s.get('all_n_violations_pred_general', 0)} | {fmt(s.get('all_agree_pred_general'))} | {fmt(s.get('all_agree_pred_hom'))} | {fmt(s.get('all_share_real_opt'))} | {fmt(s.get('all_baseline_majority'))} | {s.get('n_decided', 0)}: {fmt(s.get('decided_agree_pred_general'))} / {fmt(s.get('decided_sufficiency_pred_general'))} |")
    L.append("")
    viol = [c for c in R["allacc_cells"] if c["pred_general"] and not c["real_opt"]]
    L.append("Violations (condition predicted true on probing, theta = 0 not realized-optimal on id_test): " + ("; ".join(f"{c['verifier']}/{c['primary']}/{c['basis']}/{c['source']} (n_id_test {R['per_source'][f'{c['verifier']}|{c['primary']}|{c['basis']}|{c['source']}']['n_id_test'] if c['source'] != 'all' else R['cells'][f'{c['verifier']}|{c['primary']}|{c['basis']}']['n_id_test']}): probing pi0 {c['pi0_over_beta0'][0] * 0 + R[('per_source' if c['source'] != 'all' else 'cells')][f'{c['verifier']}|{c['primary']}|{c['basis']}' + (f'|{c['source']}' if c['source'] != 'all' else '')]['pi0'][0]:.3f}, a/b {c['a_over_b'][0]:.3f} <= pi0/beta0 {c['pi0_over_beta0'][0]:.3f}; id_test pi0 {c['id_test']['pi0']:.3f}, beta0 {c['id_test']['beta0']:.3f}, a/b {c['id_test']['a_over_b']:.3f}, condition evaluated on id_test: {c['id_test']['pred_general_id_test']}; eta(0) - eta(theta={c['best_other_theta']:g}) = {fmt(c['diff_ci'], 3)}" for c in viol) or "none") + ".")
    L.append("")
    L.append("Per pooled cell (a/b, pi0/beta0, pi0/vbar from probing; realized argmax and the paired CI of eta(0) - eta(theta_best>0), theta_best fixed at the point-estimate argmax):")
    L.append("")
    L.append("| verifier | primary | basis | a/b | pi0/beta0 | pi0/vbar | pred general | pred hom | realized argmax theta | theta=0 optimal | eta(0) - best other [CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in R["allacc_cells"]:
        if c["source"] != "all":
            continue
        L.append(f"| {c['verifier']} | {c['primary']} | {c['basis']} | {c['a_over_b'][0]:.3f} | {c['pi0_over_beta0'][0]:.3f} | {c['pi0_over_vbar'][0]:.3f} | {c['pred_general']} | {c['pred_hom']} | {R['cells'][f'{c['verifier']}|{c['primary']}|{c['basis']}']['argmax']['real']:g} | {c['real_opt']} | {fmt(c['diff_ci'], 4)} |")
    L.append("")
    L.append("### (5) C3: theta_min(eps) and argmax theta -- predicted (model-based from probing) vs realized id_test bootstrap intervals")
    L.append("")
    L.append("| cell set | cells | theta_min inside CI (model) | (plug-in) | naive theta_min = 0 inside | exact match | mean abs dev | argmax inside (model) | (plug-in) | naive 0 inside | exact |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, t, a in [("all", R["thetamin_summary"]["all"], R["argmax_summary"]["all"]), ("pooled only", R["thetamin_summary"]["pooled_only"], R["argmax_summary"]["pooled_only"])] + \
                      [(f"eps={e}", R["thetamin_summary"]["by_eps"][str(e)], {}) for e in EPS_GRID] + [(v, R["thetamin_summary"]["by_verifier"].get(v, {}), R["argmax_summary"]["by_verifier"].get(v, {})) for v in VERIFIERS]:
        if not t or not t.get("n"):
            continue
        L.append(f"| {name} | {t['n']} | {fmt(t.get('inside_model'))} | {fmt(t.get('inside_plug'))} | {fmt(t.get('naive0_inside'))} | {fmt(t.get('exact_match_model'))} | {fmt(t.get('mean_abs_dev_model'))} | {fmt(a.get('inside_model')) if a else '-'} | {fmt(a.get('inside_plug')) if a else '-'} | {fmt(a.get('naive0_inside')) if a else '-'} | {fmt(a.get('exact_match_model')) if a else '-'} |")
    L.append("")
    L.append("Pooled cells (dollars): theta_min(eps) predicted vs realized [95% bootstrap interval], and argmax theta:")
    L.append("")
    L.append("| verifier | primary | " + " | ".join(f"theta_min({e}) pred / real [CI]" for e in EPS_GRID) + " | argmax pred (model / plug-in) / real [CI] |")
    L.append("|---|---|" + "---|" * len(EPS_GRID) + "---|")
    for v in VERIFIERS:
        for m in PRIMARIES:
            c = R["cells"].get(f"{v}|{m}|usd")
            if not c:
                continue
            cells_e = []
            for e in EPS_GRID:
                t = c["theta_min"][str(e)]
                f_ = lambda x: "reject" if x > 1 else f"{x:g}"
                cells_e.append(f"{f_(t['pred_model'])} / {f_(t['real'])} [{f_(t['real_ci'][0])}, {f_(t['real_ci'][1])}]")
            a = c["argmax"]
            L.append(f"| {v} | {m} | " + " | ".join(cells_e) + f" | {a['pred_model']:g} / {a['pred_plug']:g} / {a['real']:g} [{a['real_ci'][0]:g}, {a['real_ci'][1]:g}] |")
    L.append("")
    L.append("### (6) C4: monotone likelihood ratio per verifier (probing; bins = the distinct Sigma values when there are at most 8, else {0}, (0,.25], (.25,.5], (.5,.75], (.75,1), {1}; LR = P(bin|Y=1)/P(bin|Y=0); bins with n < 15 ignored for the monotonicity count)")
    L.append("")
    L.append("| verifier | primary | n distinct Sigma | bins kept | LR by bin (n) | LR violations | ROC concavity violations | MLR holds |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, mm in R["mlr"].items():
        v, m, split = k.split("|")
        if split != "probing":
            continue
        lrs = ", ".join(f"{b['bin']}: {b['LR']:.2f} ({b['n']})" if np.isfinite(b["LR"]) else f"{b['bin']}: inf ({b['n']})" for b in mm["bins"])
        L.append(f"| {v} | {m} | {R['roc'][k]['n_distinct']} | {mm['n_bins_kept']} | {lrs} | {mm['lr_violations']} | {mm['roc_concavity_violations']} | {mm['mlr_holds']} |")
    L.append("")
    L.append(f"Realized admissible sets {{theta : Q(theta) >= -eps}} on id_test are intervals [theta_min, reject-all] in {R['thetamin_summary']['interval_share_real']:.3f} of the (verifier x primary x basis x eps x source) cells; by verifier: " + ", ".join(f"{v} {s:.3f}" for v, s in R["thetamin_summary"]["interval_share_real_by_verifier"].items()) + ".")
    L.append("")
    L.append("### (7) C5: conditional independence Sigma _||_ Y_B | Y (probing; Spearman trend of Y_B in Sigma within each correctness stratum, permutation p-value with 2000 permutations; beta1(theta) = P(Y_B = 1 | Y = 1, Sigma >= theta) range over theta with >= 20 accepted)")
    L.append("")
    L.append("| verifier | primary | split | n(Y=1) | rho | p | within-source rho (p) [strata; informative pts] | beta1(theta) range | n(Y=0) | rho | p | within-source rho (p) [strata; informative pts] | beta0(theta) range | P(Y_B=1; Y=1, bin) | P(Y_B=1; Y=0, bin) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, cc in R["cond_indep"].items():
        v, m, split = k.split("|")
        g1, g0 = cc["given_correct"], cc["given_incorrect"]
        t1 = ", ".join(f"{t['bin']}: {t['p_yb'][0]:.2f} ({t['n']})" for t in g1["table"] if t["n"] >= 5)
        t0 = ", ".join(f"{t['bin']}: {t['p_yb'][0]:.2f} ({t['n']})" for t in g0["table"] if t["n"] >= 5)
        def _ws(gg: Dict[str, Any]) -> str:
            sup = gg.get("within_source_support") or {}
            return f"{fmt(gg.get('rho_within_source'), 2)} ({fmt(gg.get('p_perm_within_source'))}) [{','.join(sup.get('strata', [])) or '-'}; {sup.get('n_informative', 0)}{' DEGENERATE' if gg.get('within_source_degenerate') else ''}]"
        L.append(f"| {v} | {m} | {split} | {g1['n']} | {fmt(g1.get('rho'), 2)} | {fmt(g1.get('p_perm'))} | {_ws(g1)} | {fmt(g1['beta_theta_range'][0], 2)}-{fmt(g1['beta_theta_range'][1], 2)} | {g0['n']} | {fmt(g0.get('rho'), 2)} | {fmt(g0.get('p_perm'))} | {_ws(g0)} | {fmt(g0['beta_theta_range'][0], 2)}-{fmt(g0['beta_theta_range'][1], 2)} | {t1} | {t0} |")
    L.append("")
    L.append("### (8) C6: cheap imperfect verifier at its floor-respecting theta vs oracle cascade vs no verifier (id_test; IN-SAMPLE theta chosen on id_test, OOS theta chosen from probing primitives; common task subset = judged tasks where V3 exists)")
    L.append("")
    L.append("| primary | basis | n | eps | verifier | theta* IS | r IS [CI] | Q IS | e IS | TPR/FPR IS | r - r_oracle(theta=1) IS [CI] | r - r_accept-all IS [CI] | eta - eta_B IS [CI] | theta* OOS | r OOS [CI] | r pred OOS (covered) | floor sign OOS / realized | Q OOS [CI] | eta - eta_oracle OOS [CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, e in R["c6"].items():
        m, basis = key.split("|")
        for eps in EPS_GRID:
            row = e["eps"][str(eps)]
            for v in e["verifiers"]:
                r_ = row[v]
                eb = R["eta_B"][basis][0]
                L.append(f"| {m} | {basis} | {e['n_id_test']} | {eps} | {v} | {('reject' if r_['theta_is'] > 1 else f'{r_['theta_is']:g}')} | {fmt(r_['r_is'], 2)} | {r_['Q_is'][0]:+.3f} | {r_['e_is'][0]:.3f} | {r_['TPR_is']:.2f}/{r_['FPR_is']:.2f} | "
                         f"{fmt([x / eb for x in r_['d_vs_oracle_is']], 2)} | {fmt([x / eb for x in r_['d_vs_acceptall_is']], 2)} | {fmt([x / eb for x in r_['d_vs_ref_is']], 2)} | {('reject' if r_['theta_oos'] > 1 else f'{r_['theta_oos']:g}')} | {fmt(r_['r_oos'], 2)} | {r_['r_pred_oos'][0]:.2f} ({'yes' if r_['r_pred_covered_oos'] else 'no'}) | {'+' if r_['floor_sign_oos'] >= 0 else '-'} / {'+' if r_['sign_real_oos'] >= 0 else '-'}{'' if r_['decided_real_oos'] else ' (undecided)'} | {fmt(r_['Q_oos'])} | {fmt([x / eb for x in r_['d_vs_oracle_oos']], 2)} |")
        L.append(f"| {m} | {basis} | {e['n_id_test']} | - | oracle theta=1 (X1 control) | 1 | {fmt(e['oracle_theta1_r'], 2)} | | | | | | | | | | | | |")
        L.append(f"| {m} | {basis} | {e['n_id_test']} | - | accept-all (no verifier) | 0 | {fmt(e['accept_all_r'], 2)} | | | | | | | | | | | | |")
    L.append("")
    L.append("Differences in the table are in units of eta_B (paired bootstrap). The oracle at theta = 1 has TPR = 1, FPR = 0 (up to the 0.8 partial-credit rows).")
    L.append("")
    # ROC table
    L.append("### (9) ROC comparison and efficiency at the floor-respecting theta (id_test, dollars, eps = 0.02)")
    L.append("")
    L.append("| verifier | primary | AUROC probing | AUROC id_test | n distinct Sigma | neutral share (V1) | empty responses (truth = 1) | verifier cost per task (s / tokens / units / $) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in VERIFIERS:
        for m in PRIMARIES:
            ro, rt = R["roc"].get(f"{v}|{m}|probing"), R["roc"].get(f"{v}|{m}|id_test")
            if not rt:
                continue
            cost = res["verifier_cost"].get(f"{v}|{m}", {})
            L.append(f"| {v} | {m} | {ro['auroc']:.3f} | {rt['auroc']:.3f} | {rt['n_distinct']} | {fmt(rt.get('share_neutral'), 2)} | {rt['empty_responses']} ({rt['empty_but_truth1']}) | {cost.get('mean_seconds', 0):.3f} s / {cost.get('mean_tokens', 0):.0f} / {cost.get('mean_units', 0):.3f} / {cost.get('mean_usd', 0):.5f} |")
    L.append("")
    L.append("### Sensitivities")
    sj = res.get("sensitivity_judge_price", {})
    if sj:
        L.append("Judge dollar price (V3 cells, dollars, pooled): r at theta = 0.5 and at the realized argmax under 'primary' (main), 'free' and 'opus' conventions: " + "; ".join(f"{k}: {v}" for k, v in sj.items()))
    t05 = res.get("truth_ge_05", {})
    if t05:
        L.append(f"Truth >= 0.5 variant: max |r(theta) difference| over all pooled cells and thresholds = {t05.get('max_abs_r_diff', float('nan')):.4f}; pi0 differences {t05.get('pi0_diff', {})}.")
    L.append(f"Empty recorded responses (data gap in response_matrix.json; treated as UNVERIFIABLE and dropped for that primary in the main analysis): {res.get('empty_responses')}.")
    ex = res.get("sensitivity_include_empty", {})
    if ex:
        L.append("Counting them as failed instead: " + "; ".join(f"{k}: {v}" for k, v in ex.items()))
    L.append("")
    L.append("## Claims")
    L.append("")
    L.append("| id | kind | statement | result | verdict |")
    L.append("|---|---|---|---|---|")
    esc = lambda x: str(x).replace("|", "\\|")
    for c in res["claims"]:
        L.append(f"| {c['id']} | {esc(c['kind'])} | {esc(c['statement'])} | {esc(c['result'])} | **{c['verdict']}** |")
    L.append("")
    L.append("## Hypothesis violations measured")
    L.append(f"* Constant reference cost b on misses: cbar_h(theta)/cbar and vbar_h(theta)/vbar per cell are in `results.json:cells[*].real_cbar_h / real_vbar_h` (selection by acceptance changes both; the exact matched form eq:matched-reference-condition would use them).")
    L.append("* Conditional independence Sigma _||_ Y_B | Y: table (7) and claim C5.")
    L.append("* Homogeneous reference success (beta1 = beta0 = vbar): table (1) shows beta1 > vbar > beta0 for every primary, so the FPR coefficient (vbar - beta0) of eq:app-operating-gain is positive and the homogeneous floor is a strict specialization; C1 reports how much sign accuracy this costs.")
    L.append("* Unit values: the released scores are binary up to 0.04% of rows; the raw-score variant is in `results.json:truth_ge_05`.")
    L.append("* Verifier cost: V1/V2 are free in dollars and cost sandbox seconds; V3 costs judge tokens (priced at the primary's blended $/token; 'free' / 'opus' sensitivities).")
    L.append("")
    L.append("## Audit")
    for k, v in res["audit"].items():
        L.append(f"* `{k}`: {v}")
    L.append("")
    L.append("## Budget and runtime")
    bu = res["budget"]
    L.append(f"* Relay: X8-tagged live calls {bu['live_calls']}, tokens {bu['tokens']:,}, ledger units {bu['units']:.1f} of the {ALLOC_UNITS:,.0f} allocation ({bu['units'] / ALLOC_UNITS:.2%}; the allocation was raised from 80,000 by the coordinator for the full grok-4.6 judge run). "
             f"Ledger rows are billed at the per-model rate stored in common/budget.json at call time (default 0.05 units/token before a model is calibrated), so the same tokens re-priced at the rates stored now come to {bu.get('units_at_stored_rates', float('nan')):.1f} units. Per model: {bu.get('by_model')}. Cached calls re-spend nothing.")
    L.append(f"* grok-4.6 units/token calibration (coordinator request; `PackyClient.calibrate_units_per_token`, sequential judge calls with the relay counter polled, median of the low cluster of clean windows): {res.get('grok_rate_calibration_summary')}.")
    L.append(f"* Sandbox: {res['sandbox_stats']}")
    L.append(f"* Runtime of this run: {res['runtime_seconds']:.0f} s (sandbox and judge stages were cached from earlier stage runs if present: {res.get('stage_times', {})}).")
    L.append("")
    L.append("## Deviations from the task protocol")
    for d in res["deviations"]:
        L.append(f"* {d}")
    L.append("")
    L.append("## Limitations")
    L.append("* id_test has only 334 runnable EXEC-CLEAN tasks; per-source cells on id_test are small (15-112 tasks), so many per-source realized CIs are wide and undecided.")
    L.append("* Truth is the released score; on humaneval/he/bcb the sandbox disagrees with it in 1-7% of rows (infra_smoke (b)), so V2 is near-oracle, not oracle; the oracle control uses the released score itself.")
    L.append("* V1 is neutral (0.5) on every mbpp task (no examples in the prompt) and structurally fails on most bcb doctests (multi-line or non-literal expected outputs), exactly as ACRouter's tool would; this is the imperfection being studied, not a bug.")
    L.append("* The dollar cost of the judge is a convention (primary's blended $/token); the seconds basis mixes recorded API latencies with sandbox wall time measured on this machine. The judge latency (mean 33-39 s, median 24 s per call) was measured under 4-10 concurrent calls on a shared relay, so V3's seconds basis is pessimistic; even at a quarter of that latency V3 would remain far below the reference in seconds (r about 0.1 at every theta).")
    L.append("* The judge calibration has 60 items (AUROC CI width about 0.2); the selected configuration is the best of five on that set, so its calibration AUROC is optimistically biased; the main-sample AUROC of V3 (table 9) is the unbiased figure.")
    L.append("* With a/b around 0.05 in dollars, accept-all is efficiency-optimal for every verifier in dollars; the seconds basis (where Opus is often the fastest model) supplies the discriminating cells for C2 and the negative-sign cells for C1.")
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_summary(res: Dict[str, Any], R: Dict[str, Any], calib: Optional[Dict[str, Any]]) -> None:
    S: List[str] = []
    S.append(f"# X8 summary for the paper (app:operating-point; CodeRouterBench EXEC-CLEAN, recorded responses; primitives on probing (500 tasks), realized on id_test (334 tasks); reference always-Opus; seed {SEED}){' -- ' + res['interim_note'] if res.get('interim') else ''}")
    cl = {c["id"]: c for c in res["claims"]}
    ss = R["sign_summary"]["all"]
    S.append(f"1. [OUT-OF-SAMPLE, C1 {cl['C1']['verdict']}] Floor sign without FPR: {ss.get('agree_pred_hom', float('nan')):.3f} agreement on {ss.get('n_decided', 0)} CI-decided distinct operating points (naive with-FPR {ss.get('agree_pred_naive', float('nan')):.3f}, general affine form {ss.get('agree_pred_gen', float('nan')):.3f}; always-+ {ss.get('baseline_always_plus', float('nan')):.3f}); dollars {R['sign_summary']['by_basis']['usd'].get('agree_pred_hom', float('nan')):.3f} (n={R['sign_summary']['by_basis']['usd'].get('n_decided', 0)}), seconds {R['sign_summary']['by_basis']['sec'].get('agree_pred_hom', float('nan')):.3f} (n={R['sign_summary']['by_basis']['sec'].get('n_decided', 0)}).")
    aa = R["allacc_summary"]["all"]
    S.append(f"2. [OUT-OF-SAMPLE, C2 {cl['C2']['verdict']}] All-accept condition a beta0 <= pi0 b: predicted in {aa.get('all_n_pred_true_pred_general', 0)}/{aa.get('n_cells', 0)} cells, realized-optimal in {aa.get('all_sufficiency_pred_general', float('nan')):.3f} of those ({aa.get('all_n_violations_pred_general', 0)} violations); overall agreement {aa.get('all_agree_pred_general', float('nan')):.3f} vs majority {aa.get('all_baseline_majority', float('nan')):.3f}; dollars share theta=0 optimal {R['allacc_summary']['by_basis']['usd'].get('all_share_real_opt', float('nan')):.2f}, seconds {R['allacc_summary']['by_basis']['sec'].get('all_share_real_opt', float('nan')):.2f}.")
    tm, am = R["thetamin_summary"]["all"], R["argmax_summary"]["all"]
    S.append(f"3. [OUT-OF-SAMPLE, C3 {cl['C3']['verdict']}] theta_min(eps) predicted from Q >= -eps inside the realized interval in {tm.get('inside_model', float('nan')):.3f} of {tm.get('n', 0)} cells (naive 0: {tm.get('naive0_inside', float('nan')):.3f}); argmax theta inside in {am.get('inside_model', float('nan')):.3f} of {am.get('n', 0)} (exact {am.get('exact_match_model', float('nan')):.3f}).")
    mlr_by_v = {}
    for k, v in R["mlr"].items():
        vv, mm, split = k.split("|")
        if split == "probing":
            mlr_by_v.setdefault(vv, []).append(v["mlr_holds"])
    S.append(f"4. [DIAGNOSTIC, C4 {cl['C4']['verdict']}] MLR on probing: " + ", ".join(f"{v} {sum(l)}/{len(l)} primaries" for v, l in mlr_by_v.items()) +
             f" (V1's likelihood ratio is non-monotone because its neutral value 0.5 = 'no visible example' is less indicative of correctness than a failed visible test); realized admissible sets are intervals in {R['thetamin_summary']['interval_share_real']:.3f} of cells.")
    ci_rows = {k: v for k, v in R["cond_indep"].items() if k.endswith("|probing") and np.isfinite(v["given_incorrect"].get("rho", float("nan")))}
    if not ci_rows:
        ci_rows = {"none": {"given_correct": {"rho": float("nan"), "p_perm": float("nan"), "beta_theta_range": [float("nan"), float("nan")]}, "given_incorrect": {"rho": float("nan"), "p_perm": float("nan"), "beta_theta_range": [float("nan"), float("nan")]}}}
    S.append(f"5. [DIAGNOSTIC, C5 {cl['C5']['verdict']}] Sigma _||_ Y_B | Y: pooled permutation tests reject in {sum(1 for v in ci_rows.values() for st in ('given_correct', 'given_incorrect') if np.isfinite(v[st].get('p_perm', float('nan'))) and v[st]['p_perm'] < 0.05)}/{sum(1 for v in ci_rows.values() for st in ('given_correct', 'given_incorrect') if np.isfinite(v[st].get('p_perm', float('nan'))))} strata "
             f"(V1 given incorrect: rho {ci_rows.get('V1|Qwen3-Max|probing', {}).get('given_incorrect', {}).get('rho', float('nan')):+.2f}; V2 given incorrect: {ci_rows.get('V2|Qwen3-Max|probing', {}).get('given_incorrect', {}).get('rho', float('nan')):+.2f}), within-source tests (non-degenerate) reject in {sum(1 for v in ci_rows.values() for st in ('given_correct', 'given_incorrect') if np.isfinite(v[st].get('p_perm_within_source', float('nan'))) and not v[st].get('within_source_degenerate') and v[st]['p_perm_within_source'] < 0.05)}/{sum(1 for v in ci_rows.values() for st in ('given_correct', 'given_incorrect') if np.isfinite(v[st].get('p_perm_within_source', float('nan'))) and not v[st].get('within_source_degenerate'))}: the pooled dependence runs through the source mix (e.g. V1 = 0.5 marks mbpp prompts without examples, which Opus also fails more often), not through the score within a source; beta1(theta) moves by at most {max((v['given_correct']['beta_theta_range'][1] - v['given_correct']['beta_theta_range'][0]) for v in ci_rows.values()):.2f} over theta, beta0(theta) by up to {max((v['given_incorrect']['beta_theta_range'][1] - v['given_incorrect']['beta_theta_range'][0]) for v in ci_rows.values()):.2f}.")
    S.append(f"6. [IN-SAMPLE, C6 {cl['C6']['verdict']}] {cl['C6']['result'][:1200]}")
    for v in VERIFIERS:
        for m in PRIMARIES:
            c = R["cells"].get(f"{v}|{m}|usd")
            if c and m == "Qwen3-Max":
                j5, j1 = THETAS.index(0.5), THETAS.index(1.0)
                S.append(f"7{v}. [OUT-OF-SAMPLE] {VERIFIER_LABEL[v]} / Qwen3-Max (dollars): pi0 {c['pi0'][0]:.3f}, beta1 {c['beta1'][0]:.3f}, beta0 {c['beta0'][0]:.3f}, AUROC id_test {R['roc'][f'{v}|Qwen3-Max|id_test']['auroc']:.3f}; r(0) {c['real_r'][0][0]:.2f}, r(0.5) {c['real_r'][j5][0]:.2f} [{c['real_r'][j5][1]:.2f},{c['real_r'][j5][2]:.2f}], r(1) {c['real_r'][j1][0]:.2f}; Q(0.5) {c['real_Q'][j5][0]:+.3f}, e(0.5) {c['real_e'][j5][0]:.3f}; theta_min(0.02) pred {c['theta_min']['0.02']['pred_model']:g} real {c['theta_min']['0.02']['real']:g}.")
    if calib:
        cj = cl.get("J", {})
        S.append(f"8. [LIVE] Judge calibration: {cj.get('result', '')[:500]} -> {cj.get('verdict')}.")
    bu = res["budget"]
    S.append(f"9. Budget: {bu['live_calls']} live calls, {bu['tokens']:,} tokens, {bu['units']:.0f} ledger units ({bu['units'] / ALLOC_UNITS:.1%} of the {ALLOC_UNITS:,.0f} allocation; at the stored per-model rates {bu.get('units_at_stored_rates', float('nan')):.0f}); grok-4.6 rate calibration: {res.get('grok_rate_calibration_summary')}; deviations: {'; '.join(res['deviations'])[:700]}")
    with open(os.path.join(OUT, "summary_for_paper.md"), "w") as f:
        f.write("\n".join(S[:15]) + "\n")


# ============================================================================================ 9. main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["sample", "sandbox", "calib", "judge", "analysis", "all"])
    ap.add_argument("--dry", action="store_true", help="no live relay calls (uses cached judge results only)")
    ap.add_argument("--quick", action="store_true", help="skip per-source cells and permutation tests")
    ap.add_argument("--no-judge", action="store_true", help="offline parts only (V1, V2, oracle): skip calibration/judge stages and analyse without V3 (interim results.json/README)")
    ap.add_argument("--collect", action="store_true", help="collect missing sandbox and model-judge records")
    args = ap.parse_args()
    stage_times: Dict[str, float] = {}
    M = load_matrix()
    samples = build_samples(M) if args.collect else json.load(open(os.path.join(CACHE, "samples.json")))
    log(f"samples: probing {samples['n_probing']}, id_test {samples['n_id_test']}; {samples['note']}")
    if args.stage == "sample":
        return
    t = time.time()
    sb = run_sandbox_stage(samples) if args.collect else json.load(open(os.path.join(CACHE, "sandbox.json")))
    stage_times["sandbox"] = time.time() - t
    if args.stage == "sandbox":
        return
    client = None
    calib = None
    jd = None
    if args.no_judge:
        # interim mode: report the cached calibration table if present, but do not touch the judge stage
        calib = load_json(os.path.join(CACHE, "calibration.json"), None)
        if calib is not None and not calib.get("best"):
            calib = None
    elif not args.collect:
        calib = json.load(open(os.path.join(CACHE, "calibration.json")))
        jd = json.load(open(os.path.join(CACHE, "judge.json")))
        jd["model"] = calib["configs"][calib["best"]]["model"]
        jd["units_per_token"] = load_json(BUDGET_PATH, {}).get("units_per_token_by_model", {}).get(jd["model"], packy.DEFAULT_UNITS_PER_TOKEN)
    else:
        try:
            client = make_client()
        except Exception as e:  # pragma: no cover
            log(f"packy client unavailable: {e}")
    if client is not None:
        t = time.time()
        calib = run_calibration(client, M, samples, dry=args.dry)
        stage_times["calib"] = time.time() - t
        if args.stage == "calib":
            return
        if calib.get("best"):
            t = time.time()
            jd = run_judge_stage(client, calib, samples, dry=args.dry)
            jd["model"] = calib["configs"][calib["best"]]["model"]
            jd["units_per_token"] = client.units_per_token(jd["model"])
            stage_times["judge"] = time.time() - t
        if args.stage == "judge":
            return
    t = time.time()
    rng = np.random.default_rng(SEED)
    D = assemble(M, samples, sb, jd, truth_level=1.0)
    R = analyze(D, rng, judge_price="primary", quick=args.quick)
    log("analysis (truth >= 1) done")
    interim = bool(args.no_judge) or (jd is not None and jd.get("plan") is not None and len(jd.get("rows", {})) < jd["plan"].get("n_calls_planned", 0))
    res: Dict[str, Any] = dict(experiment_id="X8_operating_point", seed=SEED, nboot=B_BOOT, thetas=R["thetas"], eps_grid=EPS_GRID, primaries=PRIMARIES, judged_primaries=JUDGED_PRIMARIES,
                               interim=interim, interim_note=("INTERIM: offline parts only (V1, V2, oracle); V3 judge rows not (fully) available yet" if interim else ""),
                               reference=REF, samples=dict(n_probing=samples["n_probing"], n_id_test=samples["n_id_test"], pool_counts=samples["pool_counts"], note=samples["note"],
                                                           probing_by_source=dict(Counter(t.split("_")[1] for t in samples["probing"])), id_test_by_source=dict(Counter(t.split("_")[1] for t in samples["id_test"]))),
                               price_per_token=D["price_per_token"], judge_model=D["judge_model"], judge_units_per_token=D["judge_units_per_token"])
    # data checks
    sub = M["df"][M["df"].task_id.isin(samples["probing"] + samples["id_test"])]
    bcb_meta = M["meta"][M["meta"].source == "bcb"]
    res["data_checks"] = dict(fractional_score_share=float(((sub.score > 0) & (sub.score < 1)).mean()), n_rows=int(len(sub)),
                              bcb_allzero_excluded=int(bcb_meta.allzero.sum()), bcb_allzero_excluded_by_split=bcb_meta[bcb_meta.allzero].groupby("split").size().to_dict(), bcb_kept=int((~bcb_meta.allzero).sum()),
                              always_opus_id_test_total_usd_all2919=float(M["C"].loc[M["meta"].split == "id_test", REF].sum()),
                              mean_cost_usd={m: float(sub[sub.model == m].cost_usd.mean()) for m in PRIMARIES + [REF]}, mean_latency_s={m: float(sub[sub.model == m].latency_ms.mean() / 1000) for m in PRIMARIES + [REF]},
                              pass_rate_ge1={m: float((sub[sub.model == m].score >= 1).mean()) for m in PRIMARIES + [REF]})
    # verifier cost summary (id_test)
    Tt = D["id_test"]
    res["verifier_cost"] = {}
    for v in VERIFIERS:
        for m in PRIMARIES:
            if m not in Tt["SIG"].get(v, {}):
                continue
            ok = np.isfinite(Tt["SIG"][v][m])
            e: Dict[str, float] = dict(n=int(ok.sum()), mean_seconds=float(Tt["TV"][v][m][ok].mean()) if v != "oracle" else 0.0)
            if v == "V3":
                e.update(mean_tokens=float(Tt["JTOK"][m][ok].mean()), mean_units=float(Tt["JUNITS"][m][ok].mean()), mean_usd=float((Tt["JTOK"][m][ok] * D["price_per_token"][m]).mean()),
                         total_tokens_id_test=float(Tt["JTOK"][m][ok].sum()), total_units_id_test=float(Tt["JUNITS"][m][ok].sum()))
            else:
                e.update(mean_tokens=0.0, mean_units=0.0, mean_usd=0.0)
            res["verifier_cost"][f"{v}|{m}"] = e
    # sensitivities: judge price, truth >= 0.5, exclude empties
    res["sensitivity_judge_price"] = {}
    if jd and jd.get("rows"):
        for jp in ("free", "opus"):
            Rj = analyze(D, np.random.default_rng(SEED), judge_price=jp, quick=True)
            for m in JUDGED_PRIMARIES:
                c0, c1 = R["cells"].get(f"V3|{m}|usd"), Rj["cells"].get(f"V3|{m}|usd")
                if c0 and c1:
                    j5 = THETAS.index(0.5)
                    res["sensitivity_judge_price"][f"{m}|{jp}"] = dict(r_theta05_main=c0["real_r"][j5][0], r_theta05=c1["real_r"][j5][0], argmax_main=c0["argmax"]["real"], argmax=c1["argmax"]["real"],
                                                                        r_argmax=c1["real_r"][THETAS.index(c1["argmax"]["real"])][0])
    D05 = assemble(M, samples, sb, jd, truth_level=0.5)
    R05 = analyze(D05, np.random.default_rng(SEED), judge_price="primary", quick=True)
    mx = 0.0
    for k, c in R["cells"].items():
        c5 = R05["cells"].get(k)
        if c5:
            mx = max(mx, float(np.max(np.abs(np.array(c["real_r"])[:, 0] - np.array(c5["real_r"])[:, 0]))))
    res["truth_ge_05"] = dict(max_abs_r_diff=mx, pi0_diff={k: R05["cells"][k]["pi0"][0] - R["cells"][k]["pi0"][0] for k in R["cells"] if k in R05["cells"]},
                              cells={k: dict(pi0=v["pi0"], real_r=v["real_r"], TPR_probing=v["TPR_probing"], FPR_probing=v["FPR_probing"]) for k, v in R05["cells"].items()})
    # sensitivity: count empty recorded responses as failed instead of dropping them as unverifiable
    res["sensitivity_include_empty"] = {}
    res["empty_responses"] = {m: dict(probing=int(D["probing"]["EMPTY"][m].sum()), id_test=int(Tt["EMPTY"][m].sum()),
                                      truth1_probing=int((D["probing"]["EMPTY"][m] & (D["probing"]["YP"][m] == 1)).sum()), truth1_id_test=int((Tt["EMPTY"][m] & (Tt["YP"][m] == 1)).sum())) for m in PRIMARIES}
    if any(v["probing"] + v["id_test"] for v in res["empty_responses"].values()):
        Dx = assemble(M, samples, sb, jd, truth_level=1.0, include_empty=True)
        Rx = analyze(Dx, np.random.default_rng(SEED), judge_price="primary", quick=True)
        for m in PRIMARIES:
            if res["empty_responses"][m]["probing"] + res["empty_responses"][m]["id_test"] == 0:
                continue
            for v in ("V1", "V2"):
                c0, c1 = R["cells"].get(f"{v}|{m}|usd"), Rx["cells"].get(f"{v}|{m}|usd")
                if c0 and c1:
                    j1 = THETAS.index(1.0)
                    res["sensitivity_include_empty"][f"{v}|{m}"] = dict(n_empty=res["empty_responses"][m], TPR1_probing_main=c0["TPR_probing"][j1], TPR1_probing_incl=c1["TPR_probing"][j1], r1_main=c0["real_r"][j1][0], r1_incl=c1["real_r"][j1][0])
    stage_times["analysis"] = time.time() - t
    # figures
    t = time.time()
    fig_eta_theta(R, os.path.join(FIG, "fig1_eta_theta.pdf"))
    fig_Q_theta(R, os.path.join(FIG, "fig2_Q_theta.pdf"))
    fig_roc(R, os.path.join(FIG, "fig3_roc_iso_efficiency.pdf"))
    fig_likelihood_ratio(R, os.path.join(FIG, "fig6_likelihood_ratio.pdf"))
    if calib and calib.get("truth"):
        fig_judge_calibration(calib, os.path.join(FIG, "fig4_judge_calibration.pdf"))
    if not args.quick:
        fig_cond_indep(R, os.path.join(FIG, "fig5_conditional_independence.pdf"))
    stage_times["figures"] = time.time() - t
    # claims, audit, budget
    res["claims"] = claims_from(R, calib, D)
    res["audit"] = audit(D, R, calib)
    units, calls, toks = x8_units_spent()
    by_model: Dict[str, Dict[str, float]] = defaultdict(lambda: dict(calls=0, tokens=0, units=0.0))
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("tag") == EXP and not r.get("cached"):
                    bm = by_model[r["model"]]
                    bm["calls"] += 1
                    bm["tokens"] += int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
                    bm["units"] += float(r.get("units_est") or 0.0)
    rates_now = load_json(BUDGET_PATH, {}).get("units_per_token_by_model", {})
    for mname, bm in by_model.items():
        bm["rate_now"] = float(rates_now.get(mname, packy.DEFAULT_UNITS_PER_TOKEN))
        bm["units_at_rate_now"] = bm["tokens"] * bm["rate_now"]
    res["budget"] = dict(units=units, live_calls=calls, tokens=toks, allocation=ALLOC_UNITS, by_model=dict(by_model), units_at_stored_rates=float(sum(bm["units_at_rate_now"] for bm in by_model.values())), rates_now=rates_now)
    grc = load_json(os.path.join(CACHE, "grok_rate_calibration.json"), None)
    res["grok_rate_calibration"] = grc
    res["grok_rate_calibration_summary"] = ({k: grc.get(k) for k in ("estimate", "estimate_min", "low_cluster", "n_windows", "n_clean", "stored", "error")} if grc else "not run")
    sbs = Counter()
    tv = []
    for k, r in sb.items():
        sbs[k.split("|")[0] + ":" + str(r.get("kind") or r.get("failure_kind"))] += 1
        tv.append(float(r.get("elapsed_s") or 0.0))
    res["sandbox_stats"] = dict(rows=len(sb), total_sandbox_seconds=float(np.sum(tv)), kinds=dict(sbs))
    res["judge_rows_n"] = len(jd["rows"]) if jd else 0
    res["judge_rows_error"] = sum(1 for r in jd["rows"].values() if r.get("error")) if jd else 0
    res["judge_rows_parse_fail"] = sum(1 for r in jd["rows"].values() if not r.get("parse_ok") and not r.get("error")) if jd else 0
    res["deviations"] = [
        samples["note"] or "id_test sample as requested",
        ((f"V3 judged {jd['plan'].get('primaries')} on id_test {jd['plan']['k_id_test']}/{samples['n_id_test']} and probing {jd['plan']['k_probing']}/{samples['n_probing']} tasks ({jd['plan']['n_calls_planned']} of the designed 1,668 calls); "
          + ("the plan was capped to the X8 allocation at the ledger rate" if jd["plan"].get("constrained") else
             "full designed sample by coordinator decision (the 80,000-unit allocation was raised to 400,000; grok-4.6's relay rate was calibrated at 0.0309 units/token before the run); the selected judge grok-4.6 costs 3.1-3.4k tokens and 24 s (median) per call, so the judge stage took about 3 h wall-clock (17:33-20:25, including the retry pass) at 4-10 concurrent calls"))
         if jd and jd.get("plan") else "V3 not run"),
        "grok-4.6 and deepseek-flash calibration calls used max_tokens 1024 (reasoning models); qwen3-max and deepseek-v4-pro without thinking used the judge default 320",
        "The judge's dollar cost is priced at the primary's blended $/token (no relay price list); tokens, ledger units and seconds are reported alongside",
        "The seconds basis adds sandbox wall time measured on this machine for V1/V2 (not a recorded quantity)",
        "Truth >= 0.5 variant reported only as a difference summary (scores are binary up to 0.04% of rows)",
        "Runtime rule (<= 2 h from scratch): with the delivered caches run.py completes in about 30 s; from an EMPTY relay cache the LIVE stages need about 3.5 h (2 min sandbox, 30 min five-config calibration, 3 h grok-4.6 judging bounded by the model's 24-35 s latency at concurrency 4-10), i.e. the judge stage alone exceeds the rule; stages can be run separately with --stage and every relay call is cached, so a re-run never re-spends",
    ]
    if calib and calib.get("flag_low_auroc"):
        res["deviations"].append(f"FLAG: best judge AUROC {calib.get('best_auroc'):.3f} < 0.75; proceeded as instructed")
    if jd and jd.get("rows"):
        res["deviations"].append(f"V3 rows: {len(jd['rows'])} judged; {res['judge_rows_error']} relay-error rows (HTTP 400 auth flake) skipped as unverifiable; {res['judge_rows_parse_fail']} unparseable replies imputed at p_correct = 0.5; judge stage run at JUDGE_WORKERS = 4 (first 360 rows) then 10 (coordinator raised the grok-4.6 cap)")
    res["calibration"] = {k: v for k, v in (calib or {}).items() if k != "items"} if calib else None
    res["judge_plan"] = jd.get("plan") if jd else None
    res["analysis"] = R
    res["stage_times"] = stage_times
    res["runtime_seconds"] = time.time() - T0
    # per-row csv for transparency
    for split in ("probing", "id_test"):
        A = D[split]
        rows = []
        for i, tid in enumerate(A["tids"]):
            row = dict(task_id=tid, source=A["src"][i], Y_B=A["YB"][i], opus_cost_usd=A["CB"]["usd"][i], opus_latency_s=A["CB"]["sec"][i])
            for m in PRIMARIES:
                row[f"Y_{m}"] = A["YP"][m][i]
                row[f"cost_{m}"] = A["CP"][m]["usd"][i]
                row[f"lat_{m}"] = A["CP"][m]["sec"][i]
                for v in VERIFIERS:
                    if m in A["SIG"].get(v, {}):
                        row[f"{v}_{m}"] = A["SIG"][v][m][i]
                        if v != "oracle":
                            row[f"{v}_sec_{m}"] = A["TV"][v][m][i]
                if m in A["JTOK"]:
                    row[f"V3_tokens_{m}"] = A["JTOK"][m][i]
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"rows_{split}.csv"), index=False)
    save_json(os.path.join(OUT, "results.json"), res)
    write_reports(res, D, R, calib, jd, samples)
    write_summary(res, R, calib)
    log("claims:")
    for c in res["claims"]:
        log(f"  {c['id']} [{c['kind']}] {c['verdict']}: {c['result'][:200]}")
    log(f"budget: {res['budget']}")
    log(f"done in {res['runtime_seconds']:.0f} s")


if __name__ == "__main__":
    main()

