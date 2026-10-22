#!/usr/bin/env python3
"""
X13_memory -- router memory (Section "What Should a Router Remember?", Proposition thm:memory,
eq:noise-threshold, Proposition prop:cascade-records, eq:ttl / eq:fifo, Remarks
rem:state-dependent-writes / rem:accumulating-stores, Appendix app:retention) replayed on the
RECORDED CodeRouterBench outcomes with REAL sandbox verifiers.

What is real: every value (released score), every cost (csv cost_usd / latency), every recorded
response, every verifier verdict (the visible-test sandbox W1 and the hidden-test sandbox W2 are
executed on the recorded responses).  What is constructed (SEMI-SYNTHETIC): the arrival order
(id_test tasks in random orders), the exploration coin that fields a uniformly random model, the
expiration coins, and the retention rule.  No ACRouter run logs (memory writes, verifier
sub-signals, kNN neighbours) exist in the released package -- only per-task decisions -- so the
memory is REPLAYED, not read (README, "Deviations").

Phases (``python run.py --phase sandbox`` populates the verdict cache; ``--phase all`` runs
everything, re-using the cache):
  A  data: csv outcomes, response_matrix, prompts/tests via common.sources, population = id_test
     tasks with an execution-based verifier (12 dimension x source cells), TF-IDF MiniBatchKMeans
     k=50 prompt clusters as the proxy for ACRouter's embedding-kNN store (no embedding model here).
  B  sandbox: W1 visible-test verifier (doctest / public examples in the prompt, else a syntax
     check -- ACRouter's run_visible_tests / check_syntax), W2 hidden-test verifier
     (common.sandbox.verify_response with the official tests), for every (task, model) pair of the
     population and of a probing subsample (primitives are fitted on probing only).
  C  replay: 50 streams, record process per cell (newest record), fixed-process (exploration
     writes) and state-dependent (every accepted attempt writes) modes, TTL / event FIFO / LRU /
     unbounded retention, expiration delta, three verifiers (W0 oracle, W1, W2).
  D  tests of thm:memory (i), eq:noise-threshold (ii), prop:cascade-records (iii), eq:ttl/eq:fifo
     vs the mean-span substitution (iv), state-dependent writes (v), serve-without-verification (vi).
  E  results.json, figures/*.pdf, README.md, summary_for_paper.md.

Deterministic (seed 20260917).  numpy / pandas / scipy / scikit-learn / matplotlib; the only API use is
the optional LLM-judge writer through ``common.packy`` (tag X13): W3 = deepseek-v4-pro (thinking off)
on a 300-task id_test subsample + 15 probing tasks per cell, and W3g = grok-4.6 (X8's best-calibrated
judge) as a budget-capped 2-cell pilot on the same pairs.  Both are cached; a re-run spends nothing.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=SyntaxWarning)  # model code parsed by ast inside the verifiers

OUT = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(OUT)
sys.path.insert(0, EX)
from common import sandbox, sources  # noqa: E402

FIG = os.path.join(OUT, "figures")
CACHE = os.path.join(OUT, "cache")
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 20260917
DATA = os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")
CSV = os.path.join(DATA, "coderouterbench", "id_results_long.csv")
RESPONSE_MATRIX = os.path.join(DATA, "matrices", "phase1_acrouter_v2", "response_matrix.json")
VERDICT_CACHE = os.path.join(CACHE, "verdicts.jsonl")

RUNNABLE_SOURCES = ("humaneval", "he", "mbpp", "bcb", "cruxeval", "hex", "lcb", "ds1000")
ALLZERO_EXCLUDE_SOURCES = ("bcb", "lcb", "ds1000")  # released harness failures visible in infra_smoke (b)
N_CLUSTERS = 50
PROBE_ALL = True  # verify the whole probing population of the 12 cells (2,044 tasks); the caps below apply only if False
PROBE_PER_CELL = 60
PROBE_PER_CLUSTER = 10
SANDBOX_WORKERS = 8
LCB_MAX_CASES = 25
VIS_MAX_EXAMPLES = 5  # ACRouter's run_visible_tests caps at 5
W1B_THRESHOLD = 0.7  # ACRouter tool aggregate 0.3 syntax + 0.4 tests + 0.3 quality

T0 = time.time()


def log(*a: Any) -> None:
    """Timestamped progress line."""
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ======================================================================================= Phase A


def load_outcomes() -> Dict[str, Any]:
    """Outcome matrices (score, cost, latency) and the task table from id_results_long.csv."""
    df = pd.read_csv(CSV)
    df["src"] = df.task_id.str.split("_").str[1]
    df["lang"] = df.task_id.str.split("_").str[2]
    tasks = df.drop_duplicates("task_id")[["task_id", "split", "source_split", "dimension", "src", "lang"]].reset_index(drop=True)
    models = sorted(df.model.unique())
    tix = {t: i for i, t in enumerate(tasks.task_id)}
    mix = {m: i for i, m in enumerate(models)}
    T, M = len(tasks), len(models)
    S = np.full((T, M), np.nan)
    C = np.full((T, M), np.nan)
    LAT = np.full((T, M), np.nan)
    ti = df.task_id.map(tix).values
    mi = df.model.map(mix).values
    S[ti, mi] = df.score.values
    C[ti, mi] = df.cost_usd.values
    LAT[ti, mi] = df.latency_ms.values / 1000.0
    assert not np.isnan(S).any() and not np.isnan(C).any() and not np.isnan(LAT).any()
    assert T == 9999 and M == 8
    tasks["allzero"] = S.max(1) <= 0
    tasks["cell12"] = tasks.dimension + "|" + tasks.src
    return dict(df=df, tasks=tasks, models=models, tix=tix, mix=mix, S=S, C=C, LAT=LAT)


def candidate_mask(tasks: pd.DataFrame) -> np.ndarray:
    """Tasks with an execution-based verifier: runnable sources, python hex only, no test-generation
    dimension, and no released all-zero tasks of bcb / lcb / ds1000 (released harness failures)."""
    ok = tasks.src.isin(RUNNABLE_SOURCES).to_numpy(copy=True)
    ok &= ~((tasks.src == "hex") & (tasks.lang != "python")).values
    ok &= (tasks.dimension != "test_generation").values
    ok &= ~(tasks.allzero & tasks.src.isin(ALLZERO_EXCLUDE_SOURCES)).values
    return ok


def load_tasks(task_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Prompts / tests for the given ids via common.sources (memoised inside sources)."""
    out = {}
    for tid in task_ids:
        out[tid] = sources.load_task(tid)
    return out


def fit_clusters(probe_prompts: List[str], test_prompts: List[str], k: int = N_CLUSTERS) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """TF-IDF + MiniBatchKMeans(k) fitted on PROBING prompts; id_test prompts assigned to the nearest
    centroid.  A stand-in for ACRouter's embedding-keyed kNN store (no embedding model available)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32)
    Xp = vec.fit_transform(probe_prompts)
    Xt = vec.transform(test_prompts)
    km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5, batch_size=1024, max_iter=200)
    lab_p = km.fit_predict(Xp)
    lab_t = km.predict(Xt)
    info = dict(vocab=int(len(vec.vocabulary_)), inertia=float(km.inertia_), n_probe=len(probe_prompts), n_test=len(test_prompts))
    return lab_p, lab_t, info


# ======================================================================================= Phase B: verifiers


def check_syntax(code: str) -> bool:
    """ACRouter tool 1 (check_syntax): the extracted code parses."""
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def estimate_quality(code: str, prompt: str) -> float:
    """ACRouter tool 3 (estimate_quality), copied from AGENT_ROUTER.py (structural heuristic)."""
    if not code or len(code.strip()) < 5:
        return 0.0
    score = 0.0
    lines = code.strip().split("\n")
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if len(code_lines) >= 2:
        score += 0.3
    if re.search(r"(def |class )\w+", code):
        score += 0.2
    if len(code_lines) >= 5:
        score += 0.2
    if "return " in code:
        score += 0.15
    p = (prompt or "").lower()
    needs_import = any(lib in p for lib in ["pandas", "numpy", "torch", "sklearn", "matplotlib"])
    if needs_import and "import " in code:
        score += 0.15
    elif not needs_import:
        score += 0.15
    return min(score, 1.0)


_DOCTEST_RE = re.compile(r"^\s*>>>\s?(.*)$")
_CONT_RE = re.compile(r"^\s*\.\.\.\s?(.*)$")


def visible_examples(prompt: str) -> List[Tuple[str, str]]:
    """Doctest-style examples visible in the prompt: (source statement, expected output text).

    A ``>>>`` line (with ``...`` continuations) followed by output lines up to the next blank /
    ``>>>`` line.  Statements without output (setup such as ``random.seed(42)``) are returned with
    an empty expected text and are executed but not counted.  Mirrors ACRouter's run_visible_tests
    (which used a one-line regex); nothing outside the prompt is used.
    """
    out: List[Tuple[str, str]] = []
    lines = (prompt or "").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _DOCTEST_RE.match(lines[i])
        if not m:
            i += 1
            continue
        src = [m.group(1)]
        i += 1
        while i < len(lines) and _CONT_RE.match(lines[i]):
            src.append(_CONT_RE.match(lines[i]).group(1))
            i += 1
        exp: List[str] = []
        while i < len(lines) and lines[i].strip() and not _DOCTEST_RE.match(lines[i]):
            s = lines[i].strip()
            if s.startswith('"""') or s.startswith("'''") or s.startswith("```"):
                break
            exp.append(s)
            i += 1
        stmt = "\n".join(src)
        if not exp and re.search(r"==|!=|\bis\b|>=|<=", stmt) and not re.match(r"^\s*(import|from|assert)\b", stmt):
            # ``>>> f(19) == 'xix'`` with no output line: a boolean example (HumanEval style); expected True
            exp = ["True"]
        out.append((stmt, "\n".join(exp)))
    return out


_VIS_HARNESS = '''

import sys as _sys
_examples = {examples!r}
_n = 0
_p = 0
_g = globals()
for _src, _exp in _examples:
    if not _exp:
        try:
            exec(_src, _g)
        except Exception:
            pass
        continue
    if _n >= {cap}:
        break
    _n += 1
    try:
        _res = eval(_src, _g)
    except SyntaxError:
        try:
            exec(_src, _g)
            _res = None
        except Exception:
            continue
    except Exception:
        continue
    _ok = False
    try:
        if repr(_res) == _exp or str(_res) == _exp:
            _ok = True
        else:
            try:
                _ok = (_res == eval(_exp, _g))
                if not isinstance(_ok, bool):
                    _ok = bool(_ok)
            except Exception:
                _ok = False
    except Exception:
        _ok = False
    _p += int(bool(_ok))
print("__VIS__", _p, _n)
'''


def run_visible_tests(task: Dict[str, Any], code: str, examples: List[Tuple[str, str]]) -> Tuple[int, int, float]:
    """Run the visible examples against the extracted code in the sandbox; (passed, counted, seconds)."""
    if not examples or not code.strip():
        return 0, 0, 0.0
    if task["source"] in ("bcb", "humaneval", "he", "hex"):
        code = sandbox._with_prompt_imports(code, task.get("raw_prompt"))  # same assembly as verify_response
    script = _VIS_HARNESS.format(examples=examples, cap=VIS_MAX_EXAMPLES)
    r = sandbox.run_python_tests(code, script, timeout_s=10)
    m = re.search(r"__VIS__ (\d+) (\d+)", r.get("stdout") or "")
    if not m:
        n = min(sum(1 for _, e in examples if e), VIS_MAX_EXAMPLES)
        return 0, max(n, 1), float(r.get("elapsed_s", 0.0))
    return int(m.group(1)), int(m.group(2)), float(r.get("elapsed_s", 0.0))


def verify_pair(task: Dict[str, Any], resp: str) -> Dict[str, Any]:
    """Both sandbox verifiers on one recorded response.

    W2 (hidden tests): ``sandbox.verify_response`` with the official test material (lcb capped at
    LCB_MAX_CASES cases, public first).  W1 (visible tests): the prompt's doctest examples (lcb: the
    public cases) must all pass; when the prompt shows no example the verifier can only check that
    the extracted code parses (ACRouter's check_syntax).  W1b: ACRouter's tool aggregate
    0.3*syntax + 0.4*tests(0.5 when none) + 0.3*estimate_quality >= 0.7.
    """
    src = task["source"]
    text = resp or ""
    empty = not text.strip()
    row: Dict[str, Any] = {"empty": bool(empty)}
    # ---- W2 hidden tests
    kw = {"max_cases": LCB_MAX_CASES} if src == "lcb" else {}
    r2 = sandbox.verify_response(task, text, **kw)
    row.update(w2_pass=bool(r2.get("passed")), w2_frac=float(r2.get("frac") or 0.0), w2_sec=float(r2.get("elapsed_s") or 0.0), w2_kind=r2.get("failure_kind"))
    # ---- W1 visible tests
    if empty:
        row.update(w1_syntax=False, w1_n=0, w1_p=0, w1_sec=0.0, w1_frac=None, w1_quality=0.0, w1_accept=False, w1b_tool=0.0, w1b_accept=False)
        return row
    if src == "cruxeval":
        code = sandbox.extract_prediction(text) or ""
        syntax_ok = check_syntax(code)
        n = p = 0
        sec = 0.0
    elif src == "lcb":
        code = sandbox.extract_code(text, "python", entry_point=task["tests"].get("func_name") or None)
        tests = task["tests"]
        if tests.get("type") == "functional" and tests.get("func_name") and not re.search(rf"def\s+{re.escape(tests['func_name'])}\s*\(", code):
            alt = sandbox.extract_code(text, "python", entry_point="Solution")
            if "class Solution" in alt:
                code = alt
        syntax_ok = check_syntax(code)
        pub = list(tests.get("public") or [])
        if pub and syntax_ok:
            rp = sandbox.run_python_tests(code, {"type": tests.get("type"), "func_name": tests.get("func_name"), "public": pub, "private": []},
                                          entry_point=tests.get("func_name"), timeout_s=6)
            n, p, sec = int(rp.get("n_cases") or len(pub)), int(rp.get("n_passed") or 0), float(rp.get("elapsed_s") or 0.0)
        else:
            n, p, sec = (len(pub), 0, 0.0) if pub else (0, 0, 0.0)
    else:
        code = sandbox.extract_code(text, "python", entry_point=task.get("entry_point"))
        syntax_ok = check_syntax(code)
        ex = visible_examples(task.get("prompt") or "")
        counted = [e for e in ex if e[1]]
        if counted and syntax_ok:
            p, n, sec = run_visible_tests(task, code, ex)
        else:
            n, p, sec = (min(len(counted), VIS_MAX_EXAMPLES), 0, 0.0) if counted else (0, 0, 0.0)
    frac = (p / n) if n > 0 else None
    quality = estimate_quality(code, task.get("prompt") or "")
    tool = 0.3 * float(syntax_ok) + 0.4 * (frac if frac is not None else 0.5) + 0.3 * quality
    accept = (p == n) if n > 0 else bool(syntax_ok)
    row.update(w1_syntax=bool(syntax_ok), w1_n=int(n), w1_p=int(p), w1_sec=float(sec), w1_frac=frac, w1_quality=float(quality),
               w1_accept=bool(accept), w1b_tool=float(tool), w1b_accept=bool(tool >= W1B_THRESHOLD))
    return row


def load_verdict_cache() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Cached (task_id, model) -> verdict rows."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if os.path.exists(VERDICT_CACHE):
        with open(VERDICT_CACHE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                out[(r["task_id"], r["model"])] = r
    return out


def run_sandbox_phase(task_dict: Dict[str, Dict[str, Any]], task_ids: Sequence[str], models: Sequence[str], rm: Dict[str, Dict[str, str]],
                      S: np.ndarray, tix: Dict[str, int], mix: Dict[str, int], workers: int = SANDBOX_WORKERS, limit: Optional[int] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Verify every (task, model) recorded response with W1 and W2; results cached in VERDICT_CACHE."""
    cache = load_verdict_cache()
    jobs = [(tid, m) for tid in task_ids for m in models if (tid, m) not in cache]
    if limit is not None:
        jobs = jobs[:limit]
    log(f"sandbox phase: {len(cache)} cached, {len(jobs)} to run ({len(task_ids)} tasks x {len(models)} models) with {workers} workers")
    if not jobs:
        return cache
    lock = threading.Lock()
    fh = open(VERDICT_CACHE, "a", encoding="utf-8")
    done = 0
    t_start = time.time()
    src_time: Dict[str, List[float]] = {}

    def work(job: Tuple[str, str]) -> Tuple[Tuple[str, str], Dict[str, Any]]:
        tid, m = job
        task = task_dict[tid]
        t0 = time.time()
        row = verify_pair(task, rm.get(tid, {}).get(m, ""))
        row.update(task_id=tid, model=m, source=task["source"], truth=float(S[tix[tid], mix[m]]), wall=round(time.time() - t0, 4))
        return job, row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            job, row = fut.result()
            with lock:
                cache[job] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
                src_time.setdefault(row["source"], []).append(row["wall"])
                if done % 500 == 0 or done == len(jobs):
                    fh.flush()
                    el = time.time() - t_start
                    log(f"  sandbox {done}/{len(jobs)} ({el / 60:.1f} min, eta {el / done * (len(jobs) - done) / 60:.1f} min)")
    fh.close()
    log("sandbox mean wall s per source: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in sorted(src_time.items())))
    return cache


# ======================================================================================= Phase A driver


def build_population(D: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Population (id_test with verifier), probing candidates, prompt clusters, probing subsample."""
    tasks = D["tasks"]
    cand = candidate_mask(tasks)
    cand_ids = list(tasks.task_id[cand])
    log(f"loading prompts/tests for {len(cand_ids)} candidate tasks (both splits)")
    tdict = load_tasks(cand_ids)
    ok = np.array([bool(tdict[t]["found"] and tdict[t]["runnable"]) for t in cand_ids])
    cand_ok = [t for t, o in zip(cand_ids, ok) if o]
    split = tasks.set_index("task_id").split
    pop = [t for t in cand_ok if split[t] == "id_test"]
    probe = [t for t in cand_ok if split[t] == "probing"]
    log(f"population id_test {len(pop)} tasks; probing candidates {len(probe)}")
    # clusters: fit on probing prompts, assign id_test
    lab_p, lab_t, kinfo = fit_clusters([tdict[t]["prompt"] or "" for t in probe], [tdict[t]["prompt"] or "" for t in pop])
    cl_p = dict(zip(probe, [int(x) for x in lab_p]))
    cl_t = dict(zip(pop, [int(x) for x in lab_t]))
    # probing subsample: <= PROBE_PER_CELL per dimension x source cell, union <= PROBE_PER_CLUSTER per cluster
    # (PROBE_ALL: the whole probing population of the 12 cells; affordable at ~0.03 s per sandbox job)
    cell_of = tasks.set_index("task_id").cell12
    if PROBE_ALL:
        probe_sub = sorted(probe)
        log(f"probing subsample = all {len(probe_sub)} probing tasks; clusters: {kinfo}")
        return dict(tdict=tdict, pop=pop, probe=probe, probe_sub=probe_sub, cl_t=cl_t, cl_p=cl_p, kinfo=kinfo)
    chosen = set()
    by_cell: Dict[str, List[str]] = {}
    for t in probe:
        by_cell.setdefault(cell_of[t], []).append(t)
    for c, lst in sorted(by_cell.items()):
        lst = sorted(lst)
        pick = lst if len(lst) <= PROBE_PER_CELL else list(rng.choice(lst, size=PROBE_PER_CELL, replace=False))
        chosen.update(pick)
    by_cl: Dict[int, List[str]] = {}
    for t in probe:
        by_cl.setdefault(cl_p[t], []).append(t)
    for c, lst in sorted(by_cl.items()):
        lst = sorted(lst)
        have = [t for t in lst if t in chosen]
        need = PROBE_PER_CLUSTER - len(have)
        if need > 0:
            rest = [t for t in lst if t not in chosen]
            if rest:
                chosen.update(rest if len(rest) <= need else list(rng.choice(rest, size=need, replace=False)))
    probe_sub = sorted(chosen)
    log(f"probing subsample {len(probe_sub)} tasks; clusters: {kinfo}")
    return dict(tdict=tdict, pop=pop, probe=probe, probe_sub=probe_sub, cl_t=cl_t, cl_p=cl_p, kinfo=kinfo)


def main_phase_sandbox(limit: Optional[int] = None, workers: int = SANDBOX_WORKERS) -> Dict[str, Any]:
    """Populate the verdict cache (idempotent); returns the population dict."""
    D = load_outcomes()
    rng = np.random.default_rng(SEED)
    P = build_population(D, rng)
    rm = json.load(open(RESPONSE_MATRIX, "r", encoding="utf-8"))
    ids = list(P["pop"]) + [t for t in P["probe_sub"] if t not in set(P["pop"])]
    run_sandbox_phase(P["tdict"], ids, D["models"], rm, D["S"], D["tix"], D["mix"], workers=workers, limit=limit)
    P["D"] = D
    P["rm"] = rm
    return P


# ======================================================================================= Phase B2: optional LLM judge (W3)

JUDGE_MODEL = "deepseek-v4-pro"
JUDGE_WORKERS = 4
JUDGE_TAG = "X13"
JUDGE_TEST_TASKS = 300
JUDGE_PROBE_PER_CELL = 15
JUDGE_CACHE = os.path.join(CACHE, "judge_verdicts.jsonl")
JUDGE_UNIT_CAP = 25000.0  # ledger units this experiment may spend (allocation 30,000)


def judge_subsample(P: Dict[str, Any], rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    """Deterministic subsamples for W3: JUDGE_TEST_TASKS id_test tasks stratified by cell (proportional,
    >= 5 per cell) and JUDGE_PROBE_PER_CELL probing tasks per cell."""
    tasks = P["D"]["tasks"].set_index("task_id")
    cell_of = tasks.cell12

    def strat(ids: List[str], total: Optional[int], per_cell: Optional[int]) -> List[str]:
        by: Dict[str, List[str]] = {}
        for t in ids:
            by.setdefault(cell_of[t], []).append(t)
        out: List[str] = []
        n_all = len(ids)
        for c, lst in sorted(by.items()):
            lst = sorted(lst)
            k = per_cell if per_cell is not None else max(5, int(round(total * len(lst) / n_all)))
            k = min(k, len(lst))
            out += list(rng.choice(lst, size=k, replace=False)) if k < len(lst) else lst
        return sorted(out)

    test_sub = strat(list(P["pop"]), JUDGE_TEST_TASKS, None)
    probe_sub = strat(list(P["probe"]), None, JUDGE_PROBE_PER_CELL)
    return test_sub, probe_sub


GROK_MODEL = "grok-4.6"
GROK_WORKERS = 4
GROK_CACHE = os.path.join(CACHE, "judge_verdicts_grok.jsonl")
GROK_META = os.path.join(CACHE, "judge_meta_grok.json")
GROK_UNIT_CAP = 24000.0  # ledger units (default 0.05 units/token: ~150 units per grok call); the X13 allocation is 30,000
GROK_CELLS = ("code_generation|mbpp", "code_completion|mbpp")  # short prompts (grok reasons ~3-7k tokens per call); one cell where the judge is structurally handicapped (dim1 mbpp has no function name), one where judge quality matters
GROK_TEST_PER_CELL = 6
GROK_PROBE_PER_CELL = 3
GROK_MAX_TOKENS = 1024  # as in X8's calibration


def load_judge_cache(path: str = JUDGE_CACHE) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Cached judge rows keyed by (task_id, model)."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    out[(r["task_id"], r["model"])] = r
    return out


def run_judge_generic(P: Dict[str, Any], model: str, cache_path: str, test_sub: List[str], probe_sub: List[str], *, unit_cap: float, workers: int,
                      max_tokens: int, batch: int = 200, limit: Optional[int] = None, label: str = "judge") -> Dict[str, Any]:
    """LLM-judge the recorded responses of ``test_sub`` + ``probe_sub`` (all 8 models) with ``model``.

    Every call goes through ``common.packy`` (disk cache, ledger, budget guard) with tag ``X13``;
    the parsed verdicts are appended to ``cache_path``.  Empty recorded responses are not judged
    (unverifiable).  Stops when this experiment's session spend exceeds ``unit_cap`` ledger units.
    Rows whose relay call failed after packy's retries carry ``error`` and ``verdict='unknown'``; they
    are retried on the next run (not cached as verdicts).
    """
    from common import judge as judge_mod
    from common import packy

    ids = test_sub + [t for t in probe_sub if t not in set(test_sub)]
    models = P["D"]["models"]
    rm = P["rm"]
    cache = {k: v for k, v in load_judge_cache(cache_path).items() if v.get("verdict") in ("correct", "incorrect")}
    jobs = [(t, m) for t in ids for m in models if (t, m) not in cache and (rm.get(t, {}).get(m, "") or "").strip()]
    n_empty = sum(1 for t in ids for m in models if not (rm.get(t, {}).get(m, "") or "").strip())
    if limit is not None:
        jobs = jobs[:limit]
    log(f"{label} ({model}): {len(test_sub)} id_test + {len(probe_sub)} probing tasks; {len(cache)} cached, {len(jobs)} calls to make, {n_empty} empty responses skipped")
    meta: Dict[str, Any] = {"model": model, "thinking": False, "max_tokens": max_tokens, "test_tasks": test_sub, "probe_tasks": probe_sub, "n_empty_skipped": n_empty, "calls_made": 0,
                            "units_session": 0.0, "tokens_session": 0, "n_error_rows": 0, "unit_cap": unit_cap, "stopped_by_cap": False}
    if not jobs:
        return meta
    client = packy.PackyClient(quiet=True)
    us0 = client.usage_summary()
    fh = open(cache_path, "a", encoding="utf-8")
    done = 0
    n_err = 0
    t_start = time.time()
    for b0 in range(0, len(jobs), batch):
        bt = jobs[b0:b0 + batch]
        items = [{"task": P["tdict"][t], "response_text": rm[t][m]} for t, m in bt]
        res = judge_mod.judge_many(client, items, model=model, workers=workers, tag=JUDGE_TAG, thinking=False, max_tokens=max_tokens)
        for (t, m), r in zip(bt, res):
            row = {"task_id": t, "model": m, "judge": model, "verdict": r.get("verdict"), "p_correct": r.get("p_correct"), "confidence": r.get("confidence"),
                   "parse_ok": r.get("parse_ok"), "cached": r.get("cached"), "tokens": (r.get("usage") or {}).get("total_tokens"), "error": r.get("error"),
                   "finish_reason": r.get("finish_reason"), "latency_s": r.get("latency_s"), "truth": float(P["D"]["S"][P["D"]["tix"][t], P["D"]["mix"][m]])}
            n_err += int(bool(r.get("error")) or r.get("verdict") not in ("correct", "incorrect"))
            cache[(t, m)] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        done += len(bt)
        us = client.usage_summary()
        spent = us["session_units_est"]
        el = time.time() - t_start
        log(f"  {label} {done}/{len(jobs)} ({el / 60:.1f} min, eta {el / done * (len(jobs) - done) / 60:.1f} min); session units {spent:.1f}, ledger total {us['ledger_units_total']:.0f}, error rows {n_err}")
        if spent > unit_cap:
            log(f"{label}: session unit cap {unit_cap} reached, stopping")
            meta["stopped_by_cap"] = True
            break
    fh.close()
    us1 = client.usage_summary()
    sess = us1["session"].get(model, {})
    meta.update(calls_made=done, n_error_rows=n_err, units_session=float(us1["session_units_est"]), tokens_session=int(sess.get("prompt_tokens", 0) + sess.get("completion_tokens", 0)) if sess else 0,
                ledger_before=float(us0["ledger_units_total"]), ledger_after=float(us1["ledger_units_total"]), session_detail=sess)
    client.close()
    return meta


def run_judge_phase(P: Dict[str, Any], workers: int = JUDGE_WORKERS, limit: Optional[int] = None) -> Dict[str, Any]:
    """W3: deepseek-v4-pro (thinking off) on the 300-task id_test subsample and 15 probing tasks per cell."""
    rng = np.random.default_rng(SEED + 13)
    test_sub, probe_sub = judge_subsample(P, rng)
    return run_judge_generic(P, JUDGE_MODEL, JUDGE_CACHE, test_sub, probe_sub, unit_cap=JUDGE_UNIT_CAP, workers=workers, max_tokens=320, limit=limit, label="judge W3")


def run_grok_pilot(P: Dict[str, Any], judge_meta: Dict[str, Any], workers: int = GROK_WORKERS, limit: Optional[int] = None) -> Dict[str, Any]:
    """W3g: grok-4.6 (X8's best-calibrated judge, AUROC 0.842) on a budget-capped 2-cell pilot.

    Tasks are drawn from the deepseek W3 subsample (GROK_TEST_PER_CELL id_test and GROK_PROBE_PER_CELL
    probing tasks per cell of GROK_CELLS), so both judges are compared on identical (task, model)
    pairs.  A full W3 with grok (300 tasks x 8 models, ~3,100 tokens per call) would cost ~570k
    ledger units at the default rate, twice the shared cap, so it is not attempted.
    """
    tasks = P["D"]["tasks"].set_index("task_id")
    rng = np.random.default_rng(SEED + 17)
    test_sub: List[str] = []
    probe_sub: List[str] = []
    for c in GROK_CELLS:
        tt = sorted(t for t in judge_meta.get("test_tasks", []) if tasks.cell12[t] == c)
        pp = sorted(t for t in judge_meta.get("probe_tasks", []) if tasks.cell12[t] == c)
        test_sub += list(rng.choice(tt, size=min(GROK_TEST_PER_CELL, len(tt)), replace=False)) if tt else []
        probe_sub += list(rng.choice(pp, size=min(GROK_PROBE_PER_CELL, len(pp)), replace=False)) if pp else []
    meta = run_judge_generic(P, GROK_MODEL, GROK_CACHE, sorted(test_sub), sorted(probe_sub), unit_cap=GROK_UNIT_CAP, workers=workers, max_tokens=GROK_MAX_TOKENS, batch=40, limit=limit, label="judge W3g pilot")
    meta["cells"] = list(GROK_CELLS)
    return meta


# ======================================================================================= Phase C: replay engine

N_STREAMS = 50
N_ROUNDS = 60000
LATE = 10000
EPS_FP = 0.25  # exploration probability of the fixed-process (logging) writer
EPS_SD = 0.05  # ACRouter's explore_rate for the state-dependent (every accepted attempt writes) mode
TTL_GRID = [50, 200, 1000]
FIFO_GRID = [50, 200, 1000, 5000]
DELTAS = [0.0, 0.01]
COV_E_GRID = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
COV_M_GRID = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
NBOOT = 1000
BIG = 10 ** 9
VERIFIER_NAMES = {"W0": "oracle (released score >= 1)", "W1": "visible-test sandbox (ACRouter run_visible_tests / check_syntax)", "W2": "hidden-test sandbox (near-oracle)", "W3": "LLM judge deepseek-v4-pro (thinking off)"}


class Pop:
    """Outcome, verdict and cell arrays of one task population (id_test replay population or probing)."""

    def __init__(self, key: str, ids: List[str], D: Dict[str, Any], cells: Dict[str, np.ndarray], cell_names: Dict[str, List[str]],
                 prior: np.ndarray, ACC: Dict[str, np.ndarray], VSEC: Dict[str, np.ndarray]):
        self.key = key
        self.ids = list(ids)
        self.n = len(ids)
        self.idx = np.array([D["tix"][t] for t in ids], dtype=np.int64)
        self.S = D["S"][self.idx]
        self.C = D["C"][self.idx]
        self.LAT = D["LAT"][self.idx]
        self.TRUE = self.S >= 1.0
        self.M = self.S.shape[1]
        self.opus = D["mix"]["claude-opus-4-6"]
        self.cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}
        self.cell_names = cell_names
        self.J = {k: len(v) for k, v in cell_names.items()}
        self.prior = np.asarray(prior, dtype=np.int64)
        self.ACC = {k: np.asarray(v, dtype=bool) for k, v in ACC.items()}
        self.VSEC = {k: np.asarray(v, dtype=float) for k, v in VSEC.items()}
        self.ACC["W0"] = self.TRUE.copy()
        self.VSEC["W0"] = np.zeros_like(self.S)

    def nu(self, cellset: str) -> np.ndarray:
        """Cell frequencies of the population (the arrival law of the replayed stream)."""
        return np.bincount(self.cells[cellset], minlength=self.J[cellset]) / self.n


_STREAM_CACHE: Dict[Tuple[int, str, int], Dict[str, np.ndarray]] = {}


def draw_streams(n_tasks: int, kind: str = "iid") -> Dict[str, np.ndarray]:
    """Common random numbers for all configurations on a population of ``n_tasks`` tasks.

    kind 'iid': N_ROUNDS independent draws with replacement from the population (the independent
    arrivals of Section sec:memory; X4's construction).  kind 'perm': ONE random order of the
    population per stream (n_tasks rounds; every task once), used for the finite-horizon check.
    U_EXP: exploration uniforms; MEXP: explored model; U_LIFE: expiration uniforms of the write made
    in that round.  Seeded by SEED, n_tasks and the kind.
    """
    key = (n_tasks, kind, N_ROUNDS)
    if key in _STREAM_CACHE:
        return _STREAM_CACHE[key]
    rng = np.random.default_rng([SEED, n_tasks, 0 if kind == "iid" else 1])
    if kind == "iid":
        N = N_ROUNDS
        TID = rng.integers(0, n_tasks, size=(N_STREAMS, N), dtype=np.int64)
    else:
        N = n_tasks
        TID = np.stack([rng.permutation(n_tasks) for _ in range(N_STREAMS)]).astype(np.int64)
    out = {"TID": TID, "U_EXP": rng.random((N_STREAMS, N), dtype=np.float32), "MEXP": rng.integers(0, 8, size=(N_STREAMS, N), dtype=np.int64),
           "U_LIFE": rng.random((N_STREAMS, N), dtype=np.float64)}
    _STREAM_CACHE[key] = out
    return out


def route_outcomes(pop: Pop, route: str, verifier: str, basis: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per-task, per-fielded-model value and cost under a routing variant (dollars or seconds basis).

    direct: deliver the model's output.  cascade: attempt the model, an oracle verifier decides,
    Opus fallback on a miss (V = 1 on a hit, Opus score otherwise; D = attempt + fallback cost).
    cascade_v: the same with verifier ``verifier`` deciding (false accepts deliver the wrong output).
    serve: as direct (records served from the store are handled in the engine).
    """
    Cb = pop.C if basis == "dollars" else pop.LAT
    S = pop.S
    if route in ("direct", "serve"):
        return S.copy(), Cb.copy()
    S_op = S[:, pop.opus][:, None]
    C_op = Cb[:, pop.opus][:, None]
    if route == "cascade":
        hit = pop.TRUE
        return np.where(hit, 1.0, S_op), Cb + (~hit) * C_op
    if route == "cascade_v":
        acc = pop.ACC[verifier]
        return np.where(acc, S, S_op), Cb + (~acc) * C_op
    raise ValueError(route)


def replay(pop: Pop, cellset: str, verifier: str, mode: str, retention: str, rpar: float, *, route: str = "direct", c_ell: float = 0.0,
           k_readout: int = 1, basis: str = "dollars", want_cov: bool = False, stream: str = "iid", eps_override: Optional[float] = None) -> Dict[str, Any]:
    """Replay the record process on N_STREAMS streams (vectorised over streams, loop over rounds).

    mode 'FP': fixed process -- on exploration rounds (prob EPS_FP) both routers field a uniformly
    random model and its verified outcome is written; other rounds use the store without writing.
    mode 'SD': state-dependent -- every round the informed router fields its model (record model if
    covered else the prior; epsilon-greedy exploration EPS_SD), verifies it and writes if accepted.
    retention: 'geo' (expiration prob rpar per round), 'ttl' (E = rpar rounds), 'fifo' (last M = rpar
    write events), 'lru' (capacity rpar distinct cells, least-recently-accessed eviction), or 'none'.
    Returns late-window sums per stream / cell / state / task, all-round cumulative gains (for the
    finite-horizon identity) and diagnostic counts.  stream 'perm' = one random pass (late window =
    every round).
    """
    st = draw_streams(pop.n, stream)
    TID, U_EXP, MEXP, U_LIFE = st["TID"], st["U_EXP"], st["MEXP"], st["U_LIFE"]
    S_, N = TID.shape
    late_n = N if stream == "perm" else LATE
    eps = eps_override if eps_override is not None else (EPS_FP if mode == "FP" else EPS_SD)
    EXPL = U_EXP < eps
    cell = pop.cells[cellset]
    J = pop.J[cellset]
    Mn = pop.M
    VR, DR = route_outcomes(pop, route, verifier, basis)
    TRUE = pop.TRUE
    ACC = pop.ACC[verifier]
    VS = pop.VSEC[verifier] if basis == "seconds" else None
    prior = pop.prior
    ar = np.arange(S_)
    delta = float(rpar) if retention == "geo" else 0.0
    if delta > 0:
        LIFE = np.floor(np.log(U_LIFE) / np.log(1.0 - delta)).astype(np.int64)
    else:
        LIFE = np.full((S_, N), BIG, dtype=np.int64)
    E = int(rpar) if retention == "ttl" else None
    Mf = int(rpar) if retention == "fifo" else None
    K = int(rpar) if retention == "lru" else None
    rec_model = np.full((S_, J), -1, np.int64)
    rec_type = np.zeros((S_, J), np.int8)
    rec_round = np.full((S_, J), -BIG, np.int64)
    rec_widx = np.zeros((S_, J), np.int64)
    rec_life = np.zeros((S_, J), np.int64)
    n_writes = np.zeros(S_, np.int64)
    last_access = np.full((S_, J), -1, np.int64)
    if k_readout > 1:
        rec_hist = np.full((S_, J, k_readout), -1, np.int64)
        rec_hist_type = np.zeros((S_, J, k_readout), np.int8)
    late0 = N - late_n
    sumVI = np.zeros(S_); sumDI = np.zeros(S_); sumVU = np.zeros(S_); sumDU = np.zeros(S_)
    allVI = np.zeros(S_); allDI = np.zeros(S_); allVU = np.zeros(S_); allDU = np.zeros(S_)  # every round (finite-horizon identity)
    cell_n = np.zeros((S_, J)); cell_ns = np.zeros((S_, J, 3))
    cell_dV_s = np.zeros((S_, J, 3)); cell_dD_s = np.zeros((S_, J, 3))
    cell_VU = np.zeros((S_, J)); cell_DU = np.zeros((S_, J)); cell_VI = np.zeros((S_, J)); cell_DI = np.zeros((S_, J))
    sw_cnt = np.zeros((J, 3, 3))  # state x write type (0 none, 1 informative, 2 erroneous)
    recm_cnt = np.zeros((J, 3, Mn))  # record model distribution by state (arrivals in state I/W)
    tVI = np.zeros(pop.n); tDI = np.zeros(pop.n); tVU = np.zeros(pop.n); tDU = np.zeros(pop.n)
    if want_cov:
        Eg = np.asarray(COV_E_GRID)[None, :]; Mg = np.asarray(COV_M_GRID)[None, :]
        cov_E = np.zeros((S_, J, len(COV_E_GRID))); cov_M = np.zeros((S_, J, len(COV_M_GRID)))
        cov_n = np.zeros((S_, J)); cov_nM = np.zeros((S_, J, len(COV_M_GRID)))  # FIFO denominators count arrivals after the FIFO of M has filled
    for n in range(N):
        t = TID[:, n]
        j = cell[t]
        ex = EXPL[:, n]
        me = MEXP[:, n]
        rt = rec_type[ar, j]
        has = rt > 0
        if retention == "geo":
            valid = has & (n - rec_round[ar, j] - 1 <= rec_life[ar, j])
        elif retention == "ttl":
            valid = has & (n - rec_round[ar, j] <= E)
        elif retention == "fifo":
            valid = has & (n_writes - rec_widx[ar, j] <= Mf - 1)
        else:
            valid = has
        state = np.where(valid, rt, 0).astype(np.int64)
        recm = np.where(valid, rec_model[ar, j], -1)
        if k_readout > 1:
            hist = rec_hist[ar, j]  # (S_, K), newest first
            onehot = (hist[:, :, None] == np.arange(Mn)[None, None, :]) & (hist[:, :, None] >= 0)
            score = onehot.sum(1).astype(float) + 1e-3 * onehot[:, 0, :]
            maj = score.argmax(1)
            # state = type of the newest record of the chosen model
            match = (hist == maj[:, None])
            first = np.where(match.any(1), match.argmax(1), 0)
            maj_type = rec_hist_type[ar, j, first]
            recm = np.where(valid, maj, -1)
            state = np.where(valid, maj_type, 0).astype(np.int64)
        pm = prior[t]
        inf_m = np.where(state > 0, recm, pm)
        fI = np.where(ex, me, inf_m)
        fU = np.where(ex, me, pm)
        VI = VR[t, fI]; DI = DR[t, fI]; VU = VR[t, fU]; DU = DR[t, fU]
        if route == "serve":
            served = (state > 0) & ~ex
            VI = np.where(served, (state == 1).astype(float), VI)
            DI = np.where(served, c_ell, DI)
        if mode == "FP":
            dow = ex & ACC[t, me]
            wm = me
            if VS is not None:
                vcost = np.where(ex, VS[t, me], 0.0)
                DI = DI + vcost; DU = DU + vcost
        else:
            dow = ACC[t, fI]
            wm = fI
            if VS is not None:
                DI = DI + VS[t, fI]; DU = DU + VS[t, fU]
        wt = np.where(TRUE[t, wm], 1, 2).astype(np.int64)
        wtype0 = np.where(dow, wt, 0)
        allVI += VI; allDI += DI; allVU += VU; allDU += DU
        if n >= late0:
            sumVI += VI; sumDI += DI; sumVU += VU; sumDU += DU
            cell_n[ar, j] += 1
            cell_ns[ar, j, state] += 1
            cell_dV_s[ar, j, state] += VI - VU
            cell_dD_s[ar, j, state] += DI - DU
            cell_VU[ar, j] += VU; cell_DU[ar, j] += DU; cell_VI[ar, j] += VI; cell_DI[ar, j] += DI
            np.add.at(sw_cnt, (j, state, wtype0), 1.0)
            rec_ok = state > 0
            if rec_ok.any():
                np.add.at(recm_cnt, (j[rec_ok], state[rec_ok], recm[rec_ok]), 1.0)
            np.add.at(tVI, t, VI); np.add.at(tDI, t, DI); np.add.at(tVU, t, VU); np.add.at(tDU, t, DU)
            if want_cov:
                age_r = np.where(has, n - rec_round[ar, j], BIG)
                age_e = np.where(has, n_writes - rec_widx[ar, j], BIG)
                filled = n_writes[:, None] >= Mg  # the FIFO of length M holds M events only after M writes
                cov_E[ar, j] += (age_r[:, None] <= Eg).astype(float)
                cov_M[ar, j] += ((age_e[:, None] <= Mg - 1) & filled).astype(float)
                cov_nM[ar, j] += filled.astype(float)
                cov_n[ar, j] += 1.0
        # reads refresh recency (LRU)
        if K is not None:
            last_access[ar, j] = np.where(valid, n, last_access[ar, j])
        ws = np.flatnonzero(dow)
        if ws.size:
            jw = j[ws]
            if K is not None:
                new_cell = rec_type[ws, jw] == 0
                count = (rec_type[ws] > 0).sum(1)
                need = new_cell & (count >= K)
                if need.any():
                    wsn = ws[need]
                    la = last_access[wsn].astype(np.int64).copy()
                    la[rec_type[wsn] == 0] = BIG
                    la[np.arange(len(wsn)), jw[need]] = BIG
                    victim = la.argmin(1)
                    rec_type[wsn, victim] = 0
                    rec_model[wsn, victim] = -1
                    if k_readout > 1:
                        rec_hist[wsn, victim] = -1; rec_hist_type[wsn, victim] = 0
            n_writes[ws] += 1
            rec_model[ws, jw] = wm[ws]
            rec_type[ws, jw] = wt[ws].astype(np.int8)
            rec_round[ws, jw] = n
            rec_widx[ws, jw] = n_writes[ws]
            rec_life[ws, jw] = LIFE[ws, n]
            last_access[ws, jw] = n
            if k_readout > 1:
                rec_hist[ws, jw, 1:] = rec_hist[ws, jw, :-1]
                rec_hist[ws, jw, 0] = wm[ws]
                rec_hist_type[ws, jw, 1:] = rec_hist_type[ws, jw, :-1]
                rec_hist_type[ws, jw, 0] = wt[ws].astype(np.int8)
    out = dict(sumVI=sumVI, sumDI=sumDI, sumVU=sumVU, sumDU=sumDU, allVI=allVI, allDI=allDI, allVU=allVU, allDU=allDU, cell_n=cell_n, cell_ns=cell_ns, cell_dV_s=cell_dV_s, cell_dD_s=cell_dD_s,
               cell_VU=cell_VU, cell_DU=cell_DU, cell_VI=cell_VI, cell_DI=cell_DI, sw_cnt=sw_cnt, recm_cnt=recm_cnt, tVI=tVI, tDI=tDI, tVU=tVU, tDU=tDU,
               n_writes_total=int(n_writes.sum()), n_writes_stream=n_writes.copy(), rounds=N, late=late_n, stream=stream)
    if want_cov:
        out.update(cov_E=cov_E, cov_M=cov_M, cov_n=cov_n, cov_nM=cov_nM)
    return out


def finite_horizon_pred(pr: Dict[str, Any], N: int) -> float:
    """RHS of eq:finite-memory: sum_j nu_j G_j [N - (1 - alpha_j^N) / (1 - alpha_j)] (empty store)."""
    a = pr["alpha"]
    with np.errstate(invalid="ignore", divide="ignore"):
        br = np.where(a < 1, N - (1 - a ** N) / (1 - a), 0.0)
    return float((pr["nu"] * pr["G"] * br).sum())


def finite_horizon_real(R: Dict[str, Any], price: float) -> Tuple[float, float, float]:
    """Realized cumulative gain sum_n [(V_I - price D_I) - (V_U - price D_U)] over all rounds, stream CI."""
    g = (R["allVI"] - R["allVU"]) - price * (R["allDI"] - R["allDU"])
    idx = boot_idx(len(g))
    est = g[idx].mean(1)
    return float(g.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


# ======================================================================================= Phase D: primitives, predictions, statistics

_BOOT_IDX: Dict[int, np.ndarray] = {}
_WBOOT: Dict[int, np.ndarray] = {}


def boot_idx(S_: int) -> np.ndarray:
    """Stream-bootstrap index matrix (NBOOT, S_), seeded."""
    if S_ not in _BOOT_IDX:
        _BOOT_IDX[S_] = np.random.default_rng([SEED, 1, S_]).integers(0, S_, size=(NBOOT, S_))
    return _BOOT_IDX[S_]


def wboot(n: int) -> np.ndarray:
    """Task-cluster bootstrap weights (NBOOT, n): multinomial resampling of tasks, seeded."""
    if n not in _WBOOT:
        _WBOOT[n] = np.random.default_rng([SEED, 2, n]).multinomial(n, np.ones(n) / n, size=NBOOT).astype(float)
    return _WBOOT[n]


def eta_diff_streams(R: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """Realized eta_inf - eta_U (late window, pooled ratio of sums) with a paired stream-bootstrap CI."""
    VI, DI, VU, DU = R["sumVI"], R["sumDI"], R["sumVU"], R["sumDU"]
    point = VI.sum() / DI.sum() - VU.sum() / DU.sum()
    idx = boot_idx(len(VI))
    est = VI[idx].sum(1) / DI[idx].sum(1) - VU[idx].sum(1) / DU[idx].sum(1)
    return float(point), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), float(VI.sum() / DI.sum()), float(VU.sum() / DU.sum())


def eta_diff_tasks(R: Dict[str, Any]) -> Tuple[float, float, float]:
    """The same difference under task-cluster resampling (paired: both policies on the same tasks)."""
    W = wboot(len(R["tVI"]))
    est = (W @ R["tVI"]) / (W @ R["tDI"]) - (W @ R["tVU"]) / (W @ R["tDU"])
    point = R["tVI"].sum() / R["tDI"].sum() - R["tVU"].sum() / R["tDU"].sum()
    return float(point), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def gain_per_round(R: Dict[str, Any], price: float) -> Tuple[float, float, float]:
    """Realized mean per-round gain (V_I - price D_I) - (V_U - price D_U) in the late window, stream CI."""
    g = ((R["sumVI"] - R["sumVU"]) - price * (R["sumDI"] - R["sumDU"])) / R["late"]
    idx = boot_idx(len(g))
    est = g[idx].mean(1)
    return float(g.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def cell_gain_realized(R: Dict[str, Any], price: float, state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell realized gain per arrival at ``price`` (all states, or a given record state), stream CI.

    Returns (point, lo, hi, n_arrivals) over cells; the point is the pooled ratio over streams.
    """
    if state is None:
        dV = R["cell_dV_s"].sum(2); dD = R["cell_dD_s"].sum(2); nn = R["cell_n"]
    else:
        dV = R["cell_dV_s"][:, :, state]; dD = R["cell_dD_s"][:, :, state]; nn = R["cell_ns"][:, :, state]
    num = dV - price * dD  # (S_, J)
    idx = boot_idx(num.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        point = num.sum(0) / nn.sum(0)
        est = num[idx].sum(1) / nn[idx].sum(1)  # (NBOOT, J)
        lo = np.nanpercentile(est, 2.5, axis=0); hi = np.nanpercentile(est, 97.5, axis=0)
    return point, lo, hi, nn.sum(0)


def cell_means(cell: np.ndarray, J: int, X: np.ndarray) -> np.ndarray:
    """Per-cell means of the columns of X (n, M) -> (J, M)."""
    cnt = np.bincount(cell, minlength=J).astype(float)
    out = np.zeros((J, X.shape[1]))
    for m in range(X.shape[1]):
        out[:, m] = np.bincount(cell, weights=X[:, m], minlength=J)
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / cnt[:, None]


def probe_primitives(pp: Pop, cellset: str, verifier: str, route: str, eps: float, delta: float, nu: np.ndarray, *, c_ell: float = 0.0,
                     basis: str = "dollars", chi: Optional[np.ndarray] = None, writer: str = "explore") -> Dict[str, Any]:
    """Per-cell primitives of Section sec:memory fitted on the probing population ``pp`` (OUT-OF-SAMPLE).

    writer 'explore' (fixed process): writes come from exploration rounds (probability ``eps``) that
    field a uniformly random model: omega_j = eps * mean_m P(accept & correct | m), psi_j = eps *
    mean_m P(accept & incorrect | m); the record model in state I (W) follows the accepted-and-correct
    (accepted-and-incorrect) mixture.  writer 'policy' (the fixed-process APPROXIMATION of the
    state-dependent ACRouter writer): every round writes if the fielded model is accepted, where the
    fielded model is the uninformed router's (epsilon-greedy prior) -- the state-independent
    approximation of Remark rem:state-dependent-writes' baseline.  The uninformed action is the
    probing prior; values / costs in every state include the exploration rounds (state-independent
    term).  ``chi`` (per-cell stationary coverage) replaces the geometric fixed point when a TTL /
    FIFO / LRU rule is analysed.
    """
    VR, DR = route_outcomes(pp, route, verifier, basis)
    cell = pp.cells[cellset]
    J = pp.J[cellset]
    ACC = pp.ACC[verifier]
    TRUE = pp.TRUE
    Mn = pp.M
    cnt = np.bincount(cell, minlength=J).astype(float)
    pAT = cell_means(cell, J, (ACC & TRUE).astype(float))
    pAF = cell_means(cell, J, (ACC & ~TRUE).astype(float))
    pT = cell_means(cell, J, TRUE.astype(float))
    pA = cell_means(cell, J, ACC.astype(float))
    Vm = cell_means(cell, J, VR)
    Cm = cell_means(cell, J, DR)
    if basis == "seconds":
        VSm = cell_means(cell, J, pp.VSEC[verifier])
    else:
        VSm = np.zeros((J, Mn))
    ar_n = np.arange(pp.n)
    VRp = VR[ar_n, pp.prior]; DRp = DR[ar_n, pp.prior]
    with np.errstate(invalid="ignore", divide="ignore"):
        vU_ex = np.bincount(cell, weights=VRp, minlength=J) / cnt
        cU_ex = np.bincount(cell, weights=DRp, minlength=J) / cnt
        qU = cell_means(cell, J, (pp.prior[:, None] == np.arange(Mn)[None, :]).astype(float))
    has_data = cnt > 0  # cells without probing tasks get no prediction (omega = psi = 0) and are left out of eta_U
    if writer == "explore":
        fW = np.full((J, Mn), eps / Mn)  # per-round probability that model m is fielded AND its outcome may be written
    else:
        fW = eps / Mn + (1 - eps) * np.nan_to_num(qU)  # the uninformed router's fielding law (epsilon-greedy prior)
    wAT = fW * np.nan_to_num(pAT); wAF = fW * np.nan_to_num(pAF)
    omega = np.where(has_data, wAT.sum(1), 0.0)
    psi = np.where(has_data, wAF.sum(1), 0.0)
    r = omega + psi
    with np.errstate(invalid="ignore", divide="ignore"):
        qI = np.where(wAT.sum(1)[:, None] > 0, wAT / wAT.sum(1)[:, None], np.nan)
        qW = np.where(wAF.sum(1)[:, None] > 0, wAF / wAF.sum(1)[:, None], np.nan)
    meanV = Vm.mean(1); meanC = Cm.mean(1)
    vI_ex = np.nansum(qI * Vm, 1); cI_ex = np.nansum(qI * Cm, 1)
    vW_ex = np.nansum(qW * Vm, 1); cW_ex = np.nansum(qW * Cm, 1)
    noI = wAT.sum(1) <= 0; noW = wAF.sum(1) <= 0
    vI_ex[noI] = np.nan; cI_ex[noI] = np.nan
    vW_ex[noW] = np.nan; cW_ex[noW] = np.nan
    if route == "serve":
        vI_ex = np.where(np.isnan(vI_ex), np.nan, 1.0); cI_ex = np.where(np.isnan(cI_ex), np.nan, c_ell)
        vW_ex = np.where(np.isnan(vW_ex), np.nan, 0.0); cW_ex = np.where(np.isnan(cW_ex), np.nan, c_ell)
    # verifier seconds: exploration rounds (FP) pay the explored model's verification time; the policy writer verifies every round
    if writer == "explore":
        vsec_s = {s: eps * VSm.mean(1) for s in ("U", "I", "W")}
    else:
        vsec_s = {"U": (fW * VSm).sum(1), "I": eps * VSm.mean(1) + (1 - eps) * np.nansum(qI * VSm, 1), "W": eps * VSm.mean(1) + (1 - eps) * np.nansum(qW * VSm, 1)}
    vsec = vsec_s["U"]
    vU = (1 - eps) * vU_ex + eps * meanV; cU = (1 - eps) * cU_ex + eps * meanC + vsec_s["U"]
    vI = (1 - eps) * vI_ex + eps * meanV; cI = (1 - eps) * cI_ex + eps * meanC + vsec_s["I"]
    vW = (1 - eps) * vW_ex + eps * meanV; cW = (1 - eps) * cW_ex + eps * meanC + vsec_s["W"]
    nu_d = np.where(has_data, nu, 0.0)
    eta_U = float((nu_d * np.nan_to_num(vU)).sum() / (nu_d * np.nan_to_num(cU)).sum())
    gI = (vI - vU) - eta_U * (cI - cU)
    gW = (vW - vU) - eta_U * (cW - cU)
    alpha = (1 - delta) * (1 - nu * r)
    gI0 = np.nan_to_num(gI); gW0 = np.nan_to_num(gW)
    live = (r > 0) & (nu > 0)  # cells that arrive in the stream and receive writes
    with np.errstate(invalid="ignore", divide="ignore"):
        if chi is None:
            istar = np.where(live, nu * omega / (1 - alpha), 0.0)
            wstar = np.where(live, nu * psi / (1 - alpha), 0.0)
        else:
            istar = np.where(live, chi * omega / r, 0.0)
            wstar = np.where(live, chi * psi / r, 0.0)
        G = np.where(live, nu * (omega * gI0 + psi * gW0) / (1 - alpha), 0.0)
        err = np.where(r > 0, psi / r, np.nan)
        thr = gI / (gI - gW)
    pred_gain = float((nu * (istar * gI0 + wstar * gW0)).sum())
    out = dict(J=J, cnt=cnt, pAT=pAT, pAF=pAF, pT=pT, pA=pA, Vm=Vm, Cm=Cm, qU=qU, omega=omega, psi=psi, r=r, qI=qI, qW=qW, vU=vU, cU=cU, vI=vI, cI=cI, vW=vW, cW=cW,
               eta_U=eta_U, gI=gI, gW=gW, alpha=alpha, istar=istar, wstar=wstar, G=G, sumG=float((nu * G).sum()), pred_gain=pred_gain, err=err, thr=thr, nu=nu,
               meanV=meanV, meanC=meanC, eps=eps, delta=delta, probe_mass=float(nu_d.sum()), has_data=has_data, writer=writer)
    if route == "cascade":
        # Proposition prop:cascade-records: the record changes only the hit probability
        pUh = (1 - eps) * (np.nan_to_num(qU) * np.nan_to_num(pT)).sum(1) + eps * pT.mean(1)
        pIh = (1 - eps) * np.nansum(qI * pT, 1) + eps * pT.mean(1)
        pWh = (1 - eps) * np.nansum(qW * pT, 1) + eps * pT.mean(1)
        pIh[noI] = np.nan; pWh[noW] = np.nan
        Cb = pp.C if basis == "dollars" else pp.LAT
        with np.errstate(invalid="ignore", divide="ignore"):
            b_cell = np.bincount(cell, weights=Cb[:, pp.opus], minlength=J) / cnt
            vb_cell = np.bincount(cell, weights=pp.S[:, pp.opus], minlength=J) / cnt
        W = eta_U * b_cell + 1.0 - vb_cell
        dI = pIh - pUh; dW = pWh - pUh
        # homogenised primitives (the proposition's hypotheses imposed): a_j record-independent, b_j, v_b cell constants
        a_cell = cU - (1 - pUh) * b_cell  # any record-independent attempt cost gives the same g; take the uninformed one
        v_hom = {s: p * 1.0 + (1 - p) * vb_cell for s, p in (("U", pUh), ("I", pIh), ("W", pWh))}
        c_hom = {s: a_cell + (1 - p) * b_cell for s, p in (("U", pUh), ("I", pIh), ("W", pWh))}
        g_hom = {s: (v_hom[s] - v_hom["U"]) - eta_U * (c_hom[s] - c_hom["U"]) for s in ("I", "W")}
        with np.errstate(invalid="ignore", divide="ignore"):
            thr_W = dI / (dI - dW)
            thr_g_hom = g_hom["I"] / (g_hom["I"] - g_hom["W"])  # g-form threshold under the proposition's hypotheses
        out.update(pUh=pUh, pIh=pIh, pWh=pWh, dI=dI, dW=dW, W_eta=W, b_cell=b_cell, vb_cell=vb_cell, gI_id=dI * W, gW_id=dW * W,
                   gI_hom=g_hom["I"], gW_hom=g_hom["W"], thr_W=thr_W, thr_g_hom=thr_g_hom,
                   a_diff_I=(cI - cU) + dI * b_cell, a_diff_W=(cW - cU) + dW * b_cell)  # record-dependent attempt-cost differences (hypothesis violation)
    if route == "serve":
        with np.errstate(invalid="ignore", divide="ignore"):
            out["gW_formula"] = eta_U * (cU - c_ell) - vU  # eq. in app:cascade-records (value 0 served at cost c_ell), exploitation part
            out["effU"] = vU / cU
    return out


def sd_chain(nu_j: float, eps: float, delta: float, pAT: np.ndarray, pAF: np.ndarray, qU: np.ndarray, Vm: np.ndarray, Cm: np.ndarray, vU: float, cU: float, eta_U: float) -> Dict[str, Any]:
    """Exact stationary law of one cell's (record state, record model) chain under state-dependent writes.

    States: U, (I, m), (W, m).  Per round: with probability nu_j the cell arrives; the router fields
    the record model (state I / W) or a prior-distributed model (state U), with epsilon-greedy uniform
    exploration; the attempt is accepted-and-correct / accepted-and-incorrect / rejected with the
    probing probabilities of the fielded model; the old record expires with probability delta and an
    accepted attempt then writes the fielded model.  Returns i*, w*, the per-round gain of the cell at
    price eta_U, and the state-conditional write probabilities omega^s, psi^s for the 2x2 recurrence.
    """
    M = len(pAT)
    nan_out = dict(istar=float("nan"), wstar=float("nan"), gain=float("nan"), omega_s={k: float("nan") for k in "UIW"}, psi_s={k: float("nan") for k in "UIW"}, gI=float("nan"), gW=float("nan"), pi=None)
    if not (np.all(np.isfinite(pAT)) and np.all(np.isfinite(pAF)) and np.all(np.isfinite(qU)) and np.all(np.isfinite(Vm)) and np.all(np.isfinite(Cm)) and np.isfinite(vU) and np.isfinite(cU) and np.isfinite(eta_U)):
        return nan_out
    nS = 1 + 2 * M
    T = np.zeros((nS, nS))
    unif = np.ones(M) / M
    for x in range(nS):
        if x == 0:
            f = eps * unif + (1 - eps) * qU
            after = [(0, 1.0)]
            T[x, 0] += 1 - nu_j
        else:
            m = (x - 1) % M
            oh = np.zeros(M); oh[m] = 1.0
            f = eps * unif + (1 - eps) * oh
            after = [(0, delta), (x, 1 - delta)]
            T[x, 0] += (1 - nu_j) * delta
            T[x, x] += (1 - nu_j) * (1 - delta)
        pat = f * pAT; paf = f * pAF; prej = (f * (1 - pAT - pAF)).sum()
        for xp, pxp in after:
            T[x, 1:1 + M] += nu_j * pxp * pat
            T[x, 1 + M:] += nu_j * pxp * paf
            T[x, xp] += nu_j * pxp * prej
    A = np.vstack([T.T - np.eye(nS), np.ones((1, nS))])
    bvec = np.zeros(nS + 1); bvec[-1] = 1.0
    try:
        pi, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    except np.linalg.LinAlgError:
        return nan_out
    pi = np.clip(pi, 0, None)
    if pi.sum() <= 0:
        return nan_out
    pi /= pi.sum()
    piI = pi[1:1 + M]; piW = pi[1 + M:]
    meanV = Vm.mean(); meanC = Cm.mean()
    v_x = np.concatenate([[vU], (1 - eps) * Vm + eps * meanV, (1 - eps) * Vm + eps * meanV])
    c_x = np.concatenate([[cU], (1 - eps) * Cm + eps * meanC, (1 - eps) * Cm + eps * meanC])
    gain = nu_j * float((pi * ((v_x - vU) - eta_U * (c_x - cU))).sum())
    # state-conditional write probabilities (fielded-model mixture given the state)
    fU = eps * unif + (1 - eps) * qU
    with np.errstate(invalid="ignore", divide="ignore"):
        fI = eps * unif + (1 - eps) * (piI / piI.sum() if piI.sum() > 0 else unif)
        fW = eps * unif + (1 - eps) * (piW / piW.sum() if piW.sum() > 0 else unif)
    om = {"U": float((fU * pAT).sum()), "I": float((fI * pAT).sum()), "W": float((fW * pAT).sum())}
    ps = {"U": float((fU * pAF).sum()), "I": float((fI * pAF).sum()), "W": float((fW * pAF).sum())}
    gI = float(((piI / piI.sum()) @ ((1 - eps) * Vm + eps * meanV) - vU) - eta_U * ((piI / piI.sum()) @ ((1 - eps) * Cm + eps * meanC) - cU)) if piI.sum() > 0 else float("nan")
    gW = float(((piW / piW.sum()) @ ((1 - eps) * Vm + eps * meanV) - vU) - eta_U * ((piW / piW.sum()) @ ((1 - eps) * Cm + eps * meanC) - cU)) if piW.sum() > 0 else float("nan")
    return dict(istar=float(piI.sum()), wstar=float(piW.sum()), gain=gain, omega_s=om, psi_s=ps, gI=gI, gW=gW, pi=pi)


def two_by_two(nu_j: float, delta: float, om: Dict[str, float], ps: Dict[str, float]) -> Tuple[float, float]:
    """Stationary (i*, w*) of the affine recurrence of Remark rem:state-dependent-writes."""
    rI = om["I"] + ps["I"]; rW = om["W"] + ps["W"]
    a11 = nu_j * (om["I"] - om["U"]) + (1 - delta) * (1 - nu_j * rI)
    a12 = nu_j * (om["W"] - om["U"])
    a21 = nu_j * (ps["I"] - ps["U"])
    a22 = nu_j * (ps["W"] - ps["U"]) + (1 - delta) * (1 - nu_j * rW)
    b1 = nu_j * om["U"]; b2 = nu_j * ps["U"]
    det = (1 - a11) * (1 - a22) - a12 * a21
    if abs(det) < 1e-15:
        return float("nan"), float("nan")
    return ((1 - a22) * b1 + a12 * b2) / det, (a21 * b1 + (1 - a11) * b2) / det


def chi_square(table: np.ndarray) -> Tuple[float, float, int]:
    """Chi-square independence test of a contingency table (empty rows / columns dropped)."""
    from scipy.stats import chi2_contingency

    tab = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan"), float("nan"), 0
    stat, p, dof, _ = chi2_contingency(tab)
    return float(stat), float(p), int(dof)


def sign_label(lo: float, hi: float) -> str:
    """'pos' / 'neg' when the CI excludes 0, else 'incl0'."""
    if lo > 0:
        return "pos"
    if hi < 0:
        return "neg"
    return "incl0"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney; ties count 1/2)."""
    from scipy.stats import rankdata

    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1 = y.sum(); n0 = (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = rankdata(s)
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ======================================================================================= Phase A2: populations from the caches


def w1_fallback(text: str, python_like: bool) -> bool:
    """ACRouter's verdict when no example can be run: python -> the extracted code parses; other
    languages / prose tasks -> non-empty response longer than 10 characters (check_syntax rule)."""
    if not (text or "").strip():
        return False
    if python_like:
        return check_syntax(sandbox.extract_code(text, "python"))
    return len(text.strip()) > 10


PYTHON_SOURCES = {"humaneval", "he", "mbpp", "bcb", "cruxeval", "lcb", "ds1000", "canitedit"}
EXEC_CLEAN_SOURCES = {"humaneval", "he", "mbpp", "hex", "bcb"}  # bcb already restricted to non-all-zero tasks


def build_pops(D: Dict[str, Any], P: Dict[str, Any], vcache: Dict[Tuple[str, str], Dict[str, Any]], jcache: Dict[Tuple[str, str], Dict[str, Any]],
               judge_meta: Dict[str, Any], gcache: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None, grok_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Pop]:
    """Replay and probing populations with verdict matrices for every verifier."""
    gcache = gcache or {}
    grok_meta = grok_meta or {}
    tasks = D["tasks"]
    models = D["models"]
    Mn = len(models)
    rm = P["rm"]
    tset = tasks.set_index("task_id")
    # DimensionBest prior fitted on the whole probing split (argmax mean score per dimension)
    PROBE_ALL_IDX = np.where(tasks.split.values == "probing")[0]
    dims = sorted(tasks.dimension.unique())
    dimbest = {}
    for d in dims:
        idx = PROBE_ALL_IDX[tasks.dimension.values[PROBE_ALL_IDX] == d]
        dimbest[d] = int(D["S"][idx].mean(0).argmax())
    prior_of = lambda ids: np.array([dimbest[tset.dimension[t]] for t in ids])

    def verdicts(ids: List[str], with_w2: bool, judges: Optional[Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        judges = judges or {}
        n = len(ids)
        acc1 = np.zeros((n, Mn), bool); acc2 = np.zeros((n, Mn), bool)
        accj = {name: np.zeros((n, Mn), bool) for name in judges}
        sec1 = np.zeros((n, Mn)); sec2 = np.zeros((n, Mn))
        stats = {"w1_fallback_pairs": 0, "imputed_empty_true": 0, "empty_pairs": 0, "judge_missing_pairs": {name: 0 for name in judges}}
        for i, t in enumerate(ids):
            src = t.split("_")[1]
            py = src in PYTHON_SOURCES or (src == "hex" and t.split("_")[2] == "python")
            for k, m in enumerate(models):
                row = vcache.get((t, m))
                text = rm.get(t, {}).get(m, "") or ""
                empty = not text.strip()
                truth = bool(D["S"][D["tix"][t], k] >= 1)
                stats["empty_pairs"] += int(empty)
                if empty:
                    # unverifiable (data gap in response_matrix): every verifier is imputed the released truth
                    acc1[i, k] = truth; acc2[i, k] = truth
                    for name in judges:
                        accj[name][i, k] = truth
                    stats["imputed_empty_true"] += int(truth)
                    continue
                if row is not None:
                    acc1[i, k] = bool(row["w1_accept"]); sec1[i, k] = float(row.get("w1_sec") or 0.0)
                    acc2[i, k] = bool(row["w2_pass"]); sec2[i, k] = float(row.get("w2_sec") or 0.0)
                else:
                    acc1[i, k] = w1_fallback(text, py); stats["w1_fallback_pairs"] += 1
                    acc2[i, k] = truth  # W2 undefined here (not used for these populations)
                for name, jc in judges.items():
                    jr = jc.get((t, m))
                    if jr is None:
                        accj[name][i, k] = False
                        stats["judge_missing_pairs"][name] += 1
                    else:
                        accj[name][i, k] = (jr.get("verdict") == "correct")
        ACC = {"W1": acc1}
        VSEC = {"W1": sec1}
        if with_w2:
            ACC["W2"] = acc2; VSEC["W2"] = sec2
        for name in judges:
            ACC[name] = accj[name]; VSEC[name] = np.zeros((n, Mn))
        return ACC, VSEC, stats

    pops: Dict[str, Pop] = {}
    names12 = sorted(set(tset.cell12[t] for t in P["pop"]))
    c12 = {c: i for i, c in enumerate(names12)}
    clus_names = [f"k{i}" for i in range(N_CLUSTERS)]

    def cells_for(ids: List[str], cl: Dict[str, int], with21: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        cells = {"cell12": np.array([c12[tset.cell12[t]] for t in ids]), "clus50": np.array([cl[t] for t in ids])}
        names = {"cell12": names12, "clus50": clus_names}
        return cells, names

    # replay population (12 cells, 50 clusters): W0 / W1 / W2
    ACC, VSEC, st = verdicts(P["pop"], True)
    cells, names = cells_for(P["pop"], P["cl_t"])
    pops["c12"] = Pop("c12", P["pop"], D, cells, names, prior_of(P["pop"]), ACC, VSEC)
    pops["c12"].stats = st
    # probing population for the 12 cells / clusters
    ACCp, VSECp, stp = verdicts(P["probe"], True)
    cellsp, namesp = cells_for(P["probe"], P["cl_p"])
    pops["p12"] = Pop("p12", P["probe"], D, cellsp, namesp, prior_of(P["probe"]), ACCp, VSECp)
    pops["p12"].stats = stp
    # judged subsample (W3) and its probing counterpart
    if judge_meta and judge_meta.get("test_tasks"):
        jt = [t for t in judge_meta["test_tasks"] if t in set(P["pop"])]
        jp = [t for t in judge_meta["probe_tasks"] if t in set(P["probe"])]
        ACCj, VSECj, stj = verdicts(jt, True, {"W3": jcache})
        cj, nj = cells_for(jt, P["cl_t"])
        pops["c12j"] = Pop("c12j", jt, D, cj, nj, prior_of(jt), ACCj, VSECj)
        pops["c12j"].stats = stj
        ACCjp, VSECjp, stjp = verdicts(jp, True, {"W3": jcache})
        cjp, njp = cells_for(jp, P["cl_p"])
        pops["p12j"] = Pop("p12j", jp, D, cjp, njp, prior_of(jp), ACCjp, VSECjp)
        pops["p12j"].stats = stjp
    # grok pilot (W3g) on 2 cells, same pairs judged by deepseek (W3): both judges side by side
    if grok_meta and grok_meta.get("test_tasks") and gcache:
        gt = [t for t in grok_meta["test_tasks"] if t in set(P["pop"])]
        gp = [t for t in grok_meta["probe_tasks"] if t in set(P["probe"])]
        names2 = sorted(set(tset.cell12[t] for t in gt + gp))
        c2 = {c: i for i, c in enumerate(names2)}
        for key, ids_g in (("c2g", gt), ("p2g", gp)):
            ACCg, VSECg, stg = verdicts(ids_g, True, {"W3": jcache, "W3g": gcache})
            pops[key] = Pop(key, ids_g, D, {"cell2g": np.array([c2[tset.cell12[t]] for t in ids_g])}, {"cell2g": names2}, prior_of(ids_g), ACCg, VSECg)
            pops[key].stats = stg
    # 21-cell populations (all id_test tasks; all-zero bcb / lcb / ds1000 tasks excluded as UNGRADED): W0 and W1
    excl = tasks.allzero.values & tasks.src.isin(ALLZERO_EXCLUDE_SOURCES).values
    names21 = sorted(tasks.cell12.unique())
    c21 = {c: i for i, c in enumerate(names21)}
    for key, split in (("c21", "id_test"), ("p21", "probing")):
        ids = list(tasks.task_id[(tasks.split.values == split) & ~excl])
        ACC21, VSEC21, st21 = verdicts(ids, False)
        cells21 = {"cell21": np.array([c21[tset.cell12[t]] for t in ids])}
        pops[key] = Pop(key, ids, D, cells21, {"cell21": names21}, prior_of(ids), ACC21, VSEC21)
        pops[key].stats = st21
    # sensitivity: 21 cells with the all-zero tasks included (W0 only)
    ids_all = list(tasks.task_id[tasks.split.values == "id_test"])
    pops["c21all"] = Pop("c21all", ids_all, D, {"cell21": np.array([c21[tset.cell12[t]] for t in ids_all])}, {"cell21": names21}, prior_of(ids_all), {}, {})
    ids_pall = list(tasks.task_id[tasks.split.values == "probing"])
    pops["p21all"] = Pop("p21all", ids_pall, D, {"cell21": np.array([c21[tset.cell12[t]] for t in ids_pall])}, {"cell21": names21}, prior_of(ids_pall), {}, {})
    # EXEC-CLEAN robustness pool: the sources whose sandbox agrees with the released scores >= 95.9% (Addendum)
    for key, base in (("c9", "c12"), ("p9", "p12")):
        b = pops[base]
        keep = [i for i, t in enumerate(b.ids) if t.split("_")[1] in EXEC_CLEAN_SOURCES]
        ids9 = [b.ids[i] for i in keep]
        names9 = sorted(set(tset.cell12[t] for t in ids9))
        c9 = {c: i for i, c in enumerate(names9)}
        pops[key] = Pop(key, ids9, D, {"cell9": np.array([c9[tset.cell12[t]] for t in ids9])}, {"cell9": names9}, b.prior[keep],
                        {k: v[keep] for k, v in b.ACC.items() if k != "W0"}, {k: v[keep] for k, v in b.VSEC.items() if k != "W0"})
        pops[key].stats = {"note": f"EXEC-CLEAN subset of {base} (sources {sorted(EXEC_CLEAN_SOURCES)})"}
    # always-Opus prior (the paper's reference B and ACRouter's escalation target) as the second uninformed action
    opus = D["mix"]["claude-opus-4-6"]
    for key in ("c12", "p12", "c21", "p21"):
        base = pops[key]
        po = Pop(key + "o", base.ids, D, base.cells, base.cell_names, np.full(base.n, opus), {k: v for k, v in base.ACC.items() if k != "W0"}, {k: v for k, v in base.VSEC.items() if k != "W0"})
        po.stats = getattr(base, "stats", None)
        pops[key + "o"] = po
    pops["_dimbest"] = {d: models[m] for d, m in dimbest.items()}  # type: ignore[assignment]
    return pops


def probe_pop_for(pops: Dict[str, Pop], pk: str, verifier: str) -> Pop:
    """Probing population matching a replay population (the judged probing subsample only for W3)."""
    if pk == "c12j":
        return pops["p12j"] if verifier == "W3" else pops["p12"]
    return pops[PROBE_OF[pk]]


# ======================================================================================= Phase D/E: orchestration

FIG_COLORS = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", red="#e34948", gray="#898781", grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e")
VCOL = {"W0": FIG_COLORS["blue"], "W1": FIG_COLORS["orange"], "W2": FIG_COLORS["aqua"], "W3": FIG_COLORS["gray"], "W3g": "#4a3aa7"}
PROBE_OF = {"c12": "p12", "c21": "p21", "c12j": "p12j", "c21all": "p21all", "c12o": "p12o", "c21o": "p21o", "c9": "p9", "c2g": "p2g"}


def setup_mpl():
    """Paper figure style (9-pt serif, mathtext, no usetex, recessive axes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "text.usetex": False, "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
                         "lines.linewidth": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": FIG_COLORS["axis"], "xtick.color": FIG_COLORS["ink2"],
                         "ytick.color": FIG_COLORS["ink2"], "axes.labelcolor": FIG_COLORS["ink"], "grid.color": FIG_COLORS["grid"], "grid.linewidth": 0.6,
                         "legend.frameon": False})
    return plt


def coverage_formula(retention: str, rpar: float, nu: np.ndarray, r: np.ndarray) -> Optional[np.ndarray]:
    """Stationary coverage of eq:ttl / eq:fifo for the fixed process (None for geometric expiration)."""
    x = nu * r
    if retention == "ttl":
        return 1 - (1 - x) ** int(rpar)
    if retention == "fifo":
        beta = x.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(beta > 0, x / beta, 0.0)
        return 1 - (1 - q) ** int(rpar)
    return None


def coverage_substitution(rpar_M: int, nu: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The mean-span substitution E -> M / beta in eq:ttl (Appendix app:retention; under-estimates FIFO)."""
    x = nu * r
    beta = x.sum()
    return 1 - (1 - x) ** (rpar_M / beta) if beta > 0 else np.zeros_like(x)


def realized_coverage(R: Dict[str, Any]) -> np.ndarray:
    """Per-cell fraction of late-window arrivals that found a valid record."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return R["cell_ns"][:, :, 1:].sum((0, 2)) / R["cell_n"].sum(0)


def realized_states(R: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell realized stationary probabilities of an informative / erroneous record at arrival."""
    with np.errstate(invalid="ignore", divide="ignore"):
        n = R["cell_n"].sum(0)
        return R["cell_ns"][:, :, 1].sum(0) / n, R["cell_ns"][:, :, 2].sum(0) / n


def measured_write_probs(R: Dict[str, Any], j: int) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """State-conditional (omega^s, psi^s) and pooled (omega, psi) write probabilities per arrival of cell j."""
    tab = R["sw_cnt"][j]  # (state, wtype)
    om: Dict[str, float] = {}; ps: Dict[str, float] = {}
    for s, name in enumerate(("U", "I", "W")):
        tot = tab[s].sum()
        om[name] = float(tab[s, 1] / tot) if tot > 0 else float("nan")
        ps[name] = float(tab[s, 2] / tot) if tot > 0 else float("nan")
    tot = tab.sum()
    return om, ps, float(tab[:, 1].sum() / tot) if tot > 0 else float("nan"), float(tab[:, 2].sum() / tot) if tot > 0 else float("nan")


def ledger_spend(tag: str = JUDGE_TAG) -> Dict[str, Any]:
    """Units and tokens charged to this experiment's tag in the shared ledger (read-only), by model."""
    path = os.path.join(EX, "common", "cache", "usage_ledger.jsonl")
    out: Dict[str, Any] = {"rows": 0, "live_calls": 0, "units_est": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("tag") != tag:
                continue
            m = r.get("model", "?")
            bm = out["by_model"].setdefault(m, {"rows": 0, "live_calls": 0, "units_est": 0.0, "prompt_tokens": 0, "completion_tokens": 0})
            for d in (out, bm):
                d["rows"] += 1
                d["live_calls"] += int(not r.get("cached"))
                d["units_est"] += float(r.get("units_est") or 0.0)
                d["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
                d["completion_tokens"] += int(r.get("completion_tokens") or 0)
    return out


def fmt(x: Any, nd: int = 3) -> str:
    """Compact number formatting for tables."""
    if x is None:
        return "-"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "nan"
    if abs(xf) >= 1000:
        return f"{xf:,.0f}"
    return f"{xf:.{nd}f}"


def main_all(skip_judge: bool = False, skip_sandbox: bool = False) -> None:
    """Phases A-E (the sandbox and judge caches are filled if missing)."""
    t_all = time.time()
    P = main_phase_sandbox(limit=0 if skip_sandbox else None)
    D = P["D"]
    judge_meta_path = os.path.join(CACHE, "judge_meta.json")
    judge_meta: Dict[str, Any] = {}
    grok_meta: Dict[str, Any] = {}
    if skip_judge:
        if os.path.exists(judge_meta_path):
            judge_meta = json.load(open(judge_meta_path))
        else:  # subsample definition without any call
            ts, ps = judge_subsample(P, np.random.default_rng(SEED + 13))
            judge_meta = {"model": JUDGE_MODEL, "thinking": False, "test_tasks": ts, "probe_tasks": ps, "calls_made": 0, "skipped": True}
        if os.path.exists(GROK_META):
            grok_meta = json.load(open(GROK_META))
    else:
        try:
            judge_meta = run_judge_phase(P)
            json.dump(judge_meta, open(judge_meta_path, "w"), indent=1)
        except Exception as e:  # the judge is optional: report and continue
            log(f"judge phase skipped/failed: {type(e).__name__}: {e}")
            if os.path.exists(judge_meta_path):
                judge_meta = json.load(open(judge_meta_path))
        try:
            grok_meta = run_grok_pilot(P, judge_meta)
            json.dump(grok_meta, open(GROK_META, "w"), indent=1)
        except Exception as e:
            log(f"grok pilot skipped/failed: {type(e).__name__}: {e}")
            if os.path.exists(GROK_META):
                grok_meta = json.load(open(GROK_META))
    vcache = load_verdict_cache()
    jcache = load_judge_cache()
    gcache = {k: v for k, v in load_judge_cache(GROK_CACHE).items() if v.get("verdict") in ("correct", "incorrect")}
    pops = build_pops(D, P, vcache, jcache, judge_meta, gcache, grok_meta)
    models = D["models"]
    RES: Dict[str, Any] = {"experiment_id": "X13_memory", "seed": SEED, "streams": N_STREAMS, "rounds": N_ROUNDS, "late_window": LATE, "eps_fp": EPS_FP, "eps_sd": EPS_SD,
                           "nboot": NBOOT, "config": {}, "tables": {}, "claims": []}
    RES["config"]["dimbest_probing"] = pops["_dimbest"]
    RES["config"]["populations"] = {k: {"n": v.n, "J": v.J, "verifiers": sorted(v.ACC), "stats": getattr(v, "stats", None)} for k, v in pops.items() if isinstance(v, Pop)}
    RES["config"]["clusters"] = P["kinfo"]
    RES["config"]["judge"] = {k: v for k, v in judge_meta.items() if k not in ("test_tasks", "probe_tasks")}
    RES["config"]["judge"]["n_test_tasks"] = len(judge_meta.get("test_tasks", [])); RES["config"]["judge"]["n_probe_tasks"] = len(judge_meta.get("probe_tasks", []))
    RES["config"]["judge_grok"] = {k: v for k, v in grok_meta.items() if k not in ("test_tasks", "probe_tasks")}
    RES["config"]["judge_grok"]["n_test_tasks"] = len(grok_meta.get("test_tasks", [])); RES["config"]["judge_grok"]["n_probe_tasks"] = len(grok_meta.get("probe_tasks", []))
    RES["config"]["judge_grok"]["test_tasks"] = grok_meta.get("test_tasks", []); RES["config"]["judge_grok"]["probe_tasks"] = grok_meta.get("probe_tasks", [])
    if gcache:
        yg = np.array([float(r["truth"]) >= 1 for r in gcache.values()]); sg = np.array([float(r["p_correct"]) for r in gcache.values()])
        RES["config"]["judge_grok"]["own_check"] = {"n_pairs": int(len(yg)), "auroc": auroc(yg, sg), "accuracy": float(np.mean((sg >= 0.5) == yg))}
        # deepseek on the same pairs
        same = [(k, jcache[k]) for k in gcache if k in jcache]
        if same:
            ys = np.array([float(r["truth"]) >= 1 for _, r in same]); ss = np.array([float(r["p_correct"]) for _, r in same])
            RES["config"]["judge_grok"]["deepseek_on_same_pairs"] = {"n_pairs": int(len(ys)), "auroc": auroc(ys, ss), "accuracy": float(np.mean((ss >= 0.5) == ys))}
        all_g = load_judge_cache(GROK_CACHE)
        RES["config"]["judge_grok"]["n_error_rows_in_cache"] = sum(1 for r in all_g.values() if r.get("verdict") not in ("correct", "incorrect"))

    # ------------------------------------------------------------------ verifier diagnostics (population c12; probing p12)
    vrows = []
    for pk in ("c12", "p12", "c12j", "p12j", "c2g", "p2g"):
        if pk not in pops:
            continue
        pop = pops[pk]
        src = np.array([t.split("_")[1] for t in pop.ids])
        empty = np.array([[not (P["rm"].get(t, {}).get(m, "") or "").strip() for m in models] for t in pop.ids])
        for v in sorted(pop.ACC):
            if v == "W0":
                continue
            for s in ["all"] + sorted(set(src)):
                mask = np.ones(pop.n, bool) if s == "all" else (src == s)
                A = pop.ACC[v][mask]; Tt = pop.TRUE[mask]; E_ = empty[mask]
                ok = ~E_
                n = int(ok.sum())
                if n == 0:
                    continue
                tpr = float((A & Tt & ok).sum() / max((Tt & ok).sum(), 1)); fpr = float((A & ~Tt & ok).sum() / max((~Tt & ok).sum(), 1))
                acc_share_wrong = float((A & ~Tt & ok).sum() / max((A & ok).sum(), 1))
                vrows.append(dict(pop=pk, verifier=v, source=s, pairs=n, truth_rate=float((Tt & ok).sum() / n), accept_rate=float((A & ok).sum() / n), TPR=tpr, FPR=fpr,
                                  agreement=float(((A == Tt) & ok).sum() / n), false_accept_share=acc_share_wrong, empty_pairs=int(E_.sum())))
    RES["tables"]["verifiers"] = vrows
    # W3 own check: AUROC of p_correct vs truth on the judged pairs (non-empty)
    if jcache:
        y = []; s = []
        for (t, m), r in jcache.items():
            if r.get("p_correct") is None:
                continue
            y.append(float(r["truth"]) >= 1); s.append(float(r["p_correct"]))
        RES["config"]["judge"]["own_check"] = {"n_pairs": len(y), "auroc": auroc(np.array(y), np.array(s)), "accuracy": float(np.mean((np.array(s) >= 0.5) == np.array(y))) if y else None}
        x8 = os.path.join(EX, "X8_operating_point", "cache", "calibration.json")
        if os.path.exists(x8):
            try:
                cal = json.load(open(x8))
                RES["config"]["judge"]["x8_calibration"] = {k: {kk: v.get(kk) for kk in ("model", "thinking", "auroc", "auroc_ci", "accuracy", "mean_total_tokens")} for k, v in cal.get("configs", {}).items()}
            except Exception:
                pass

    # ------------------------------------------------------------------ settings grid
    def lru_caps(J: int) -> List[int]:
        return [int(math.ceil(J / 3)), int(math.ceil(2 * J / 3))]

    def retention_grid(J: int, full: bool = True) -> List[Tuple[str, float]]:
        g: List[Tuple[str, float]] = [("geo", 0.0), ("geo", 0.01)]
        if full:
            g += [("ttl", e) for e in TTL_GRID] + [("fifo", m) for m in FIFO_GRID] + [("lru", k) for k in lru_caps(J)]
        else:
            g += [("ttl", 200), ("fifo", 1000), ("lru", lru_caps(J)[0])]
        return g

    settings: List[Dict[str, Any]] = []

    def add(pool: str, pk: str, cs: str, v: str, mode: str, ret: str, rp: float, route: str = "direct", c_ell: float = 0.0, k: int = 1, basis: str = "dollars", stream: str = "iid", eps: Optional[float] = None) -> None:
        settings.append(dict(pool=pool, pop=pk, cellset=cs, verifier=v, mode=mode, retention=ret, rpar=rp, route=route, c_ell=c_ell, k_readout=k, basis=basis, stream=stream, eps=eps))

    for pk, cellsets, full in (("c12", ["cell12", "clus50"], True), ("c21", ["cell21"], True), ("c12j", ["cell12"], True), ("c12o", ["cell12", "clus50"], True), ("c21o", ["cell21"], True), ("c9", ["cell9"], False), ("c2g", ["cell2g"], False)):
        if pk not in pops:
            continue
        for cs in cellsets:
            J = pops[pk].J[cs]
            pilot = pk == "c2g"  # judge pilot: 12 test / 6 probing tasks -> reported, not counted in the claim pools
            for v in sorted(pops[pk].ACC):
                for ret, rp in retention_grid(J, full=full):
                    add("PILOT" if pilot else "C1", pk, cs, v, "FP", ret, rp)
                for ret, rp in retention_grid(J, full=False):
                    add("PILOT_SD" if pilot else "SD", pk, cs, v, "SD", ret, rp)
    for pk in ("c12", "c12o"):
        for v in sorted(pops[pk].ACC):
            for ret, rp in (("geo", 0.0), ("geo", 0.01), ("ttl", 200), ("fifo", 1000)):
                add("C1", pk, "cell12", v, "FP", ret, rp, route="cascade")
        for v in ("W1", "W2"):
            for ret, rp in (("geo", 0.0), ("geo", 0.01)):
                add("C1", pk, "cell12", v, "FP", ret, rp, route="cascade_v")
    mean_prior_cost = float(pops["c12"].C[np.arange(pops["c12"].n), pops["c12"].prior].mean())
    for v in ("W0", "W1", "W2"):
        for ret, rp in (("geo", 0.0), ("geo", 0.01)):
            for ce in (0.0, 0.05 * mean_prior_cost):
                add("SERVE", "c12", "cell12", v, "FP", ret, rp, route="serve", c_ell=ce)
    for ret, rp in (("geo", 0.0), ("geo", 0.01)):
        add("SENS", "c21all", "cell21", "W0", "FP", ret, rp)
    for v in ("W1", "W2"):
        add("DIAG", "c12", "clus50", v, "FP", "geo", 0.0, k=5)
        for pk in ("c12", "c12o"):
            for ret, rp in (("geo", 0.0), ("geo", 0.01), ("fifo", 1000)):
                add("C1", pk, "cell12", v, "FP", ret, rp, basis="seconds")
            add("SD", pk, "cell12", v, "SD", "geo", 0.01, basis="seconds")
    # transient check of eq:finite-memory: one random pass over the population (50 orders), empty store
    for pk in ("c12", "c12o", "c21", "c21o"):
        for v in sorted(pops[pk].ACC):
            for ret, rp in (("geo", 0.0), ("geo", 0.01)):
                add("TRANS", pk, "cell12" if pk.startswith("c12") else "cell21", v, "FP", ret, rp, stream="perm")
    # high write-rate coverage streams (every arrival's random attempt is verified and written: beta = mean accept rate ~ 0.5,
    # the regime of the example in app:retention where the mean-span substitution is visibly below eq:fifo)
    for cs in ("cell12", "clus50"):
        for v in ("W0", "W1", "W2"):
            add("COV", "c12", cs, v, "FP", "geo", 0.0, eps=1.0)
    for v in ("W0", "W1"):
        add("COV", "c21", "cell21", v, "FP", "geo", 0.0, eps=1.0)
    log(f"{len(settings)} settings to replay")

    # ------------------------------------------------------------------ run
    rows: List[Dict[str, Any]] = []
    RRES: Dict[int, Dict[str, Any]] = {}
    PRIM: Dict[int, Dict[str, Any]] = {}
    for si, s in enumerate(settings):
        pop = pops[s["pop"]]; pp = probe_pop_for(pops, s["pop"], s["verifier"])
        cs = s["cellset"]; v = s["verifier"]
        want_cov = (s["mode"] == "FP" and s["retention"] == "geo" and s["rpar"] == 0.0 and s["route"] == "direct" and s["k_readout"] == 1 and s["basis"] == "dollars" and s["stream"] == "iid")
        t0 = time.time()
        R = replay(pop, cs, v, s["mode"], s["retention"], s["rpar"], route=s["route"], c_ell=s["c_ell"], k_readout=s["k_readout"], basis=s["basis"], want_cov=want_cov, stream=s["stream"], eps_override=s.get("eps"))
        dt = time.time() - t0
        nu = pop.nu(cs)
        eps = s["eps"] if s.get("eps") is not None else (EPS_FP if s["mode"] == "FP" else EPS_SD)
        writer = "explore" if s["mode"] == "FP" else "policy"
        delta = s["rpar"] if s["retention"] == "geo" else 0.0
        pr0 = probe_primitives(pp, cs, v, s["route"], eps, delta, nu, c_ell=s["c_ell"], basis=s["basis"], writer=writer)
        chi_real = realized_coverage(R)
        if s["retention"] in ("ttl", "fifo"):
            chi = coverage_formula(s["retention"], s["rpar"], nu, pr0["r"])
        elif s["retention"] == "lru":
            chi = np.nan_to_num(chi_real)
        else:
            chi = None
        pr = probe_primitives(pp, cs, v, s["route"], eps, delta, nu, c_ell=s["c_ell"], basis=s["basis"], chi=chi, writer=writer)
        PRIM[si] = pr; RRES[si] = R
        d_eta, lo, hi, eta_inf, eta_U_real = eta_diff_streams(R)
        d_eta_t, lo_t, hi_t = eta_diff_tasks(R)
        g_real, g_lo, g_hi = gain_per_round(R, pr["eta_U"])
        pred = pr["pred_gain"]
        fh_pred = finite_horizon_pred(pr, R["rounds"])
        fh_real, fh_lo, fh_hi = finite_horizon_real(R, pr["eta_U"])
        fh_scale = R["rounds"] * float((nu * np.abs(pr["G"])).sum())
        row = dict(s, id=si, seconds=round(dt, 2), J=pop.J[cs], n_tasks=pop.n, prior=("opus" if s["pop"].endswith("o") else "dimbest"), probe_mass=pr["probe_mass"], eta_U_pred=pr["eta_U"], eta_U_real=eta_U_real, eta_inf_real=eta_inf,
                   d_eta=d_eta, d_eta_lo=lo, d_eta_hi=hi, d_eta_sign=sign_label(lo, hi), d_eta_task=d_eta_t, d_eta_task_lo=lo_t, d_eta_task_hi=hi_t, d_eta_task_sign=sign_label(lo_t, hi_t),
                   pred_gain=pred, pred_sign="pos" if pred >= 0 else "neg", sumG_geo=pr["sumG"], gain_real=g_real, gain_lo=g_lo, gain_hi=g_hi, gain_sign=sign_label(g_lo, g_hi),
                   fh_pred=fh_pred, fh_real=fh_real, fh_lo=fh_lo, fh_hi=fh_hi, fh_cover=bool(fh_lo <= fh_pred <= fh_hi), fh_norm_err=(abs(fh_real - fh_pred) / fh_scale if fh_scale > 0 else None), rounds=R["rounds"],
                   coverage_real_mean=float(np.nansum(nu * np.nan_to_num(chi_real))), coverage_pred_mean=float((nu * (pr["istar"] + pr["wstar"])).sum()),
                   share_I_real=float(R["cell_ns"][:, :, 1].sum() / R["cell_n"].sum()), share_W_real=float(R["cell_ns"][:, :, 2].sum() / R["cell_n"].sum()),
                   naive_omega_gt_psi="pos" if float((nu * pr["omega"]).sum()) >= float((nu * pr["psi"]).sum()) else "neg",
                   writes_total=R["n_writes_total"], writes_per_stream_min=int(R["n_writes_stream"].min()))
        row["agree"] = (row["pred_sign"] == row["d_eta_sign"]) if row["d_eta_sign"] != "incl0" else None
        row["fh_sign_real"] = sign_label(fh_lo, fh_hi)
        row["fh_agree"] = ((fh_pred >= 0) == (fh_real >= 0)) if row["fh_sign_real"] != "incl0" else None
        if s["retention"] == "geo" and s["route"] == "direct" and s["basis"] == "dollars" and s["k_readout"] == 1 and s["mode"] == "FP":
            # IN-SAMPLE mechanism check: primitives from the replay population itself (the identity with its own primitives)
            pr_in = probe_primitives(pop, cs, v, s["route"], eps, delta, nu, c_ell=s["c_ell"], basis=s["basis"], writer=writer)
            fh_pred_in = finite_horizon_pred(pr_in, R["rounds"])
            fh_real_in, fh_lo_in, fh_hi_in = finite_horizon_real(R, pr_in["eta_U"])
            g_in, g_lo_in, g_hi_in = gain_per_round(R, pr_in["eta_U"])
            sc_in = R["rounds"] * float((nu * np.abs(pr_in["G"])).sum())
            row.update(fh_pred_in=fh_pred_in, fh_real_in=fh_real_in, fh_lo_in=fh_lo_in, fh_hi_in=fh_hi_in, fh_cover_in=bool(fh_lo_in <= fh_pred_in <= fh_hi_in),
                       fh_norm_err_in=(abs(fh_real_in - fh_pred_in) / sc_in if sc_in > 0 else None), pred_gain_in=pr_in["pred_gain"], gain_real_in=g_in, gain_lo_in=g_lo_in, gain_hi_in=g_hi_in,
                       gain_cover_in=bool(g_lo_in <= pr_in["pred_gain"] <= g_hi_in), gain_cover=bool(g_lo <= pred <= g_hi))
        rows.append(row)
        if si % 10 == 0 or si == len(settings) - 1:
            log(f"  setting {si + 1}/{len(settings)} {s['pop']}/{cs}/{v}/{s['mode']}/{s['route']}/{s['retention']}{s['rpar']}: pred {pred:+.4f} realized d_eta {d_eta:+.3f} [{lo:+.3f},{hi:+.3f}] ({dt:.1f}s)")
    RES["tables"]["settings"] = rows

    # ------------------------------------------------------------------ (i) C1: stationary sign criterion
    def agreement(rs: List[Dict[str, Any]], key_pred: str = "pred_sign", key_real: str = "d_eta_sign") -> Dict[str, Any]:
        dec = [r for r in rs if r[key_real] != "incl0"]
        n = len(dec)
        ag = sum(1 for r in dec if r[key_pred] == r[key_real])
        pos = sum(1 for r in dec if r[key_real] == "pos")
        return dict(n_settings=len(rs), n_decided=n, agree=ag, rate=(ag / n if n else None), naive_always_pos=(pos / n if n else None), naive_always_neg=((n - pos) / n if n else None),
                    naive_omega_gt_psi=(sum(1 for r in dec if r["naive_omega_gt_psi"] == r[key_real]) / n if n else None))

    c1_rows = [r for r in rows if r["pool"] == "C1"]
    c1 = {"all": agreement(c1_rows), "task_bootstrap": agreement(c1_rows, key_real="d_eta_task_sign")}
    for key in ("verifier", "pop", "prior", "cellset", "retention", "route", "basis"):
        c1[f"by_{key}"] = {str(k): agreement([r for r in c1_rows if r[key] == k]) for k in sorted(set(r[key] for r in c1_rows))}
    c1["by_delta"] = {str(d): agreement([r for r in c1_rows if r["retention"] == "geo" and r["rpar"] == d]) for d in DELTAS}
    c1["serve"] = agreement([r for r in rows if r["pool"] == "SERVE"])
    c1["sd_naive_fp_prediction"] = agreement([r for r in rows if r["pool"] == "SD"])
    c1["pilot_c2g"] = agreement([r for r in rows if r["pool"] == "PILOT"])
    c1["pilot_c2g_by_verifier"] = {str(k): agreement([r for r in rows if r["pool"] == "PILOT" and r["verifier"] == k]) for k in sorted(set(r["verifier"] for r in rows if r["pool"] == "PILOT"))}
    c1["all_including_pilot"] = agreement(c1_rows + [r for r in rows if r["pool"] == "PILOT"])
    # finite-horizon identity eq:finite-memory: iid streams over all rounds, and one random pass (TRANS pool)
    def fh_summary(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        dec = [r for r in rs if r["fh_agree"] is not None]
        return dict(n_settings=len(rs), n_decided=len(dec), sign_agree=(float(np.mean([r["fh_agree"] for r in dec])) if dec else None),
                    ci_covers_prediction=(float(np.mean([r["fh_cover"] for r in rs])) if rs else None),
                    median_norm_err=(float(np.median([r["fh_norm_err"] for r in rs if r["fh_norm_err"] is not None])) if rs else None))

    c1["finite_horizon_iid_geo"] = fh_summary([r for r in c1_rows if r["retention"] == "geo" and r["route"] == "direct" and r["basis"] == "dollars"])
    c1["finite_horizon_one_pass"] = fh_summary([r for r in rows if r["pool"] == "TRANS"])

    def fh_in_summary(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        rs = [r for r in rs if "fh_pred_in" in r]
        return dict(n_settings=len(rs), ci_covers_prediction=(float(np.mean([r["fh_cover_in"] for r in rs])) if rs else None),
                    median_norm_err=(float(np.median([r["fh_norm_err_in"] for r in rs if r["fh_norm_err_in"] is not None])) if rs else None),
                    stationary_gain_ci_covers_prediction=(float(np.mean([r["gain_cover_in"] for r in rs])) if rs else None),
                    stationary_gain_ci_covers_prediction_oos=(float(np.mean([r["gain_cover"] for r in rs])) if rs else None),
                    stationary_sign_agreement=(float(np.mean([(r["pred_gain_in"] >= 0) == (r["gain_real_in"] >= 0) for r in rs if sign_label(r["gain_lo_in"], r["gain_hi_in"]) != "incl0"])) if rs else None))

    c1["insample_iid_geo"] = fh_in_summary([r for r in c1_rows if r["retention"] == "geo" and r["route"] == "direct" and r["basis"] == "dollars"])
    c1["insample_one_pass"] = fh_in_summary([r for r in rows if r["pool"] == "TRANS"])
    RES["tables"]["C1"] = c1

    # ------------------------------------------------------------------ (ii) C2: per-cell noise threshold
    cell_rows: List[Dict[str, Any]] = []
    for si, s in enumerate(settings):
        if not (s["pool"] in ("C1", "PILOT") and s["route"] == "direct" and s["retention"] == "geo" and s["mode"] == "FP"):
            continue
        pop = pops[s["pop"]]; cs = s["cellset"]; pr = PRIM[si]; R = RRES[si]
        price = R["sumVU"].sum() / R["sumDU"].sum()  # realized uninformed efficiency as the price of the realized gain
        pt, lo, hi, nn = cell_gain_realized(R, price)
        gI_r, gI_lo, gI_hi, nI = cell_gain_realized(R, price, state=1)
        gW_r, gW_lo, gW_hi, nW = cell_gain_realized(R, price, state=2)
        i_real, w_real = realized_states(R)
        for j in range(pop.J[cs]):
            if pr["r"][j] <= 0 or nn[j] == 0:
                continue
            hyp = bool(np.isfinite(pr["gI"][j]) and np.isfinite(pr["gW"][j]) and pr["gI"][j] > 0 > pr["gW"][j])
            num = pr["omega"][j] * np.nan_to_num(pr["gI"][j]) + pr["psi"][j] * np.nan_to_num(pr["gW"][j])
            cell_rows.append(dict(setting=si, pool=s["pool"], pop=s["pop"], prior=("opus" if s["pop"].endswith("o") else "dimbest"), cellset=cs, verifier=s["verifier"], delta=s["rpar"], cell=pop.cell_names[cs][j], j=j, nu=float(pr["nu"][j]), n_probe=int(pr["cnt"][j]),
                                  omega=float(pr["omega"][j]), psi=float(pr["psi"][j]), r=float(pr["r"][j]), err=float(pr["err"][j]), gI=float(pr["gI"][j]), gW=float(pr["gW"][j]), thr=float(pr["thr"][j]),
                                  hypothesis=hyp, pred_sign=("pos" if num >= 0 else "neg"), thr_rule=(("pos" if pr["err"][j] < pr["thr"][j] else "neg") if hyp else None),
                                  real_gain=float(pt[j]), real_lo=float(lo[j]), real_hi=float(hi[j]), real_sign=sign_label(lo[j], hi[j]), n_arrivals=int(nn[j]),
                                  gI_real=float(gI_r[j]), gI_real_lo=float(gI_lo[j]), gI_real_hi=float(gI_hi[j]), gW_real=float(gW_r[j]), gW_real_lo=float(gW_lo[j]), gW_real_hi=float(gW_hi[j]),
                                  i_real=float(i_real[j]), w_real=float(w_real[j]), i_pred=float(pr["istar"][j]), w_pred=float(pr["wstar"][j]),
                                  naive_majority=("pos" if pr["err"][j] < 0.5 else "neg")))
    RES["tables"]["cells"] = cell_rows

    def cell_agreement(rs: List[Dict[str, Any]], pred_key: str = "pred_sign") -> Dict[str, Any]:
        dec = [r for r in rs if r["real_sign"] != "incl0" and r.get(pred_key) is not None]
        n = len(dec); ag = sum(1 for r in dec if r[pred_key] == r["real_sign"]); pos = sum(1 for r in dec if r["real_sign"] == "pos")
        return dict(n_cells=len(rs), n_decided=n, agree=ag, rate=(ag / n if n else None), naive_always_pos=(pos / n if n else None), naive_always_neg=((n - pos) / n if n else None),
                    naive_majority=(sum(1 for r in dec if r["naive_majority"] == r["real_sign"]) / n if n else None))

    main_rows = [r for r in cell_rows if r["pool"] == "C1"]
    pilot_rows = [r for r in cell_rows if r["pool"] == "PILOT"]
    hyp_rows = [r for r in main_rows if r["hypothesis"]]
    c2 = {"threshold_rule_on_hypothesis_cells": cell_agreement(hyp_rows, "thr_rule"), "general_sign_all_cells": cell_agreement(main_rows, "pred_sign"),
          "n_cells_total": len(main_rows), "n_hypothesis_cells": len(hyp_rows), "hypothesis_share": (len(hyp_rows) / len(main_rows) if main_rows else None),
          "hypothesis_pattern": {}, "pilot_threshold_rule": cell_agreement([r for r in pilot_rows if r["hypothesis"]], "thr_rule"), "pilot_general_sign": cell_agreement(pilot_rows, "pred_sign")}
    for r in main_rows:
        pat = ("I+" if r["gI"] > 0 else "I-") + ("W+" if r["gW"] > 0 else "W-") if np.isfinite(r["gI"]) and np.isfinite(r["gW"]) else "undefined"
        c2["hypothesis_pattern"][pat] = c2["hypothesis_pattern"].get(pat, 0) + 1
    c2["hypothesis_cells_decided_negative"] = sum(1 for r in hyp_rows if r["real_sign"] == "neg")
    c2["hypothesis_cells_decided_positive"] = sum(1 for r in hyp_rows if r["real_sign"] == "pos")
    c2["hypothesis_cells_above_threshold"] = sum(1 for r in hyp_rows if r["err"] >= r["thr"])
    for key in ("verifier", "cellset", "pop", "prior"):
        c2[f"by_{key}"] = {str(k): cell_agreement([r for r in hyp_rows if r[key] == k], "thr_rule") for k in sorted(set(r[key] for r in hyp_rows))}
        c2[f"general_by_{key}"] = {str(k): cell_agreement([r for r in main_rows if r[key] == k], "pred_sign") for k in sorted(set(r[key] for r in main_rows))}
    # OOS check of the state-conditional primitives: predicted vs realized g^I, g^W and record probabilities
    def mae_corr(a: List[float], b: List[float]) -> Dict[str, Any]:
        a_ = np.array(a); b_ = np.array(b); ok = np.isfinite(a_) & np.isfinite(b_)
        if ok.sum() < 3:
            return dict(n=int(ok.sum()), mae=None, spearman=None, sign_agreement=None)
        from scipy.stats import spearmanr
        return dict(n=int(ok.sum()), mae=float(np.mean(np.abs(a_[ok] - b_[ok]))), spearman=float(spearmanr(a_[ok], b_[ok]).correlation), sign_agreement=float(np.mean(np.sign(a_[ok]) == np.sign(b_[ok]))))

    c2["gI_pred_vs_real"] = mae_corr([r["gI"] for r in main_rows], [r["gI_real"] for r in main_rows])
    c2["gW_pred_vs_real"] = mae_corr([r["gW"] for r in main_rows], [r["gW_real"] for r in main_rows])
    c2["i_pred_vs_real"] = mae_corr([r["i_pred"] for r in main_rows], [r["i_real"] for r in main_rows])
    c2["w_pred_vs_real"] = mae_corr([r["w_pred"] for r in main_rows], [r["w_real"] for r in main_rows])
    RES["tables"]["C2"] = c2

    # ------------------------------------------------------------------ (iii) C3: records inside a verified cascade
    casc_rows: List[Dict[str, Any]] = []
    for si, s in enumerate(settings):
        if not (s["route"] == "cascade" and s["retention"] == "geo"):
            continue
        pop = pops[s["pop"]]; cs = s["cellset"]; pr = PRIM[si]; R = RRES[si]
        price = R["sumVU"].sum() / R["sumDU"].sum()
        gI_r, gI_lo, gI_hi, nI = cell_gain_realized(R, price, state=1)
        gW_r, gW_lo, gW_hi, nW = cell_gain_realized(R, price, state=2)
        for j in range(pop.J[cs]):
            if pr["r"][j] <= 0:
                continue
            casc_rows.append(dict(setting=si, pop=s["pop"], prior=("opus" if s["pop"].endswith("o") else "dimbest"), verifier=s["verifier"], delta=s["rpar"], cell=pop.cell_names[cs][j], W_eta=float(pr["W_eta"][j]), b=float(pr["b_cell"][j]), vb=float(pr["vb_cell"][j]),
                                  pU=float(pr["pUh"][j]), pI=float(pr["pIh"][j]), pW=float(pr["pWh"][j]), dI=float(pr["dI"][j]), dW=float(pr["dW"][j]),
                                  gI=float(pr["gI"][j]), gW=float(pr["gW"][j]), gI_id=float(pr["gI_id"][j]), gW_id=float(pr["gW_id"][j]),
                                  resid_I_identity=float(pr["gI_hom"][j] - pr["gI_id"][j]) if np.isfinite(pr["gI_hom"][j]) else None,
                                  resid_W_identity=float(pr["gW_hom"][j] - pr["gW_id"][j]) if np.isfinite(pr["gW_hom"][j]) else None,
                                  gap_I_actual=float(pr["gI"][j] - pr["gI_id"][j]) if np.isfinite(pr["gI"][j]) else None, gap_W_actual=float(pr["gW"][j] - pr["gW_id"][j]) if np.isfinite(pr["gW"][j]) else None,
                                  a_diff_I=float(pr["a_diff_I"][j]) if np.isfinite(pr["a_diff_I"][j]) else None, a_diff_W=float(pr["a_diff_W"][j]) if np.isfinite(pr["a_diff_W"][j]) else None,
                                  thr_g_actual=float(pr["thr"][j]), thr_g_hom=float(pr["thr_g_hom"][j]), thr_W=float(pr["thr_W"][j]), err=float(pr["err"][j]),
                                  hyp_W=bool(np.isfinite(pr["dI"][j]) and np.isfinite(pr["dW"][j]) and pr["W_eta"][j] > 0 and pr["dI"][j] > 0 > pr["dW"][j]),
                                  hyp_g_actual=bool(np.isfinite(pr["gI"][j]) and np.isfinite(pr["gW"][j]) and pr["gI"][j] > 0 > pr["gW"][j]),
                                  gI_real=float(gI_r[j]), gI_real_lo=float(gI_lo[j]), gI_real_hi=float(gI_hi[j]), gW_real=float(gW_r[j]), gW_real_lo=float(gW_lo[j]), gW_real_hi=float(gW_hi[j])))
    RES["tables"]["cascade_cells"] = casc_rows
    res_id = [abs(r["resid_I_identity"]) for r in casc_rows if r["resid_I_identity"] is not None] + [abs(r["resid_W_identity"]) for r in casc_rows if r["resid_W_identity"] is not None]
    hypW = [r for r in casc_rows if r["hyp_W"] and np.isfinite(r["thr_g_hom"]) and np.isfinite(r["thr_W"])]
    thr_match = [abs(r["thr_g_hom"] - r["thr_W"]) for r in hypW]
    dec_match = [((r["err"] < r["thr_g_hom"]) == (r["err"] < r["thr_W"])) for r in hypW]
    both = [r for r in casc_rows if r["hyp_W"] and r["hyp_g_actual"] and np.isfinite(r["thr_g_actual"])]
    thr_gap_actual = [abs(r["thr_g_actual"] - r["thr_W"]) for r in both]
    dec_actual = [((r["err"] < r["thr_g_actual"]) == (r["err"] < r["thr_W"])) for r in both]
    sign_I = [np.sign(r["gI"]) == np.sign(r["gI_id"]) for r in casc_rows if np.isfinite(r["gI"]) and r["gI_id"] != 0]
    sign_W = [np.sign(r["gW"]) == np.sign(r["gW_id"]) for r in casc_rows if np.isfinite(r["gW"]) and r["gW_id"] != 0]
    rel_gap_I = [abs(r["gap_I_actual"]) / max(abs(r["gI_id"]), 1e-12) for r in casc_rows if r["gap_I_actual"] is not None]
    rel_gap_W = [abs(r["gap_W_actual"]) / max(abs(r["gW_id"]), 1e-12) for r in casc_rows if r["gap_W_actual"] is not None]
    a_share = [abs(pr_eta * r["a_diff_I"]) / max(abs(r["gI"]), 1e-12) for r, pr_eta in ((r, PRIM[r["setting"]]["eta_U"]) for r in casc_rows) if r["a_diff_I"] is not None and np.isfinite(r["gI"])]
    real_vs_id_I = mae_corr([r["gI_id"] for r in casc_rows], [r["gI_real"] for r in casc_rows])
    real_vs_id_W = mae_corr([r["gW_id"] for r in casc_rows], [r["gW_real"] for r in casc_rows])
    real_vs_act_I = mae_corr([r["gI"] for r in casc_rows], [r["gI_real"] for r in casc_rows])
    real_vs_act_W = mae_corr([r["gW"] for r in casc_rows], [r["gW_real"] for r in casc_rows])
    RES["tables"]["C3"] = dict(n_cells=len(casc_rows), identity_max_residual=(max(res_id) if res_id else None), n_hypothesis_cells=len(hypW),
                               threshold_gap_max=(max(thr_match) if thr_match else None), threshold_decision_agreement=(float(np.mean(dec_match)) if dec_match else None),
                               n_cells_both_hypotheses=len(both), threshold_gap_actual_median=(float(np.median(thr_gap_actual)) if thr_gap_actual else None), threshold_gap_actual_max=(max(thr_gap_actual) if thr_gap_actual else None),
                               threshold_decision_agreement_actual=(float(np.mean(dec_actual)) if dec_actual else None),
                               sign_agreement_actual_vs_identity_I=(float(np.mean(sign_I)) if sign_I else None), sign_agreement_actual_vs_identity_W=(float(np.mean(sign_W)) if sign_W else None),
                               relative_gap_actual_I_median=(float(np.median(rel_gap_I)) if rel_gap_I else None), relative_gap_actual_W_median=(float(np.median(rel_gap_W)) if rel_gap_W else None),
                               attempt_cost_term_share_of_gI_median=(float(np.median(a_share)) if a_share else None),
                               realized_gI_vs_identity=real_vs_id_I, realized_gW_vs_identity=real_vs_id_W, realized_gI_vs_actual=real_vs_act_I, realized_gW_vs_actual=real_vs_act_W,
                               cascade_settings_sign=agreement([r for r in rows if r["route"] == "cascade"]), cascade_v_settings_sign=agreement([r for r in rows if r["route"] == "cascade_v"]))

    # ------------------------------------------------------------------ (iv) C4: coverage under TTL / FIFO
    cov_rows: List[Dict[str, Any]] = []
    cov_curves: Dict[str, Any] = {}
    for si, s in enumerate(settings):
        R = RRES[si]
        if "cov_M" not in R:
            continue
        pop = pops[s["pop"]]; cs = s["cellset"]; pr = PRIM[si]; nu = pop.nu(cs)
        J = pop.J[cs]
        eps = s["eps"] if s.get("eps") is not None else EPS_FP
        idx = boot_idx(R["cov_n"].shape[0])
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chiM = R["cov_M"].sum(0) / R["cov_nM"].sum(0); chiE = R["cov_E"].sum(0) / R["cov_n"].sum(0)[:, None]
            bM = R["cov_M"][idx].sum(1) / R["cov_nM"][idx].sum(1); bE = R["cov_E"][idx].sum(1) / R["cov_n"][idx].sum(1)[:, :, None]
            loM = np.nanpercentile(bM, 2.5, axis=0); hiM = np.nanpercentile(bM, 97.5, axis=0)
            loE = np.nanpercentile(bE, 2.5, axis=0); hiE = np.nanpercentile(bE, 97.5, axis=0)
        # realized write rate per arrival (mechanism check, in-sample)
        r_real = np.zeros(J)
        for j in range(J):
            tot = R["sw_cnt"][j].sum()
            r_real[j] = R["sw_cnt"][j][:, 1:].sum() / tot if tot > 0 else 0.0
        key = f"{s['pop']}/{cs}/{s['verifier']}" + ("/eps1" if s.get("eps") == 1.0 else "")
        cov_curves[key] = dict(cells=pop.cell_names[cs], nu=nu.tolist(), r_probe=pr["r"].tolist(), r_real=r_real.tolist(), M_grid=COV_M_GRID, E_grid=COV_E_GRID, eps=eps, beta_probe=float((nu * pr["r"]).sum()), beta_real=float((nu * r_real).sum()),
                               real_M=np.nan_to_num(chiM).tolist(), real_E=np.nan_to_num(chiE).tolist(), lo_M=np.nan_to_num(loM).tolist(), hi_M=np.nan_to_num(hiM).tolist(), n_filled_M=R["cov_nM"].sum(0).tolist(),
                               lo_E=np.nan_to_num(loE).tolist(), hi_E=np.nan_to_num(hiE).tolist(), n_arrivals=R["cov_n"].sum(0).tolist(), min_writes_per_stream=int(R["n_writes_stream"].min()))
        for mi, Mv in enumerate(COV_M_GRID):
            f_pred = coverage_formula("fifo", Mv, nu, pr["r"]); sub = coverage_substitution(Mv, nu, pr["r"])
            f_in = coverage_formula("fifo", Mv, nu, r_real); sub_in = coverage_substitution(Mv, nu, r_real)
            for j in range(J):
                if R["cov_nM"][:, j, mi].sum() == 0 or pr["r"][j] <= 0:
                    continue
                cov_rows.append(dict(kind="fifo", key=key, eps=eps, cell=pop.cell_names[cs][j], param=Mv, real=float(chiM[j, mi]), lo=float(loM[j, mi]), hi=float(hiM[j, mi]), n=float(R["cov_nM"][:, j, mi].sum()),
                                     formula=float(f_pred[j]), substitution=float(sub[j]), formula_insample=float(f_in[j]), substitution_insample=float(sub_in[j])))
        for ei, Ev in enumerate(COV_E_GRID):
            t_pred = coverage_formula("ttl", Ev, nu, pr["r"]); t_in = coverage_formula("ttl", Ev, nu, r_real)
            for j in range(J):
                if R["cov_n"][:, j].sum() == 0 or pr["r"][j] <= 0:
                    continue
                cov_rows.append(dict(kind="ttl", key=key, eps=eps, cell=pop.cell_names[cs][j], param=Ev, real=float(chiE[j, ei]), lo=float(loE[j, ei]), hi=float(hiE[j, ei]), n=float(R["cov_n"][:, j].sum()),
                                     formula=float(t_pred[j]), substitution=None, formula_insample=float(t_in[j]), substitution_insample=None))
    RES["tables"]["coverage_rows"] = cov_rows
    RES["tables"]["coverage_curves"] = cov_curves

    def cov_summary(rs: List[Dict[str, Any]], sat: float = 0.995, min_n: float = 200.0) -> Dict[str, Any]:
        """Fit statistics on unsaturated points (realized < sat) with at least min_n arrivals."""
        un = [r for r in rs if r["real"] < sat and r["n"] >= min_n]
        out = dict(n_points=len(rs), n_unsaturated=len(un))
        if rs:
            out["mae_formula"] = float(np.mean([abs(r["real"] - r["formula"]) for r in rs]))
            out["mae_formula_insample"] = float(np.mean([abs(r["real"] - r["formula_insample"]) for r in rs]))
        if un:
            out["mae_formula_unsat"] = float(np.mean([abs(r["real"] - r["formula"]) for r in un]))
            out["mae_formula_insample_unsat"] = float(np.mean([abs(r["real"] - r["formula_insample"]) for r in un]))
            out["bias_formula_unsat"] = float(np.mean([r["real"] - r["formula"] for r in un]))
            out["bias_formula_insample_unsat"] = float(np.mean([r["real"] - r["formula_insample"] for r in un]))
            out["formula_in_ci_share"] = float(np.mean([r["lo"] <= r["formula"] <= r["hi"] for r in un]))
            out["formula_insample_in_ci_share"] = float(np.mean([r["lo"] <= r["formula_insample"] <= r["hi"] for r in un]))
            if rs and rs[0]["kind"] == "fifo":
                out["mae_substitution_unsat"] = float(np.mean([abs(r["real"] - r["substitution"]) for r in un]))
                out["mae_substitution_insample_unsat"] = float(np.mean([abs(r["real"] - r["substitution_insample"]) for r in un]))
                out["substitution_underestimates_share"] = float(np.mean([r["substitution"] < r["real"] for r in un]))
                out["substitution_underestimates_share_insample"] = float(np.mean([r["substitution_insample"] < r["real"] for r in un]))
                out["substitution_below_ci_share"] = float(np.mean([r["substitution"] < r["lo"] for r in un]))
                out["substitution_insample_below_ci_share"] = float(np.mean([r["substitution_insample"] < r["lo"] for r in un]))
                out["substitution_in_ci_share"] = float(np.mean([r["lo"] <= r["substitution"] <= r["hi"] for r in un]))
                out["formula_below_real_share"] = float(np.mean([r["formula"] < r["real"] for r in un]))
                out["substitution_below_formula_share"] = float(np.mean([r["substitution"] < r["formula"] - 1e-12 for r in un]))
                out["formula_beats_substitution_share"] = float(np.mean([abs(r["real"] - r["formula"]) < abs(r["real"] - r["substitution"]) for r in un]))
                out["formula_beats_substitution_share_insample"] = float(np.mean([abs(r["real"] - r["formula_insample"]) < abs(r["real"] - r["substitution_insample"]) for r in un]))
                # detectable points: the predicted gap eq:fifo - substitution exceeds the realized CI half-width
                det = [r for r in un if (r["formula_insample"] - r["substitution_insample"]) > 0.5 * (r["hi"] - r["lo"])]
                out["n_detectable"] = len(det)
                out["max_predicted_gap"] = float(max(r["formula_insample"] - r["substitution_insample"] for r in un))
                if det:
                    out["det_formula_beats_substitution_share_insample"] = float(np.mean([abs(r["real"] - r["formula_insample"]) < abs(r["real"] - r["substitution_insample"]) for r in det]))
                    out["det_substitution_underestimates_share_insample"] = float(np.mean([r["substitution_insample"] < r["real"] for r in det]))
                    out["det_substitution_below_ci_share_insample"] = float(np.mean([r["substitution_insample"] < r["lo"] for r in det]))
                    out["det_formula_in_ci_share_insample"] = float(np.mean([r["lo"] <= r["formula_insample"] <= r["hi"] for r in det]))
                    out["det_mae_formula_insample"] = float(np.mean([abs(r["real"] - r["formula_insample"]) for r in det]))
                    out["det_mae_substitution_insample"] = float(np.mean([abs(r["real"] - r["substitution_insample"]) for r in det]))
                    out["det_formula_beats_substitution_share"] = float(np.mean([abs(r["real"] - r["formula"]) < abs(r["real"] - r["substitution"]) for r in det]))
                    out["det_substitution_underestimates_share"] = float(np.mean([r["substitution"] < r["real"] for r in det]))
        return out

    c4 = {"fifo": cov_summary([r for r in cov_rows if r["kind"] == "fifo"]), "ttl": cov_summary([r for r in cov_rows if r["kind"] == "ttl"])}
    c4["fifo_main"] = cov_summary([r for r in cov_rows if r["kind"] == "fifo" and r["eps"] < 1.0])
    c4["fifo_highbeta"] = cov_summary([r for r in cov_rows if r["kind"] == "fifo" and r["eps"] == 1.0])
    c4["ttl_main"] = cov_summary([r for r in cov_rows if r["kind"] == "ttl" and r["eps"] < 1.0])
    c4["ttl_highbeta"] = cov_summary([r for r in cov_rows if r["kind"] == "ttl" and r["eps"] == 1.0])
    c4["fifo_by_key"] = {k: cov_summary([r for r in cov_rows if r["kind"] == "fifo" and r["key"] == k]) for k in sorted(cov_curves)}
    c4["ttl_by_key"] = {k: cov_summary([r for r in cov_rows if r["kind"] == "ttl" and r["key"] == k]) for k in sorted(cov_curves)}
    c4["fifo_by_M"] = {str(m): cov_summary([r for r in cov_rows if r["kind"] == "fifo" and r["param"] == m]) for m in COV_M_GRID}
    c4["fifo_highbeta_by_M"] = {str(m): cov_summary([r for r in cov_rows if r["kind"] == "fifo" and r["param"] == m and r["eps"] == 1.0]) for m in COV_M_GRID}
    c4["ttl_by_E"] = {str(e): cov_summary([r for r in cov_rows if r["kind"] == "ttl" and r["param"] == e]) for e in COV_E_GRID}
    c4["beta"] = {k: dict(probe=v["beta_probe"], real=v["beta_real"], eps=v["eps"]) for k, v in cov_curves.items()}
    c4["fifo_grid_settings"] = {}  # realized coverage of the FIFO settings of the main grid vs eq:fifo (mean over cells, nu-weighted)
    for si, s in enumerate(settings):
        if s["retention"] in ("fifo", "ttl") and s["pool"] == "C1" and s["route"] == "direct":
            pop = pops[s["pop"]]; nu = pop.nu(s["cellset"]); pr = PRIM[si]
            chi_r = realized_coverage(RRES[si]); chi_f = coverage_formula(s["retention"], s["rpar"], nu, pr["r"])
            key = f"{s['pop']}/{s['cellset']}/{s['verifier']}/{s['retention']}{int(s['rpar'])}"
            c4["fifo_grid_settings"][key] = dict(real=float(np.nansum(nu * np.nan_to_num(chi_r))), formula=float((nu * chi_f).sum()),
                                                 substitution=(float((nu * coverage_substitution(int(s["rpar"]), nu, pr["r"])).sum()) if s["retention"] == "fifo" else None),
                                                 min_writes_per_stream=int(RRES[si]["n_writes_stream"].min()))
    RES["tables"]["C4"] = c4

    # ------------------------------------------------------------------ (v) C5: state-dependent writes
    sd_cells: List[Dict[str, Any]] = []
    for si, s in enumerate(settings):
        if s["pool"] not in ("SD", "C1") or s["route"] != "direct" or s["retention"] != "geo" or s["k_readout"] != 1 or s["basis"] != "dollars":
            continue
        pop = pops[s["pop"]]; cs = s["cellset"]; pr = PRIM[si]; R = RRES[si]; nu = pop.nu(cs)
        eps = EPS_FP if s["mode"] == "FP" else EPS_SD
        i_real, w_real = realized_states(R)
        n_arr = R["cell_n"].sum(0)
        for j in range(pop.J[cs]):
            if n_arr[j] == 0 or pr["r"][j] <= 0:
                continue
            stat, pval, dof = chi_square(R["sw_cnt"][j])
            om, ps, om_pool, ps_pool = measured_write_probs(R, j)
            i22, w22 = two_by_two(nu[j], s["rpar"], om, ps) if all(np.isfinite([om[k] for k in om] + [ps[k] for k in ps])) else (float("nan"), float("nan"))
            alpha_hat = (1 - s["rpar"]) * (1 - nu[j] * (om_pool + ps_pool))
            if np.isfinite(alpha_hat) and alpha_hat < 1:
                i_fp_in = nu[j] * om_pool / (1 - alpha_hat); w_fp_in = nu[j] * ps_pool / (1 - alpha_hat)
            else:
                i_fp_in = w_fp_in = float("nan")
            rec = dict(setting=si, pop=s["pop"], cellset=cs, verifier=s["verifier"], mode=s["mode"], delta=s["rpar"], cell=pop.cell_names[cs][j], nu=float(nu[j]), n_arrivals=int(n_arr[j]),
                       chi2=stat, p_value=pval, dof=dof, omega_s=om, psi_s=ps, omega_pool=om_pool, psi_pool=ps_pool, i_real=float(i_real[j]), w_real=float(w_real[j]),
                       i_fp_oos=float(pr["istar"][j]), w_fp_oos=float(pr["wstar"][j]), i_2x2=float(i22), w_2x2=float(w22), i_fp_insample=float(i_fp_in), w_fp_insample=float(w_fp_in))
            if s["mode"] == "SD":
                ch = sd_chain(nu[j], eps, s["rpar"], pr["pAT"][j], pr["pAF"][j], pr["qU"][j], pr["Vm"][j], pr["Cm"][j], float(pr["vU"][j]), float(pr["cU"][j]), pr["eta_U"])
                rec.update(i_chain=ch["istar"], w_chain=ch["wstar"], gain_chain=ch["gain"], omega_s_chain=ch["omega_s"], psi_s_chain=ch["psi_s"], gI_chain=ch["gI"], gW_chain=ch["gW"])
            rec["prior"] = "opus" if s["pop"].endswith("o") else "dimbest"
            sd_cells.append(rec)
    RES["tables"]["sd_cells"] = sd_cells

    def share(i: float, w: float) -> float:
        return w / (i + w) if (i + w) > 0 else float("nan")

    c5: Dict[str, Any] = {}
    for mode in ("SD", "FP"):
        rs = [r for r in sd_cells if r["mode"] == mode]
        pv = np.array([r["p_value"] for r in rs if np.isfinite(r["p_value"])])
        c5[mode] = dict(n_cells=len(rs), n_tested=int(len(pv)), rejected_at_0_01=int((pv < 0.01).sum()), rejected_at_0_001=int((pv < 0.001).sum()), share_rejected_0_01=(float((pv < 0.01).mean()) if len(pv) else None),
                        mae_i_fp_oos=float(np.nanmean([abs(r["i_fp_oos"] - r["i_real"]) for r in rs])) if rs else None,
                        mae_w_fp_oos=float(np.nanmean([abs(r["w_fp_oos"] - r["w_real"]) for r in rs])) if rs else None,
                        mae_i_2x2=float(np.nanmean([abs(r["i_2x2"] - r["i_real"]) for r in rs])) if rs else None,
                        mae_w_2x2=float(np.nanmean([abs(r["w_2x2"] - r["w_real"]) for r in rs])) if rs else None,
                        mae_i_fp_insample=float(np.nanmean([abs(r["i_fp_insample"] - r["i_real"]) for r in rs])) if rs else None,
                        mae_w_fp_insample=float(np.nanmean([abs(r["w_fp_insample"] - r["w_real"]) for r in rs])) if rs else None,
                        mae_share_fp_oos=float(np.nanmean([abs(share(r["i_fp_oos"], r["w_fp_oos"]) - share(r["i_real"], r["w_real"])) for r in rs])) if rs else None,
                        mae_share_2x2=float(np.nanmean([abs(share(r["i_2x2"], r["w_2x2"]) - share(r["i_real"], r["w_real"])) for r in rs])) if rs else None)
        if mode == "SD" and rs:
            c5[mode].update(mae_i_chain=float(np.nanmean([abs(r["i_chain"] - r["i_real"]) for r in rs])), mae_w_chain=float(np.nanmean([abs(r["w_chain"] - r["w_real"]) for r in rs])),
                            mae_share_chain=float(np.nanmean([abs(share(r["i_chain"], r["w_chain"]) - share(r["i_real"], r["w_real"])) for r in rs])))
    # SD settings: sign of the realized gain vs the naive fixed-process prediction and the chain prediction
    sd_set_rows = []
    for si, s in enumerate(settings):
        if s["pool"] != "SD" or s["retention"] != "geo":
            continue
        rs = [r for r in sd_cells if r["setting"] == si]
        row = next(r for r in rows if r["id"] == si)
        chain_gain = float(np.nansum([r["gain_chain"] for r in rs]))
        sd_set_rows.append(dict(id=si, pop=s["pop"], cellset=s["cellset"], verifier=s["verifier"], delta=s["rpar"], real_sign=row["d_eta_sign"], d_eta=row["d_eta"], d_eta_lo=row["d_eta_lo"], d_eta_hi=row["d_eta_hi"],
                                naive_fp_pred=row["pred_gain"], naive_fp_sign=row["pred_sign"], chain_pred=chain_gain, chain_sign=("pos" if chain_gain >= 0 else "neg"), gain_real=row["gain_real"]))
    dec = [r for r in sd_set_rows if r["real_sign"] != "incl0"]
    c5["sd_settings"] = sd_set_rows
    c5["sd_sign_naive_fp"] = (float(np.mean([r["naive_fp_sign"] == r["real_sign"] for r in dec])) if dec else None)
    c5["sd_sign_chain"] = (float(np.mean([r["chain_sign"] == r["real_sign"] for r in dec])) if dec else None)
    c5["sd_n_decided"] = len(dec)
    RES["tables"]["C5"] = c5

    # ------------------------------------------------------------------ (vi) C6: serve without verification
    serve_rows: List[Dict[str, Any]] = []
    for si, s in enumerate(settings):
        if s["route"] != "serve":
            continue
        pop = pops[s["pop"]]; cs = s["cellset"]; pr = PRIM[si]; R = RRES[si]
        price = R["sumVU"].sum() / R["sumDU"].sum()
        gW_r, gW_lo, gW_hi, nW = cell_gain_realized(R, price, state=2)
        gI_r, gI_lo, gI_hi, nI = cell_gain_realized(R, price, state=1)
        for j in range(pop.J[cs]):
            if pr["r"][j] <= 0:
                continue
            serve_rows.append(dict(setting=si, verifier=s["verifier"], delta=s["rpar"], c_ell=s["c_ell"], cell=pop.cell_names[cs][j], effU=float(pr["effU"][j]), eta_U=pr["eta_U"], cU=float(pr["cU"][j]),
                                  gW_formula=float(pr["gW_formula"][j]), gW=float(pr["gW"][j]), gI=float(pr["gI"][j]), gW_real=float(gW_r[j]), gW_real_lo=float(gW_lo[j]), gW_real_hi=float(gW_hi[j]), nW=int(nW[j]),
                                  gI_real=float(gI_r[j]), nI=int(nI[j]), hyp_fails=bool(np.isfinite(pr["gW"][j]) and pr["gW"][j] >= 0)))
    RES["tables"]["serve_cells"] = serve_rows
    c6 = {}
    for ce in sorted(set(r["c_ell"] for r in serve_rows)):
        rs = [r for r in serve_rows if r["c_ell"] == ce and r["delta"] == 0.0]
        c6[f"c_ell={ce:.5f}"] = dict(n_cells=len(rs), cells_gW_formula_nonneg=sum(1 for r in rs if r["gW_formula"] >= 0), cells_gW_pred_nonneg=sum(1 for r in rs if r["hyp_fails"]),
                                     cells_gW_real_ci_nonneg=sum(1 for r in rs if np.isfinite(r["gW_real_lo"]) and r["gW_real_lo"] >= 0), cells_gW_real_ci_neg=sum(1 for r in rs if np.isfinite(r["gW_real_hi"]) and r["gW_real_hi"] < 0),
                                     sign_agreement_pred_vs_real=(float(np.mean([(r["gW"] >= 0) == (r["gW_real"] >= 0) for r in rs if np.isfinite(r["gW_real"]) and np.isfinite(r["gW"])])) if rs else None))
    RES["tables"]["C6"] = c6

    # ------------------------------------------------------------------ claims
    def verdict(cond: Optional[bool]) -> str:
        return "PASS" if cond else ("FAIL" if cond is not None else "INCONCLUSIVE")

    a1 = c1["all"]
    fh1 = c1["finite_horizon_iid_geo"]; fh2 = c1["finite_horizon_one_pass"]
    ap1 = c1["pilot_c2g"]; ai1 = c1["all_including_pilot"]
    RES["claims"].append(dict(id="C1", statement="sign(sum_j nu_j G_j) from probing primitives equals the sign of the realized eta_inf - eta_U in >= 95% of CI-decided (verifier, retention, delta, cell set, prior, route, basis) settings of the fixed-process replay",
                              kind="SEMI-SYNTHETIC (recorded outcomes, real verifiers, constructed iid stream) / OUT-OF-SAMPLE primitives",
                              result=f"{a1['agree']}/{a1['n_decided']} decided settings agree ({fmt(a1['rate'])}); naive always-positive {fmt(a1['naive_always_pos'])}, always-negative {fmt(a1['naive_always_neg'])}, omega>psi {fmt(a1['naive_omega_gt_psi'])}; task-bootstrap decisions {fmt(c1['task_bootstrap']['rate'])} ({c1['task_bootstrap']['n_decided']} decided); by prior " + ", ".join(f"{k} {fmt(v['rate'])} ({v['n_decided']})" for k, v in c1["by_prior"].items()) + "; by verifier " + ", ".join(f"{k} {fmt(v['rate'])} ({v['n_decided']})" for k, v in c1["by_verifier"].items()) + f"; finite-horizon identity eq:finite-memory: prediction inside the realized CI in {fmt(fh1['ci_covers_prediction'])} of iid settings (median normalized error {fmt(fh1['median_norm_err'])}, sign agreement {fmt(fh1['sign_agree'])} on {fh1['n_decided']}) and in {fmt(fh2['ci_covers_prediction'])} of single random passes (median normalized error {fmt(fh2['median_norm_err'])}, sign agreement {fmt(fh2['sign_agree'])} on {fh2['n_decided']}); EXCLUDED from the pool and reported separately: the 2-cell judge pilot (12 test / 6 probing tasks, primitives from 3 probing tasks per cell) agrees in {ap1['agree']}/{ap1['n_decided']} ({fmt(ap1['rate'])}); pooled with it {ai1['agree']}/{ai1['n_decided']} ({fmt(ai1['rate'])})",
                              verdict=verdict(a1["rate"] is not None and a1["rate"] >= 0.95) if a1["n_decided"] >= 10 else "INCONCLUSIVE"))
    a2 = c2["threshold_rule_on_hypothesis_cells"]
    g2 = c2["general_sign_all_cells"]
    both_signs = c2["hypothesis_cells_decided_negative"] >= 3 and c2["hypothesis_cells_decided_positive"] >= 3
    if a2["n_decided"] >= 10 and both_signs:
        v2 = verdict(a2["rate"] is not None and a2["rate"] >= 0.90 and all(a2["rate"] >= (a2[k] or 0) for k in ("naive_always_pos", "naive_always_neg", "naive_majority")))
    else:
        v2 = "INCONCLUSIVE"
    RES["claims"].append(dict(id="C2", statement="eq:noise-threshold (psi_j/r_j < g^I/(g^I-g^W)) predicts the sign of the realized per-cell stationary gain in >= 90% of CI-decided cells with g^I > 0 > g^W (exploratory threshold 0.90), beating the naive baselines; INCONCLUSIVE when the decided hypothesis cells do not contain both signs (no discriminating power)",
                              kind="SEMI-SYNTHETIC / OUT-OF-SAMPLE primitives",
                              result=f"{a2['agree']}/{a2['n_decided']} decided hypothesis cells agree ({fmt(a2['rate'])}); decided negative {c2['hypothesis_cells_decided_negative']}, positive {c2['hypothesis_cells_decided_positive']}; cells above the threshold {c2['hypothesis_cells_above_threshold']}/{c2['n_hypothesis_cells']}; naive always-pos {fmt(a2['naive_always_pos'])}, always-neg {fmt(a2['naive_always_neg'])}, majority rule (psi/r<1/2) {fmt(a2['naive_majority'])}; the premise g^I>0>g^W holds in only {c2['n_hypothesis_cells']}/{c2['n_cells_total']} cell rows (sign patterns {c2['hypothesis_pattern']}): the record (a model name) carries cost information, so both record types usually have the same sign; judge pilot (excluded, 3 probing tasks per cell): {c2['pilot_threshold_rule']['agree']}/{c2['pilot_threshold_rule']['n_decided']}",
                              verdict=v2))
    RES["claims"].append(dict(id="C2b", statement="per-cell form of thm:memory without the sign premise: sign(omega_j g_j^I + psi_j g_j^W) from probing equals the sign of the realized per-cell stationary gain in >= 90% of CI-decided cells, beating the naive baselines",
                              kind="SEMI-SYNTHETIC / OUT-OF-SAMPLE primitives",
                              result=f"{g2['agree']}/{g2['n_decided']} decided cells agree ({fmt(g2['rate'])}); naive always-pos {fmt(g2['naive_always_pos'])}, always-neg {fmt(g2['naive_always_neg'])}, majority rule {fmt(g2['naive_majority'])}; by verifier " + ", ".join(f"{k} {fmt(v['rate'])} ({v['n_decided']})" for k, v in c2["general_by_verifier"].items()) + "; by cell set " + ", ".join(f"{k} {fmt(v['rate'])} ({v['n_decided']})" for k, v in c2["general_by_cellset"].items()) + f"; judge pilot (excluded, 3 probing tasks per cell): {c2['pilot_general_sign']['agree']}/{c2['pilot_general_sign']['n_decided']}",
                              verdict=verdict(g2["rate"] is not None and g2["rate"] >= 0.90 and all(g2["rate"] >= (g2[k] or 0) for k in ("naive_always_pos", "naive_always_neg", "naive_majority"))) if g2["n_decided"] >= 10 else "INCONCLUSIVE"))
    c3 = RES["tables"]["C3"]
    RES["claims"].append(dict(id="C3", statement="prop:cascade-records: g_j^s = Delta_j^s W_{eta_U,j} under record-independent a_j, b_j, v_h, v_b (identity), and the W-form threshold Delta^I/(Delta^I-Delta^W) equals the g-form threshold under the same hypotheses",
                              kind="IDENTITY (holds by algebra; the record-independent-cost hypothesis is violated in the data and the violation is measured)",
                              result=f"max identity residual {fmt(c3['identity_max_residual'], 12)} over {c3['n_cells']} cell rows; W-form vs g-form threshold under the hypotheses: max gap {fmt(c3['threshold_gap_max'], 10)}, decision agreement {fmt(c3['threshold_decision_agreement'])} on {c3['n_hypothesis_cells']} cells with Delta^I>0>Delta^W; VIOLATION: with the actual (model-dependent) attempt costs sign(g^s) = sign(Delta^s W) in {fmt(c3['sign_agreement_actual_vs_identity_I'])} (I) / {fmt(c3['sign_agreement_actual_vs_identity_W'])} (W) of cells, median relative gap {fmt(c3['relative_gap_actual_I_median'])} / {fmt(c3['relative_gap_actual_W_median'])}, the attempt-cost term eta_U (a^I - a^U) is {fmt(c3['attempt_cost_term_share_of_gI_median'])} x |g^I| at the median; realized state-conditional gains track the actual g (Spearman I {fmt(c3['realized_gI_vs_actual'].get('spearman'))}, W {fmt(c3['realized_gW_vs_actual'].get('spearman'))}) better than Delta W (I {fmt(c3['realized_gI_vs_identity'].get('spearman'))}, W {fmt(c3['realized_gW_vs_identity'].get('spearman'))}); realized cascade sign agreement with sum nu G {fmt(c3['cascade_settings_sign']['rate'])} ({c3['cascade_settings_sign']['n_decided']} decided), cascade with the imperfect verifier deciding {fmt(c3['cascade_v_settings_sign']['rate'])} ({c3['cascade_v_settings_sign']['n_decided']} decided)",
                              verdict=verdict(c3["identity_max_residual"] is not None and c3["identity_max_residual"] < 1e-9 and (c3["threshold_gap_max"] is None or c3["threshold_gap_max"] < 1e-9))))
    f4 = c4["fifo"]; f4m = c4["fifo_main"]; f4h = c4["fifo_highbeta"]
    n_det = (f4.get("n_detectable") or 0)
    v4 = verdict(f4.get("det_formula_beats_substitution_share_insample") is not None and f4["det_formula_beats_substitution_share_insample"] >= 0.75 and f4["det_substitution_underestimates_share_insample"] >= 0.90
                 and f4["mae_formula_insample_unsat"] < f4["mae_substitution_insample_unsat"]) if n_det >= 20 else "INCONCLUSIVE"
    RES["claims"].append(dict(id="C4", statement="eq:fifo fits the realized event-FIFO coverage better than the mean-span substitution E -> M/beta in eq:ttl, and the substitution under-estimates coverage (tested where the predicted gap exceeds the sampling noise)",
                              kind="SEMI-SYNTHETIC (coverage of the replayed iid write process; probing r_j = OOS, realized write rates = in-sample mechanism check)",
                              result=f"all unsaturated FIFO points (real < 0.995, >= 200 arrivals after the FIFO filled): eq:fifo inside the realized 95% CI in {fmt(f4.get('formula_insample_in_ci_share'))} (in-sample rates; OOS {fmt(f4.get('formula_in_ci_share'))}), bias {fmt(f4.get('bias_formula_insample_unsat'), 4)}; MAE eq:fifo {fmt(f4.get('mae_formula_insample_unsat'), 4)} vs substitution {fmt(f4.get('mae_substitution_insample_unsat'), 4)} (OOS {fmt(f4.get('mae_formula_unsat'), 4)} vs {fmt(f4.get('mae_substitution_unsat'), 4)}); substitution < eq:fifo in {fmt(f4.get('substitution_below_formula_share'))} of points (Bernoulli), max predicted gap {fmt(f4.get('max_predicted_gap'), 4)}; DETECTABLE points (gap > CI half-width): {n_det}, eq:fifo closer in {fmt(f4.get('det_formula_beats_substitution_share_insample'))} (OOS {fmt(f4.get('det_formula_beats_substitution_share'))}), substitution below realized in {fmt(f4.get('det_substitution_underestimates_share_insample'))} (OOS {fmt(f4.get('det_substitution_underestimates_share'))}), below the CI in {fmt(f4.get('det_substitution_below_ci_share_insample'))}, eq:fifo in CI {fmt(f4.get('det_formula_in_ci_share_insample'))}, MAE {fmt(f4.get('det_mae_formula_insample'), 4)} vs {fmt(f4.get('det_mae_substitution_insample'), 4)}; main stream beta~{fmt(np.mean([v['real'] for v in c4['beta'].values() if v['eps'] < 1]), 3)}: {f4m.get('n_detectable')} detectable of {f4m.get('n_unsaturated')}; high-beta stream (eps=1, beta~{fmt(np.mean([v['real'] for v in c4['beta'].values() if v['eps'] == 1]), 3)}): {f4h.get('n_detectable')} detectable of {f4h.get('n_unsaturated')}, eq:fifo closer in {fmt(f4h.get('det_formula_beats_substitution_share_insample'))}; TTL: eq:ttl in CI {fmt(c4['ttl'].get('formula_insample_in_ci_share'))} (OOS {fmt(c4['ttl'].get('formula_in_ci_share'))}), MAE {fmt(c4['ttl'].get('mae_formula_insample_unsat'), 4)} (OOS {fmt(c4['ttl'].get('mae_formula_unsat'), 4)})",
                              verdict=v4))
    s5 = c5.get("SD", {})
    RES["claims"].append(dict(id="C5", statement="under state-dependent writes (every accepted attempt writes) the write type is not independent of the record state in some cells, and the 2x2 recurrence / exact chain predicts the stationary record probabilities better than the fixed-process formula",
                              kind="diagnostic (SEMI-SYNTHETIC; chi-square on pooled late-window arrivals; chain OUT-OF-SAMPLE, 2x2 IN-SAMPLE)",
                              result=f"chi-square rejects independence at 0.01 in {s5.get('rejected_at_0_01')}/{s5.get('n_tested')} SD cell rows (fixed-process control: {c5.get('FP', {}).get('rejected_at_0_01')}/{c5.get('FP', {}).get('n_tested')}); MAE of (i*, w*): fixed-process OOS {fmt(s5.get('mae_i_fp_oos'))}/{fmt(s5.get('mae_w_fp_oos'))}, exact chain OOS {fmt(s5.get('mae_i_chain'))}/{fmt(s5.get('mae_w_chain'))}, 2x2 in-sample {fmt(s5.get('mae_i_2x2'))}/{fmt(s5.get('mae_w_2x2'))}, fixed-process in-sample {fmt(s5.get('mae_i_fp_insample'))}/{fmt(s5.get('mae_w_fp_insample'))}; realized-sign agreement in SD settings: naive fixed-process {fmt(c5['sd_sign_naive_fp'])} vs chain {fmt(c5['sd_sign_chain'])} ({c5['sd_n_decided']} decided)",
                              verdict=verdict((s5.get("rejected_at_0_01") or 0) >= 1 and s5.get("mae_w_chain") is not None and s5["mae_w_chain"] < s5["mae_w_fp_oos"] and s5["mae_w_2x2"] < s5["mae_w_fp_insample"])))
    k0 = [k for k in c6 if k.startswith("c_ell=0.00000")]
    c6a = c6[k0[0]] if k0 else {}
    RES["claims"].append(dict(id="C6", statement="serving records without re-verification (lookup cost c_ell ~ 0) makes g_j^W >= 0 in some cells (the hypothesis g^W < 0 of thm:memory fails), as prop:cascade-records warns",
                              kind="diagnostic (SEMI-SYNTHETIC accounting variant; predicted from probing, realized in the replay)",
                              result=f"at c_ell = 0: g^W formula eta_U(c^U - c_ell) - v^U >= 0 in {c6a.get('cells_gW_formula_nonneg')}/{c6a.get('n_cells')} cell rows (W0/W1/W2 pooled), realized state-W gain CI >= 0 in {c6a.get('cells_gW_real_ci_nonneg')} and < 0 in {c6a.get('cells_gW_real_ci_neg')}; predicted/realized sign agreement {fmt(c6a.get('sign_agreement_pred_vs_real'))}; " + "; ".join(f"{k}: formula>=0 in {v['cells_gW_formula_nonneg']}/{v['n_cells']}, realized CI>=0 in {v['cells_gW_real_ci_nonneg']}" for k, v in c6.items()),
                              verdict=verdict((c6a.get("cells_gW_formula_nonneg") or 0) >= 1 and (c6a.get("cells_gW_real_ci_nonneg") or 0) >= 1)))

    # ------------------------------------------------------------------ audit (independent recomputation of 3 headline numbers)
    RES["audit"] = audit(pops, settings, rows, PRIM, RRES)

    # ------------------------------------------------------------------ figures, outputs
    RES["spend"] = ledger_spend()
    RES["runtime_s"] = round(time.time() - t_all, 1)
    make_figures(RES, pops, settings, PRIM, RRES, cell_rows, cov_curves, sd_cells, rows)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=_json_default)
    write_readme(RES, pops)
    log(f"done in {RES['runtime_s']} s")
    for c in RES["claims"]:
        print(f"{c['id']}: {c['verdict']} -- {c['result'][:220]}")


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy scalars / arrays / NaN."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def audit(pops: Dict[str, Pop], settings: List[Dict[str, Any]], rows: List[Dict[str, Any]], PRIM: Dict[int, Dict[str, Any]], RRES: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Three headline numbers recomputed with independent code.

    (1) the late-window sums of stream 0 for the main setting, re-simulated with a scalar Python loop
    and a dict store; (2) eta_U of the same setting via pandas group-bys; (3) eq:fifo for one cell with
    Python floats (math) against the numpy value.
    """
    out: Dict[str, Any] = {}
    si = next(i for i, s in enumerate(settings) if s["pop"] == "c12" and s["cellset"] == "cell12" and s["verifier"] == "W2" and s["mode"] == "FP" and s["retention"] == "geo" and s["rpar"] == 0.0 and s["route"] == "direct" and s["k_readout"] == 1 and s["basis"] == "dollars")
    s = settings[si]; pop = pops["c12"]; R = RRES[si]
    st = draw_streams(pop.n)
    cell = pop.cells["cell12"].tolist(); prior = pop.prior.tolist()
    S = pop.S.tolist(); C = pop.C.tolist(); ACC = pop.ACC["W2"].tolist()
    store: Dict[int, int] = {}
    vi = di = vu = du = 0.0
    N = st["TID"].shape[1]
    for n in range(N):
        t = int(st["TID"][0, n]); j = cell[t]; ex = bool(st["U_EXP"][0, n] < EPS_FP); me = int(st["MEXP"][0, n])
        rec = store.get(j)
        fI = me if ex else (rec if rec is not None else prior[t])
        fU = me if ex else prior[t]
        if n >= N - LATE:
            vi += S[t][fI]; di += C[t][fI]; vu += S[t][fU]; du += C[t][fU]
        if ex and ACC[t][me]:
            store[j] = me
    out["audit1_stream0_late_sums"] = dict(engine=[float(R["sumVI"][0]), float(R["sumDI"][0]), float(R["sumVU"][0]), float(R["sumDU"][0])], scalar=[vi, di, vu, du],
                                           max_abs_diff=float(max(abs(R["sumVI"][0] - vi), abs(R["sumDI"][0] - di), abs(R["sumVU"][0] - vu), abs(R["sumDU"][0] - du))))
    # (2) eta_U with pandas
    pp = pops["p12"]
    dfp = pd.DataFrame({"cell": pp.cells["cell12"], "prior": pp.prior})
    dfp["v_prior"] = pp.S[np.arange(pp.n), pp.prior]; dfp["c_prior"] = pp.C[np.arange(pp.n), pp.prior]
    dfp["v_mean"] = pp.S.mean(1); dfp["c_mean"] = pp.C.mean(1)
    g = dfp.groupby("cell")[["v_prior", "c_prior", "v_mean", "c_mean"]].mean()
    nu = pd.Series(pop.nu("cell12"))
    vU = (1 - EPS_FP) * g["v_prior"] + EPS_FP * g["v_mean"]; cU = (1 - EPS_FP) * g["c_prior"] + EPS_FP * g["c_mean"]
    eta_pd = float((nu * vU.reindex(range(12)).fillna(0)).sum() / (nu * cU.reindex(range(12)).fillna(0)).sum())
    out["audit2_eta_U_pandas"] = dict(pandas=eta_pd, engine=PRIM[si]["eta_U"], abs_diff=abs(eta_pd - PRIM[si]["eta_U"]))
    # (3) eq:fifo with python floats, on the median-frequency cell at M = 50 (unsaturated), against the numpy value and the realized coverage
    pr = PRIM[si]; nuv = pop.nu("cell12")
    beta = sum(float(nuv[j]) * float(pr["r"][j]) for j in range(12))
    order = np.argsort(nuv * pr["r"]); j0 = int(order[len(order) // 2]); Mv = 50
    q = float(nuv[j0]) * float(pr["r"][j0]) / beta
    chi_py = 1.0 - (1.0 - q) ** Mv
    chi_np = float(coverage_formula("fifo", Mv, nuv, pr["r"])[j0])
    sub_py = 1.0 - (1.0 - float(nuv[j0]) * float(pr["r"][j0])) ** (Mv / beta)
    mi = COV_M_GRID.index(Mv)
    with np.errstate(invalid="ignore", divide="ignore"):
        real = float(R["cov_M"][:, j0, mi].sum() / R["cov_nM"][:, j0, mi].sum()) if "cov_M" in R else float("nan")
    out["audit3_fifo_formula"] = dict(cell=pop.cell_names["cell12"][j0], M=Mv, q_j=q, beta=beta, python=chi_py, numpy=chi_np, abs_diff=abs(chi_py - chi_np), realized=real,
                                      substitution_python=sub_py, substitution_below_formula=bool(sub_py < chi_py))
    out["setting_audited"] = {k: s[k] for k in ("pop", "cellset", "verifier", "mode", "retention", "rpar", "route")}
    return out


def make_figures(RES: Dict[str, Any], pops: Dict[str, Pop], settings: List[Dict[str, Any]], PRIM: Dict[int, Dict[str, Any]], RRES: Dict[int, Dict[str, Any]],
                 cell_rows: List[Dict[str, Any]], cov_curves: Dict[str, Any], sd_cells: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    """Four figures, one per main claim (C2 noise threshold, C4 coverage, C1 stationary prediction, C5 state dependence)."""
    plt = setup_mpl()
    SIGN_COL = {"pos": FIG_COLORS["blue"], "neg": FIG_COLORS["red"], "incl0": FIG_COLORS["gray"]}
    # ---- Fig 1: (psi/r, threshold) scatter coloured by realized sign (FP, delta = 0, direct; both priors); filled = premise g^I > 0 > g^W holds
    vers = [v for v in ("W1", "W2", "W3") if any(r["verifier"] == v for r in cell_rows)]
    fig, axes = plt.subplots(1, max(len(vers), 1), figsize=(2.3 * max(len(vers), 1) + 0.6, 2.6), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], vers):
        rs = [r for r in cell_rows if r["verifier"] == v and r["delta"] == 0.0 and r["pop"] in ("c12", "c12j", "c21", "c9", "c12o", "c21o") and np.isfinite(r["thr"]) and r["real_sign"] != "incl0"]
        ax.plot([0, 1], [0, 1], color=FIG_COLORS["axis"], lw=0.8, zorder=1)
        ax.axhline(-0.2, color=FIG_COLORS["grid"], lw=0.6)
        jit = np.random.default_rng(7)
        for cs, mk, ms in (("cell12", "o", 20), ("cell9", "D", 14), ("cell21", "s", 16), ("clus50", "^", 12)):
            for hyp in (False, True):
                sub = [r for r in rs if r["cellset"] == cs and r["hypothesis"] == hyp]
                if not sub:
                    continue
                x = [min(r["err"], 1.0) for r in sub]
                # cells where the premise fails have no meaningful threshold: they sit in the strip at y = -0.2
                y = [min(max(r["thr"], 0.0), 1.05) for r in sub] if hyp else list(-0.2 + jit.uniform(-0.05, 0.05, size=len(sub)))
                cols = [SIGN_COL[r["real_sign"]] for r in sub]
                ax.scatter(x, y, s=ms, marker=mk, c=(cols if hyp else "none"), edgecolors=cols, linewidths=0.8, alpha=0.9 if hyp else 0.55, zorder=3 if hyp else 2)
        ax.set_xlabel(r"erroneous write share $\psi_j/r_j$ (%s)" % v)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.3, 1.1)
        ax.set_yticks([-0.2, 0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["n/a", "0", "0.25", "0.5", "0.75", "1"])
        ax.grid(True, axis="y")
    axes[0][0].set_ylabel(r"threshold $g_j^I/(g_j^I-g_j^W)$")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=SIGN_COL["pos"], ls="", ms=5, label="realized cell gain > 0"), Line2D([], [], marker="o", color=SIGN_COL["neg"], ls="", ms=5, label="realized cell gain < 0"),
               Line2D([], [], marker="o", markerfacecolor="none", color=FIG_COLORS["ink2"], ls="", ms=5, label=r"hollow (strip n/a): premise $g^I>0>g^W$ fails"),
               Line2D([], [], marker="o", color=FIG_COLORS["ink2"], ls="", ms=4, label="12 cells"), Line2D([], [], marker="D", color=FIG_COLORS["ink2"], ls="", ms=3.5, label="9 exec-clean cells"),
               Line2D([], [], marker="s", color=FIG_COLORS["ink2"], ls="", ms=4, label="21 cells"), Line2D([], [], marker="^", color=FIG_COLORS["ink2"], ls="", ms=4, label="50 clusters")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_noise_threshold.pdf"), bbox_inches="tight"); plt.close(fig)
    # ---- Fig 2: coverage vs M (eq:fifo vs substitution; high-beta stream where the gap is visible) and vs E (eq:ttl; main stream) for four cells
    key_h = next((k for k in cov_curves if k.startswith("c12/cell12/W2") and k.endswith("/eps1")), None)
    key_m = next((k for k in cov_curves if k == "c12/cell12/W2"), next(iter(cov_curves), None))
    if key_m is not None:
        ccm = cov_curves[key_m]; cch = cov_curves[key_h] if key_h else ccm
        nur = np.array(ccm["nu"]) * np.array(ccm["r_probe"])
        order = np.argsort(nur)
        picks = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]] if len(order) >= 4 else list(order)
        fig, axes = plt.subplots(3, len(picks), figsize=(1.9 * len(picks) + 0.7, 5.3), squeeze=False)
        Mg = np.array(ccm["M_grid"]); Eg = np.array(ccm["E_grid"])
        tb = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)
        for k, j in enumerate(picks):
            ax = axes[0][k]
            nu_h = np.array(cch["nu"]); r_h = np.array(cch["r_real"])  # in-sample write rates: the formula's own mechanism
            f_line = np.array([coverage_formula("fifo", m, nu_h, r_h)[j] for m in Mg]); s_line = np.array([coverage_substitution(m, nu_h, r_h)[j] for m in Mg])
            ax.plot(Mg, f_line, color=FIG_COLORS["blue"], lw=1.6, label="eq:fifo")
            ax.plot(Mg, s_line, color=FIG_COLORS["orange"], lw=1.6, label=r"substitution $E\to M/\beta$ in eq:ttl")
            yv = np.array(cch["real_M"][j]); lo = np.array(cch["lo_M"][j]); hi = np.array(cch["hi_M"][j])
            filled = np.array(cch.get("n_filled_M", [[1] * len(Mg)] * len(cch["cells"]))[j]) > 0
            ax.errorbar(Mg[filled], yv[filled], yerr=[np.clip(yv - lo, 0, None)[filled], np.clip(hi - yv, 0, None)[filled]], fmt="o", ms=3, color=FIG_COLORS["ink"], ecolor=FIG_COLORS["ink2"], elinewidth=0.6, capsize=0, zorder=3, label="realized (95% CI)")
            ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03); ax.grid(True, axis="y")
            ax.set_xlabel("FIFO length $M$ (write events)")
            ax.text(0.97, 0.06, f"{ccm['cells'][j]}\n" + r"$q_j$=%.3f, $\beta$=%.2f" % (nu_h[j] * r_h[j] / max((nu_h * r_h).sum(), 1e-12), (nu_h * r_h).sum()), transform=ax.transAxes, fontsize=7, va="bottom", ha="right", color=FIG_COLORS["ink2"], bbox=tb)
            ax3 = axes[1][k]  # the gap: predicted eq:fifo - substitution vs realized - substitution (points with CI)
            ax3.axhline(0, color=FIG_COLORS["axis"], lw=0.8)
            ax3.plot(Mg, f_line - s_line, color=FIG_COLORS["blue"], lw=1.6, label="eq:fifo $-$ substitution")
            ax3.errorbar(Mg[filled], (yv - s_line)[filled], yerr=[np.clip(yv - lo, 0, None)[filled], np.clip(hi - yv, 0, None)[filled]], fmt="o", ms=3, color=FIG_COLORS["ink"], ecolor=FIG_COLORS["ink2"], elinewidth=0.6, capsize=0, zorder=3, label="realized $-$ substitution")
            ax3.set_xscale("log"); ax3.grid(True, axis="y")
            ax3.set_xlabel("FIFO length $M$ (write events)")
            ax2 = axes[2][k]
            nu_m = np.array(ccm["nu"]); r_m = np.array(ccm["r_probe"])
            ax2.plot(Eg, [coverage_formula("ttl", e, nu_m, r_m)[j] for e in Eg], color=FIG_COLORS["blue"], lw=1.6, label="eq:ttl (probing $r_j$)")
            yv = np.array(ccm["real_E"][j]); lo = np.array(ccm["lo_E"][j]); hi = np.array(ccm["hi_E"][j])
            ax2.errorbar(Eg, yv, yerr=[np.clip(yv - lo, 0, None), np.clip(hi - yv, 0, None)], fmt="o", ms=3, color=FIG_COLORS["ink"], ecolor=FIG_COLORS["ink2"], elinewidth=0.6, capsize=0, zorder=3, label="realized")
            ax2.set_xscale("log"); ax2.set_ylim(-0.03, 1.03); ax2.grid(True, axis="y")
            ax2.set_xlabel("TTL $E$ (rounds)")
            ax2.text(0.97, 0.06, r"$\nu_j r_j$=%.4f" % nur[j], transform=ax2.transAxes, fontsize=7, va="bottom", ha="right", color=FIG_COLORS["ink2"], bbox=tb)
            if k > 0:
                for a in (ax, ax2, ax3):
                    a.tick_params(labelleft=False)
        axes[0][0].set_ylabel(r"FIFO coverage (high-$\beta$)"); axes[1][0].set_ylabel("coverage gap"); axes[2][0].set_ylabel("TTL coverage (main)")
        gl = [a.get_ylim() for a in axes[1]]
        for a in axes[1]:
            a.set_ylim(min(g[0] for g in gl), max(g[1] for g in gl))
        h0, l0 = axes[0][0].get_legend_handles_labels(); h1, l1 = axes[1][0].get_legend_handles_labels()
        fig.legend(h0 + [h1[1]], l0 + [l1[1]], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_coverage_fifo_ttl.pdf"), bbox_inches="tight"); plt.close(fig)
    # ---- Fig 3: predicted stationary gain vs realized gain per round (C1 pool, direct + cascade)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    pool = [r for r in rows if r["pool"] == "C1" and r["route"] in ("direct", "cascade", "cascade_v") and r["basis"] == "dollars"]
    if pool:
        allv = [abs(r["pred_gain"]) for r in pool] + [abs(r["gain_real"]) for r in pool]
        lim = 1.05 * max(allv) if allv else 1.0
        ax.plot([-lim, lim], [-lim, lim], color=FIG_COLORS["axis"], lw=0.8, zorder=1)
        ax.axhline(0, color=FIG_COLORS["grid"], lw=0.8); ax.axvline(0, color=FIG_COLORS["grid"], lw=0.8)
        for v in ("W0", "W1", "W2", "W3", "W3g"):
            for cs, mk in (("cell12", "o"), ("cell9", "o"), ("cell2g", "o"), ("cell21", "s"), ("clus50", "^")):
                sub = [r for r in pool if r["verifier"] == v and r["cellset"] == cs]
                if not sub:
                    continue
                ax.errorbar([r["pred_gain"] for r in sub], [r["gain_real"] for r in sub], yerr=[[r["gain_real"] - r["gain_lo"] for r in sub], [r["gain_hi"] - r["gain_real"] for r in sub]],
                            fmt=mk, ms=3.5, color=VCOL[v], ecolor=VCOL[v], elinewidth=0.6, capsize=0, alpha=0.9)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"predicted stationary gain per round $\sum_j \nu_j G_j$ (probing)")
        ax.set_ylabel(r"realized $\overline{(V-\eta_U D)}_{\infty} - \overline{(V-\eta_U D)}_U$")
        from matplotlib.lines import Line2D
        hv = [Line2D([], [], marker="o", color=VCOL[v], ls="", ms=4, label=v) for v in ("W0", "W1", "W2", "W3", "W3g") if any(r["verifier"] == v for r in pool)]
        hs = [Line2D([], [], marker=mk, color=FIG_COLORS["ink2"], ls="", ms=4, label=lab) for mk, lab in (("o", "12 / 9 / 2 cells"), ("s", "21 cells"), ("^", "50 clusters"))]
        ax.legend(handles=hv + hs, loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_stationary_prediction.pdf"), bbox_inches="tight"); plt.close(fig)
    # ---- Fig 4: state-dependent writes: realized vs predicted stationary erroneous share
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    rs = [r for r in sd_cells if r["mode"] == "SD" and r["delta"] == 0.01 and r["cellset"] in ("cell12", "cell21") and r["verifier"] in ("W1", "W2")]

    def sh(i: float, w: float) -> float:
        return w / (i + w) if (i + w) > 0 else float("nan")

    if rs:
        ax.plot([0, 1], [0, 1], color=FIG_COLORS["axis"], lw=0.8)
        x = [sh(r["i_real"], r["w_real"]) for r in rs]
        ax.scatter(x, [sh(r["i_fp_oos"], r["w_fp_oos"]) for r in rs], s=16, marker="o", color=FIG_COLORS["orange"], label="fixed-process formula (probing)")
        ax.scatter(x, [sh(r["i_chain"], r["w_chain"]) for r in rs], s=16, marker="^", color=FIG_COLORS["blue"], label="state-dependent chain (probing)")
        ax.scatter(x, [sh(r["i_2x2"], r["w_2x2"]) for r in rs], s=12, marker="s", facecolors="none", edgecolors=FIG_COLORS["aqua"], label=r"2$\times$2 recurrence (measured $\omega^s,\psi^s$)")
        ax.set_xlabel(r"realized stationary erroneous share $w_j^*/(i_j^*+w_j^*)$")
        ax.set_ylabel("predicted share")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(True, axis="y")
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_state_dependent_writes.pdf"), bbox_inches="tight"); plt.close(fig)


def write_readme(RES: Dict[str, Any], pops: Dict[str, Pop]) -> None:
    """README.md and summary_for_paper.md from the results."""
    T = RES["tables"]
    L: List[str] = []
    L.append("# X13_memory -- router memory replayed on recorded outcomes with real verifiers\n")
    L.append(f"Generated by `run.py` (seed {SEED}); runtime {RES['runtime_s']} s; {RES['streams']} streams x {RES['rounds']} rounds, late window {RES['late_window']} rounds; {RES['nboot']} bootstrap resamples (paired over streams; task-cluster resampling as a second decision rule).\n")
    L.append("## What this experiment is (and is not)\n")
    L.append("Proposition thm:memory, eq:noise-threshold, prop:cascade-records, eq:ttl / eq:fifo and Remark rem:state-dependent-writes are tested on an ACRouter-style record store REPLAYED over the recorded CodeRouterBench outcomes. **No ACRouter run logs exist in the released package** (`outputs/current/id_decisions.jsonl` holds one chosen model per task and no memory writes, verifier sub-signals or kNN neighbours), so the memory writes are generated here: the verifier is run on the RECORDED responses (`response_matrix.json`) and the record process is simulated. Everything that is an outcome (score, cost, latency, response text, sandbox verdict) is real; the arrival order, the exploration coin, the expiration coins and the retention rule are constructed. Evidence kind: SEMI-SYNTHETIC with OUT-OF-SAMPLE primitives (probing) unless marked otherwise.\n")
    L.append("## Setup\n")
    cfg = RES["config"]
    L.append(f"- Data: `$ACROUTER_DATA/coderouterbench/id_results_long.csv` (scores, `cost_usd`, latency), `matrices/phase1_acrouter_v2/response_matrix.json` (recorded responses); prompts / tests via `common/sources.py`; verifiers via `common/sandbox.py`.")
    L.append(f"- Replay population: id_test tasks with an execution-based verifier: sources humaneval, he (completion), mbpp, bcb, cruxeval, hex (python), lcb, ds1000; test-generation prompts excluded; **all-zero bcb / lcb / ds1000 tasks excluded as UNGRADED** (released grader failure, Addendum 2) -> {cfg['populations']['c12']['n']} tasks in 12 dimension x source cells (the 9 other cells have no local verifier). Probing counterpart: {cfg['populations']['p12']['n']} tasks (primitives are fitted there only).")
    L.append(f"- 21-cell population (oracle W0 and the syntax-only W1): all id_test tasks minus the all-zero bcb / lcb / ds1000 tasks -> {cfg['populations']['c21']['n']} tasks (probing {cfg['populations']['p21']['n']}); sensitivity `c21all` keeps every task ({cfg['populations']['c21all']['n']}). EXEC-CLEAN robustness pool `c9`: the 12-cell population restricted to humaneval / he / mbpp / hex / bcb -> {cfg['populations']['c9']['n']} tasks in {cfg['populations']['c9']['J']['cell9']} cells (probing {cfg['populations']['p9']['n']}). Populations suffixed `o` use always-Opus as the uninformed action instead of DimensionBest.")
    L.append(f"- Empty recorded responses (data gap in response_matrix.json; {cfg['populations']['c12']['stats']['empty_pairs']} of {8 * cfg['populations']['c12']['n']} pairs in the replay population, {cfg['populations']['c12']['stats']['imputed_empty_true']} of them with released score 1) are unverifiable: every verifier is imputed the released truth on them (Addendum 2); they are excluded from the verifier operating-point table.")
    L.append(f"- Cluster store (proxy for ACRouter's embedding-kNN memory; **no embedding model is available here**): TF-IDF (1-2 grams, {cfg['clusters']['vocab']} features) + MiniBatchKMeans k={N_CLUSTERS} fitted on the probing prompts, id_test prompts assigned to the nearest centroid.")
    L.append(f"- Prior (uninformed action, ACRouter's DimensionBest warm start) fitted on probing: " + ", ".join(f"{d}: {m}" for d, m in cfg["dimbest_probing"].items()) + ".")
    L.append(f"- Stream: each of {RES['streams']} streams draws {RES['rounds']} tasks i.i.d. with replacement from the population (the independent arrivals of sec:memory; nu_j = the population's cell frequencies); the store starts empty; stationary statistics use the last {RES['late_window']} rounds; the finite-horizon identity is also checked on ONE random pass over the population (TRANS pool).")
    L.append(f"- Writer / record process. Fixed process (FP, the hypothesis of thm:memory): with probability eps = {EPS_FP} a round explores a uniformly random model for BOTH routers and the verifier's accept writes (model) to the arriving cell; a write is INFORMATIVE if the model is truly correct (released score >= 1) and ERRONEOUS if the verifier accepted an incorrect output. State-dependent (SD, ACRouter-like): every round the informed router fields the record model (prior if uncovered; epsilon-greedy {EPS_SD}), verifies and writes if accepted. Informed action = model of the cell's newest valid record; uninformed = prior. Retention: geometric expiration delta in {DELTAS}, TTL E in {TTL_GRID}, event FIFO M in {FIFO_GRID}, LRU over distinct cells (capacity ceil(J/3), ceil(2J/3)).")
    L.append("- Verifiers (all on recorded responses): W0 oracle = released score >= 1; W1 = ACRouter's tools: the prompt's doctest examples (public cases for lcb) run in the sandbox, a syntax check when the prompt shows no example (mbpp descriptions, cruxeval, ds1000), non-empty text for non-python tasks; W2 = hidden-test sandbox (`sandbox.verify_response`; lcb capped at 25 cases; empty recorded responses with released score 1 are unverifiable and imputed from the released score); W3 = LLM judge deepseek-v4-pro (thinking off) on a 300-task id_test subsample and 15 probing tasks per cell.")
    j = cfg.get("judge", {})
    jg = cfg.get("judge_grok", {})
    x8 = j.get("x8_calibration", {})
    L.append(f"- W3 status: X8's 60-item calibration table (`X8_operating_point/cache/calibration.json`) gives AUROC " + ", ".join(f"{k} {fmt(v.get('auroc'))} [{fmt(v.get('auroc_ci', [None, None])[0])}, {fmt(v.get('auroc_ci', [None, None])[1])}]" for k, v in x8.items()) + f". W3 = deepseek-v4-pro (thinking off; AUROC {fmt(x8.get('dsv4pro_nothink', {}).get('auroc'))} in X8's table, below the protocol's 0.75 gate) was run on the 300-task subsample per the coordinator's first instruction and is reported as the NOISY-WRITER case, not as a trustworthy verifier; own check on its {j.get('own_check', {}).get('n_pairs')} judged (task, model) pairs: AUROC {fmt(j.get('own_check', {}).get('auroc'))}, accuracy {fmt(j.get('own_check', {}).get('accuracy'))}; error rows 0. W3g = grok-4.6 (X8's best judge, AUROC {fmt(x8.get('grok46', {}).get('auroc'))}; ~{fmt(x8.get('grok46', {}).get('mean_total_tokens'), 0)} tokens and ~30-130 s per call) could not be run on the full subsample: at the uncalibrated ledger rate a call costs ~150-230 units, so the 300-task W3 would need ~0.6M ledger units, twice the shared cap. It was run as a budget-capped PILOT on {jg.get('n_test_tasks')} id_test + {jg.get('n_probe_tasks')} probing tasks of the cells {jg.get('cells')} drawn from the deepseek subsample (same pairs for both judges): {jg.get('calls_made')} calls, {jg.get('units_session', 0):.0f} ledger units, cap reached: {jg.get('stopped_by_cap')}; error rows in cache: {jg.get('n_error_rows_in_cache')}; own check AUROC {fmt(jg.get('own_check', {}).get('auroc'))} (accuracy {fmt(jg.get('own_check', {}).get('accuracy'))}) on {jg.get('own_check', {}).get('n_pairs')} pairs vs deepseek {fmt(jg.get('deepseek_on_same_pairs', {}).get('auroc'))} on the same pairs.")
    sp = RES["spend"]
    L.append(f"- API spend (tag X13, shared ledger): {sp['live_calls']} live calls, {sp['prompt_tokens']:,} prompt + {sp['completion_tokens']:,} completion tokens, {sp['units_est']:.1f} ledger units; by model: " + "; ".join(f"{m} {v['live_calls']} live calls, {v['prompt_tokens'] + v['completion_tokens']:,} tokens, {v['units_est']:.1f} units" for m, v in sp["by_model"].items()) + " (deepseek-v4-pro at the calibrated 0.00034 units/token, grok-4.6 at the default 0.05). No other API use.")
    L.append("- Cost basis: dollars (csv `cost_usd`) unless a row says seconds (latency + verifier wall-clock).\n")
    # verifier table
    L.append("## Verifier operating points on the replay population (non-empty recorded responses; truth = released score >= 1)\n")
    L.append("| population | verifier | source | pairs | truth rate | accept rate | TPR | FPR | agreement | erroneous share of accepts (psi/r) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in T["verifiers"]:
        if (r["pop"] in ("c12", "c12j", "c2g") and (r["source"] == "all" or r["pop"] == "c12")) or (r["pop"] == "c2g"):
            L.append(f"| {r['pop']} | {r['verifier']} | {r['source']} | {r['pairs']} | {fmt(r['truth_rate'])} | {fmt(r['accept_rate'])} | {fmt(r['TPR'])} | {fmt(r['FPR'])} | {fmt(r['agreement'])} | {fmt(r['false_accept_share'])} |")
    L.append("")
    g_rows = [r for r in T["cells"] if r["pop"] == "c2g" and r["delta"] == 0.0]
    if g_rows:
        L.append("### Judge pilot: the same 2 cells and (task, model) pairs under every verifier (FP writer, delta = 0, dollars)\n")
        jg = cfg.get("judge_grok", {})
        L.append(f"Grok-4.6 vs deepseek-v4-pro on identical pairs: AUROC {fmt(jg.get('own_check', {}).get('auroc'))} vs {fmt(jg.get('deepseek_on_same_pairs', {}).get('auroc'))} ({jg.get('own_check', {}).get('n_pairs')} pairs), both far below X8's calibration values (0.842 / 0.656) because the pilot cells are structurally hard for a judge: the dim1 mbpp prompt is the bare task description without the function name the MBPP+ harness calls, so a correct-looking function with another name fails the hidden tests (truth rate 0.15, judge erroneous share of accepts ~0.85 for both judges; the syntax-only W1 accepts everything there). The pilot's replay populations (12 test / 6 probing tasks; primitives from 3 probing tasks per cell) are too small for out-of-sample primitives, so the pilot's C1 / C2 rows are reported here and in the tables but EXCLUDED from the claim pools (C1 pilot: {T['C1']['pilot_c2g']['agree']}/{T['C1']['pilot_c2g']['n_decided']} decided settings agree; pooled with the main settings the C1 rate would be {fmt(T['C1']['all_including_pilot']['rate'])}).\n")
        L.append("| cell | verifier | n probe | omega | psi | psi/r | g^I | g^W | predicted sign | realized gain/arrival [CI] |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(g_rows, key=lambda r: (r["cell"], r["verifier"])):
            L.append(f"| {r['cell']} | {r['verifier']} | {r['n_probe']} | {fmt(r['omega'], 4)} | {fmt(r['psi'], 4)} | {fmt(r['err'])} | {r['gI']:+.4f} | {r['gW']:+.4f} | {r['pred_sign']} | {r['real_gain']:+.4f} [{r['real_lo']:+.4f}, {r['real_hi']:+.4f}] ({r['real_sign']}) |")
        L.append("")
    # claims
    L.append("## Claims\n")
    L.append("| id | statement | kind | result | verdict |")
    L.append("|---|---|---|---|---|")
    for c in RES["claims"]:
        L.append(f"| {c['id']} | {c['statement']} | {c['kind']} | {c['result']} | **{c['verdict']}** |")
    L.append("")
    # C1 tables
    c1 = T["C1"]
    L.append("## (i) Stationary sign criterion (C1): sign agreement on CI-decided settings (fixed-process pool)\n")
    L.append("| slice | settings | decided | agree | rate | always + | always - | omega>psi |")
    L.append("|---|---|---|---|---|---|---|---|")

    def c1line(name: str, a: Dict[str, Any]) -> str:
        return f"| {name} | {a['n_settings']} | {a['n_decided']} | {a['agree']} | {fmt(a['rate'])} | {fmt(a['naive_always_pos'])} | {fmt(a['naive_always_neg'])} | {fmt(a['naive_omega_gt_psi'])} |"

    L.append(c1line("all (stream bootstrap)", c1["all"]))
    L.append(c1line("all (task-cluster bootstrap decisions)", c1["task_bootstrap"]))
    for key in ("by_verifier", "by_pop", "by_cellset", "by_retention", "by_route", "by_delta"):
        for k, a in c1[key].items():
            L.append(c1line(f"{key[3:]} = {k}", a))
    L.append(c1line("serve-without-verification variant", c1["serve"]))
    L.append(c1line("state-dependent writes, naive fixed-process prediction (diagnostic)", c1["sd_naive_fp_prediction"]))
    L.append(c1line("judge pilot c2g (EXCLUDED from the pool: 3 probing tasks per cell)", c1["pilot_c2g"]))
    for k, a in c1["pilot_c2g_by_verifier"].items():
        L.append(c1line(f"judge pilot c2g, verifier = {k}", a))
    L.append(c1line("pool + pilot (for reference)", c1["all_including_pilot"]))
    L.append("")
    fh1 = c1["finite_horizon_iid_geo"]; fh2 = c1["finite_horizon_one_pass"]; fi1 = c1["insample_iid_geo"]; fi2 = c1["insample_one_pass"]
    L.append("Finite-horizon identity eq:finite-memory (cumulative gain from the empty store): OOS = probing primitives; IN-SAMPLE = primitives of the replay population itself (mechanism check of the identity with its own primitives).\n")
    L.append("| stream | settings | OOS: prediction in realized CI | OOS: median normalized error | OOS: sign agreement (decided) | IN-SAMPLE: prediction in CI | IN-SAMPLE: median normalized error | IN-SAMPLE: stationary gain prediction in CI | OOS: stationary gain prediction in CI |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(f"| iid, {RES['rounds']} rounds, geometric | {fh1['n_settings']} | {fmt(fh1['ci_covers_prediction'])} | {fmt(fh1['median_norm_err'])} | {fmt(fh1['sign_agree'])} ({fh1['n_decided']}) | {fmt(fi1['ci_covers_prediction'])} | {fmt(fi1['median_norm_err'])} | {fmt(fi1['stationary_gain_ci_covers_prediction'])} | {fmt(fi1['stationary_gain_ci_covers_prediction_oos'])} |")
    L.append(f"| one random pass (n_tasks rounds) | {fh2['n_settings']} | {fmt(fh2['ci_covers_prediction'])} | {fmt(fh2['median_norm_err'])} | {fmt(fh2['sign_agree'])} ({fh2['n_decided']}) | {fmt(fi2['ci_covers_prediction'])} | {fmt(fi2['median_norm_err'])} | {fmt(fi2['stationary_gain_ci_covers_prediction'])} | {fmt(fi2['stationary_gain_ci_covers_prediction_oos'])} |")
    L.append("")
    L.append("Reading this table: the CIs are stream-bootstrap CIs (arrival order, exploration and expiration coins), which do not include the task-sampling uncertainty of the probing fit, so a probing (OOS) prediction inside a stream CI is a strict test that mostly fails while the SIGN and the size (normalized error = |realized - predicted| / (N sum_j nu_j |G_j|)) agree; with the replay population's own primitives (IN-SAMPLE) the identity eq:finite-memory reproduces the realized cumulative gain within the stream CI in most settings, which is the mechanism check that the replay implements the record process of sec:memory.\n")
    L.append("Per-setting table (dollars unless noted): predicted stationary gain per round sum_j nu_j G_j (probing) vs realized eta_inf - eta_U (late window, paired stream-bootstrap 95% CI) and realized gain per round at the predicted price.\n")
    L.append("| pool | pop | cells | verifier | mode | route | retention | eta_U pred | eta_U real | eta_inf real | pred gain | realized d_eta [CI] | realized gain/round [CI] | coverage real/pred | agree |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in T["settings"]:
        ret = f"{r['retention']} {r['rpar']:g}" + (f" c_ell={r['c_ell']:.4f}" if r["route"] == "serve" else "") + (f" k={r['k_readout']}" if r["k_readout"] > 1 else "") + (" [seconds]" if r["basis"] == "seconds" else "")
        ag = "-" if r["agree"] is None else ("yes" if r["agree"] else "NO")
        L.append(f"| {r['pool']} | {r['pop']} | {r['cellset']} | {r['verifier']} | {r['mode']} | {r['route']} | {ret} | {fmt(r['eta_U_pred'], 2)} | {fmt(r['eta_U_real'], 2)} | {fmt(r['eta_inf_real'], 2)} | {r['pred_gain']:+.4f} | {r['d_eta']:+.3f} [{r['d_eta_lo']:+.3f}, {r['d_eta_hi']:+.3f}] | {r['gain_real']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}] | {fmt(r['coverage_real_mean'])}/{fmt(r['coverage_pred_mean'])} | {ag} |")
    L.append("")
    # C2
    c2 = T["C2"]
    L.append("## (ii) Per-cell noise threshold (C2)\n")
    L.append(f"Cells (rows over verifier x delta x cell set, FP direct geometric settings): {c2['n_cells_total']}; hypothesis g^I > 0 > g^W holds in {c2['n_hypothesis_cells']} ({fmt(c2['hypothesis_share'])}).\n")
    L.append("| slice | cells | decided | agree | rate | always + | always - | majority (psi/r < 1/2) |")
    L.append("|---|---|---|---|---|---|---|---|")

    def c2line(name: str, a: Dict[str, Any]) -> str:
        return f"| {name} | {a['n_cells']} | {a['n_decided']} | {a['agree']} | {fmt(a['rate'])} | {fmt(a['naive_always_pos'])} | {fmt(a['naive_always_neg'])} | {fmt(a['naive_majority'])} |"

    L.append(c2line("threshold rule, hypothesis cells", c2["threshold_rule_on_hypothesis_cells"]))
    L.append(c2line("general sign(omega g^I + psi g^W), all cells", c2["general_sign_all_cells"]))
    for key in ("by_verifier", "by_cellset", "by_pop"):
        for k, a in c2[key].items():
            L.append(c2line(f"{key[3:]} = {k} (threshold rule)", a))
    L.append("")
    L.append(f"OOS check of the state-conditional primitives over all cell rows: g^I predicted vs realized MAE {fmt(c2['gI_pred_vs_real'].get('mae'), 4)}, Spearman {fmt(c2['gI_pred_vs_real'].get('spearman'))}, sign agreement {fmt(c2['gI_pred_vs_real'].get('sign_agreement'))}; g^W: MAE {fmt(c2['gW_pred_vs_real'].get('mae'), 4)}, Spearman {fmt(c2['gW_pred_vs_real'].get('spearman'))}, sign agreement {fmt(c2['gW_pred_vs_real'].get('sign_agreement'))}; record probabilities i*: MAE {fmt(c2['i_pred_vs_real'].get('mae'), 4)}, w*: MAE {fmt(c2['w_pred_vs_real'].get('mae'), 4)}.\n")
    L.append("Per-cell table for the 12 cells (W1 and W2, delta = 0; dollars):\n")
    L.append("| verifier | cell | nu | omega | psi | psi/r | g^I | g^W | threshold | predicted | realized gain/arrival [CI] | g^I real | g^W real | i* pred/real | w* pred/real |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in T["cells"]:
        if r["pop"] == "c12" and r["cellset"] == "cell12" and r["delta"] == 0.0 and r["verifier"] in ("W1", "W2"):
            L.append(f"| {r['verifier']} | {r['cell']} | {fmt(r['nu'])} | {fmt(r['omega'], 4)} | {fmt(r['psi'], 4)} | {fmt(r['err'])} | {r['gI']:+.4f} | {r['gW']:+.4f} | {fmt(r['thr'])} | {r['pred_sign']}{'' if r['hypothesis'] else ' (hyp. fails)'} | {r['real_gain']:+.4f} [{r['real_lo']:+.4f}, {r['real_hi']:+.4f}] | {r['gI_real']:+.4f} | {r['gW_real']:+.4f} | {fmt(r['i_pred'])}/{fmt(r['i_real'])} | {fmt(r['w_pred'])}/{fmt(r['w_real'])} |")
    L.append("")
    # C3
    c3 = T["C3"]
    L.append("## (iii) Records inside a verified cascade (C3)\n")
    L.append(f"Cascade variant: the informed / prior model is attempted, the ORACLE decides (hit = released score >= 1), Opus is the fallback; writes come from the exploration attempts verified by W0 / W1 / W2 (W3 on the judged subsample). IDENTITY: with the proposition's hypotheses imposed (record-independent a_j; b_j, v_h = 1, v_b = Opus's cell mean), g^s = Delta^s W_eta with max residual {fmt(c3['identity_max_residual'], 14)} over {c3['n_cells']} cell rows, and the g-form and W-form thresholds coincide (max gap {fmt(c3['threshold_gap_max'], 10)}, decision agreement {fmt(c3['threshold_decision_agreement'])} on {c3['n_hypothesis_cells']} cells with Delta^I > 0 > Delta^W). VIOLATION MEASURED: the actual attempt cost depends on the fielded model (a record is a model name, and the models differ up to 20x in price), so the actual g^s = Delta^s W - eta_U (a^s - a^U): sign(g^s) = sign(Delta^s W) in {fmt(c3['sign_agreement_actual_vs_identity_I'])} (I) / {fmt(c3['sign_agreement_actual_vs_identity_W'])} (W) of cells, median relative gap {fmt(c3['relative_gap_actual_I_median'])} / {fmt(c3['relative_gap_actual_W_median'])}; the attempt-cost term is {fmt(c3['attempt_cost_term_share_of_gI_median'])} x |g^I| at the median; both sign premises (Delta and actual g) hold together in {c3['n_cells_both_hypotheses']} cells, where the two thresholds differ by {fmt(c3['threshold_gap_actual_median'])} (median) and give the same decision in {fmt(c3['threshold_decision_agreement_actual'])}. Realized state-conditional gains: vs the actual g, Spearman I {fmt(c3['realized_gI_vs_actual'].get('spearman'))} / W {fmt(c3['realized_gW_vs_actual'].get('spearman'))} (MAE {fmt(c3['realized_gI_vs_actual'].get('mae'), 4)} / {fmt(c3['realized_gW_vs_actual'].get('mae'), 4)}); vs Delta^s W, Spearman I {fmt(c3['realized_gI_vs_identity'].get('spearman'))} / W {fmt(c3['realized_gW_vs_identity'].get('spearman'))} (MAE {fmt(c3['realized_gI_vs_identity'].get('mae'), 4)} / {fmt(c3['realized_gW_vs_identity'].get('mae'), 4)}). Realized sign of eta_inf - eta_U in the cascade settings agrees with sum nu G in {fmt(c3['cascade_settings_sign']['rate'])} ({c3['cascade_settings_sign']['n_decided']} decided); cascade with the imperfect verifier deciding (cascade_v, false accepts deliver the wrong output): {fmt(c3['cascade_v_settings_sign']['rate'])} ({c3['cascade_v_settings_sign']['n_decided']} decided).\n")
    L.append("| prior | verifier | cell | W_eta | p^U | p^I | p^W | Delta^I W | Delta^W W | g^I actual | g^W actual | g^I real | g^W real | thr g-form (hyp.) | thr W-form | thr g-form (actual) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in T["cascade_cells"]:
        if r["delta"] == 0.0 and r["verifier"] in ("W1", "W2"):
            L.append(f"| {r['prior']} | {r['verifier']} | {r['cell']} | {fmt(r['W_eta'], 4)} | {fmt(r['pU'])} | {fmt(r['pI'])} | {fmt(r['pW'])} | {r['gI_id']:+.4f} | {r['gW_id']:+.4f} | {r['gI']:+.4f} | {r['gW']:+.4f} | {r['gI_real']:+.4f} | {r['gW_real']:+.4f} | {fmt(r['thr_g_hom'])} | {fmt(r['thr_W'])} | {fmt(r['thr_g_actual'])} |")
    L.append("")
    # C4
    c4 = T["C4"]
    L.append("## (iv) Coverage under TTL and event FIFO (C4)\n")
    L.append(f"Realized coverage (share of late-window arrivals finding a record written within the last E rounds / among the last M write events, the latter counted only after the FIFO has filled) from the unbounded fixed-process replays on the iid streams; formulas use the stream's nu_j and either probing r_j (OOS) or the realized write rates (in-sample mechanism check). Points: cells x grid values with realized coverage < 0.995 and >= 200 arrivals. 'Detectable' = the predicted gap eq:fifo - substitution exceeds the realized CI half-width. The main stream writes on exploration rounds only (beta = sum nu_j r_j ~ {fmt(np.mean([v['real'] for v in c4['beta'].values() if v['eps'] < 1]), 3)}); the high-beta stream verifies and writes every arrival's random attempt (beta ~ {fmt(np.mean([v['real'] for v in c4['beta'].values() if v['eps'] == 1]), 3)}), the regime of the example in app:retention.\n")
    L.append("| rule / stream | points | unsat. | formula in CI (in-sample / OOS) | bias formula (in-sample) | MAE formula / substitution (in-sample) | MAE formula / substitution (OOS) | subst. < realized (in-sample) | detectable | det.: formula closer | det.: subst. < realized | det.: subst. < CI | det.: MAE formula / subst. |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, a in (("event FIFO, all", c4["fifo"]), ("event FIFO, main stream", c4["fifo_main"]), ("event FIFO, high-beta stream", c4["fifo_highbeta"]), ("TTL, all", c4["ttl"]), ("TTL, main", c4["ttl_main"]), ("TTL, high-beta", c4["ttl_highbeta"])):
        L.append(f"| {name} | {a['n_points']} | {a['n_unsaturated']} | {fmt(a.get('formula_insample_in_ci_share'))} / {fmt(a.get('formula_in_ci_share'))} | {fmt(a.get('bias_formula_insample_unsat'), 4)} | {fmt(a.get('mae_formula_insample_unsat'), 4)} / {fmt(a.get('mae_substitution_insample_unsat'), 4)} | {fmt(a.get('mae_formula_unsat'), 4)} / {fmt(a.get('mae_substitution_unsat'), 4)} | {fmt(a.get('substitution_underestimates_share_insample'))} | {a.get('n_detectable', '-')} | {fmt(a.get('det_formula_beats_substitution_share_insample'))} | {fmt(a.get('det_substitution_underestimates_share_insample'))} | {fmt(a.get('det_substitution_below_ci_share_insample'))} | {fmt(a.get('det_mae_formula_insample'), 4)} / {fmt(a.get('det_mae_substitution_insample'), 4)} |")
    L.append("")
    L.append("| M (high-beta stream) | unsaturated points | formula in CI | MAE eq:fifo | MAE substitution | subst. < realized | detectable | det.: formula closer | det.: subst. < realized |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m, a in c4["fifo_highbeta_by_M"].items():
        L.append(f"| {m} | {a['n_unsaturated']} | {fmt(a.get('formula_insample_in_ci_share'))} | {fmt(a.get('mae_formula_insample_unsat'), 4)} | {fmt(a.get('mae_substitution_insample_unsat'), 4)} | {fmt(a.get('substitution_underestimates_share_insample'))} | {a.get('n_detectable', '-')} | {fmt(a.get('det_formula_beats_substitution_share_insample'))} | {fmt(a.get('det_substitution_underestimates_share_insample'))} |")
    L.append("")
    L.append("Main-grid retention settings: nu-weighted realized coverage vs eq:ttl / eq:fifo (probing r_j) and the substitution.\n")
    L.append("| setting | realized | formula | substitution | min writes per stream |")
    L.append("|---|---|---|---|---|")
    for k, a in c4["fifo_grid_settings"].items():
        L.append(f"| {k} | {fmt(a['real'])} | {fmt(a['formula'])} | {fmt(a['substitution'])} | {a['min_writes_per_stream']} |")
    L.append("")
    # C5
    c5 = T["C5"]
    L.append("## (v) State-dependent writes (C5)\n")
    L.append("| mode | cell rows | tested | chi2 rejections (0.01) | MAE i*: FP OOS / chain OOS / 2x2 in-sample / FP in-sample | MAE w*: same order | MAE erroneous share: FP OOS / chain / 2x2 |")
    L.append("|---|---|---|---|---|---|---|")
    for mode in ("SD", "FP"):
        a = c5.get(mode, {})
        L.append(f"| {mode} | {a.get('n_cells')} | {a.get('n_tested')} | {a.get('rejected_at_0_01')} | {fmt(a.get('mae_i_fp_oos'))} / {fmt(a.get('mae_i_chain'))} / {fmt(a.get('mae_i_2x2'))} / {fmt(a.get('mae_i_fp_insample'))} | {fmt(a.get('mae_w_fp_oos'))} / {fmt(a.get('mae_w_chain'))} / {fmt(a.get('mae_w_2x2'))} / {fmt(a.get('mae_w_fp_insample'))} | {fmt(a.get('mae_share_fp_oos'))} / {fmt(a.get('mae_share_chain'))} / {fmt(a.get('mae_share_2x2'))} |")
    L.append("")
    L.append(f"SD settings (geometric retention): realized sign agreement of eta_inf - eta_U with the naive fixed-process prediction {fmt(c5['sd_sign_naive_fp'])} vs the exact chain {fmt(c5['sd_sign_chain'])} ({c5['sd_n_decided']} decided).\n")
    L.append("| pop | cells | verifier | delta | realized d_eta [CI] | naive FP pred | chain pred |")
    L.append("|---|---|---|---|---|---|---|")
    for r in c5["sd_settings"]:
        L.append(f"| {r['pop']} | {r['cellset']} | {r['verifier']} | {r['delta']} | {r['d_eta']:+.3f} [{r['d_eta_lo']:+.3f}, {r['d_eta_hi']:+.3f}] | {r['naive_fp_pred']:+.4f} | {r['chain_pred']:+.4f} |")
    L.append("")
    # C6
    L.append("## (vi) Serving records without verification (C6)\n")
    L.append("| c_ell | cell rows | g^W formula >= 0 | predicted g^W >= 0 | realized g^W CI >= 0 | realized g^W CI < 0 | sign agreement |")
    L.append("|---|---|---|---|---|---|---|")
    for k, a in T["C6"].items():
        L.append(f"| {k} | {a['n_cells']} | {a['cells_gW_formula_nonneg']} | {a['cells_gW_pred_nonneg']} | {a['cells_gW_real_ci_nonneg']} | {a['cells_gW_real_ci_neg']} | {fmt(a['sign_agreement_pred_vs_real'])} |")
    L.append("")
    L.append("| verifier | c_ell | cell | v^U/c^U | eta_U | g^W formula | g^W pred | g^W real [CI] |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in T["serve_cells"]:
        if r["delta"] == 0.0 and r["verifier"] == "W1":
            L.append(f"| {r['verifier']} | {r['c_ell']:.5f} | {r['cell']} | {fmt(r['effU'], 1)} | {fmt(r['eta_U'], 1)} | {r['gW_formula']:+.4f} | {r['gW']:+.4f} | {r['gW_real']:+.4f} [{r['gW_real_lo']:+.4f}, {r['gW_real_hi']:+.4f}] |")
    L.append("")
    # hypothesis violations
    L.append("## Hypothesis violations measured\n")
    L.append(f"- g^I > 0 > g^W (eq:noise-threshold's premise) holds in {c2['n_hypothesis_cells']}/{c2['n_cells_total']} cell rows (sign patterns of (g^I, g^W): {c2['hypothesis_pattern']}; 'undefined' = no accepted-and-correct or no accepted-and-incorrect write of any model in the probing cell). A record is a model name, and the eight models differ up to 20x in price, so in dollars both record types usually move value and cost in the same direction: relative to an Opus prior any record points to a cheaper model (both g > 0), relative to a cheap prior (algorithm: qwen3.5-plus, data_science: kimi-k2.5) to a dearer one (both g < 0). The general per-cell sign rule sign(omega g^I + psi g^W) does not need the premise (agreement {fmt(c2['general_sign_all_cells']['rate'])} on {c2['general_sign_all_cells']['n_decided']} decided cells).")
    L.append(f"- prop:cascade-records assumes a record-independent attempt cost a_j; here the attempt cost is the fielded model's price, and the term eta_U (a^s - a^U) is {fmt(c3['attempt_cost_term_share_of_gI_median'])} x |g^I| at the median, so sign(g^s) = sign(Delta^s W) only in {fmt(c3['sign_agreement_actual_vs_identity_I'])} / {fmt(c3['sign_agreement_actual_vs_identity_W'])} of cells (median relative gap {fmt(c3['relative_gap_actual_I_median'])} / {fmt(c3['relative_gap_actual_W_median'])}); the identity itself holds to {fmt(c3['identity_max_residual'], 12)}.")
    L.append(f"- Write-type independence of the record state (standing assumption of sec:memory) is rejected at 0.01 in {c5.get('SD', {}).get('rejected_at_0_01')}/{c5.get('SD', {}).get('n_tested')} cell rows under ACRouter-like writes and in {c5.get('FP', {}).get('rejected_at_0_01')}/{c5.get('FP', {}).get('n_tested')} under exploration writes (control).")
    a6 = T["C6"].get(k0(T), {})
    L.append(f"- Serving without verification (c_ell = 0): the hypothesis g^W < 0 of thm:memory fails in {a6.get('cells_gW_formula_nonneg')}/{a6.get('n_cells')} cell rows by the formula of app:cascade-records and the realized state-W gain has a CI at or above 0 in {a6.get('cells_gW_real_ci_nonneg')} of them (table (vi)).")
    L.append("- The i.i.d.-arrival assumption is met by construction on the replay streams (draws with replacement); the released id_test order is not a stream, so the arrival law of a deployed ACRouter is not tested here.\n")
    # audit
    A = RES["audit"]
    L.append("## Audit (independent recomputation)\n")
    L.append(f"1. Stream 0 late-window sums (V_I, D_I, V_U, D_U) of the setting {A['setting_audited']} re-simulated with a scalar Python loop and a dict store: engine {[round(x, 6) for x in A['audit1_stream0_late_sums']['engine']]}, scalar {[round(x, 6) for x in A['audit1_stream0_late_sums']['scalar']]}, max |diff| {A['audit1_stream0_late_sums']['max_abs_diff']:.2e}.")
    L.append(f"2. eta_U of the same setting via pandas group-bys: {A['audit2_eta_U_pandas']['pandas']:.6f} vs engine {A['audit2_eta_U_pandas']['engine']:.6f} (|diff| {A['audit2_eta_U_pandas']['abs_diff']:.2e}).")
    a3 = A["audit3_fifo_formula"]
    L.append(f"3. eq:fifo for cell {a3['cell']} (q_j = {a3['q_j']:.4f}, beta = {a3['beta']:.4f}) at M = {a3['M']} with Python floats: {a3['python']:.6f} vs numpy {a3['numpy']:.6f} (|diff| {a3['abs_diff']:.2e}); realized coverage in the replay {a3['realized']:.4f}; the mean-span substitution gives {a3['substitution_python']:.6f} (below the formula: {a3['substitution_below_formula']}).\n")
    # deviations / limitations
    L.append("## Deviations from the protocol and limitations\n")
    L.append("- ACRouter's real Memory logs (采集 A) do not exist; the memory is replayed on recorded outcomes with our verifiers, so the results test the record theory on an ACRouter-STYLE store, not the released system. Records carry (model) only; rejected attempts write nothing (no negative records).")
    L.append("- Only 12 of the 21 dimension x source cells have an execution-based verifier; the 21-cell rows use the oracle (released score) or the syntax-only / non-empty W1 rule, whose truth for the Proxy+ dimensions is the released judge score.")
    sens = [r for r in T["settings"] if r["pool"] == "SENS"]
    L.append("- All-zero bcb / lcb / ds1000 tasks (released grader failure, Addendum 2) are excluded from every cell estimate; the sensitivity rows `c21all` (all 2,919 id_test tasks, oracle W0) give " + "; ".join(f"delta {r['rpar']}: predicted {r['pred_gain']:+.4f}, realized d_eta {r['d_eta']:+.3f} [{r['d_eta_lo']:+.3f}, {r['d_eta_hi']:+.3f}]" for r in sens) + " (same signs as the excluded-task rows).")
    L.append("- The kNN store is approximated by TF-IDF k-means clusters (k = 50) because no embedding model is available; the top-k (k = 5) majority readout is a diagnostic row only (the newest-record weights of thm:memory do not describe it, rem:accumulating-stores).")
    L.append("- W3 (LLM judge): X8's calibration ranks grok-4.6 best (AUROC 0.842) and deepseek-v4-pro poor (0.656). The 300-task W3 was run with deepseek-v4-pro (thinking off) per the coordinator's first instruction, before the grok result was known; a full grok W3 is unaffordable in ledger units (~150-230 units per call at the uncalibrated default rate, ~0.6M for 300 tasks x 8 models against a shared cap of 300k), so grok was run as a 2-cell pilot on the same pairs as deepseek (144 calls, 13.2k ledger units, 0 relay-error rows; the packy patch for the intermittent HTTP 400 'Incorrect API key' was in place and no row failed). The pilot is a judge comparison, not a claim pool: its populations are too small for out-of-sample primitives and are excluded from C1 / C2 / C5 (numbers reported separately).")
    L.append("- Empty recorded responses (data gap) are unverifiable: every verifier is imputed the released truth on them; counts are in `results.json` (`config.populations.*.stats`).")
    L.append("- The LRU prediction uses the realized coverage (no closed form for finite J); TTL / FIFO predictions use eq:ttl / eq:fifo with probing r_j.")
    L.append("- The replay mixes exploration rounds into every state's value and cost (as X4 did), so eta_U here is the efficiency of the exploring uninformed router, not of the pure prior.")
    L.append("- Streams are long (60,000 rounds) so that a FIFO of 5,000 write events fills; the 867-task population is therefore visited ~69 times per stream (the record process, not the task pool, is the object of study).")
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    # summary for paper
    S: List[str] = []
    c1 = T["C1"]["all"]; c2t = T["C2"]["threshold_rule_on_hypothesis_cells"]; f = T["C4"]["fifo"]; s5 = T["C5"].get("SD", {})
    S.append(f"X13 memory (SEMI-SYNTHETIC replay of an ACRouter-style record store on recorded CodeRouterBench outcomes; real sandbox verifiers; probing primitives, id_test realized; no ACRouter memory logs exist):")
    S.append(f"1. thm:memory sign criterion: sign(sum nu_j G_j) = sign(eta_inf - eta_U) in {c1['agree']}/{c1['n_decided']} CI-decided fixed-process settings ({fmt(c1['rate'])}; naive always-+ {fmt(c1['naive_always_pos'])}, omega>psi {fmt(c1['naive_omega_gt_psi'])}) -> {RES['claims'][0]['verdict']}.")
    S.append(f"2. eq:noise-threshold per cell: {c2t['agree']}/{c2t['n_decided']} decided hypothesis cells ({fmt(c2t['rate'])}; majority-rule baseline {fmt(c2t['naive_majority'])}); premise g^I>0>g^W holds in {T['C2']['n_hypothesis_cells']}/{T['C2']['n_cells_total']} cell rows -> {RES['claims'][1]['verdict']}.")
    S.append(f"3. prop:cascade-records identity residual {fmt(T['C3']['identity_max_residual'], 12)}; W-form = g-form threshold (max gap {fmt(T['C3']['threshold_gap_max'], 6)}); with model-dependent attempt costs sign(g^s) = sign(Delta^s W) in {fmt(T['C3']['sign_agreement_actual_vs_identity_I'])}/{fmt(T['C3']['sign_agreement_actual_vs_identity_W'])} of cells -> {RES['claims'][2]['verdict']} (IDENTITY).")
    S.append(f"4. eq:fifo vs mean-span substitution: eq:fifo inside the realized 95% CI in {fmt(f.get('formula_insample_in_ci_share'))} of {f.get('n_unsaturated')} unsaturated FIFO points (realized write rates; probing rates {fmt(f.get('formula_in_ci_share'))}); on the {f.get('n_detectable')} points where the predicted gap exceeds the CI half-width eq:fifo is closer in {fmt(f.get('det_formula_beats_substitution_share_insample'))} and the substitution under-estimates in {fmt(f.get('det_substitution_underestimates_share_insample'))} (MAE {fmt(f.get('det_mae_formula_insample'), 4)} vs {fmt(f.get('det_mae_substitution_insample'), 4)}); max predicted gap only {fmt(f.get('max_predicted_gap'), 3)} at these beta -> {RES['claims'][3]['verdict']}.")
    S.append(f"5. State-dependent writes: independence rejected in {s5.get('rejected_at_0_01')}/{s5.get('n_tested')} cells; MAE of w*: fixed-process {fmt(s5.get('mae_w_fp_oos'))} vs chain {fmt(s5.get('mae_w_chain'))} vs 2x2 {fmt(s5.get('mae_w_2x2'))}; sign agreement naive FP {fmt(T['C5']['sd_sign_naive_fp'])} vs chain {fmt(T['C5']['sd_sign_chain'])} -> {RES['claims'][4]['verdict']} (diagnostic).")
    k00 = [k for k in T["C6"] if k.startswith("c_ell=0.00000")]
    a6 = T["C6"][k00[0]] if k00 else {}
    S.append(f"6. Serve-without-verification: g^W >= 0 in {a6.get('cells_gW_formula_nonneg')}/{a6.get('n_cells')} cell rows at c_ell = 0 (realized CI >= 0 in {a6.get('cells_gW_real_ci_nonneg')}) -> {RES['claims'][5]['verdict']} (diagnostic).")
    vw = {r["verifier"]: r for r in T["verifiers"] if r["pop"] == "c12" and r["source"] == "all"}
    S.append("7. Verifier operating points (12-cell population): " + "; ".join(f"{v} TPR {fmt(r['TPR'])} FPR {fmt(r['FPR'])} erroneous share of accepts {fmt(r['false_accept_share'])}" for v, r in vw.items()) + ".")
    jg = RES["config"].get("judge_grok", {})
    S.append(f"8. Judge writers: W3 deepseek-v4-pro AUROC {fmt(RES['config'].get('judge', {}).get('own_check', {}).get('auroc'))} on {RES['config'].get('judge', {}).get('own_check', {}).get('n_pairs')} judged pairs (X8 table 0.656), erroneous share of accepts {fmt(next((r['false_accept_share'] for r in T['verifiers'] if r['pop'] == 'c12j' and r['verifier'] == 'W3' and r['source'] == 'all'), None))} vs W1 {fmt(next((r['false_accept_share'] for r in T['verifiers'] if r['pop'] == 'c12j' and r['verifier'] == 'W1' and r['source'] == 'all'), None))}, W2 {fmt(next((r['false_accept_share'] for r in T['verifiers'] if r['pop'] == 'c12j' and r['verifier'] == 'W2' and r['source'] == 'all'), None))}; grok-4.6 pilot (2 mbpp cells, {jg.get('own_check', {}).get('n_pairs')} pairs): AUROC {fmt(jg.get('own_check', {}).get('auroc'))} vs deepseek {fmt(jg.get('deepseek_on_same_pairs', {}).get('auroc'))} on the same pairs -- judges cannot see the function name the MBPP+ harness requires.")
    S.append(f"Runtime {RES['runtime_s']} s; API spend {RES['spend']['units_est']:.0f} ledger units ({RES['spend']['prompt_tokens'] + RES['spend']['completion_tokens']:,} tokens: deepseek {RES['spend']['by_model'].get('deepseek-v4-pro', {}).get('units_est', 0):.0f} units at the calibrated rate, grok {RES['spend']['by_model'].get('grok-4.6', {}).get('units_est', 0):.0f} at the default rate).")
    with open(os.path.join(OUT, "summary_for_paper.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(S) + "\n")


def k0(T: Dict[str, Any]) -> str:
    """Key of the c_ell = 0 serve summary."""
    ks = [k for k in T["C6"] if k.startswith("c_ell=0.00000")]
    return ks[0] if ks else ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["sandbox", "judge", "grok", "all"])
    ap.add_argument("--limit", type=int, default=None, help="sandbox/judge phase: cap the number of jobs (testing)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="short streams (6,000 rounds, late window 1,000) for a smoke run")
    ap.add_argument("--skip-judge", action="store_true", help="do not make judge calls (use the cached verdicts only)")
    ap.add_argument("--skip-sandbox", action="store_true", help="do not run sandbox jobs (use the cached verdicts only)")
    ap.add_argument("--collect", action="store_true", help="collect missing sandbox and model-judge records")
    args = ap.parse_args()
    if not args.collect:
        args.skip_judge = True
        args.skip_sandbox = True
        if args.phase != "all":
            ap.error("Individual collection phases require --collect; use --phase all to replay cached records")
    if args.quick:
        N_ROUNDS, LATE = 6000, 1000
    if args.phase == "sandbox":
        main_phase_sandbox(limit=args.limit, workers=args.workers or SANDBOX_WORKERS)
    elif args.phase in ("judge", "grok"):
        D = load_outcomes()
        P = build_population(D, np.random.default_rng(SEED))
        P["D"] = D
        P["rm"] = json.load(open(RESPONSE_MATRIX, "r", encoding="utf-8"))
        if args.phase == "judge":
            meta = run_judge_phase(P, workers=args.workers or JUDGE_WORKERS, limit=args.limit)
            json.dump(meta, open(os.path.join(CACHE, "judge_meta.json"), "w"), indent=1)
        else:
            jm = json.load(open(os.path.join(CACHE, "judge_meta.json")))
            meta = run_grok_pilot(P, jm, workers=args.workers or GROK_WORKERS, limit=args.limit)
            json.dump(meta, open(GROK_META, "w"), indent=1)
        print(json.dumps({k: v for k, v in meta.items() if k not in ("test_tasks", "probe_tasks")}, indent=1))
    else:
        main_all(skip_judge=args.skip_judge, skip_sandbox=args.skip_sandbox)

