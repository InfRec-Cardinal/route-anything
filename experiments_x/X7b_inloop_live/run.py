#!/usr/bin/env python
"""X7-b / X10-live: in-loop routing as repeated threshold decisions, with REAL retries (LIVE).

Design (see README.md for the full protocol)
--------------------------------------------
* Tasks: EXEC-CLEAN id_test tasks (humaneval, he dim4, mbpp dim1/dim4, hex python, bcb) that are runnable in the
  local sandbox and not all-zero across the 8 recorded models (uniform exclusion; the Addendum applies it to bcb, we
  apply it to every source because the recorded Opus terminal fallback and W_lambda are meaningless on such tasks).
  Stratified by source with seed 20260917; the frame has fewer than 300 tasks so the whole frame is used.
* Models: qwen3.5-plus (cheap), qwen3-max, kimi-k2.5 (benchmark models with recorded attempt 0) and deepseek-v4-pro
  (NON-BENCHMARK cheap model, thinking disabled, no recorded attempt 0; its dollar price is a PROXY = Qwen3-Max list
  price, its primary bases are tokens and seconds).
* Chain per (task, root model): attempt 1 = fresh generation at T=0.7.  If the verifier (hidden-test sandbox =
  truth) rejects: attempt 2/3 (a) RETRY the same model with the failure feedback appended (sandbox error message,
  <= 800 chars), (b) SWITCH to the next-stronger model (qwen3.5-plus -> qwen3-max -> kimi-k2.5; deepseek -> qwen3-max)
  with the same feedback; plus (c) a FRESH retry (same model, no feedback, seed=2) at attempt 2 for the salvage test.
  Terminal fallback after K=3 = the recorded Opus outcome (csv score / cost / latency).
* Verifiers: hidden tests (common.sandbox.verify_response; = truth for live attempts) and, as the imperfect variant,
  visible tests (doctest / assert examples in the prompt, ACRouter's run_visible_tests semantics).
* Every call goes through common.packy (disk cache; re-runs never re-spend); tag X7b; per-model workers
  qwen3.5-plus 4, qwen3-max 4, kimi-k2.5 2, deepseek-v4-pro 4; partial results persisted every 50 chains.

Usage: run.py [--limit N] [--models a,b] [--fresh] [--no-figs]
Outputs (this directory): attempts.jsonl (raw log), recorded.jsonl, partial_progress.json, results.json, README.md,
summary_for_paper.md, figures/*.pdf.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

T0 = time.time()
OUT = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(OUT)
sys.path.insert(0, EX)
from common.packy import PackyClient, BudgetExceeded, ModelUnavailable, RelayError  # noqa: E402
from common import sources, sandbox  # noqa: E402

# ================================================================================================ configuration
SEED = 20260917
B = 1000
TAG = "X7b"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")
CSV = f"{DATA}/coderouterbench/id_results_long.csv"
RESP = f"{DATA}/matrices/phase1_acrouter_v2/response_matrix.json"
PRICES = f"{DATA}/coderouterbench/models.json"
BASE = "claude-opus-4-6"
LIVE_MODELS = ["qwen3.5-plus", "qwen3-max", "kimi-k2.5", "deepseek-v4-pro", "deepseek-flash"]
CSV_NAME = {"qwen3.5-plus": "qwen3.5-plus", "qwen3-max": "Qwen3-Max", "kimi-k2.5": "kimi-k2.5"}  # deepseek models: none
NEXT = {"qwen3.5-plus": "qwen3-max", "qwen3-max": "kimi-k2.5", "kimi-k2.5": None, "deepseek-v4-pro": "qwen3-max", "deepseek-flash": "qwen3-max"}  # switch-2 target
# switch-3 target: the paper's chain qwen3.5-plus -> qwen3-max -> kimi-k2.5 is kept in full; the two non-benchmark roots stop at
# qwen3-max (kimi-k2.5 is capped at 2 concurrent calls shared by all experiments and generated ~10 tok/s under load in the smoke run,
# so the ~50 extra long-context kimi calls would have pushed the run past the 2-hour wall-clock limit).
NEXT3 = {"qwen3.5-plus": "kimi-k2.5", "qwen3-max": None, "kimi-k2.5": None, "deepseek-v4-pro": None, "deepseek-flash": None}
WORKERS = {"qwen3.5-plus": 4, "qwen3-max": 4, "kimi-k2.5": 2, "deepseek-v4-pro": 4, "deepseek-flash": 4}
NON_BENCHMARK = {"deepseek-v4-pro", "deepseek-flash"}  # deepseek-flash added per the coordinator's Addendum-2 message (second cheap non-benchmark model)
TEMP = 0.7
# 2048 completion tokens for the code-only outputs (the smoke run cut one qwen3-max bcb answer at 1024); deepseek-flash
# reasons before answering and its reasoning tokens are billed inside the completion budget (at 2048 it returned empty
# content on 22/26 bcb calls), so it gets 6144.
MAX_TOKENS = 2048
MAX_TOKENS_BY_MODEL = {"deepseek-flash": 6144}
FEEDBACK_CHARS = 800
K_MAX = 3
SOURCES = ["humaneval", "he", "mbpp", "hex", "bcb"]
N_TARGET = 300
LAMBDA_FIXED = [1.0, 10.0]
BASES = ["usd", "sec"]
EPS = [0.0, 0.01, 0.02, 0.05]
# Ledger units at the DEFAULT 0.05 units/token rate (only deepseek-v4-pro has a calibrated rate, 0.00034); the coordinator
# ruled (Addendum 2) that the allocation is not a constraint and that attempts/tasks must not be cut for budget reasons, and
# raised X7b's allocation to 200,000 units for the deepseek-flash re-run pass; this soft stop sits just below it (packy's
# global cap is the hard stop).  If it is hit, the run stops and reports what was not re-run instead of skipping silently.
SOFT_CAP_UNITS = 195000.0
# deepseek-flash truncation protocol (coordinator, 2026-09-17 18:55): a flash attempt whose 6,144-token completion budget was
# spent entirely on reasoning (finish_reason 'length', empty content) is re-issued with the same messages at 16,384 tokens
# under a separate tag; the original row is kept in attempts_flash_truncated.jsonl; a re-run that still truncates is CENSORED
# (excluded from p estimates).  Downstream attempts of such a chain are regenerated from the re-run's real verdict.
FLASH_RERUN_MAX_TOKENS = 16384
RERUN_TAG = "X7b:flash_rerun"
TRUNC_PATH = os.path.join(OUT, "attempts_flash_truncated.jsonl")
THETAS_VIS = [1.0, 0.5]
SYS_PROMPT = ("You are an expert Python programmer. Reply with the complete solution in a single ```python code block "
              "and nothing else.")
FIX_INSTR = ("\n\nFix the code and return the complete corrected solution in a single ```python code block and nothing else.")
ATTEMPTS_PATH = os.path.join(OUT, "attempts.jsonl")
RECORDED_PATH = os.path.join(OUT, "recorded.jsonl")
PROGRESS_PATH = os.path.join(OUT, "partial_progress.json")

# palette (dataviz skill reference instance; same slots as X7-a): retry = slot 1, switch = slot 2, fresh = slot 3
COL = {"retry": "#2a78d6", "switch": "#eb6834", "fresh": "#1baf7a", "k1": "#4a3aa7", "opus": "#898781", "ink": "#0b0b0b",
       "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}


def log(*a: Any) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ================================================================================================ estimators
def boot_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """(B+1) x n; row 0 = ones (point estimate), rows 1..B = multinomial bootstrap counts (tasks resampled with replacement)."""
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    return np.vstack([np.ones((1, n)), W])


def wmean(W: np.ndarray, mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Weighted mean of x over the tasks in mask for every bootstrap row (paired across all quantities)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.full(W.shape[0], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (W[:, idx] @ x[idx]) / W[:, idx].sum(1)


def wsum(W: np.ndarray, mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.full(W.shape[0], np.nan)
    return W[:, idx] @ x[idx]


def ci(v: np.ndarray) -> List[float]:
    v = np.asarray(v, float)
    w = v[1:][np.isfinite(v[1:])]
    if not np.isfinite(v[0]) or len(w) < 50:
        return [float(v[0]) if np.isfinite(v[0]) else float("nan"), float("nan"), float("nan")]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]


def excl(t: List[float], x: float = 0.0) -> bool:
    return bool(np.isfinite(t[1]) and np.isfinite(t[2]) and (t[1] > x or t[2] < x))


def wilson(k: int, n: int) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan"), float("nan")]
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, c - h, c + h]


def f3(t: List[float], nd: int = 3) -> str:
    if t is None or not np.isfinite(t[0]):
        return "n/a"
    if not np.isfinite(t[1]):
        return f"{t[0]:.{nd}f}"
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def fm(t: List[float], s: float = 1000.0, nd: int = 2) -> str:
    if t is None or not np.isfinite(t[0]):
        return "n/a"
    if not np.isfinite(t[1]):
        return f"{s * t[0]:.{nd}f}"
    return f"{s * t[0]:.{nd}f} [{s * t[1]:.{nd}f}, {s * t[2]:.{nd}f}]"


def table(hdr: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ================================================================================================ data
def load_prices() -> Dict[str, Tuple[float, float]]:
    js = json.load(open(PRICES))
    pr = {m["model"]: (float(m["input_per_1m"]), float(m["output_per_1m"])) for m in js["models"]}
    out = {}
    for live, csvn in CSV_NAME.items():
        out[live] = pr[csvn]
    out["deepseek-v4-pro"] = pr["Qwen3-Max"]  # PROXY price (the deepseek models are not in the benchmark's price list)
    out["deepseek-flash"] = pr["qwen3.5-plus"]  # PROXY price: cheapest listed tier (flash-class model)
    return out


PRICE = load_prices()


def usd_cost(model: str, pin: int, pout: int) -> float:
    a, b = PRICE[model]
    return (pin * a + pout * b) / 1e6


def build_frame() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    df = pd.read_csv(CSV)
    idt = df[df.split == "id_test"].copy()
    S = idt.pivot(index="task_id", columns="model", values="score")
    Cst = idt.pivot(index="task_id", columns="model", values="cost_usd")
    L = idt.pivot(index="task_id", columns="model", values="latency_ms") / 1000.0
    TT = idt.pivot(index="task_id", columns="model", values="total_tokens")
    allzero = S.max(axis=1) <= 0
    rows = []
    tasks: Dict[str, Dict[str, Any]] = {}
    counts = {"frame_candidates": 0, "excluded_not_runnable": 0, "excluded_allzero": 0, "excluded_other_variant": 0}
    for tid in S.index:
        parts = tid.split("_")
        dim, src = parts[0], parts[1]
        if src not in SOURCES:
            continue
        if src == "hex" and "_python_" not in tid:
            counts["excluded_other_variant"] += 1
            continue
        if src in ("he", "mbpp") and dim == "dim9":
            counts["excluded_other_variant"] += 1
            continue
        counts["frame_candidates"] += 1
        if allzero[tid]:
            counts["excluded_allzero"] += 1
            continue
        t = sources.load_task(tid)
        if not t["runnable"]:
            counts["excluded_not_runnable"] += 1
            continue
        tasks[tid] = t
        rows.append(dict(task_id=tid, source=src, dim=dim, opus_score=float(S.loc[tid, BASE]), opus_usd=float(Cst.loc[tid, BASE]),
                         opus_sec=float(L.loc[tid, BASE]), opus_tok=float(TT.loc[tid, BASE]),
                         **{f"rec_score_{m}": float(S.loc[tid, CSV_NAME[m]]) for m in CSV_NAME},
                         **{f"rec_usd_{m}": float(Cst.loc[tid, CSV_NAME[m]]) for m in CSV_NAME},
                         **{f"rec_sec_{m}": float(L.loc[tid, CSV_NAME[m]]) for m in CSV_NAME},
                         **{f"rec_tok_{m}": float(TT.loc[tid, CSV_NAME[m]]) for m in CSV_NAME}))
    F = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    counts["frame_size"] = int(len(F))
    return F, counts, tasks


def stratified_sample(F: pd.DataFrame, n_target: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(F) <= n_target:
        return F.copy()
    per = F.groupby("source").size()
    quota = per * n_target / len(F)
    base = np.floor(quota).astype(int)
    rem = n_target - int(base.sum())
    frac = (quota - base).sort_values(ascending=False)
    for s in frac.index[:rem]:
        base[s] += 1
    keep = []
    for s, n in base.items():
        ids = F[F.source == s].task_id.to_numpy()
        keep.extend(sorted(rng.choice(ids, size=int(n), replace=False)))
    return F[F.task_id.isin(keep)].reset_index(drop=True)


# ================================================================================================ prompts / verification
def first_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = task["prompt"]
    if task["source"] == "mbpp" and task["dim"] == "dim1" and task.get("entry_point"):
        # The MBPP+ harness calls the reference entry point; the released prompt did not show it (10-35 % of the recorded
        # responses guess it), so without the name the failure set would be dominated by naming mismatches that the
        # NameError feedback fixes trivially.  Deviation from attempt 0, documented in the README.
        prompt = prompt.rstrip() + f"\n\nName the function `{task['entry_point']}`."
    return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": prompt}]


def make_feedback(res: Dict[str, Any]) -> str:
    """Error message of the failed hidden-test run, truncated to FEEDBACK_CHARS (tail of stderr = last traceback frames)."""
    if res.get("timed_out"):
        head = "timeout: the tests did not finish within the time limit"
    else:
        head = str(res.get("error") or "the tests failed")
    tail = (res.get("stderr") or "").strip()
    if not tail and res.get("stdout"):
        tail = str(res["stdout"]).strip()
    text = head
    room = FEEDBACK_CHARS - len(head) - 1
    if tail and room > 40:
        text = head + "\n" + (tail[-room:] if len(tail) > room else tail)
    return text[:FEEDBACK_CHARS]


def followup_messages(prev: List[Dict[str, str]], prev_text: str, feedback: str) -> List[Dict[str, str]]:
    return list(prev) + [{"role": "assistant", "content": (prev_text or "")[:6000]},
                         {"role": "user", "content": "Your solution failed the tests.\nError:\n" + feedback + FIX_INSTR}]


_VIS_DOCTEST = re.compile(r">>>\s*(.+?)\n\s*(.+)$", re.M)
_VIS_ASSERT = re.compile(r"assert\s+(.+?)\s*==\s*(.+)$", re.M)


def visible_examples(task: Dict[str, Any]) -> List[Tuple[str, str]]:
    """ACRouter run_visible_tests extraction: '>>> call \\n expected' pairs (else 'assert call == expected'), cap 5.
    Deviation: pairs whose expected line is itself a '>>>' / '...' line (multi-statement doctests) are dropped."""
    prompt = task.get("raw_prompt") or task.get("prompt") or ""
    ex = [(c, e) for c, e in _VIS_DOCTEST.findall(prompt) if not e.strip().startswith((">>>", "..."))]
    if not ex:
        ex = _VIS_ASSERT.findall(prompt)
    return ex[:5]


def _prompt_imports(code: str, prompt: Optional[str]) -> str:
    missing = [l.rstrip() for l in (prompt or "").split("\n") if re.match(r"^(import\s+\S|from\s+\S+\s+import\s)", l) and l.rstrip() not in code]
    return ("\n".join(missing) + "\n" + code) if missing else code


def solution_code(task: Dict[str, Any], text: str) -> str:
    code = sandbox.extract_code(text or "", "python", entry_point=task.get("entry_point"))
    if task["source"] in ("bcb", "humaneval", "he", "hex"):
        code = _prompt_imports(code, task.get("raw_prompt"))
    return code


def verify_visible(task: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Imperfect verifier: visible examples only.  score = pass fraction (0.5 = neutral when no examples), syntax_ok."""
    code = solution_code(task, text)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(code, "<sol>", "exec")
        syntax_ok = bool(code.strip())
    except Exception:
        syntax_ok = False
    ex = visible_examples(task)
    if not ex or not syntax_ok:
        return dict(score=0.5 if syntax_ok else 0.0, n=len(ex), n_pass=0, syntax_ok=syntax_ok, elapsed_s=0.0)
    harness = ["\nimport json as _j", "_res = []"]
    for call, exp in ex:
        harness.append("try:\n    _r = " + call.strip() + "\n    _e = eval(" + repr(exp.strip()) + ")\n    _res.append(bool(_r == _e))\nexcept BaseException:\n    _res.append(False)")
    harness.append("print('__VIS__ ' + _j.dumps(_res))")
    r = sandbox.run_python_tests(code, "\n".join(harness), timeout_s=10)
    m = re.search(r"__VIS__ (\[.*\])", r.get("stdout", ""))
    res = json.loads(m.group(1)) if m else [False] * len(ex)
    return dict(score=sum(res) / len(ex), n=len(ex), n_pass=int(sum(res)), syntax_ok=True, elapsed_s=float(r.get("elapsed_s", 0.0)))


def vis_accept(row: Dict[str, Any], theta: float) -> bool:
    return bool(row.get("vis_syntax_ok")) and float(row.get("vis_score", 0.0)) >= theta


# ================================================================================================ live collection
class SoftBudget(Exception):
    pass


def is_truncated(r: Dict[str, Any]) -> bool:
    """A completion whose whole budget went to reasoning: finish_reason 'length' and no visible content."""
    return r.get("finish_reason") == "length" and not (r.get("text") or "").strip()


class Collector:
    def __init__(self, client: PackyClient, tasks: Dict[str, Dict[str, Any]], frame: pd.DataFrame) -> None:
        self.c = client
        self.tasks = tasks
        self.frame = frame.set_index("task_id")
        self.sems = {m: threading.Semaphore(WORKERS[m]) for m in LIVE_MODELS}
        self.lock = threading.Lock()
        self.rows: List[Dict[str, Any]] = []
        self.done_jobs = 0
        self.n_jobs = 0
        self.stop = False
        self._tag_units = 0.0
        self._tag_scan = 0.0
        self.errors: List[str] = []
        self.flash_rerun = False  # second pass: re-issue truncated deepseek-flash attempts at FLASH_RERUN_MAX_TOKENS
        self.n_truncated_saved = 0
        self.rerun_not_completed = 0

    # ---- budget guard on THIS experiment's tags (packy guards only the global cap)
    def tag_units(self, force: bool = False) -> float:
        if not force and time.time() - self._tag_scan < 10:
            return self._tag_units
        total = 0.0
        if os.path.exists(self.c.ledger_path):
            with open(self.c.ledger_path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if str(r.get("tag") or "").startswith(TAG):
                        total += float(r.get("units_est") or 0.0)
        self._tag_units, self._tag_scan = total, time.time()
        return total

    def call(self, model: str, messages: List[Dict[str, str]], seed: Optional[int] = None, max_tokens: Optional[int] = None, tag: str = TAG) -> Dict[str, Any]:
        if self.stop:
            raise SoftBudget("stopped")
        extra: Dict[str, Any] = {}
        if model == "deepseek-v4-pro":
            extra["thinking"] = {"type": "disabled"}  # deepseek-flash keeps its default reasoning (that is the model)
        if seed is not None:
            extra["seed"] = int(seed)
        mt = int(max_tokens) if max_tokens is not None else MAX_TOKENS_BY_MODEL.get(model, MAX_TOKENS)
        with self.sems[model]:
            if self.tag_units() > SOFT_CAP_UNITS:
                self.stop = True
                raise SoftBudget(f"X7b ledger units {self._tag_units:.0f} > soft cap {SOFT_CAP_UNITS:.0f}")
            r = self.c.chat(model, messages, max_tokens=mt, temperature=TEMP, tag=tag, extra=extra or None)
        r["max_tokens_used"] = mt
        return r

    def _call_safe(self, model: str, messages: List[Dict[str, str]], seed: Optional[int], max_tokens: Optional[int] = None, tag: str = TAG) -> Dict[str, Any]:
        try:
            return self.call(model, messages, seed, max_tokens=max_tokens, tag=tag)
        except RelayError as e:  # retries exhausted: keep the chain's earlier rows, mark this attempt as an API error
            return {"text": "", "usage": {}, "latency_s": 0.0, "cached": False, "finish_reason": "error", "error": str(e)[:300], "cache_key": None,
                    "max_tokens_used": int(max_tokens) if max_tokens is not None else MAX_TOKENS_BY_MODEL.get(model, MAX_TOKENS)}

    def _row(self, task: Dict[str, Any], root: str, branch: str, k: int, model: str, seed: Optional[int], r: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        text = r.get("text") or ""
        hid = sandbox.verify_response(task, text)
        vis = verify_visible(task, text)
        u = r.get("usage") or {}
        pin, pout, ptot = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), int(u.get("total_tokens") or 0)
        row = dict(job_id=f"{task['task_id']}|{root}", task_id=task["task_id"], source=task["source"], dim=task["dim"], root_model=root, branch=branch, k=k,
                   model=model, seed=seed, cached=bool(r.get("cached")), finish_reason=r.get("finish_reason"), api_error=bool(r.get("error")),
                   max_tokens=int(r.get("max_tokens_used") or 0), rerun=False, censored=False, collection_pass=("rerun" if self.flash_rerun else "main"),
                   prompt_tokens=pin, completion_tokens=pout, total_tokens=ptot, latency_s=float(r.get("latency_s") or 0.0),
                   sandbox_s=float(hid.get("elapsed_s") or 0.0), cost_usd=usd_cost(model, pin, pout),
                   truth_pass=bool(hid.get("passed")), truth_frac=float(hid.get("frac") or 0.0), failure_kind=hid.get("failure_kind"), error=hid.get("error"),
                   vis_score=vis["score"], vis_n=vis["n"], vis_n_pass=vis["n_pass"], vis_syntax_ok=vis["syntax_ok"], vis_s=vis["elapsed_s"],
                   text=text[:6000], code=(hid.get("code") or "")[:4000], cache_key=r.get("cache_key"), ts=time.time())
        row["cost_sec"] = row["latency_s"] + row["sandbox_s"]
        return row, hid

    def _save_truncated(self, row: Dict[str, Any]) -> None:
        with self.lock:
            with open(TRUNC_PATH, "a") as fh:
                fh.write(json.dumps(jsonable(dict(row, truncated_original=True)), ensure_ascii=False) + "\n")
            self.n_truncated_saved += 1

    def attempt(self, task: Dict[str, Any], root: str, branch: str, k: int, model: str, messages: List[Dict[str, str]], seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r = self._call_safe(model, messages, seed)
        row, hid = self._row(task, root, branch, k, model, seed, r)
        if self.flash_rerun and model == "deepseek-flash" and not row["api_error"] and is_truncated(r):
            self._save_truncated(row)  # the original attempt: a_k paid, nothing returned (p = 0 by construction)
            r2 = self._call_safe(model, messages, seed, max_tokens=FLASH_RERUN_MAX_TOKENS, tag=RERUN_TAG)
            row, hid = self._row(task, root, branch, k, model, seed, r2)
            row["rerun"] = True
            row["censored"] = (not row["api_error"]) and is_truncated(r2)  # still no content at 16k: excluded from p estimates
        return row, hid

    def chain(self, tid: str, root: str) -> List[Dict[str, Any]]:
        task = self.tasks[tid]
        rows: List[Dict[str, Any]] = []
        msgs1 = first_messages(task)
        r1, h1 = self.attempt(task, root, "root", 1, root, msgs1)
        rows.append(r1)
        if r1["api_error"]:
            return rows
        if r1["truth_pass"]:
            return rows
        fb1 = make_feedback(h1)
        msgs2 = followup_messages(msgs1, r1["text"], fb1)
        # (c) fresh retry: same model, same prompt, no feedback, second sample
        rf, _ = self.attempt(task, root, "fresh", 2, root, msgs1, seed=2)
        rows.append(rf)
        # (a) retry with feedback
        r2, h2 = self.attempt(task, root, "retry", 2, root, msgs2)
        rows.append(r2)
        if not r2["api_error"] and not r2["truth_pass"]:
            msgs3 = followup_messages(msgs2, r2["text"], make_feedback(h2))
            r3, _ = self.attempt(task, root, "retry", 3, root, msgs3)
            rows.append(r3)
        # (b) switch to the next-stronger model with the same feedback
        m2 = NEXT[root]
        if m2 is not None:
            s2, hs2 = self.attempt(task, root, "switch", 2, m2, msgs2)
            rows.append(s2)
            m3 = NEXT3[root]
            if m3 is not None and not s2["api_error"] and not s2["truth_pass"]:
                msgs3s = followup_messages(msgs2, s2["text"], make_feedback(hs2))
                s3, _ = self.attempt(task, root, "switch", 3, m3, msgs3s)
                rows.append(s3)
        return rows

    def run_jobs(self, jobs: List[Tuple[str, str]], done: Dict[str, List[Dict[str, Any]]]) -> None:
        self.n_jobs = len(jobs)
        pending = [j for j in jobs if f"{j[0]}|{j[1]}" not in done]
        for j in jobs:
            jid = f"{j[0]}|{j[1]}"
            if jid in done:
                self.rows.extend(done[jid])
                self.done_jobs += 1
        log(f"jobs: {len(jobs)} total, {len(jobs) - len(pending)} restored from attempts.jsonl, {len(pending)} to run")
        self._execute(pending)
        self.progress()

    def rerun_jobs(self, job_ids: List[str]) -> None:
        """Second pass: replay the given chains with the deepseek-flash truncation fallback (cached steps are free).
        A chain whose replay does not complete (soft budget cap, relay error) keeps its ORIGINAL rows, so nothing vanishes."""
        drop = set(job_ids)
        originals: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.rows:
            if r["job_id"] in drop:
                originals.setdefault(r["job_id"], []).append(r)
        self.rows = [r for r in self.rows if r["job_id"] not in drop]
        pending = [(jid.rsplit("|", 1)[0], jid.rsplit("|", 1)[1]) for jid in job_ids]
        self.n_jobs += len(pending)
        self.flash_rerun = True
        log(f"flash re-run pass: {len(pending)} chains with truncated deepseek-flash attempts replayed at max_tokens {FLASH_RERUN_MAX_TOKENS} (tag {RERUN_TAG})")
        self._execute(pending)
        replayed = {r["job_id"] for r in self.rows}
        missing = [jid for jid in job_ids if jid not in replayed]
        for jid in missing:
            self.rows.extend(originals.get(jid, []))
        self.rerun_not_completed = len(missing)
        if missing:
            log(f"flash re-run pass: {len(missing)} chains NOT replayed (stopped: {self.stop}); their original truncated rows are kept")
        self.progress()

    def _execute(self, pending: List[Tuple[str, str]]) -> None:
        if not pending:
            return
        pools = {m: ThreadPoolExecutor(max_workers=WORKERS[m], thread_name_prefix=f"chain-{m}") for m in LIVE_MODELS}
        futs = {}
        for tid, m in pending:
            futs[pools[m].submit(self.chain, tid, m)] = (tid, m)
        try:
            for fut in as_completed(futs):
                tid, m = futs[fut]
                try:
                    rows = fut.result()
                    if rows:
                        rows[-1]["job_complete"] = True
                    with self.lock:
                        self.rows.extend(rows)
                        with open(ATTEMPTS_PATH, "a") as fh:
                            for r in rows:
                                fh.write(json.dumps(jsonable(r), ensure_ascii=False) + "\n")
                        self.done_jobs += 1
                        if self.done_jobs % 50 == 0 or self.done_jobs == self.n_jobs:
                            self.progress()
                except SoftBudget as e:
                    if not self.stop:
                        log("soft budget cap reached:", e)
                    self.stop = True
                except (BudgetExceeded, ModelUnavailable) as e:
                    log("fatal packy error:", e)
                    self.stop = True
                except Exception as e:  # noqa: BLE001
                    msg = f"{tid}|{m}: {type(e).__name__}: {str(e)[:300]}"
                    self.errors.append(msg)
                    log("job error:", msg)
                    traceback.print_exc()
        finally:
            for p in pools.values():
                p.shutdown(wait=True, cancel_futures=True)

    def progress(self) -> None:
        us = self.c.usage_summary()
        prog = dict(ts=time.time(), elapsed_s=round(time.time() - T0, 1), jobs_done=self.done_jobs, jobs_total=self.n_jobs, rows=len(self.rows),
                    tag_units=self.tag_units(force=True), session=us["session"], ledger_units_total=us["ledger_units_total"], stopped=self.stop,
                    errors=self.errors[-20:])
        with open(PROGRESS_PATH, "w") as fh:
            json.dump(jsonable(prog), fh, indent=1)
        log(f"progress: {self.done_jobs}/{self.n_jobs} jobs, {len(self.rows)} rows, X7b units {prog['tag_units']:.0f}, ledger total {us['ledger_units_total']:.0f}")


REISSUE: Dict[str, int] = {"api_error_rows_all_runs": 0, "jobs_reissued": 0}
VIEW_A_PATH = os.path.join(OUT, "attempts_view_a_6144.jsonl")
VIEW_B_PATH = os.path.join(OUT, "attempts_view_b_rerun16k.jsonl")


def load_job_runs(path: str) -> Dict[str, List[List[Dict[str, Any]]]]:
    """job_id -> list of COMPLETE runs (each a list of rows, file order).  attempts.jsonl is append-only: the main live pass
    wrote one complete run per chain; the deepseek-flash re-run pass appended a second complete run for the replayed
    chains.  View (a) = the first complete run without relay-error rows (6,144-token budget for deepseek-flash, truncations
    counted as failures); view (b) = the last run of a chain that has more than one (the 16,384-token re-run view)."""
    out: Dict[str, List[List[Dict[str, Any]]]] = {}
    if not os.path.exists(path):
        return out
    by: Dict[str, List[Dict[str, Any]]] = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        by.setdefault(r["job_id"], []).append(r)
    for jid, rows in by.items():
        runs: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        for r in rows:
            cur.append(r)
            if r.get("job_complete"):
                runs.append(cur)
                cur = []
        if runs:
            out[jid] = runs
    return out


def with_budget(r: Dict[str, Any], view: str) -> Dict[str, Any]:
    """Copy of an attempt row with the explicit completion budget it was generated under and the view it enters."""
    r = dict(r)
    r["budget"] = int(r.get("max_tokens") or MAX_TOKENS_BY_MODEL.get(r["model"], MAX_TOKENS))
    r["view"] = view
    r.setdefault("rerun", False)
    r.setdefault("censored", False)
    r.setdefault("collection_pass", "main")
    return r


def load_done_jobs(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Complete jobs persisted in attempts.jsonl (last complete run of each job).  A job whose last run contains an API-error
    row (relay error after packy's retries, e.g. the intermittent HTTP 400 'Incorrect API key' flake reported by the
    coordinator) is NOT returned, so a re-run re-issues it: its successful attempts are cache hits (free) and only the
    failed call is repeated through the patched client."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    REISSUE["api_error_rows_all_runs"] = 0
    REISSUE["jobs_reissued"] = 0
    if not os.path.exists(path):
        return out
    by: Dict[str, List[Dict[str, Any]]] = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("api_error"):
            REISSUE["api_error_rows_all_runs"] += 1
        by.setdefault(r["job_id"], []).append(r)
    for jid, rows in by.items():
        if any(r.get("job_complete") for r in rows):
            # keep only the rows of the last complete run of this job (in case of duplicates)
            idx = max(i for i, r in enumerate(rows) if r.get("job_complete"))
            start = 0
            for i in range(idx - 1, -1, -1):
                if rows[i].get("job_complete"):
                    start = i + 1
                    break
            last = rows[start:idx + 1]
            if any(r.get("api_error") for r in last):
                REISSUE["jobs_reissued"] += 1
                continue
            out[jid] = last
    return out


def verify_recorded(tasks: Dict[str, Dict[str, Any]], frame: pd.DataFrame, rm: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Attempt 0: sandbox-verify the recorded responses of the benchmark models (truth = released score)."""
    if os.path.exists(RECORDED_PATH):
        rows = [json.loads(l) for l in open(RECORDED_PATH) if l.strip()]
        R = pd.DataFrame(rows)
        want = {(t, m) for t in frame.task_id for m in CSV_NAME}
        if want.issubset(set(zip(R.task_id, R.model))):
            return R[[(t, m) in want for t, m in zip(R.task_id, R.model)]].reset_index(drop=True)
    jobs = [(t, m) for t in frame.task_id for m in CSV_NAME]

    def one(t: str, m: str) -> Dict[str, Any]:
        d = rm.get(t, {})
        text = d.get(CSV_NAME[m])
        if text is None and m == "qwen3-max":
            text = d.get("通义千问Max")
        text = text or ""
        hid = sandbox.verify_response(tasks[t], text)
        vis = verify_visible(tasks[t], text)
        return dict(task_id=t, model=m, source=tasks[t]["source"], empty=not text.strip(), sandbox_pass=bool(hid.get("passed")), failure_kind=hid.get("failure_kind"),
                    sandbox_s=float(hid.get("elapsed_s") or 0.0), vis_score=vis["score"], vis_n=vis["n"], vis_syntax_ok=vis["syntax_ok"])

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(lambda tm: one(*tm), jobs):
            rows.append(r)
    R = pd.DataFrame(rows)
    with open(RECORDED_PATH, "w") as fh:
        for r in rows:
            fh.write(json.dumps(jsonable(r), ensure_ascii=False) + "\n")
    return R


# ================================================================================================ analysis
class Arrays:
    """Per-root-model per-task arrays (n,) with NaN where the attempt does not exist."""

    def __init__(self, n: int) -> None:
        z = lambda: np.full(n, np.nan)  # noqa: E731
        self.pass_ = {}
        self.cost = {}
        self.vis = {}
        self.syn = {}
        self.fk = {}
        for key in ["1", "f2", "r2", "r3", "s2", "s3"]:
            self.pass_[key] = z()
            self.vis[key] = z()
            self.syn[key] = z()
            self.fk[key] = np.array([None] * n, dtype=object)
            for b in ["usd", "sec", "tok", "api_sec", "lat"]:
                self.cost[(key, b)] = z()
        self.model = {}
        self.api_err = np.zeros(n, bool)
        self.censored = np.zeros(n, int)


KEY = {("root", 1): "1", ("fresh", 2): "f2", ("retry", 2): "r2", ("retry", 3): "r3", ("switch", 2): "s2", ("switch", 3): "s3"}


def build_arrays(A: pd.DataFrame, tasks_order: List[str]) -> Dict[str, Arrays]:
    pos = {t: i for i, t in enumerate(tasks_order)}
    out = {}
    for m in LIVE_MODELS:
        ar = Arrays(len(tasks_order))
        sub = A[A.root_model == m]
        for r in sub.itertuples(index=False):
            key = KEY.get((r.branch, int(r.k)))
            if key is None or r.task_id not in pos:
                continue
            i = pos[r.task_id]
            if r.api_error:
                ar.api_err[i] = True
                continue
            if getattr(r, "censored", False):
                ar.censored[i] += 1  # excluded from p estimates: the attempt stays NaN (as if not made)
                ar.model[key] = r.model
                continue
            ar.pass_[key][i] = 1.0 if r.truth_pass else 0.0
            ar.cost[(key, "usd")][i] = r.cost_usd
            ar.cost[(key, "sec")][i] = r.cost_sec
            ar.cost[(key, "tok")][i] = r.total_tokens
            ar.cost[(key, "api_sec")][i] = r.latency_s
            ar.vis[key][i] = r.vis_score
            ar.syn[key][i] = 1.0 if r.vis_syntax_ok else 0.0
            ar.fk[key][i] = r.failure_kind
            ar.model[key] = r.model
        out[m] = ar
    return out


def chain_arrays(ar: Arrays, branch: str, K: int, b_cost: np.ndarray, vb: np.ndarray, basis: str, theta: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Per-task V, D, H (resolved by the chain), A (attempt cost), n_att for the policy: attempt 1, then `branch` attempts
    up to K, then Opus.  theta=None -> hidden-test verifier (= truth); else the visible verifier at threshold theta."""
    n = len(vb)
    keys = ["1"] + ([f"{'r' if branch == 'retry' else 's'}{k}" for k in range(2, K + 1)] if K >= 2 else [])
    V = np.zeros(n)
    D = np.zeros(n)
    H = np.zeros(n)
    Acost = np.zeros(n)
    natt = np.zeros(n)
    alive = np.ones(n)  # still un-committed
    valid = ~np.isnan(ar.pass_["1"])
    for key in keys:
        p = ar.pass_[key]
        c = ar.cost[(key, basis)]
        exists = ~np.isnan(p)
        att = alive * exists
        natt += att
        cc = np.where(exists, c, 0.0)
        Acost += att * cc
        D += att * cc
        if theta is None:
            acc = np.where(exists, p, 0.0)
        else:
            acc = np.where(exists, (ar.vis[key] >= theta) & (ar.syn[key] > 0.5), False).astype(float)
        commit = att * acc
        V += commit * np.where(exists, p, 0.0)  # truth value of the committed output (1 if truth pass, 0 otherwise)
        H += commit
        alive = alive * (1 - acc)  # a task stays alive until an attempt is accepted; missing attempts leave it for the terminal
    # terminal fallback for un-committed tasks
    V += alive * vb
    D += alive * b_cost
    out = dict(V=V, D=D, H=H, A=Acost, natt=natt, valid=valid)
    return out


def pol_stats(W: np.ndarray, mask: np.ndarray, ch: Dict[str, np.ndarray], vb: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    V, D, H, A = ch["V"], ch["D"], ch["H"], ch["A"]
    EV, ED = wmean(W, mask, V), wmean(W, mask, D)
    vbar, cbar = wmean(W, mask, vb), wmean(W, mask, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = wsum(W, mask, V) / wsum(W, mask, D)
        eta_B = wsum(W, mask, vb) / wsum(W, mask, b)
        Q = EV - vbar
        S = cbar - ED
        net = Q + eta_B * S
        avoided = wmean(W, mask, H * b)
        attempt = wmean(W, mask, A)
        Qhit = wmean(W, mask, H * (np.where(H > 0, V, 0.0) - vb))  # accepted quality change vs Opus on the same tasks
        e = wmean(W, mask, (H > 0) * (V < vb - 1e-12))
        out = dict(EV=EV, ED=ED, vbar=vbar, cbar=cbar, eta=eta, eta_B=eta_B, r=eta / eta_B, Q=Q, S=S, net=net, net_per_task_cost=net / eta_B,
                   avoided=avoided, attempt=attempt, Qhit=Qhit, e=e, p=wmean(W, mask, H), cost_per_resolved=ED / EV, natt=wmean(W, mask, ch["natt"]))
        for eps in EPS:
            out[f"floor_{eps}"] = (EV >= vbar - eps - 1e-12).astype(float)
    return out


def cell(W: np.ndarray, ar: Arrays, branch: str, k: int, src_mask: np.ndarray, lam: float, basis: str, b_cost: np.ndarray, vb: np.ndarray,
         marg: Dict[str, np.ndarray], half_mask: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """One (branch, k) segment: state set = tasks where the branch's earlier attempts failed and attempt k exists."""
    kk = f"{'r' if branch == 'retry' else 's'}{k}"
    p = ar.pass_[kk]
    state = ~np.isnan(p) & src_mask & ~ar.api_err
    if k == 3:
        prev = ar.pass_[kk[0] + "2"]
        state &= (prev == 0)
    else:
        state &= (ar.pass_["1"] == 0)
    est_mask = state if half_mask is None else (state & half_mask)
    if est_mask.sum() < 3:
        return None
    a = ar.cost[(kk, basis)]
    pk = wmean(W, est_mask, p)
    ak = wmean(W, est_mask, a)
    bk = wmean(W, est_mask, b_cost)
    vbk = wmean(W, est_mask, vb)
    vh = 1.0
    Wl = lam * bk + vh - vbk
    seg = wmean(W, est_mask, p * (vh - vb + lam * b_cost) - lam * a)   # realized per-segment gain, attempt k then Opus
    # option value of attempt k+1 (same branch)
    d_next = np.zeros_like(pk)
    p_next_marg = None
    if k == 2:
        nk = kk[0] + "3"
        pn = ar.pass_[nk]
        nstate = ~np.isnan(pn) & (p == 0) & src_mask & ~ar.api_err
        nest = nstate if half_mask is None else (nstate & half_mask)
        if nest.sum() >= 3:
            an = ar.cost[(nk, basis)]
            p3, a3, b3, vb3 = wmean(W, nest, pn), wmean(W, nest, an), wmean(W, nest, b_cost), wmean(W, nest, vb)
            d_next = np.maximum(0.0, p3 * (lam * b3 + vh - vb3) - lam * a3)
            mm = ar.model.get(nk)
            if mm in marg:
                pm3 = wmean(W, src_mask & ~np.isnan(marg[mm]), marg[mm])
                p_next_marg = np.maximum(0.0, pm3 * (lam * b3 + vh - vb3) - lam * a3)
    G = pk * Wl + (1 - pk) * d_next - lam * ak
    G_seg = pk * Wl - lam * ak
    m_called = ar.model.get(kk)
    marg_mask = src_mask if half_mask is None else (src_mask & half_mask)  # the baseline is estimated on the same half as the rule
    pm = wmean(W, marg_mask & ~np.isnan(marg[m_called]), marg[m_called]) if m_called in marg else np.full_like(pk, np.nan)
    d_marg = p_next_marg if p_next_marg is not None else d_next
    G_marg = pm * Wl + (1 - pm) * d_marg - lam * ak
    G_marg_seg = pm * Wl - lam * ak
    # realized value of "attempt k then continue as the rule prescribes" (k=2: attempt 3 iff d_next>0 at the point estimate)
    if k == 2 and d_next[0] > 0:
        nk = kk[0] + "3"
        pn = np.where(np.isnan(ar.pass_[nk]), 0.0, ar.pass_[nk])
        an = np.where(np.isnan(ar.cost[(nk, basis)]), 0.0, ar.cost[(nk, basis)])
        Vc = p * vh + (1 - p) * (pn * vh + (1 - pn) * vb)
        Dc = a + (1 - p) * (an + (1 - pn) * b_cost)
        real = wmean(W, est_mask if half_mask is None else (state & ~half_mask), Vc - lam * Dc - (vb - lam * b_cost))
    else:
        real = seg if half_mask is None else wmean(W, state & ~half_mask, p * (vh - vb + lam * b_cost) - lam * a)
    if half_mask is not None:
        seg = wmean(W, state & ~half_mask, p * (vh - vb + lam * b_cost) - lam * a)
        n_real = int((state & ~half_mask).sum())
    else:
        n_real = int(state.sum())
    return dict(branch=branch, k=k, basis=basis, lam=float(lam), n_state=int(est_mask.sum()), n_real=n_real, model_called=m_called,
                p=ci(pk), a=ci(ak), b=ci(bk), v_b=ci(vbk), W=ci(Wl), delta_next=ci(d_next), p_marg=ci(pm),
                G=ci(G), G_seg=ci(G_seg), G_marg=ci(G_marg), G_marg_seg=ci(G_marg_seg), real_seg=ci(seg), real_opt=ci(real))


def flash_budget_views(A: pd.DataFrame, Ab: pd.DataFrame) -> Dict[str, Any]:
    """Budget-matched statements for deepseek-flash (coordinator point 1).

    (a) HEADLINE view: every chain's first complete run, i.e. the 6,144-token completion budget; an attempt that spent the
        whole budget on reasoning returned nothing, paid a_k and counts as a failure (p = 0 by construction).
    (b) RE-RUN view: the replayed chains' last run; the truncated attempts were re-issued at 16,384 tokens (same messages,
        temperature 0.7, seed where the original had one); re-runs still truncated are censored.  Downstream attempts of a
        replayed chain were regenerated at the 6,144 budget (and re-issued at 16,384 if they truncated).
    Also the COUNTERFACTUAL population p_1 at 16,384: the 6,144-budget roots that finished ('stop') are taken as their own
    16k outcome (a larger budget cannot change a completion that ended before the cap), the 34 truncated roots take their
    re-run outcome; reported with censored roots as failures and excluding them.  Wilson 95% intervals (counts, not the
    paired bootstrap)."""
    fa = A[(A.root_model == "deepseek-flash") & (~A.api_error)]
    ra = fa[(fa.branch == "root") & (fa.k == 1)]
    trunc_a = (ra.finish_reason == "length") & (ra.text.str.strip() == "")
    out: Dict[str, Any] = dict(view_a=dict(n_roots=int(len(ra)), n_truncated_roots=int(trunc_a.sum()), p1_truncations_as_failures=wilson(int(ra.truth_pass.sum()), int(len(ra))),
                                           p1_non_truncated_roots=wilson(int(ra[~trunc_a].truth_pass.sum()), int((~trunc_a).sum())),
                                           root_cost=dict(n=int((~trunc_a).sum()), mean_completion_tokens=float(ra[~trunc_a].completion_tokens.mean()) if (~trunc_a).any() else float("nan"),
                                                          mean_usd_proxy=float(ra[~trunc_a].cost_usd.mean()) if (~trunc_a).any() else float("nan"), mean_latency_s=float(ra[~trunc_a].latency_s.mean()) if (~trunc_a).any() else float("nan")),
                                           truncated_root_cost=dict(n=int(trunc_a.sum()), mean_completion_tokens=float(ra[trunc_a].completion_tokens.mean()) if trunc_a.any() else float("nan"),
                                                                    mean_usd_proxy=float(ra[trunc_a].cost_usd.mean()) if trunc_a.any() else float("nan"))))
    if not len(Ab):
        out["view_b"] = None
        out["counterfactual_p1_at_16k"] = None
        return out
    fb = Ab[(Ab.root_model == "deepseek-flash") & (~Ab.api_error)]
    rb = fb[(fb.branch == "root") & (fb.k == 1)]
    rr = rb[rb.rerun]
    ok16 = rr[~rr.censored]
    fail_jobs = set(rb[(~rb.truth_pass) & (~rb.censored)].job_id)

    def k2(branch: str) -> Dict[str, Any]:
        s = fb[(fb.branch == branch) & (fb.k == 2) & fb.job_id.isin(fail_jobs) & (~fb.censored)]
        return dict(n=int(len(s)), n_censored_excluded=int(((fb.branch == branch) & (fb.k == 2) & fb.job_id.isin(fail_jobs) & fb.censored).sum()),
                    p=wilson(int(s.truth_pass.sum()), int(len(s))), n_at_16k=int((s.budget == 16384).sum()), model=(s.model.iloc[0] if len(s) else None),
                    mean_completion_tokens=float(s.completion_tokens.mean()) if len(s) else float("nan"), mean_usd_proxy=float(s.cost_usd.mean()) if len(s) else float("nan"))

    rows16 = fb[(fb.model == "deepseek-flash") & (fb.budget == 16384)]
    ok16_all = rows16[~rows16.censored]

    def cost_stats(s: pd.DataFrame) -> Dict[str, Any]:
        """Means (and the median of completion tokens) of the relay-reported completion_tokens and the proxy-priced cost_usd."""
        return dict(n=int(len(s)), n_censored=int(s.censored.sum()) if len(s) else 0,
                    mean_completion_tokens=float(s.completion_tokens.mean()) if len(s) else float("nan"), median_completion_tokens=float(s.completion_tokens.median()) if len(s) else float("nan"),
                    mean_total_tokens=float(s.total_tokens.mean()) if len(s) else float("nan"), mean_usd_proxy=float(s.cost_usd.mean()) if len(s) else float("nan"),
                    mean_latency_s=float(s.latency_s.mean()) if len(s) else float("nan"), statistic="mean over the listed rows (median only for completion tokens); fields completion_tokens / total_tokens / cost_usd (proxy price) / latency_s")

    rows6_b = fb[(fb.model == "deepseek-flash") & (fb.budget != 16384)]
    out["view_b"] = dict(n_chains=int(fb.job_id.nunique()), n_root_rerun=int(len(rr)), n_root_censored=int(rr.censored.sum()),
                         cost_by_row_basis=dict(root_reruns_16k_all=cost_stats(rr), root_reruns_16k_non_censored=cost_stats(ok16), all_16k_rows_all_branches=cost_stats(rows16),
                                                all_16k_rows_all_branches_non_censored=cost_stats(ok16_all), censored_16k_rows=cost_stats(rows16[rows16.censored]),
                                                budget_6144_flash_rows_in_view_b=cost_stats(rows6_b)),
                         p1_16k_excluding_censored=wilson(int(ok16.truth_pass.sum()), int(len(ok16))), p1_16k_censored_as_failures=wilson(int(ok16.truth_pass.sum()), int(len(rr))),
                         n_fail_roots_16k=len(fail_jobs), k2=dict(fresh=k2("fresh"), retry=k2("retry"), switch=k2("switch")),
                         rows_16k=dict(n=int(len(rows16)), n_censored=int(rows16.censored.sum()), mean_completion_tokens_all=float(rows16.completion_tokens.mean()) if len(rows16) else float("nan"),
                                       mean_completion_tokens_non_censored=float(ok16_all.completion_tokens.mean()) if len(ok16_all) else float("nan"),
                                       mean_usd_proxy_all=float(rows16.cost_usd.mean()) if len(rows16) else float("nan"), mean_usd_proxy_non_censored=float(ok16_all.cost_usd.mean()) if len(ok16_all) else float("nan"),
                                       mean_latency_s=float(rows16.latency_s.mean()) if len(rows16) else float("nan"), pass_rate_non_censored=float(ok16_all.truth_pass.mean()) if len(ok16_all) else float("nan")),
                         chain_cost_per_task_usd=float(fb.groupby("job_id").cost_usd.sum().mean()) if len(fb) else float("nan"),
                         chain_cost_per_task_tokens=float(fb.groupby("job_id").total_tokens.sum().mean()) if len(fb) else float("nan"))
    passes_nt, n_nt = int(ra[~trunc_a].truth_pass.sum()), int((~trunc_a).sum())
    passes_rr = int(ok16.truth_pass.sum())
    out["counterfactual_p1_at_16k"] = dict(definition="6,144-budget roots that finished keep their outcome; the 34 truncated roots take their 16,384 re-run outcome",
                                           censored_as_failures=wilson(passes_nt + passes_rr, n_nt + int(len(rr))), excluding_censored=wilson(passes_nt + passes_rr, n_nt + int(len(ok16))),
                                           n_population=int(n_nt + len(rr)), n_censored=int(rr.censored.sum()))
    return out


def sign_agreement(cells: List[Dict[str, Any]], pred: str, real: str, other: Optional[str] = None) -> Dict[str, Any]:
    """Sign agreement of predictor `pred` with the realized `real` on CI-decided cells, with naive baselines.  When `other` is
    given, also the agreement restricted to the DISCRIMINATING cells where `pred` and `other` predict different signs."""
    dec = [c for c in cells if excl(c[real])]
    agree = sum(1 for c in dec if (c[pred][0] >= 0) == (c[real][0] >= 0))
    pos = sum(1 for c in dec if c[real][0] >= 0)
    out = dict(n_cells=len(cells), n_decided=len(dec), n_agree=agree, rate=wilson(agree, len(dec)), always_attempt=wilson(pos, len(dec)),
               always_stop=wilson(len(dec) - pos, len(dec)))
    if other is not None:
        dis = [c for c in dec if np.isfinite(c[other][0]) and ((c[pred][0] >= 0) != (c[other][0] >= 0))]
        out["discriminating"] = dict(n=len(dis), n_pred_right=sum(1 for c in dis if (c[pred][0] >= 0) == (c[real][0] >= 0)),
                                     rate=wilson(sum(1 for c in dis if (c[pred][0] >= 0) == (c[real][0] >= 0)), len(dis)))
    # minority-sign predictions: cells where `pred` predicts the sign that the naive majority of this group gets wrong
    # (the group's majority realized sign is the better naive baseline); this is where the rule can beat 'always X'
    if dec:
        maj_attempt = pos >= (len(dec) - pos)
        mino = [c for c in dec if (c[pred][0] >= 0) != maj_attempt]
        out["minority"] = dict(majority="attempt" if maj_attempt else "stop", n=len(mino), n_pred_right=sum(1 for c in mino if (c[pred][0] >= 0) == (c[real][0] >= 0)),
                               rate=wilson(sum(1 for c in mino if (c[pred][0] >= 0) == (c[real][0] >= 0)), len(mino)))
    return out


# ================================================================================================ figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": COL["axis"],
                         "axes.labelcolor": COL["ink"], "xtick.color": COL["muted"], "ytick.color": COL["muted"], "axes.grid": True, "grid.color": COL["grid"],
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
    return plt


SHORT = {"qwen3.5-plus": "qwen3.5-plus", "qwen3-max": "qwen3-max", "kimi-k2.5": "kimi-k2.5", "deepseek-v4-pro": "ds-v4-pro*", "deepseek-flash": "ds-flash*"}


def fig_conditional(res: Dict[str, Any], plt) -> None:
    models = [m for m in LIVE_MODELS if m in res["per_model"]]
    fig, axes = plt.subplots(1, len(models), figsize=(2.1 * len(models), 2.9), sharey=True, squeeze=False)
    axes = axes[0]
    handles: Dict[str, Any] = {}
    for ax, m in zip(axes, models):
        s = res["per_model"][m]["all"]
        pts = {"fresh": [(1, s["p1"]), (2, s.get("p_f2"))], "retry": [(1, s["p1"]), (2, s.get("p_r2")), (3, s.get("p_r3"))], "switch": [(1, s["p1"]), (2, s.get("p_s2")), (3, s.get("p_s3"))]}
        off = {"retry": -0.07, "switch": 0.07, "fresh": 0.0}
        for name, series in pts.items():
            xs, ys, lo, hi = [], [], [], []
            for k, t in series:
                if t is None or not np.isfinite(t[0]):
                    continue
                xs.append(k + off[name]); ys.append(t[0]); lo.append(t[0] - t[1] if np.isfinite(t[1]) else 0); hi.append(t[2] - t[0] if np.isfinite(t[2]) else 0)
            if len(xs) > 1:
                h = ax.errorbar(xs, ys, yerr=[lo, hi], color=COL[name], lw=1.6, marker={"retry": "o", "switch": "s", "fresh": "^"}[name], ms=4.5, capsize=2,
                                label={"retry": "retry, same model + feedback", "switch": "switch to the next-stronger model", "fresh": "fresh retry, same model, no feedback"}[name])
                handles.setdefault(name, h)
        ax.axhline(s["p1"][0], color=COL["muted"], lw=0.9, ls="--")
        ax.set_xticks([1, 2, 3]); ax.set_xlim(0.6, 3.4); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(f"attempt k\n{SHORT[m]} (n fail = {s['n_fail1']})", fontsize=8)
    axes[0].set_ylabel("P(pass at k | earlier failed)")
    axes[-1].text(3.35, res["per_model"][models[-1]]["all"]["p1"][0] - 0.05, "marginal p₁", color=COL["muted"], fontsize=7, ha="right", va="top")
    order = [n for n in ["fresh", "retry", "switch"] if n in handles]
    fig.legend([handles[n] for n in order], [handles[n].get_label() for n in order], loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_conditional_success.pdf"))
    plt.close(fig)


def fig_pred_real(res: Dict[str, Any], plt) -> None:
    """Split-half OUT-OF-SAMPLE cells at λ = η_B: rule value from one half vs realized ΔJ on the other (pooled = large, per-source = small)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, basis in zip(axes, BASES):
        cells = [c for c in res["cells"]["oos"] if c["basis"] == basis and c["lam_name"] == "eta_B"]
        for br, mk in [("retry", "o"), ("switch", "s")]:
            for pooled in (True, False):
                cc = [c for c in cells if c["branch"] == br and ((c["source"] == "all") == pooled)]
                if not cc:
                    continue
                x = np.array([c["G"][0] for c in cc]); y = np.array([c["real_opt"][0] for c in cc])
                lo = np.array([c["real_opt"][0] - c["real_opt"][1] if np.isfinite(c["real_opt"][1]) else 0 for c in cc])
                hi = np.array([c["real_opt"][2] - c["real_opt"][0] if np.isfinite(c["real_opt"][2]) else 0 for c in cc])
                dec = np.array([excl(c["real_opt"]) for c in cc])
                size = 26 if pooled else 11
                ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=COL[br], elinewidth=0.5, alpha=0.35)
                if dec.any():
                    ax.scatter(x[dec], y[dec], s=size, marker=mk, color=COL[br], edgecolor="white", linewidth=0.5, zorder=3,
                               label=(f"{br}, CI-decided" if pooled else None))
                if (~dec).any():
                    ax.scatter(x[~dec], y[~dec], s=size, marker=mk, facecolor="none", edgecolor=COL[br], linewidth=0.7, zorder=3,
                               label=(f"{br}, undecided" if pooled else None))
        ax.axhline(0, color=COL["axis"], lw=0.8); ax.axvline(0, color=COL["axis"], lw=0.8)
        lim = max([abs(v) for v in list(ax.get_xlim()) + list(ax.get_ylim())] + [1e-3])
        ax.plot([-lim, lim], [-lim, lim], color=COL["grid"], lw=0.8, ls=":")
        ax.set_xlabel("predicted G_k  (λ = η_B per $)" if basis == "usd" else "predicted G_k  (λ = η_B per s)", fontsize=8)
        ax.set_ylabel("realized ΔJ_λ per task in the state" if basis == "usd" else "")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[1].text(0.02, 0.02, "large = pooled sources, small = per source", transform=axes[1].transAxes, fontsize=7, color=COL["muted"])
    fig.text(0.5, 0.005, "G_k = p_k W_λ + (1−p_k) Δ_{k+1} − λ a_k, estimated on one half of the tasks; realized ΔJ_λ on the other half", ha="center", fontsize=7, color=COL["muted"])
    fig.tight_layout(w_pad=2.5, rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_predicted_vs_realized.pdf"))
    plt.close(fig)


def fig_cost(res: Dict[str, Any], plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    # Opus = the reference, drawn as an unfilled ink outline (not a categorical series); the three policies use
    # validated slots (violet, blue, orange) plus hatching as the secondary (non-colour) encoding.
    pols = [("direct Opus", None, "////"), ("attempt 1 -> Opus", "k1", "//"), ("retry K=3 -> Opus", "retry", "\\\\"), ("switch K=3 -> Opus", "switch", "..")]
    models = [m for m in LIVE_MODELS if m in res["policies"]]
    for ax, basis in zip(axes, BASES):
        width = 0.19
        for j, (pname, col, hatch) in enumerate(pols):
            xs, ys, lo, hi = [], [], [], []
            for i, m in enumerate(models):
                pol = res["policies"].get(m, {}).get(basis, {})
                if pname.startswith("switch") and pname not in pol:
                    pol_name = "switch K=2 -> Opus"
                else:
                    pol_name = pname
                t = pol.get(pol_name, {}).get("cost_per_resolved")
                if t is None or not np.isfinite(t[0]):
                    continue
                xs.append(i + (j - 1.5) * (width + 0.02)); ys.append(t[0]); lo.append(t[0] - t[1] if np.isfinite(t[1]) else 0); hi.append(t[2] - t[0] if np.isfinite(t[2]) else 0)
            if xs:
                kw = dict(color="none", edgecolor=COL["ink"], linewidth=0.8) if col is None else dict(color=COL[col], edgecolor="white", linewidth=0.8)
                ax.bar(xs, ys, width=width, hatch=hatch, label=pname.replace("K=3", "K≤3"), yerr=[lo, hi], error_kw=dict(elinewidth=0.7, ecolor=COL["ink"], capsize=1.5), **kw)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([SHORT[m] for m in models], fontsize=7, rotation=15)
        ax.set_ylabel("$ per resolved task (E[D]/E[V])" if basis == "usd" else "seconds per resolved task (E[D]/E[V])")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_cost_per_resolved.pdf"))
    plt.close(fig)


# ================================================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="use only the first N sampled tasks (smoke test)")
    ap.add_argument("--models", type=str, default=",".join(LIVE_MODELS))
    ap.add_argument("--fresh", action="store_true", help="ignore attempts.jsonl / recorded.jsonl (calls are still cached by packy)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--analyze-only", action="store_true", help="no live calls: analyse the complete jobs persisted in attempts.jsonl")
    ap.add_argument("--collect", action="store_true", help="collect new model responses using configured API credentials")
    args = ap.parse_args()
    args.analyze_only = args.analyze_only or not args.collect
    models = [m for m in args.models.split(",") if m]
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- frame
    F, counts, tasks = build_frame()
    log("frame:", counts)
    F = stratified_sample(F, N_TARGET, rng)
    order = rng.permutation(len(F))
    F = F.iloc[order].reset_index(drop=True)
    if args.limit:
        F = F.iloc[: args.limit].reset_index(drop=True)
    log(f"tasks used: {len(F)}", F.source.value_counts().to_dict())
    tasks = {t: tasks[t] for t in F.task_id}
    if args.fresh:
        for p in (ATTEMPTS_PATH, RECORDED_PATH):
            if os.path.exists(p):
                os.remove(p)

    # ---------------------------------------------------------------- recorded attempt 0
    rm = json.load(open(RESP))
    R = verify_recorded(tasks, F, rm)
    log("recorded responses verified:", len(R))

    # ---------------------------------------------------------------- live chains
    client = PackyClient(quiet=True, offline=args.analyze_only)
    units_before = 0.0
    col = Collector(client, tasks, F)
    units_before = col.tag_units(force=True)
    done = {} if args.fresh else load_done_jobs(ATTEMPTS_PATH)
    jobs = [(t, m) for t in F.task_id for m in models]
    if args.analyze_only:
        jobs = [j for j in jobs if f"{j[0]}|{j[1]}" in done]
        log(f"analyze-only: {len(jobs)} complete jobs restored, no live calls")
    col.run_jobs(jobs, done)
    # ---- deepseek-flash truncation re-run pass (coordinator's request): chains holding a truncated 6,144-token flash attempt
    trunc_jobs = sorted({r["job_id"] for r in col.rows if r["model"] == "deepseek-flash" and not r.get("rerun") and not r.get("api_error") and is_truncated(r)})
    if trunc_jobs and not args.analyze_only and not col.stop:
        col.rerun_jobs(trunc_jobs)
    elif trunc_jobs:
        log(f"NOTE: {len(trunc_jobs)} chains still hold truncated deepseek-flash attempts (analyze-only or stopped): not re-run")
    # ---- de-duplicated, budget-explicit views (coordinator points 1 and 3).  Every persisted complete run is read back:
    #   view (a): first complete run of every chain  -> the 6,144-token deepseek-flash budget, truncations = failures (HEADLINE)
    #   view (b): last run of the chains that were replayed in the re-run pass -> the 16,384-token re-run view (separate section)
    runs = load_job_runs(ATTEMPTS_PATH)
    keep_tasks = set(F.task_id)
    rows_a: List[Dict[str, Any]] = []
    rows_b: List[Dict[str, Any]] = []
    for jid, rs in runs.items():
        tid, root = jid.rsplit("|", 1)
        if tid not in keep_tasks or root not in models:
            continue
        clean = [run for run in rs if not any(r.get("api_error") for r in run)] or rs
        rows_a.extend(with_budget(r, "a_6144") for r in clean[0])
        if len(rs) > 1:
            rows_b.extend(with_budget(r, "b_rerun16k") for r in rs[-1])
    for path, rows_ in ((VIEW_A_PATH, rows_a), (VIEW_B_PATH, rows_b)):
        with open(path, "w") as fh:
            for r in rows_:
                fh.write(json.dumps(jsonable(r), ensure_ascii=False) + "\n")
    COLS = ["job_id", "task_id", "source", "dim", "root_model", "branch", "k", "model", "seed", "cached", "finish_reason", "api_error", "max_tokens", "rerun", "censored", "collection_pass",
            "prompt_tokens", "completion_tokens", "total_tokens", "latency_s", "sandbox_s", "cost_usd", "cost_sec", "truth_pass", "truth_frac", "failure_kind", "error", "vis_score", "vis_n",
            "vis_n_pass", "vis_syntax_ok", "vis_s", "text", "code", "cache_key", "ts", "budget", "view"]
    A = pd.DataFrame(rows_a, columns=COLS)
    Ab = pd.DataFrame(rows_b, columns=COLS)
    for D in (A, Ab):
        for c_, dflt in [("rerun", False), ("censored", False), ("max_tokens", 0), ("collection_pass", "main"), ("api_error", False), ("cached", False), ("truth_pass", False)]:
            D[c_] = D[c_].fillna(dflt).astype(type(dflt))
        D["text"] = D["text"].fillna("")
    A = A.reset_index(drop=True)
    Ab = Ab.reset_index(drop=True)
    units_after = col.tag_units(force=True)
    log(f"view (a) rows: {len(A)} from {sum(1 for jid in runs if jid.rsplit('|', 1)[0] in keep_tasks)} chains; view (b) rows: {len(Ab)} from {Ab.job_id.nunique() if len(Ab) else 0} replayed chains; "
        f"X7b ledger units (both tags): {units_before:.0f} -> {units_after:.0f}")
    # truncated originals (kept out of the analysis set; reported as a measured phenomenon).  Two kinds: RE-ISSUED (saved by the
    # re-run pass when it met the truncated attempt on the chain's path) and SUPERSEDED (truncated in the main pass, but the
    # replayed chain no longer reaches that attempt because the re-run root/retry passed).  Both were paid; the file holds both.
    trunc_rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if os.path.exists(TRUNC_PATH):
        for line in open(TRUNC_PATH):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["task_id"] in set(F.task_id):
                r.setdefault("superseded", False)
                trunc_rows[(r["job_id"], r["branch"], r["k"], r.get("seed"))] = r
    if os.path.exists(ATTEMPTS_PATH):
        for line in open(ATTEMPTS_PATH):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["model"] == "deepseek-flash" and not r.get("rerun") and not r.get("api_error") and is_truncated(r) and r["task_id"] in set(F.task_id):
                key = (r["job_id"], r["branch"], r["k"], r.get("seed"))
                if key not in trunc_rows:
                    trunc_rows[key] = dict(r, truncated_original=True, superseded=True)
    if trunc_rows:
        with open(TRUNC_PATH, "w") as fh:
            for key in sorted(trunc_rows, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
                fh.write(json.dumps(jsonable(trunc_rows[key]), ensure_ascii=False) + "\n")
    TR = pd.DataFrame(list(trunc_rows.values())) if trunc_rows else pd.DataFrame(columns=["job_id", "branch", "k", "completion_tokens", "cost_usd", "total_tokens", "superseded"])
    still_trunc = int(((Ab.model == "deepseek-flash") & (~Ab.rerun) & (~Ab.api_error) & (Ab.finish_reason == "length") & (Ab.text.str.strip() == "")).sum()) if len(Ab) else 0

    # ---------------------------------------------------------------- arrays
    tasks_order = list(F.task_id)
    n = len(tasks_order)
    src = F.source.to_numpy()
    vb = F.opus_score.to_numpy(float)
    bcost = {"usd": F.opus_usd.to_numpy(float), "sec": F.opus_sec.to_numpy(float), "tok": F.opus_tok.to_numpy(float)}
    AR = build_arrays(A, tasks_order)
    W = boot_weights(n, np.random.default_rng(SEED))
    allm = np.ones(n, bool)
    src_masks = {"all": allm, **{s: src == s for s in SOURCES}}
    # split halves (stratified by source) for the OUT-OF-SAMPLE sign test
    half = np.zeros(n, bool)
    rng2 = np.random.default_rng(SEED + 1)
    for s in SOURCES:
        idx = np.where(src == s)[0]
        pick = rng2.choice(idx, size=len(idx) // 2, replace=False)
        half[pick] = True
    marg = {m: AR[m].pass_["1"] for m in models}
    eta_B = {b: ci(wsum(W, allm, vb) / wsum(W, allm, bcost[b])) for b in BASES}
    lambdas = {b: [("eta_B", eta_B[b][0])] + [(str(l), l) for l in LAMBDA_FIXED] for b in BASES}

    res: Dict[str, Any] = dict(experiment="X7b_inloop_live", seed=SEED, n_bootstrap=B, n_tasks=n, frame_counts=counts, sources={s: int((src == s).sum()) for s in SOURCES},
                               models=models, non_benchmark=sorted(NON_BENCHMARK), chain={m: [NEXT[m], NEXT3[m]] for m in LIVE_MODELS}, workers=WORKERS, temperature=TEMP,
                               max_tokens=MAX_TOKENS, max_tokens_by_model=MAX_TOKENS_BY_MODEL,
                               feedback_chars=FEEDBACK_CHARS, prices_per_1m=PRICE, eta_B=eta_B, lambdas=lambdas, per_model={}, cells={}, claims=[], policies={},
                               visible={}, recorded={}, spend={}, deviations=[], runtime_s=None)

    # ---------------------------------------------------------------- recorded attempt 0 agreement
    rec = {}
    for m in CSV_NAME:
        if m not in models:
            continue
        Rm = R[R.model == m].set_index("task_id").reindex(tasks_order)
        rel = F[f"rec_score_{m}"].to_numpy(float) >= 1.0
        sb = Rm.sandbox_pass.to_numpy(bool)
        ne = ~Rm["empty"].to_numpy(bool)  # empty recorded responses are unverifiable (coordinator note 4), not failures
        rec[m] = dict(n=int(n), n_verifiable=int(ne.sum()), released_pass=ci(wmean(W, allm, rel.astype(float))), sandbox_pass_verifiable=ci(wmean(W, ne, sb.astype(float))),
                      released_pass_verifiable=ci(wmean(W, ne, rel.astype(float))),
                      agreement=float((rel == sb)[ne].mean()) if ne.any() else float("nan"), false_pass=int((sb & ~rel & ne).sum()), false_fail=int((~sb & rel & ne).sum()), empty=int((~ne).sum()),
                      empty_released_pass=int((rel & ~ne).sum()),
                      per_source={s: dict(n=int((src == s).sum()), released=float(rel[src == s].mean()) if (src == s).any() else float("nan"),
                                          sandbox=float(sb[(src == s) & ne].mean()) if ((src == s) & ne).any() else float("nan"),
                                          agreement=float((rel == sb)[(src == s) & ne].mean()) if ((src == s) & ne).any() else float("nan")) for s in SOURCES},
                      live_attempt1_pass=ci(wmean(W, allm & ~np.isnan(AR[m].pass_["1"]), AR[m].pass_["1"])),
                      rec_cost_usd=ci(wmean(W, allm, F[f"rec_usd_{m}"].to_numpy(float))), live_cost_usd=ci(wmean(W, ~np.isnan(AR[m].cost[("1", "usd")]), AR[m].cost[("1", "usd")])),
                      rec_sec=ci(wmean(W, allm, F[f"rec_sec_{m}"].to_numpy(float))), live_api_sec=ci(wmean(W, ~np.isnan(AR[m].cost[("1", "api_sec")]), AR[m].cost[("1", "api_sec")])),
                      rec_tok=ci(wmean(W, allm, F[f"rec_tok_{m}"].to_numpy(float))), live_tok=ci(wmean(W, ~np.isnan(AR[m].cost[("1", "tok")]), AR[m].cost[("1", "tok")])))
    res["recorded"] = rec

    # ---------------------------------------------------------------- per-model conditional success, C1, C2, C4
    for m in models:
        ar = AR[m]
        pm: Dict[str, Any] = {}
        for sname, smask in src_masks.items():
            ok = smask & ~ar.api_err & ~np.isnan(ar.pass_["1"])
            fail1 = ok & (ar.pass_["1"] == 0)
            d: Dict[str, Any] = dict(n=int(ok.sum()), n_fail1=int(fail1.sum()))
            d["p1"] = ci(wmean(W, ok, ar.pass_["1"]))
            for key in ["f2", "r2", "s2"]:
                mk = fail1 & ~np.isnan(ar.pass_[key])
                d[f"p_{key}"] = ci(wmean(W, mk, ar.pass_[key])) if mk.sum() >= 3 else [float("nan")] * 3
                d[f"n_{key}"] = int(mk.sum())
                for b in ["usd", "sec", "tok"]:
                    d[f"a_{key}_{b}"] = ci(wmean(W, mk, ar.cost[(key, b)])) if mk.sum() >= 3 else [float("nan")] * 3
            for key in ["r3", "s3"]:
                prev = key[0] + "2"
                mk = fail1 & (ar.pass_[prev] == 0) & ~np.isnan(ar.pass_[key])
                d[f"p_{key}"] = ci(wmean(W, mk, ar.pass_[key])) if mk.sum() >= 3 else [float("nan")] * 3
                d[f"n_{key}"] = int(mk.sum())
                for b in ["usd", "sec", "tok"]:
                    d[f"a_{key}_{b}"] = ci(wmean(W, mk, ar.cost[(key, b)])) if mk.sum() >= 3 else [float("nan")] * 3
            # first-attempt cost
            for b in ["usd", "sec", "tok"]:
                d[f"a_1_{b}"] = ci(wmean(W, ok, ar.cost[("1", b)]))
            # paired differences (same bootstrap rows)
            mkf = fail1 & ~np.isnan(ar.pass_["f2"]); mkr = fail1 & ~np.isnan(ar.pass_["r2"]); mks = fail1 & ~np.isnan(ar.pass_["s2"])
            p1v = wmean(W, ok, ar.pass_["1"])
            if mkf.sum() >= 3:
                d["C1_fresh_minus_marginal"] = ci(wmean(W, mkf, ar.pass_["f2"]) - p1v)
            if mkr.sum() >= 3:
                d["C1_retry_minus_marginal"] = ci(wmean(W, mkr, ar.pass_["r2"]) - p1v)
            both = mkf & mkr
            if both.sum() >= 3:
                d["C4_retryfb_minus_fresh"] = ci(wmean(W, both, ar.pass_["r2"]) - wmean(W, both, ar.pass_["f2"]))
                d["n_C4"] = int(both.sum())
                # salvage value in the paper's units: the failed attempt's observations (feedback) raise the retry's success by Δp on the
                # failure state; at price λ = η_B (per $) the reusable work is worth Δp W_λ − λ Δa per task in the state, with W_λ = λ b + 1 − v_b
                # on the failure set and Δa the extra prompt cost of carrying the feedback (score units; divide by η_B for $).
                for basis in BASES:
                    lamB = wsum(W, allm, vb) / wsum(W, allm, bcost[basis])
                    Wl = lamB * wmean(W, both, bcost[basis]) + 1.0 - wmean(W, both, vb)
                    dp = wmean(W, both, ar.pass_["r2"]) - wmean(W, both, ar.pass_["f2"])
                    da = wmean(W, both, ar.cost[("r2", basis)]) - wmean(W, both, ar.cost[("f2", basis)])
                    d[f"salvage_value_{basis}"] = ci(dp * Wl - lamB * da)
                    d[f"salvage_value_{basis}_in_cost_units"] = ci((dp * Wl - lamB * da) / lamB)
                # by failure kind of attempt 1
                d["C4_by_failure_kind"] = {}
                for fk in sorted({x for x in ar.fk["1"][both] if x}):
                    mfk = both & (ar.fk["1"] == fk)
                    if mfk.sum() >= 3:
                        d["C4_by_failure_kind"][fk] = dict(n=int(mfk.sum()), p_retry=ci(wmean(W, mfk, ar.pass_["r2"])), p_fresh=ci(wmean(W, mfk, ar.pass_["f2"])),
                                                           diff=ci(wmean(W, mfk, ar.pass_["r2"]) - wmean(W, mfk, ar.pass_["f2"])))
            both2 = mkr & mks
            if both2.sum() >= 3:
                d["C2_switch_minus_retry_p"] = ci(wmean(W, both2, ar.pass_["s2"]) - wmean(W, both2, ar.pass_["r2"]))
                d["n_C2"] = int(both2.sum())
                for b in ["usd", "sec", "tok"]:
                    d[f"C2_switch_minus_retry_a_{b}"] = ci(wmean(W, both2, ar.cost[("s2", b)]) - wmean(W, both2, ar.cost[("r2", b)]))
                # Opus as the switch target (terminal): p = recorded Opus pass on the same failure set
                d["opus_pass_on_fail1"] = ci(wmean(W, fail1, (vb >= 1.0).astype(float)))
                d["opus_cost_usd_on_fail1"] = ci(wmean(W, fail1, bcost["usd"]))
            if m == "kimi-k2.5":
                d["opus_pass_on_fail1"] = ci(wmean(W, fail1, (vb >= 1.0).astype(float))) if fail1.sum() >= 3 else [float("nan")] * 3
                d["opus_cost_usd_on_fail1"] = ci(wmean(W, fail1, bcost["usd"])) if fail1.sum() >= 3 else [float("nan")] * 3
            # failure kinds of attempt 1
            d["failure_kinds_1"] = {str(k): int(v) for k, v in pd.Series(ar.fk["1"][fail1]).value_counts().items()}
            # attempt-1 pass rate excluding truncated outputs (finish_reason=length -> empty or cut content; a max_tokens artefact, not a model failure)
            r1 = A[(A.root_model == m) & (A.branch == "root") & (A.k == 1) & (~A.api_error) & A.task_id.isin(np.array(tasks_order)[ok])]
            notlen = r1[r1.finish_reason != "length"]
            d["n_length_1"] = int((r1.finish_reason == "length").sum())
            d["p1_excl_length"] = wilson(int(notlen.truth_pass.sum()), int(len(notlen)))
            # share of fresh retries whose extracted code is byte-identical to attempt 1 (the sampling channel of error correlation)
            sub_m = A[(A.root_model == m) & A.task_id.isin(np.array(tasks_order)[fail1]) & (~A.api_error)]
            code1 = {r.task_id: (r.code or "").strip() for r in sub_m[(sub_m.branch == "root")].itertuples(index=False)}
            fr = sub_m[sub_m.branch == "fresh"]
            if len(fr):
                ident = np.array([(r.code or "").strip() == code1.get(r.task_id, None) for r in fr.itertuples(index=False)], bool)
                d["fresh_identical_code"] = dict(n=int(len(fr)), share=float(ident.mean()), pass_when_identical=float(fr.truth_pass.to_numpy()[ident].mean()) if ident.any() else float("nan"),
                                                 pass_when_different=float(fr.truth_pass.to_numpy()[~ident].mean()) if (~ident).any() else float("nan"))
            d["finish_length_1"] = int(((A.root_model == m) & (A.k == 1) & (A.finish_reason == "length") & A.task_id.isin(np.array(tasks_order)[smask])).sum())
            pm[sname] = d
        res["per_model"][m] = pm

    # ---------------------------------------------------------------- rule cells (IN-SAMPLE and split-half OOS)
    cells_in: List[Dict[str, Any]] = []
    cells_oos: List[Dict[str, Any]] = []
    for m in models:
        ar = AR[m]
        for branch in ["retry", "switch"]:
            for k in [2, 3]:
                for sname, smask in src_masks.items():
                    for basis in BASES:
                        for lname, lam in lambdas[basis]:
                            c = cell(W, ar, branch, k, smask, lam, basis, bcost[basis], vb, marg)
                            if c is not None:
                                c.update(model=m, source=sname, lam_name=lname)
                                cells_in.append(c)
                            for hm in (half, ~half):
                                c2 = cell(W, ar, branch, k, smask, lam, basis, bcost[basis], vb, marg, half_mask=hm)
                                if c2 is not None:
                                    c2.update(model=m, source=sname, lam_name=lname, est_half="A" if hm is half else "B")
                                    cells_oos.append(c2)
    res["cells"] = dict(insample=cells_in, oos=cells_oos)

    def agg(cells: List[Dict[str, Any]], real: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        groups = {"all": lambda c: True, "usd": lambda c: c["basis"] == "usd", "sec": lambda c: c["basis"] == "sec", "lam=eta_B": lambda c: c["lam_name"] == "eta_B",
                  "lam=1": lambda c: c["lam_name"] == "1.0", "lam=10": lambda c: c["lam_name"] == "10.0", "k=2": lambda c: c["k"] == 2, "k=3": lambda c: c["k"] == 3,
                  "retry": lambda c: c["branch"] == "retry", "switch": lambda c: c["branch"] == "switch", "pooled-source": lambda c: c["source"] == "all",
                  "per-source": lambda c: c["source"] != "all"}
        for g, f in groups.items():
            cc = [c for c in cells if f(c)]
            pk, mk = ("G", "G_marg") if real == "real_opt" else ("G_seg", "G_marg_seg")
            out[g] = dict(rule=sign_agreement(cc, pk, real, other=mk), marginal=sign_agreement(cc, mk, real, other=pk))
        return out

    res["sign_agreement"] = dict(insample_opt=agg(cells_in, "real_opt"), insample_seg=agg(cells_in, "real_seg"), oos_opt=agg(cells_oos, "real_opt"), oos_seg=agg(cells_oos, "real_seg"))

    # ---------------------------------------------------------------- policies (eta/eta_B, net gain, cost per resolved) and visible-verifier chains (C5)
    for m in models:
        ar = AR[m]
        res["policies"][m] = {}
        ok = ~ar.api_err & ~np.isnan(ar.pass_["1"])
        for basis in BASES:
            P: Dict[str, Any] = {}
            direct = dict(V=vb.copy(), D=bcost[basis].copy(), H=np.zeros(n), A=np.zeros(n), natt=np.zeros(n))
            P["direct Opus"] = {k: ci(v) for k, v in pol_stats(W, ok, direct, vb, bcost[basis]).items()}
            P["attempt 1 -> Opus"] = {k: ci(v) for k, v in pol_stats(W, ok, chain_arrays(ar, "retry", 1, bcost[basis], vb, basis), vb, bcost[basis]).items()}
            for br in ["retry", "switch"]:
                for K in [2, 3]:
                    if br == "switch" and NEXT[m] is None:
                        continue
                    if br == "switch" and K == 3 and NEXT3[m] is None:
                        continue
                    P[f"{br} K={K} -> Opus"] = {k: ci(v) for k, v in pol_stats(W, ok, chain_arrays(ar, br, K, bcost[basis], vb, basis), vb, bcost[basis]).items()}
            if m in CSV_NAME and basis in ("usd", "sec"):
                s0 = (F[f"rec_score_{m}"].to_numpy(float) >= 1.0).astype(float)
                c0 = F[f"rec_{basis}_{m}"].to_numpy(float)
                rec_ch = dict(V=s0 + (1 - s0) * vb, D=c0 + (1 - s0) * bcost[basis], H=s0, A=c0, natt=np.ones(n))
                P["recorded attempt 0 -> Opus (oracle theta=1)"] = {k: ci(v) for k, v in pol_stats(W, ok, rec_ch, vb, bcost[basis]).items()}
            res["policies"][m][basis] = P
        # visible verifier (imperfect) -- attempt-level rates and chain replay
        vis: Dict[str, Any] = {}
        rows_m = A[(A.root_model == m) & (~A.api_error)]
        for theta in THETAS_VIS:
            Hv = rows_m.apply(lambda r: vis_accept(r, theta), axis=1).to_numpy(bool) if len(rows_m) else np.zeros(0, bool)
            tr = rows_m.truth_pass.to_numpy(bool)
            hasex = rows_m.vis_n.to_numpy() > 0
            d = dict(theta=theta, n_attempts=int(len(rows_m)), accept_rate=float(Hv.mean()) if len(Hv) else float("nan"),
                     e_false_accept=float((Hv & ~tr).mean()) if len(Hv) else float("nan"), e_false_accept_ci=wilson(int((Hv & ~tr).sum()), int(len(Hv))),
                     false_reject=float((~Hv & tr).mean()) if len(Hv) else float("nan"),
                     TPR=float(Hv[tr].mean()) if tr.sum() else float("nan"), FPR=float(Hv[~tr].mean()) if (~tr).sum() else float("nan"), n_truth_fail=int((~tr).sum()),
                     coverage_with_examples=float(hasex.mean()) if len(rows_m) else float("nan"),
                     with_examples=dict(n=int(hasex.sum()), e=float((Hv & ~tr)[hasex].mean()) if hasex.any() else float("nan"), fr=float((~Hv & tr)[hasex].mean()) if hasex.any() else float("nan"),
                                        TPR=float(Hv[hasex & tr].mean()) if (hasex & tr).any() else float("nan"), FPR=float(Hv[hasex & ~tr].mean()) if (hasex & ~tr).any() else float("nan")),
                     without_examples=dict(n=int((~hasex).sum()), e=float((Hv & ~tr)[~hasex].mean()) if (~hasex).any() else float("nan"), fr=float((~Hv & tr)[~hasex].mean()) if (~hasex).any() else float("nan")),
                     per_source={s: dict(n=int((rows_m.source == s).sum()), e=float((Hv & ~tr)[rows_m.source.to_numpy() == s].mean()) if (rows_m.source == s).sum() else float("nan"),
                                         fr=float((~Hv & tr)[rows_m.source.to_numpy() == s].mean()) if (rows_m.source == s).sum() else float("nan")) for s in SOURCES})
            chains = {}
            for basis in BASES:
                for br in ["retry", "switch"]:
                    if br == "switch" and NEXT[m] is None:
                        continue
                    K = 3 if (br == "retry" or NEXT3[m] is not None) else 2
                    hid_ch = chain_arrays(ar, br, K, bcost[basis], vb, basis)
                    vis_ch = chain_arrays(ar, br, K, bcost[basis], vb, basis, theta=theta)
                    sh, sv = pol_stats(W, ok, hid_ch, vb, bcost[basis]), pol_stats(W, ok, vis_ch, vb, bcost[basis])
                    # channels of the quality change: false rejects that end at an Opus miss (−), false accepts of a wrong output (−), and
                    # tasks the visible chain commits later than the hidden one (0 by construction of v_h = 1)
                    fr_to_opus_miss = (hid_ch["H"] > 0) & (vis_ch["H"] == 0) & (vb < 1.0)
                    fa_commit = (vis_ch["H"] > 0) & (vis_ch["V"] < 1.0)
                    chains[f"{br} K={K} {basis}"] = dict(EV_hidden=ci(sh["EV"]), EV_visible=ci(sv["EV"]), Q_loss=ci(sv["EV"] - sh["EV"]), ED_hidden=ci(sh["ED"]), ED_visible=ci(sv["ED"]),
                                                         D_change=ci(sv["ED"] - sh["ED"]), e_chain=ci(sv["e"]), e_hidden=ci(sh["e"]), eta_r_hidden=ci(sh["r"]), eta_r_visible=ci(sv["r"]),
                                                         net_hidden=ci(sh["net"]), net_visible=ci(sv["net"]), floors_visible={str(eps): ci(sv[f"floor_{eps}"]) for eps in EPS},
                                                         floors_hidden={str(eps): ci(sh[f"floor_{eps}"]) for eps in EPS},
                                                         false_reject_to_opus=ci(wmean(W, ok, ((hid_ch["H"] > 0) & (vis_ch["H"] == 0)).astype(float))),
                                                         loss_false_reject=ci(wmean(W, ok, fr_to_opus_miss * (1.0 - vb))), loss_false_accept=ci(wmean(W, ok, fa_commit * (hid_ch["V"] - vis_ch["V"]))))
            d["chains"] = chains
            vis[str(theta)] = d
        res["visible"][m] = vis

    # ---------------------------------------------------------------- flash truncation report (re-run statistics from view b)
    fl = Ab[(Ab.root_model == "deepseek-flash")] if len(Ab) else A.iloc[0:0]
    n_trunc_a = int(((A.model == "deepseek-flash") & (~A.api_error) & (A.finish_reason == "length") & (A.text.str.strip() == "")).sum())
    res["flash_truncation"] = dict(
        n_truncated_rows_in_view_a=n_trunc_a,
        protocol=f"6,144-token flash attempts with finish_reason 'length' and empty content were re-issued with the same messages at {FLASH_RERUN_MAX_TOKENS} tokens (tag {RERUN_TAG}); "
                 "originals kept in attempts_flash_truncated.jsonl; re-runs still truncated are censored (excluded from p estimates); downstream attempts of a re-run chain were regenerated from the re-run's verdict.",
        n_truncated_original=int(len(TR)), n_reissued=int((~TR.superseded.astype(bool)).sum()) if len(TR) else 0, n_superseded=int(TR.superseded.astype(bool).sum()) if len(TR) else 0,
        by_branch_k={f"{b} k={k}": int(n) for (b, k), n in TR.groupby(["branch", "k"]).size().items()} if len(TR) else {},
        original_completion_tokens=int(TR.completion_tokens.sum()) if len(TR) else 0, original_total_tokens=int(TR.total_tokens.sum()) if len(TR) else 0,
        original_usd_proxy=float(TR.cost_usd.sum()) if len(TR) else 0.0,
        n_rerun_rows=int(fl.rerun.sum()), n_censored=int(fl.censored.sum()), rerun_rows_pass_rate=float(fl[fl.rerun & ~fl.censored].truth_pass.mean()) if (fl.rerun & ~fl.censored).any() else float("nan"),
        rerun_tokens=int(fl[fl.rerun].total_tokens.sum()), rerun_usd_proxy=float(fl[fl.rerun].cost_usd.sum()),
        rerun_completion_tokens_mean=float(fl[fl.rerun].completion_tokens.mean()) if fl.rerun.any() else float("nan"),
        n_still_truncated_not_rerun=still_trunc, chains_replayed=int(fl[fl.rerun].job_id.nunique()), chains_not_replayed=int(getattr(col, "rerun_not_completed", 0)),
        n_regenerated_downstream_rows=int(((fl.collection_pass == "rerun") & (~fl.rerun) & (~fl.cached) & (fl.k > 1)).sum()),
        regenerated_downstream_tokens=int(fl[(fl.collection_pass == "rerun") & (~fl.rerun) & (~fl.cached)].total_tokens.sum()))

    # ---------------------------------------------------------------- view (b): the replayed chains at the 16,384 budget, and the counterfactual
    res["flash_views"] = flash_budget_views(A, Ab)

    ol = A[(A.model != "deepseek-flash") & (A.finish_reason == "length") & (~A.api_error)]
    res["other_truncation"] = dict(n=int(len(ol)), n_empty=int((ol.text.str.strip() == "").sum()), n_pass=int(ol.truth_pass.sum()),
                                   by_model={m: int(v) for m, v in ol.model.value_counts().items()}, by_branch_k={f"{b} k={k}": int(v) for (b, k), v in ol.groupby(["branch", "k"]).size().items()},
                                   failure_kinds={str(k): int(v) for k, v in ol.failure_kind.value_counts().items()})

    # ---------------------------------------------------------------- spend
    us = client.usage_summary()
    per_model_spend: Dict[str, Dict[str, float]] = {}
    per_tag_units: Dict[str, float] = {}
    for line in open(client.ledger_path) if os.path.exists(client.ledger_path) else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        tg = str(r.get("tag") or "")
        if not tg.startswith(TAG):
            continue
        per_tag_units[tg] = per_tag_units.get(tg, 0.0) + float(r.get("units_est") or 0.0)
        d = per_model_spend.setdefault(r["model"], dict(calls=0, live_calls=0, cached=0, prompt_tokens=0, completion_tokens=0, units_est=0.0))
        d["calls"] += 1
        d["live_calls"] += 0 if r.get("cached") else 1
        d["cached"] += 1 if r.get("cached") else 0
        if not r.get("cached"):
            d["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
            d["completion_tokens"] += int(r.get("completion_tokens") or 0)
        d["units_est"] += float(r.get("units_est") or 0.0)
    A_all = pd.concat([A, Ab], ignore_index=True) if len(Ab) else A  # every analysed row of both views (spend statistics)
    live_rows = A_all[~A_all.cached]
    alloc = float((client.load_budget().get("allocations_units") or {}).get("X7b_X10_live_retries", 100000))
    # live-pass runtimes from the preserved console logs (the analysis re-run itself takes seconds)
    pass_runtimes: Dict[str, Optional[float]] = {}
    for name in ("run_main_pass.log", "run_rerun_pass.log"):
        p = os.path.join(OUT, name)
        pass_runtimes[name] = None
        if os.path.exists(p):
            m_ = re.findall(r"done in ([0-9.]+) s", open(p, errors="replace").read())
            if m_:
                pass_runtimes[name] = float(m_[-1])
    res["spend"] = dict(tag=TAG, tags=sorted(per_tag_units), ledger_units_by_tag=per_tag_units, ledger_by_model=per_model_spend, ledger_units_tag_total=sum(d["units_est"] for d in per_model_spend.values()),
                        ledger_units_all_experiments=us["ledger_units_total"], cap_units=us["cap_units"], allocation_units=alloc, live_pass_runtimes_s=pass_runtimes,
                        attempt_rows_view_a=int(len(A)), attempt_rows_view_b=int(len(Ab)), attempt_rows=int(len(A_all)), live_attempt_rows=int(len(live_rows)), cached_attempt_rows=int(A_all.cached.sum()),
                        attempts_jsonl_rows_total=int(sum(len(run) for rs in runs.values() for run in rs)),
                        attempt_tokens_by_model={m: int(A_all[A_all.model == m].total_tokens.sum()) for m in models},
                        attempt_usd_by_model={m: float(A_all[A_all.model == m].cost_usd.sum()) for m in models}, api_errors=int(A.api_error.sum()), job_errors=col.errors,
                        api_error_rows_all_runs=int(REISSUE["api_error_rows_all_runs"]), jobs_reissued_this_run=int(REISSUE["jobs_reissued"]),
                        stopped_by_soft_cap=bool(col.stop), units_before=units_before, units_after=units_after,
                        live_wallclock_min=float((A_all.ts.max() - A_all.ts.min()) / 60.0) if len(A_all) else float("nan"))

    # ---------------------------------------------------------------- claims
    def verdict_from(diffs: List[List[float]], want: str) -> Tuple[str, str]:
        dec = [d for d in diffs if excl(d)]
        if not diffs:
            return "NOT-INSTANTIABLE", "no estimable cell"
        if not dec:
            return "INCONCLUSIVE", f"0/{len(diffs)} CIs exclude 0"
        good = sum(1 for d in dec if (d[0] < 0) == (want == "neg"))
        if good == len(dec) and len(dec) >= max(1, len(diffs) // 2):
            return "PASS", f"{good}/{len(dec)} decided cells in the predicted direction ({len(diffs)} cells)"
        if good == 0:
            return "FAIL", f"0/{len(dec)} decided cells in the predicted direction ({len(diffs)} cells)"
        return "INCONCLUSIVE", f"{good}/{len(dec)} decided cells in the predicted direction ({len(diffs)} cells)"

    claims = []
    c1 = [res["per_model"][m]["all"].get("C1_fresh_minus_marginal") for m in models if res["per_model"][m]["all"].get("C1_fresh_minus_marginal")]
    v, note = verdict_from(c1, "neg")
    claims.append(dict(id="C1", kind="LIVE", statement="p_{retry|fail} < marginal success (positive error correlation across attempts of the same model)",
                       result="fresh retry (no feedback) minus marginal p_1, pooled per model: " + "; ".join(f"{m} {f3(res['per_model'][m]['all'].get('C1_fresh_minus_marginal', [np.nan]*3))}" for m in models)
                       + " | with feedback: " + "; ".join(f"{m} {f3(res['per_model'][m]['all'].get('C1_retry_minus_marginal', [np.nan]*3))}" for m in models), verdict=v, note=note))
    c2p = [res["per_model"][m]["all"].get("C2_switch_minus_retry_p") for m in models if res["per_model"][m]["all"].get("C2_switch_minus_retry_p")]
    v2, note2 = verdict_from(c2p, "pos")
    c2a = {b: [res["per_model"][m]["all"].get(f"C2_switch_minus_retry_a_{b}") for m in models if res["per_model"][m]["all"].get(f"C2_switch_minus_retry_a_{b}")] for b in ["usd", "sec", "tok"]}
    cost_dir = {b: verdict_from(c2a[b], "neg")[1] for b in c2a}
    # joint verdict: success part decided in favour AND the $ cost of the switch attempt not significantly higher
    cost_higher = [d for d in c2a["usd"] if excl(d) and d[0] > 0]
    if v2 == "PASS" and cost_higher:
        v2 = "INCONCLUSIVE"
        note2 += "; switch attempt is significantly MORE expensive in $ for %d model(s), so the joint claim (higher success at equal-or-lower cost) does not hold as stated -- the per-$ trade-off is what the rule (C3) prices" % len(cost_higher)
    claims.append(dict(id="C2", kind="LIVE", statement="p_{switch|fail} > p_{retry|fail} at equal or lower incremental cost",
                       result="Δp (switch − retry, attempt 2, paired on the failure set): " + "; ".join(f"{m} {f3(res['per_model'][m]['all'].get('C2_switch_minus_retry_p', [np.nan]*3))}" for m in models if NEXT[m])
                       + " | Δa $ : " + "; ".join(f"{m} {fm(res['per_model'][m]['all'].get('C2_switch_minus_retry_a_usd', [np.nan]*3), 1000, 2)} m$" for m in models if NEXT[m])
                       + " | Δa s : " + "; ".join(f"{m} {f3(res['per_model'][m]['all'].get('C2_switch_minus_retry_a_sec', [np.nan]*3), 2)}" for m in models if NEXT[m]),
                       verdict=v2, note=f"success part: {note2}; cost part (switch cheaper?): $ {cost_dir['usd']}, s {cost_dir['sec']}, tokens {cost_dir['tok']}"))
    sa = res["sign_agreement"]
    oos_all = sa["oos_opt"]["all"]
    ins_all = sa["insample_opt"]["all"]
    r_rule, r_marg = oos_all["rule"], oos_all["marginal"]
    best_naive = max(r_rule["always_attempt"][0], r_rule["always_stop"][0]) if r_rule["n_decided"] else float("nan")
    # the naive baselines are basis-specific ($ cells are almost all 'attempt', s cells almost all 'stop'); the rule must also beat
    # the best naive rate WITHIN each basis, otherwise its pooled agreement is just 'attempt in $, stop in s'
    per_basis_ok = all(sa["oos_opt"][b]["rule"]["rate"][0] >= max(sa["oos_opt"][b]["rule"]["always_attempt"][0], sa["oos_opt"][b]["rule"]["always_stop"][0]) - 1e-12
                       for b in BASES if sa["oos_opt"][b]["rule"]["n_decided"] > 0)
    within = "; ".join(f"{b}: rule {sa['oos_opt'][b]['rule']['rate'][0]:.3f} vs best naive {max(sa['oos_opt'][b]['rule']['always_attempt'][0], sa['oos_opt'][b]['rule']['always_stop'][0]):.3f}"
                       for b in BASES if sa["oos_opt"][b]["rule"]["n_decided"] > 0)
    if r_rule["n_decided"] == 0:
        v3, note3 = "INCONCLUSIVE", "no CI-decided cells"
    elif r_rule["n_agree"] >= r_marg["n_agree"] and r_rule["rate"][0] >= 0.8 and r_rule["rate"][0] >= best_naive and per_basis_ok:
        v3, note3 = "PASS", f"rule >= marginal-rate predictor, >= best pooled naive baseline ({best_naive:.3f}) and >= best naive baseline within each basis ({within}); the margin over the marginal-rate predictor is small (see C3b)"
    elif r_rule["rate"][0] >= 0.8 and r_rule["rate"][0] >= best_naive:
        v3, note3 = "INCONCLUSIVE", f"rule agrees (>= 0.8, >= best pooled naive {best_naive:.3f}) but is below the marginal-rate predictor or below a within-basis naive baseline ({within})"
    else:
        v3, note3 = "FAIL", f"rule agreement below 0.8 or below the best naive baseline ({best_naive:.3f}) on OOS decided cells ({within})"
    claims.append(dict(id="C3", kind="LIVE (split-half OUT-OF-SAMPLE; in-sample also reported)",
                       statement="per-segment sign of p_k W_λ + (1−p_k)Δ_{k+1} − λ a_k matches the realized ΔJ_λ sign on CI-decided (source, k) cells, above the naive baselines (always attempt / always stop) and not below the marginal-rate predictor",
                       result=f"OOS: rule {r_rule['n_agree']}/{r_rule['n_decided']} ({r_rule['rate'][0]:.3f}) vs marginal-rate {r_marg['n_agree']}/{r_marg['n_decided']} ({r_marg['rate'][0]:.3f}); "
                              f"always-attempt {oos_all['rule']['always_attempt'][0]:.3f}, always-stop {oos_all['rule']['always_stop'][0]:.3f}; "
                              f"discriminating cells (predictors disagree): rule right {r_rule.get('discriminating', {}).get('n_pred_right', 0)}/{r_rule.get('discriminating', {}).get('n', 0)}; "
                              f"$ cells: rule {sa['oos_opt']['usd']['rule']['n_agree']}/{sa['oos_opt']['usd']['rule']['n_decided']} (always-attempt {sa['oos_opt']['usd']['rule']['always_attempt'][0]:.3f}); "
                              f"s cells: rule {sa['oos_opt']['sec']['rule']['n_agree']}/{sa['oos_opt']['sec']['rule']['n_decided']} (always-stop {sa['oos_opt']['sec']['rule']['always_stop'][0]:.3f}); "
                              f"in-sample rule {ins_all['rule']['n_agree']}/{ins_all['rule']['n_decided']} vs marginal {ins_all['marginal']['n_agree']}/{ins_all['marginal']['n_decided']}",
                       verdict=v3, note=note3))
    dis = r_rule.get("discriminating", {})
    n_dis, k_dis = int(dis.get("n", 0)), int(dis.get("n_pred_right", 0))
    if n_dis == 0:
        v3b, note3b = "NOT-INSTANTIABLE", "the conditional and the marginal predictor never disagree in sign on a CI-decided OOS cell"
    elif n_dis < 8:
        v3b, note3b = "INCONCLUSIVE", f"only {n_dis} discriminating OOS cells (rule right on {k_dis}); too few for a sign test"
    elif k_dis / n_dis > 0.5 and wilson(k_dis, n_dis)[1] > 0.5:
        v3b, note3b = "PASS", f"rule right on {k_dis}/{n_dis} discriminating cells, Wilson CI above 0.5"
    elif k_dis / n_dis < 0.5 and wilson(k_dis, n_dis)[2] < 0.5:
        v3b, note3b = "FAIL", f"marginal-rate predictor right on {n_dis - k_dis}/{n_dis} discriminating cells"
    else:
        v3b, note3b = "INCONCLUSIVE", f"rule right on {k_dis}/{n_dis} discriminating cells, CI includes 0.5"
    claims.append(dict(id="C3b", kind="LIVE (split-half OUT-OF-SAMPLE)",
                       statement="where the conditional rule and the marginal-rate predictor disagree in sign, the conditional rule is the one that matches the realized sign (the conditioning on failure carries information beyond the marginal rate)",
                       result=f"discriminating CI-decided OOS cells: {n_dis}; conditional rule right on {k_dis} ({f3(wilson(k_dis, n_dis))}); overall agreement rule {r_rule['n_agree']}/{r_rule['n_decided']} vs marginal {r_marg['n_agree']}/{r_marg['n_decided']}",
                       verdict=v3b, note=note3b))
    c4 = [res["per_model"][m]["all"].get("C4_retryfb_minus_fresh") for m in models if res["per_model"][m]["all"].get("C4_retryfb_minus_fresh")]
    v4, note4 = verdict_from(c4, "pos")
    claims.append(dict(id="C4", kind="LIVE", statement="feedback-conditioned retry beats a fresh retry on the same failure set (σ > 0: the failed attempt leaves reusable work)",
                       result="; ".join(f"{m} {f3(res['per_model'][m]['all'].get('C4_retryfb_minus_fresh', [np.nan]*3))} (n={res['per_model'][m]['all'].get('n_C4', 0)})" for m in models), verdict=v4, note=note4))
    qs = []
    for m in models:
        for th in THETAS_VIS:
            for key, ch in res["visible"][m][str(th)]["chains"].items():
                if key.endswith("usd"):
                    qs.append(ch["Q_loss"])
    v5q, note5q = verdict_from(qs, "neg")
    e_ci = {(m, th): res["visible"][m][str(th)]["e_false_accept_ci"] for m in models for th in THETAS_VIS}
    e_pos = [k for k, t in e_ci.items() if np.isfinite(t[1]) and t[1] > 0]
    if v5q == "PASS" and e_pos:
        v5, note5 = "PASS", f"Q loss: {note5q}; attempt-level e > 0 (Wilson CI excludes 0) for {len(e_pos)}/{len(e_ci)} (model, θ) pairs"
    elif v5q == "PASS":
        v5, note5 = "PASS", f"Q loss: {note5q}; but no (model, θ) pair has e > 0 with CI excluding 0 -- the loss runs through false REJECTS that fall back to an Opus miss, not through false accepts"
    else:
        v5, note5 = v5q, f"Q loss: {note5q}; e > 0 with CI excluding 0 for {len(e_pos)}/{len(e_ci)} (model, θ) pairs"
    claims.append(dict(id="C5", kind="LIVE", statement="the imperfect visible-test verifier has false accepts e > 0 and loses quality relative to the hidden-test verifier",
                       result="attempt-level e (θ=1 / θ=0.5): " + "; ".join(f"{m} {res['visible'][m]['1.0']['e_false_accept']:.3f} / {res['visible'][m]['0.5']['e_false_accept']:.3f}" for m in models)
                       + " | chain Q loss (retry K=3, θ=1 / θ=0.5): "
                       + "; ".join(f"{m} {f3(res['visible'][m]['1.0']['chains'].get('retry K=3 usd', {}).get('Q_loss', [np.nan]*3))} / {f3(res['visible'][m]['0.5']['chains'].get('retry K=3 usd', {}).get('Q_loss', [np.nan]*3))}" for m in models),
                       verdict=v5, note=note5))
    res["claims"] = claims
    res["runtime_s"] = round(time.time() - T0, 1)
    res["deviations"] = [
        "n = %d tasks (< 300): the EXEC-CLEAN runnable id_test frame after the uniform all-zero exclusion is the whole sample (Addendum applies the all-zero rule to bcb; here it is applied to every source because Opus's terminal outcome and W_lambda are degenerate on such tasks; 48 mbpp dim1 + 3 mbpp dim4 tasks affected)." % n,
        "deepseek-v4-pro (thinking disabled) and deepseek-flash (reasoning on, max_tokens 2048, added on the coordinator's Addendum-2 request) are NON-BENCHMARK models: no recorded attempt 0, dollar costs use PROXY prices (Qwen3-Max and qwen3.5-plus list prices respectively); their primary bases are tokens and seconds. Their switch target is qwen3-max (not in the paper's chain). deepseek-flash completion tokens include its reasoning tokens.",
        "Feedback = hidden-test error message (tail of the sandbox stderr, <= 800 chars); for assert-style suites this reveals the failing assertion, so the retry gain mixes reusable work and test-information leakage (C4 reports the split by failure kind).",
        "The visible-verifier chains (C5) are replayed on the attempts generated under hidden-test feedback (a visible-verifier false reject would have produced a different feedback text, and a falsely rejected passing attempt has no live continuation and is sent to the Opus terminal), so the visible chain's quality loss and cost increase are upper bounds; the extraction follows ACRouter's run_visible_tests except that '>>>'/'...' continuation lines are dropped from the expected side and the <= 5 examples run in one sandbox process. In this frame 77/112 bcb prompts yield at least one clean call/expected pair; the other 35 carry only multi-statement doctests (setup lines), which ACRouter's regex would pair with the next '>>>' line and always fail (systematic false rejects); here they score the neutral 0.5 like prompts without examples (mbpp dim1 has none by construction).",
        f"deepseek-flash truncation: with max_tokens 6144 its reasoning exhausted the completion budget on {res['flash_truncation']['n_truncated_original']} attempts over both passes (finish_reason=length, empty content). HEADLINE = view (a): these stay in the sample as paid failures at the 6,144 budget (budget-matched with the other 249 flash roots). Per the coordinator's protocol the truncated attempts were also re-issued with the same messages at {FLASH_RERUN_MAX_TOKENS} tokens under tag {RERUN_TAG} (originals in attempts_flash_truncated.jsonl); this is view (b), reported in its own section with its own attempt costs and never mixed into view (a); {res['flash_truncation']['n_censored']} re-runs still truncated and are censored there. Inside view (b) the retry/switch attempts of a replayed chain were regenerated from the re-run's real verdict and error message (the original downstream rows had been conditioned on an empty assistant turn and the feedback 'empty response').",
        "All live outputs are code-only (system prompt) with max_tokens 1024, temperature 0.7; recorded attempt-0 responses were free-form, so p_0 and p_1 differ in format as well as sampling.",
        "mbpp dim1 live prompts append 'Name the function `<entry_point>`' (the released prompt did not show the name and the MBPP+ harness calls it; without the name the failure set is dominated by naming mismatches that the NameError feedback fixes trivially). Attempt 0 for mbpp dim1 is therefore not prompt-matched to attempt 1.",
        "kimi-k2.5 has no live switch target (the next-stronger model is the recorded Opus terminal), so its switch branch is the terminal fallback; its 'switch' success is the recorded Opus pass rate on its failure set.",
        "Switch chains: qwen3.5-plus -> qwen3-max -> kimi-k2.5 (paper's chain, in full) and qwen3-max -> kimi-k2.5; the two non-benchmark roots switch to qwen3-max only (no third live attempt), to keep the kimi-k2.5 load (2 concurrent calls shared by all experiments) inside the 2-hour wall-clock.",
    ]
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    log("results.json written")

    # ---------------------------------------------------------------- figures
    if not args.no_figs:
        try:
            plt = setup_mpl()
            fig_conditional(res, plt)
            fig_pred_real(res, plt)
            fig_cost(res, plt)
            log("figures written")
        except Exception as e:  # noqa: BLE001
            log("figure error:", e)
            traceback.print_exc()

    # ---------------------------------------------------------------- README / summary
    write_readme(res, A, F, R)
    log(f"done in {res['runtime_s']} s")
    print("\nCLAIMS")
    for c in claims:
        print(f"  {c['id']}: {c['verdict']} -- {c['result'][:200]}")


def write_readme(res: Dict[str, Any], A: pd.DataFrame, F: pd.DataFrame, R: pd.DataFrame) -> None:
    models = res["models"]
    L: List[str] = []
    L.append("# X7-b / X10-live: in-loop routing with real retries (LIVE)\n")
    L.append(f"Generated by `run.py` (seed {SEED}, B = {B} paired task-bootstrap resamples, 95% percentile CIs; this analysis run {res['runtime_s']} s from the cache; live passes: see Deviations and limitations). "
             "Everything below is LIVE unless labelled otherwise: primitives and realized values come from the same live sample, so the rule test (C3) is reported both IN-SAMPLE and split-half OUT-OF-SAMPLE.\n")
    L.append("## Setup\n")
    L.append(f"- Tasks: {res['n_tasks']} id_test tasks from the EXEC-CLEAN sources (humaneval, he dim4 completion, mbpp dim1/dim4, hex python, bcb), runnable in the local sandbox and "
             f"not all-zero across the 8 recorded models (frame: {res['frame_counts']}). Per source: {res['sources']}. Stratified by source, seed {SEED} (the frame is smaller than the 300-task target, so the whole frame is used).")
    L.append(f"- Models: {', '.join(m + (' (NON-BENCHMARK; proxy $ price)' if m in NON_BENCHMARK else '') for m in models)}. Switch targets (attempt 2, attempt 3): {res['chain']}. Workers: {res['workers']}. Temperature {TEMP}, max_tokens {MAX_TOKENS} ({MAX_TOKENS_BY_MODEL} for the reasoning model), code-only system prompt.")
    L.append("- Verifier: hidden-test sandbox (`common.sandbox.verify_response`) = truth for live attempts; imperfect variant: visible-test verifier (doctest/assert examples in the prompt, ACRouter `run_visible_tests` semantics), accept iff syntax OK and visible pass fraction ≥ θ (θ ∈ {1, 0.5}; no examples -> neutral 0.5).")
    L.append(f"- Chain: attempt 1 fresh; on rejection attempt 2/3 = (a) retry same model with the sandbox error message (≤ {FEEDBACK_CHARS} chars) appended, (b) switch to the next-stronger model with the same context, (c) fresh retry (no feedback, seed=2) for the salvage test. Terminal fallback = recorded Opus (csv score, cost_usd, latency).")
    L.append(f"- Costs: $ = tokens × models.json list prices per 1M {res['prices_per_1m']} (deepseek = PROXY Qwen3-Max price); seconds = API latency + sandbox verification time (sequential steps add); tokens = relay-reported total tokens. Opus terminal: csv cost_usd / latency_ms.")
    L.append(f"- Prices λ: η_B per $ = {f3(res['eta_B']['usd'], 2)}, per s = {f3(res['eta_B']['sec'], 3)} (ratio of sums over the {res['n_tasks']} tasks), plus λ ∈ {{1, 10}} per $ and per s.\n")

    # recorded attempt 0
    L.append("## Recorded attempt 0 (sandbox vs released) and live attempt 1\n")
    rows = []
    for m, d in res["recorded"].items():
        rows.append([m, d["n"], f3(d["released_pass"]), f3(d["sandbox_pass_verifiable"]), f"{d['agreement']:.3f} (n={d['n_verifiable']})", d["false_pass"], d["false_fail"], f"{d['empty']} ({d['empty_released_pass']} released pass)",
                     f3(d["live_attempt1_pass"]), fm(d["rec_cost_usd"]) + " / " + fm(d["live_cost_usd"]), f3(d["rec_sec"], 1) + " / " + f3(d["live_api_sec"], 1), f3(d["rec_tok"], 0) + " / " + f3(d["live_tok"], 0)])
    L.append(table(["model", "n", "released pass (attempt 0 = truth)", "sandbox pass on verifiable attempt-0 texts", "agreement (non-empty)", "FP", "FF", "empty (unverifiable)", "LIVE attempt-1 pass", "cost m$ rec / live", "s rec / live API", "tokens rec / live"], rows))
    L.append("\nAttempt 0 = the recorded response; its truth is the released score (empty recorded texts are unverifiable, not failures). The sandbox column is a validity check of the local verifier on the same texts.\n")
    # conditional success
    L.append("## Conditional success by attempt index (pooled over sources; failure set = tasks whose live attempt 1 failed)\n")
    rows = []
    for m in models:
        d = res["per_model"][m]["all"]
        rows.append([m, d["n"], d["n_fail1"], f3(d["p1"]), f"{f3(d['p1_excl_length'])} ({d['n_length_1']} truncated)", f3(d.get("p_f2")), f3(d.get("p_r2")), f3(d.get("p_r3")), f3(d.get("p_s2")) if NEXT[m] else "terminal (Opus): " + f3(d.get("opus_pass_on_fail1", [np.nan]*3)), f3(d.get("p_s3")) if NEXT3[m] else "-"])
    L.append(table(["root model", "n", "n fail1", "p_1 (marginal)", "p_1 excl. truncated outputs [Wilson]", "p_2 fresh (no feedback)", "p_2 retry+feedback", "p_3 retry", "p_2 switch", "p_3 switch"], rows))
    L.append("\nTruncated = finish_reason `length` (the completion budget was exhausted; for the reasoning model deepseek-flash the reasoning consumed it and the content is empty). Such attempts count as failures in every chain (the verifier sees no code) but are a max_tokens artefact; the C4 table splits the retry gain by failure kind.\n")
    L.append("Per source (p_1 / p_2 fresh / p_2 retry / p_2 switch; n fail1 in brackets):\n")
    rows = []
    for m in models:
        for s in SOURCES:
            d = res["per_model"][m].get(s)
            if not d:
                continue
            rows.append([m, s, d["n"], d["n_fail1"], f3(d["p1"], 2), f3(d.get("p_f2"), 2), f3(d.get("p_r2"), 2), f3(d.get("p_s2"), 2) if NEXT[m] else "-"])
    L.append(table(["model", "source", "n", "n fail1", "p_1", "p_2 fresh", "p_2 retry", "p_2 switch"], rows))
    L.append("")
    # attempt costs
    L.append("## Attempt costs a_k on the state sets (means; $ per task ×1000, seconds incl. sandbox, tokens)\n")
    rows = []
    for m in models:
        d = res["per_model"][m]["all"]
        for key, lab in [("1", "attempt 1"), ("f2", "fresh 2"), ("r2", "retry 2"), ("r3", "retry 3"), ("s2", "switch 2"), ("s3", "switch 3")]:
            if d.get(f"a_{key}_usd") is None or not np.isfinite(d[f"a_{key}_usd"][0]):
                continue
            rows.append([m, lab, d.get(f"n_{key}", d["n"]), fm(d[f"a_{key}_usd"]), f3(d[f"a_{key}_sec"], 1), f3(d[f"a_{key}_tok"], 0)])
    L.append(table(["model", "attempt", "n", "m$", "s", "tokens"], rows))
    L.append("")
    # budget-matched deepseek-flash views
    fv = res.get("flash_views") or {}
    va, vb, cf = fv.get("view_a") or {}, fv.get("view_b"), fv.get("counterfactual_p1_at_16k")
    L.append("## deepseek-flash: budget-matched views (the headline is view a)\n")
    L.append("Every table and claim in this README except this section uses VIEW (a) = each chain's first complete run: deepseek-flash at the 6,144-token completion budget, a truncated attempt (whole budget spent on reasoning, no code returned) counted as a paid failure (p = 0 by construction). "
             "VIEW (b) = the last run of the chains replayed in the re-run pass, where the truncated attempts were re-issued with the same messages at 16,384 tokens (re-runs still truncated are censored); it is reported here only and never mixed into a view-(a) estimate. Wilson 95% intervals (counts).\n")
    rows = [["(a) 6,144 budget, truncations = failures (HEADLINE)", va.get("n_roots"), f3(va.get("p1_truncations_as_failures")), f"{va.get('n_truncated_roots')} truncated roots (each 6,144 completion tokens, {1000 * va.get('truncated_root_cost', {}).get('mean_usd_proxy', float('nan')):.2f} m$)",
             f"{va.get('root_cost', {}).get('mean_completion_tokens', float('nan')):.0f} tokens / {1000 * va.get('root_cost', {}).get('mean_usd_proxy', float('nan')):.2f} m$ / {va.get('root_cost', {}).get('mean_latency_s', float('nan')):.1f} s (n={va.get('root_cost', {}).get('n')})"]]
    rows.append(["(a) same roots, non-truncated only (capability, not budget-matched to the chain)", va.get("root_cost", {}).get("n"), f3(va.get("p1_non_truncated_roots")), "-", "as above"])
    if vb:
        cb = vb["cost_by_row_basis"]
        ra_, rn_ = cb["root_reruns_16k_all"], cb["root_reruns_16k_non_censored"]
        rows.append([f"(b) 16,384 budget: the {vb['n_root_rerun']} re-run roots of the {vb['n_chains']} replayed chains, censored excluded", vb["n_root_rerun"] - vb["n_root_censored"], f3(vb["p1_16k_excluding_censored"]), f"{vb['n_root_censored']} censored (still truncated at 16,384)",
                     f"root re-runs, non-censored (n={rn_['n']}): mean {rn_['mean_completion_tokens']:.0f} completion tokens / {1000 * rn_['mean_usd_proxy']:.2f} m$ / {rn_['mean_latency_s']:.1f} s"])
        rows.append([f"(b) same, censored counted as failures", vb["n_root_rerun"], f3(vb["p1_16k_censored_as_failures"]), "-",
                     f"root re-runs, all incl. censored (n={ra_['n']}): mean {ra_['mean_completion_tokens']:.0f} completion tokens (median {ra_['median_completion_tokens']:.0f}) / {1000 * ra_['mean_usd_proxy']:.2f} m$ / {ra_['mean_latency_s']:.1f} s"])
    L.append(table(["view", "n roots", "p_1 [Wilson]", "truncation / censoring", "root attempt cost (mean unless stated; proxy $ price)"], rows))
    L.append("")
    if vb:
        cb = vb["cost_by_row_basis"]
        L.append("Cost of the 16,384-budget rows by row basis (means of the relay-reported `completion_tokens` and the proxy-priced `cost_usd`; a censored row bills exactly 16,384 completion tokens):\n")
        lab = {"root_reruns_16k_all": "root re-runs at 16k, all (incl. censored)", "root_reruns_16k_non_censored": "root re-runs at 16k, non-censored", "all_16k_rows_all_branches": "all 16k rows, all branches (root + fresh + retry k=2 + retry k=3), incl. censored",
               "all_16k_rows_all_branches_non_censored": "all 16k rows, all branches, non-censored", "censored_16k_rows": "censored 16k rows only", "budget_6144_flash_rows_in_view_b": "deepseek-flash rows in view (b) generated at the 6,144 budget (regenerated downstream attempts)"}
        L.append(table(["rows", "n", "censored", "mean completion tokens", "median completion tokens", "mean total tokens", "mean m$", "mean latency s"],
                       [[lab[k], v["n"], v["n_censored"], f"{v['mean_completion_tokens']:.0f}", f"{v['median_completion_tokens']:.0f}", f"{v['mean_total_tokens']:.0f}", f"{1000 * v['mean_usd_proxy']:.2f}", f"{v['mean_latency_s']:.1f}"] for k, v in cb.items()]))
        L.append(f"\nReference for the root-vs-root comparison: the {va.get('root_cost', {}).get('n')} non-truncated view-(a) roots at the 6,144 budget average {va.get('root_cost', {}).get('mean_completion_tokens', float('nan')):.0f} completion tokens / {1000 * va.get('root_cost', {}).get('mean_usd_proxy', float('nan')):.2f} m$.\n")
    if vb:
        k2 = vb["k2"]
        L.append(f"View (b) failure set (root re-run failed, non-censored): {vb['n_fail_roots_16k']} tasks. Attempt-2 rates on it: fresh {f3(k2['fresh']['p'], 2)} (n={k2['fresh']['n']}, {k2['fresh']['n_at_16k']} at 16k), "
                 f"retry+feedback {f3(k2['retry']['p'], 2)} (n={k2['retry']['n']}, {k2['retry']['n_at_16k']} at 16k), switch to {k2['switch']['model']} {f3(k2['switch']['p'], 2)} (n={k2['switch']['n']}). "
                 f"Downstream attempts of a replayed chain were regenerated at the 6,144 budget and re-issued at 16,384 only if they truncated (hence the mixed budgets inside view b). Replayed-chain cost per task (all attempts on the analysed path): {1000 * vb['chain_cost_per_task_usd']:.2f} m$, {vb['chain_cost_per_task_tokens']:.0f} tokens.\n")
    if cf:
        L.append(f"COUNTERFACTUAL population p_1 at 16,384 (not a measurement of one budget: {cf['definition']}; n = {cf['n_population']}, of which {cf['n_censored']} censored): "
                 f"censored as failures {f3(cf['censored_as_failures'])}; excluding censored {f3(cf['excluding_censored'])}. It rests on {vb['n_root_rerun'] if vb else 0} root re-runs and is reported for orientation only.\n")
    # C1/C2/C4 paired differences
    L.append("## Paired differences (same failure set, same bootstrap resamples)\n")
    rows = []
    for m in models:
        d = res["per_model"][m]["all"]
        rows.append([m, f3(d.get("C1_fresh_minus_marginal")), f3(d.get("C1_retry_minus_marginal")), f3(d.get("C4_retryfb_minus_fresh")), d.get("n_C4", 0),
                     f3(d.get("C2_switch_minus_retry_p")) if NEXT[m] else "-", fm(d.get("C2_switch_minus_retry_a_usd")) if NEXT[m] else "-",
                     f3(d.get("C2_switch_minus_retry_a_sec"), 1) if NEXT[m] else "-", f3(d.get("C2_switch_minus_retry_a_tok"), 0) if NEXT[m] else "-"])
    L.append(table(["model", "C1: p_2 fresh − p_1", "p_2 retry − p_1", "C4: p_2 retry − p_2 fresh", "n", "C2: p_2 switch − p_2 retry", "Δa m$", "Δa s", "Δa tokens"], rows))
    L.append("")
    L.append("Salvage value of the failed attempt's observations (prop:salvage proxy): Δp W_λ − λ Δa on the failure state at λ = η_B, with Δp = p_2 retry+feedback − p_2 fresh and Δa the extra cost of carrying the feedback (score units per task in the state; in brackets the same divided by η_B, i.e. in $ or s):\n")
    rows = []
    for m in models:
        d = res["per_model"][m]["all"]
        if d.get("salvage_value_usd"):
            rows.append([m, d.get("n_C4", 0), f3(d["salvage_value_usd"]), fm(d["salvage_value_usd_in_cost_units"]), f3(d["salvage_value_sec"]), f3(d["salvage_value_sec_in_cost_units"], 1)])
    L.append(table(["model", "n", "$ basis (score units)", "= m$ per task in the state", "s basis (score units)", "= s per task in the state"], rows) if rows else "(no cells with n ≥ 3)")
    L.append("")
    L.append("C4 by failure kind of attempt 1 (p_2 retry+feedback vs p_2 fresh):\n")
    rows = []
    for m in models:
        for fk, d in (res["per_model"][m]["all"].get("C4_by_failure_kind") or {}).items():
            rows.append([m, fk, d["n"], f3(d["p_retry"], 2), f3(d["p_fresh"], 2), f3(d["diff"], 2)])
    L.append(table(["model", "failure kind (attempt 1)", "n", "p_2 retry", "p_2 fresh", "diff"], rows) if rows else "(no cells with n ≥ 3)")
    L.append("")
    # rule cells
    L.append("## Escalation rule (prop:attempts) per segment: predicted vs realized\n")
    L.append("Cells = (model, branch, k, source or pooled, basis, λ). `G` = p_k W_λ + (1−p_k)Δ_{k+1} − λ a_k with p_k, a_k, b, v_b estimated on the state set (v_h = 1); `realized` = paired mean of the per-task gain of attempting k (then continuing as the rule prescribes) over the direct Opus terminal. Sign agreement on CI-decided cells (realized CI excludes 0), next to the marginal-rate predictor (p_k replaced by the called model's unconditional attempt-1 rate) and the naive always-attempt / always-stop rates.\n")
    for lab, key in [("Split-half OUT-OF-SAMPLE (primitives on one half, realized on the other; both directions)", "oos_opt"), ("IN-SAMPLE", "insample_opt"), ("IN-SAMPLE, segment-only rule (Δ=0, prop:threshold) vs segment-only realized", "insample_seg")]:
        L.append(f"**{lab}**\n")
        rows = []
        for g, d in res["sign_agreement"][key].items():
            r_, mg = d["rule"], d["marginal"]
            dis = r_.get("discriminating", {})
            mino = r_.get("minority", {})
            mino_m = mg.get("minority", {})
            rows.append([g, r_["n_cells"], r_["n_decided"], f"{r_['n_agree']}/{r_['n_decided']} = {f3(r_['rate'])}", f"{mg['n_agree']}/{mg['n_decided']} = {f3(mg['rate'])}", f3(r_["always_attempt"]), f3(r_["always_stop"]),
                         f"{dis.get('n_pred_right', 0)}/{dis.get('n', 0)}" + (f" = {f3(dis['rate'])}" if dis.get("n") else ""),
                         (f"{mino.get('n_pred_right', 0)}/{mino.get('n', 0)} (rule) vs {mino_m.get('n_pred_right', 0)}/{mino_m.get('n', 0)} (marginal); majority = {mino.get('majority', '-')}") if mino else "-"])
        L.append(table(["cells", "n", "CI-decided", "rule (conditional p_k)", "marginal-rate predictor", "always attempt", "always stop", "discriminating cells: rule right / n (the two predictors disagree)",
                        "minority-sign predictions right / n (cells where the predictor goes against the group's naive majority)"], rows))
        L.append("")
    L.append("Pooled-source cells at λ = η_B (in-sample), $ basis:\n")
    rows = []
    for c in res["cells"]["insample"]:
        if c["source"] == "all" and c["lam_name"] == "eta_B" and c["basis"] == "usd":
            rows.append([c["model"], c["branch"], c["k"], c["model_called"], c["n_state"], f3(c["p"], 2), fm(c["a"]), fm(c["b"]), f3(c["v_b"], 2), f3(c["W"], 2), f3(c["delta_next"], 3), f3(c["G"], 3), f3(c["G_marg"], 3), f3(c["real_opt"], 3), "yes" if excl(c["real_opt"]) else "no"])
    L.append(table(["model", "branch", "k", "called", "n state", "p_k", "a_k m$", "b m$", "v_b", "W_λ", "Δ_{k+1}", "G rule", "G marginal", "realized ΔJ", "decided"], rows))
    L.append("")
    L.append("Per-source cells, attempt 2, λ = η_B (in-sample), $ basis (cells with ≥ 3 tasks in the state):\n")
    rows = []
    for c in res["cells"]["insample"]:
        if c["source"] != "all" and c["k"] == 2 and c["lam_name"] == "eta_B" and c["basis"] == "usd":
            rows.append([c["model"], c["branch"], c["source"], c["n_state"], f3(c["p"], 2), f3(c["p_marg"], 2), fm(c["a"]), fm(c["b"]), f3(c["v_b"], 2), f3(c["G"], 3), f3(c["G_marg"], 3), f3(c["real_opt"], 3), "yes" if excl(c["real_opt"]) else "no"])
    L.append(table(["model", "branch", "source", "n state", "p_2|fail", "marginal p", "a_2 m$", "b m$", "v_b", "G rule", "G marginal", "realized ΔJ", "decided"], rows) if rows else "(none)")
    L.append("")
    L.append("Pooled-source cells at λ = η_B (in-sample), seconds basis:\n")
    rows = []
    for c in res["cells"]["insample"]:
        if c["source"] == "all" and c["lam_name"] == "eta_B" and c["basis"] == "sec":
            rows.append([c["model"], c["branch"], c["k"], c["n_state"], f3(c["p"], 2), f3(c["a"], 1), f3(c["b"], 1), f3(c["W"], 2), f3(c["G"], 3), f3(c["G_marg"], 3), f3(c["real_opt"], 3), "yes" if excl(c["real_opt"]) else "no"])
    L.append(table(["model", "branch", "k", "n state", "p_k", "a_k s", "b s", "W_λ", "G rule", "G marginal", "realized ΔJ", "decided"], rows))
    L.append("")
    # policies
    L.append("## Policies under one accounting (reference = always-Opus; hidden-test verifier; $ basis, per task)\n")
    L.append("Net gain = Q + η_B S = avoided Opus work − attempt cost + accepted quality change (+ 0 missed-quality and + 0 recovery-cost terms: the terminal path IS the reference, so a task that is not committed receives exactly the reference's value and cost). η/η_B is the ratio of sums. Cells are not independent across λ and basis (the same state enters once per λ and per basis); the per-group rows are the honest units.\n")
    rows = []
    for m in models:
        for pname, d in res["policies"][m]["usd"].items():
            rows.append([m, pname, f3(d["EV"]), fm(d["ED"]), f3(d["r"], 2), fm(d["net_per_task_cost"]), fm(d["avoided"]), fm(d["attempt"]), f3(d["Qhit"]), fm(d["cost_per_resolved"]), f3(d["natt"], 2),
                         " ".join(("Y" if d[f"floor_{eps}"][0] >= 0.5 else "n") for eps in EPS)])
    L.append(table(["model", "policy", "E[V]", "E[D] m$", "η/η_B", "net gain m$", "avoided Opus m$", "attempts m$", "accepted quality Δ", "m$ per resolved", "mean attempts", "floor ε=0/.01/.02/.05"], rows))
    L.append("")
    L.append("Seconds basis:\n")
    rows = []
    for m in models:
        for pname, d in res["policies"][m]["sec"].items():
            rows.append([m, pname, f3(d["EV"]), f3(d["ED"], 1), f3(d["r"], 2), f3(d["cost_per_resolved"], 1)])
    L.append(table(["model", "policy", "E[V]", "E[D] s", "η/η_B (s)", "s per resolved"], rows))
    L.append("")
    # visible verifier
    L.append("## Imperfect visible-test verifier (C5)\n")
    def fx(x: Any, nd: int = 3) -> str:
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    L.append("Attempt level (all live attempts of each root model; truth = hidden tests). Visible verifier: accept iff syntax OK and visible pass fraction ≥ θ; prompts without examples score the neutral 0.5 (ACRouter), so θ=1 rejects them and θ=0.5 accepts them.\n")
    rows = []
    for m in models:
        for th in THETAS_VIS:
            d = res["visible"][m][str(th)]
            rows.append([m, th, d["n_attempts"], d["n_truth_fail"], f"{d['coverage_with_examples']:.2f}", fx(d["accept_rate"]), f3(d["e_false_accept_ci"]), fx(d["false_reject"]), fx(d["TPR"]), fx(d["FPR"]),
                         f"{fx(d['with_examples']['e'])} / {fx(d['with_examples']['fr'])} (n={d['with_examples']['n']})", f"{fx(d['without_examples']['e'])} / {fx(d['without_examples']['fr'])} (n={d['without_examples']['n']})"])
    L.append(table(["model", "θ", "attempts", "truth fails", "share with examples", "accept rate", "e = P(accept, truth fail) [Wilson]", "P(reject, truth pass)", "TPR", "FPR", "with examples: e / false-reject", "without examples: e / false-reject"], rows))
    L.append("")
    L.append("Chain replay with the visible verifier as the acceptance rule (truth still scores the committed output; $ basis). Q loss = E[V]_visible − E[V]_hidden; its two channels: false rejects that end at an Opus miss, and false accepts of a wrong output. "
             "CONSERVATIVE replay: a falsely rejected attempt that truly passed has no live continuation (the hidden-verifier chain had ended), so the task goes to the Opus terminal at cost b; the visible chain's Q loss and E[D] increase are therefore upper bounds along this chain.\n")
    rows = []
    for m in models:
        for th in THETAS_VIS:
            for key, ch in res["visible"][m][str(th)]["chains"].items():
                if key.endswith("usd"):
                    rows.append([m, th, key.replace(" usd", ""), f3(ch["EV_hidden"]), f3(ch["EV_visible"]), f3(ch["Q_loss"]), f3(ch["loss_false_reject"]), f3(ch["loss_false_accept"]), f3(ch["false_reject_to_opus"]),
                                 fm(ch["ED_hidden"]), fm(ch["ED_visible"]), f3(ch["e_chain"]), f3(ch["eta_r_hidden"], 2), f3(ch["eta_r_visible"], 2)])
    L.append(table(["model", "θ", "chain", "E[V] hidden", "E[V] visible", "Q loss", "loss via false reject → Opus miss", "loss via false accept", "P(false reject → Opus)", "E[D] hidden m$", "E[D] visible m$", "e (chain commits a wrong output)", "η/η_B hidden", "η/η_B visible"], rows))
    L.append("")
    # claims
    L.append("## Claims\n")
    L.append(table(["id", "kind", "statement", "result", "verdict", "note"], [[c["id"], c["kind"], c["statement"], c["result"], c["verdict"], c["note"]] for c in res["claims"]]))
    L.append("")
    # spend
    sp = res["spend"]
    L.append("## Spend (tag X7b; ledger units at the conservative 0.05 units/token)\n")
    rows = [[m, d["calls"], d["live_calls"], d["cached"], d["prompt_tokens"], d["completion_tokens"], f"{d['units_est']:.0f}"] for m, d in sp["ledger_by_model"].items()]
    L.append(table(["model", "calls", "live", "cached", "prompt tokens (live)", "completion tokens (live)", "units_est"], rows))
    L.append(f"\nTag total {sp['ledger_units_tag_total']:.0f} units of the {sp['allocation_units']:,.0f} allocation in common/budget.json (initially 100,000; raised by the coordinator for the deepseek-flash re-run pass; cap for all experiments {sp['cap_units']:,.0f}; ledger total {sp['ledger_units_all_experiments']:,.0f}). "
             f"Analysed attempt rows {sp['attempt_rows']} = view (a) {sp['attempt_rows_view_a']} + view (b) {sp['attempt_rows_view_b']} ({sp['live_attempt_rows']} live, {sp['cached_attempt_rows']} cache hits at analysis time); "
             f"attempts.jsonl holds {sp['attempts_jsonl_rows_total']} rows in complete runs (both passes; the views de-duplicate it: see `attempts_view_a_6144.jsonl` / `attempts_view_b_rerun16k.jsonl`, each row with an explicit `budget` field). "
             f"Attempt tokens by called model over both views {sp['attempt_tokens_by_model']}; $ by model {{{', '.join(f'{k}: {v:.3f}' for k, v in sp['attempt_usd_by_model'].items())}}}; soft-cap stop: {sp['stopped_by_soft_cap']}.\n")
    L.append(f"Ledger units by tag: {{{', '.join(f'{k}: {v:.0f}' for k, v in sp['ledger_units_by_tag'].items())}}}.\n")
    L.append(f"Relay errors: {sp['api_error_rows_all_runs']} attempt rows over all runs ended in a relay error after packy's retries (this includes the intermittent HTTP 400 'Incorrect API key' flake the coordinator reported); "
             f"{sp['jobs_reissued_this_run']} chains were re-issued in this run because their last run contained such a row (successful attempts came from the cache); "
             f"{sp['api_errors']} attempt rows in the final analysed set still carry an API error (their tasks are excluded from that root model's estimates).\n")
    ft = res["flash_truncation"]
    L.append(f"deepseek-flash truncation (measured phenomenon): in the headline view (a) {ft['n_truncated_rows_in_view_a']} attempts are truncated and count as paid failures. Over both passes {ft['n_truncated_original']} attempts ({ft['by_branch_k']}) spent their whole 6,144-token completion budget on reasoning and returned no code -- "
             f"an attempt that pays a_k and returns nothing has p = 0 by construction; they cost {ft['original_total_tokens']:,} tokens ({ft['original_completion_tokens']:,} completion) = {ft['original_usd_proxy']:.3f} $ at the proxy price and are kept in `attempts_flash_truncated.jsonl` "
             f"({ft['n_reissued']} re-issued; {ft['n_superseded']} superseded = paid in the main pass but no longer on the replayed chain's path because the re-run root/retry passed). "
             f"Re-issued with the same messages at {FLASH_RERUN_MAX_TOKENS} tokens (tag {RERUN_TAG}): {ft['n_rerun_rows']} rows, pass rate {ft['rerun_rows_pass_rate']:.3f} among the non-censored, mean completion {ft['rerun_completion_tokens_mean']:.0f} tokens, {ft['rerun_tokens']:,} tokens = {ft['rerun_usd_proxy']:.3f} $; "
             f"{ft['n_censored']} still truncated at {FLASH_RERUN_MAX_TOKENS} -> CENSORED (excluded from p estimates); {ft['chains_replayed']} chains replayed so that later attempts condition on the re-run's real verdict "
             f"({ft['n_regenerated_downstream_rows']} downstream attempts regenerated live, {ft['regenerated_downstream_tokens']:,} tokens); {ft['n_still_truncated_not_rerun']} truncated rows remain un-re-run.\n")
    ot = res["other_truncation"]
    L.append(f"Other models' truncations: {ot['n']} attempts hit the 2,048-token completion cap with content present (code cut mid-way; {ot['n_empty']} empty), by model {ot['by_model']}, by attempt {ot['by_branch_k']}; "
             f"{ot['n_pass']} of them nevertheless pass; the rest fail as {ot['failure_kinds']}. They are kept as ordinary failed attempts (the cap is part of the attempt's specification and no re-run was requested for them).\n")
    # hypothesis violations
    L.append("## Hypothesis violations measured\n")
    hv = []
    for m in models:
        d = res["per_model"][m]["all"]
        t = d.get("C1_fresh_minus_marginal") or [float("nan")] * 3
        neg = [-t[0], -t[2], -t[1]]  # p_1 - p_2^fresh|fail with the CI ends swapped
        fi = d.get("fresh_identical_code") or {}
        hv.append(f"- {m}: error correlation across attempts = p_1 − p_2^fresh|fail = {f3(neg)} (0 under independence); "
                  f"attempt-1 failure kinds {d.get('failure_kinds_1')}; truncated (finish_reason=length) attempt-1 outputs: {d.get('finish_length_1')}; "
                  f"fresh retries with code byte-identical to attempt 1: {fi.get('share', float('nan')):.2f} of {fi.get('n', 0)} (pass rate when identical {fi.get('pass_when_identical', float('nan')):.2f}, when different {fi.get('pass_when_different', float('nan')):.2f}).")
    L.extend(hv)
    L.append("- v_h = 1 by construction (the hidden-test verifier is the truth), so W_λ = λb + 1 − v_b exactly; b and v_b are miss-conditional means on each state set (they differ from the full-sample means: see the b / v_b columns above).")
    L.append("- The recorded Opus outcome is used as the terminal path; its scores on the failure sets are far below its marginal (v_b column), i.e. the failure sets are hard tasks (positive error correlation across models).\n")
    # audit
    L.append("## Audit (independent recomputation with plain Python from attempts.jsonl / the csv)\n")
    L.append(audit_text(res, A, F))
    L.append("")
    L.append("## Deviations and limitations\n")
    for d in res["deviations"]:
        L.append(f"- {d}")
    L.append("- Small per-source failure sets (humaneval, he, hex) make most per-source cells CI-undecided; pooled cells carry the evidence. The split-half OOS test halves them again.")
    L.append("- Sandbox verification time is included in a_k (seconds basis); the visible-verifier time is not (it would replace it in the visible chain).")
    prt = res["spend"].get("live_pass_runtimes_s") or {}
    L.append(f"- Wall-clock: main live pass {(prt.get('run_main_pass.log') or float('nan')) / 60:.0f} min (1,415 chains; bounded by the kimi-k2.5 cap of 2 concurrent calls shared by all experiments, ~30 s per kimi call under load), "
             f"deepseek-flash re-run pass {(prt.get('run_rerun_pass.log') or float('nan')) / 60:.0f} min; first-to-last attempt timestamp span {res['spend'].get('live_wallclock_min', float('nan')):.0f} min. "
             f"The main pass alone exceeds the 2-hour guidance for a from-scratch run; a re-run of run.py reads every call from the packy cache and finishes in seconds with zero live spend (this README was written by such a run: runtime {res['runtime_s']} s).")
    L.append("- The recorded Opus latency/cost come from the benchmark's run (another day, another relay); live latencies are relay latencies under our concurrency.")
    L.append("- Files: `attempts.jsonl` (append-only raw log of every live attempt with tokens, latency, cost, verdicts and feedback-bearing texts; complete runs are delimited by `job_complete`), `attempts_view_a_6144.jsonl` (HEADLINE analysed rows = first complete run per chain, explicit `budget` field), `attempts_view_b_rerun16k.jsonl` (the replayed chains' last run), `attempts_flash_truncated.jsonl` (the original 6,144-token deepseek-flash attempts that returned no content), `recorded.jsonl` (attempt-0 verification), `partial_progress.json`, `results.json`, `figures/`, `run_main_pass.log` / `run_rerun_pass.log` (console logs of the two live passes), `probe.py` (the 8-call pre-flight probe of the four models, tag X7b, included in the spend). `integrator_check.py` was placed here by the integrator; note that it scans every row of `attempts.jsonl` (both passes), whereas the analysed set is view (a).")
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    # summary
    S = []
    S.append(f"# X7-b / X10-live summary for the paper (LIVE; seed {SEED}; B = {B}; paired 95% CIs; n = {res['n_tasks']} EXEC-CLEAN id_test tasks; $ basis unless stated)\n")
    for c in res["claims"]:
        S.append(f"- {c['id']} [{c['kind']}] {c['verdict']}: {c['result'][:420]}")
    for m in models:
        d = res["per_model"][m]["all"]
        S.append(f"- {m}: p_1 {f3(d['p1'], 2)}, fresh p_2|fail {f3(d.get('p_f2'), 2)}, retry+feedback p_2|fail {f3(d.get('p_r2'), 2)}, switch p_2|fail {f3(d.get('p_s2'), 2) if NEXT[m] else 'Opus ' + f3(d.get('opus_pass_on_fail1', [np.nan]*3), 2)}; "
                 f"m$ per resolved: direct Opus {fm(res['policies'][m]['usd']['direct Opus']['cost_per_resolved'])}, attempt1->Opus {fm(res['policies'][m]['usd']['attempt 1 -> Opus']['cost_per_resolved'])}, retry K=3 {fm(res['policies'][m]['usd'].get('retry K=3 -> Opus', {}).get('cost_per_resolved', [np.nan]*3))}"
                 + (f", switch {fm(res['policies'][m]['usd'].get('switch K=3 -> Opus', res['policies'][m]['usd'].get('switch K=2 -> Opus', {})).get('cost_per_resolved', [np.nan]*3))}" if NEXT[m] else ""))
    sa = res["sign_agreement"]["oos_opt"]["all"]
    S.append(f"- Rule sign agreement (OOS split-half, all cells): {sa['rule']['n_agree']}/{sa['rule']['n_decided']} vs marginal-rate {sa['marginal']['n_agree']}/{sa['marginal']['n_decided']}; in-sample {res['sign_agreement']['insample_opt']['all']['rule']['n_agree']}/{res['sign_agreement']['insample_opt']['all']['rule']['n_decided']}.")
    fv = res.get("flash_views") or {}
    va, vb, cf = fv.get("view_a") or {}, fv.get("view_b"), fv.get("counterfactual_p1_at_16k")
    S.append(f"- deepseek-flash budgets: all lines above are VIEW (a) = 6,144-token budget with truncations as paid failures (p_1 {f3(va.get('p1_truncations_as_failures'), 2)} over {va.get('n_roots')} roots, {va.get('n_truncated_roots')} truncated; non-truncated roots {f3(va.get('p1_non_truncated_roots'), 2)}).")
    if vb and cf:
        cb = vb["cost_by_row_basis"]
        ra_, rn_, all16, nc16 = cb["root_reruns_16k_all"], cb["root_reruns_16k_non_censored"], cb["all_16k_rows_all_branches"], cb["all_16k_rows_all_branches_non_censored"]
        S.append(f"- VIEW (b), 16,384 re-run of the truncated attempts ({vb['n_chains']} chains, {vb['n_root_rerun']} root re-runs, {vb['n_root_censored']} censored): p_1 {f3(vb['p1_16k_excluding_censored'], 2)} excl. censored / {f3(vb['p1_16k_censored_as_failures'], 2)} censored as failures; "
                 f"root attempt cost (means, root vs root): 16k root re-runs incl. censored (n={ra_['n']}) {ra_['mean_completion_tokens']:.0f} completion tokens / {1000 * ra_['mean_usd_proxy']:.2f} m$, non-censored (n={rn_['n']}) {rn_['mean_completion_tokens']:.0f} / {1000 * rn_['mean_usd_proxy']:.2f} m$, "
                 f"vs {va.get('root_cost', {}).get('mean_completion_tokens', float('nan')):.0f} / {1000 * va.get('root_cost', {}).get('mean_usd_proxy', float('nan')):.2f} m$ for the {va.get('root_cost', {}).get('n')} non-truncated 6,144-budget roots "
                 f"(all 16k rows of all branches: n={all16['n']} {all16['mean_completion_tokens']:.0f} / {1000 * all16['mean_usd_proxy']:.2f} m$; non-censored n={nc16['n']} {nc16['mean_completion_tokens']:.0f} / {1000 * nc16['mean_usd_proxy']:.2f} m$); "
                 f"COUNTERFACTUAL population p_1 at 16k {f3(cf['censored_as_failures'], 2)} (censored = failures) / {f3(cf['excluding_censored'], 2)} (excl.), over {vb['n_root_rerun']} root re-runs -- orientation only.")
    prt = res["spend"].get("live_pass_runtimes_s") or {}
    S.append(f"- Spend: {res['spend']['ledger_units_tag_total']:.0f} ledger units (X7b {res['spend']['ledger_units_by_tag'].get('X7b', 0):.0f} + X7b:flash_rerun {res['spend']['ledger_units_by_tag'].get('X7b:flash_rerun', 0):.0f}; ~4.3M live tokens); live passes {(prt.get('run_main_pass.log') or 0) / 60:.0f} + {(prt.get('run_rerun_pass.log') or 0) / 60:.0f} min; cached re-analysis {res['runtime_s']} s.")
    with open(os.path.join(OUT, "summary_for_paper.md"), "w") as fh:
        fh.write("\n".join(S[:20]) + "\n")


def audit_text(res: Dict[str, Any], A: pd.DataFrame, F: pd.DataFrame) -> str:
    """Three headline numbers recomputed with plain Python loops (no numpy/pandas aggregation)."""
    out = []
    keep = set(F.task_id)
    # view (a) = first complete run of every chain in attempts.jsonl (plain scan of the raw file, same rule as the analysis)
    by_job: Dict[str, List[Dict[str, Any]]] = {}
    for l in open(ATTEMPTS_PATH) if os.path.exists(ATTEMPTS_PATH) else []:
        if l.strip():
            r = json.loads(l)
            by_job.setdefault(r["job_id"], []).append(r)
    all_rows: List[Dict[str, Any]] = []
    for jid, rs in by_job.items():
        first: List[Dict[str, Any]] = []
        for r in rs:
            first.append(r)
            if r.get("job_complete"):
                break
        if first and first[-1].get("job_complete") and first[0]["task_id"] in keep:
            all_rows.extend(first)
    rows = [r for r in all_rows if not r.get("api_error")]
    # (1) p_2 retry|fail pooled for the first model (tasks with any API-error row for this root are excluded, as in results.json)
    m = res["models"][0]
    bad = {r["task_id"] for r in all_rows if r["root_model"] == m and r.get("api_error")}
    fail1 = {r["task_id"] for r in rows if r["root_model"] == m and r["branch"] == "root" and r["k"] == 1 and not r["truth_pass"] and r["task_id"] not in bad}
    r2 = [r for r in rows if r["root_model"] == m and r["branch"] == "retry" and r["k"] == 2 and r["task_id"] in fail1]
    p_r2 = sum(1 for r in r2 if r["truth_pass"]) / len(r2) if r2 else float("nan")
    ref = res["per_model"][m]["all"].get("p_r2", [float("nan")] * 3)[0]

    def same(a: float, b: float) -> str:
        return "match" if (abs(a - b) < 1e-9 or (math.isnan(a) and math.isnan(b))) else "MISMATCH"

    out.append(f"1. p_2 retry+feedback | fail, {m}: plain count {p_r2:.4f} over {len(r2)} tasks vs results.json {ref:.4f} -> {same(p_r2, ref)}")
    # (2) eta_B in dollars from the frame csv columns
    sv = 0.0
    sd = 0.0
    for v, c in zip(F.opus_score.tolist(), F.opus_usd.tolist()):
        sv += v
        sd += c
    eb = sv / sd
    out.append(f"2. η_B per $: plain sums {eb:.4f} vs results.json {res['eta_B']['usd'][0]:.4f} -> {same(eb, res['eta_B']['usd'][0])}")
    # (3) visible-verifier false accepts at theta=1 for the first model
    rm_ = [r for r in rows if r["root_model"] == m]
    e = sum(1 for r in rm_ if r["vis_syntax_ok"] and r["vis_score"] >= 1.0 and not r["truth_pass"]) / len(rm_) if rm_ else float("nan")
    ref = res["visible"][m]["1.0"]["e_false_accept"]
    out.append(f"3. e (θ=1, all attempts of {m}): plain count {e:.4f} vs results.json {ref:.4f} -> {same(e, ref)}")
    # (4) the de-duplicated headline for deepseek-flash: p_1 in view (a) = first complete run per chain, truncations counted as failures
    fr = [r for r in rows if r["root_model"] == "deepseek-flash" and r["branch"] == "root" and r["k"] == 1]
    p1 = sum(1 for r in fr if r["truth_pass"]) / len(fr) if fr else float("nan")
    ref = (res.get("per_model", {}).get("deepseek-flash", {}).get("all", {}).get("p1") or [float("nan")])[0]
    n_tr = sum(1 for r in fr if r.get("finish_reason") == "length" and not (r.get("text") or "").strip())
    out.append(f"4. deepseek-flash p_1 in view (a) (first complete run per chain in attempts.jsonl, {n_tr} truncated roots counted as failures): plain count {p1:.4f} over {len(fr)} roots vs results.json {ref:.4f} -> {same(p1, ref)}")
    return "\n".join(out)


if __name__ == "__main__":
    main()

