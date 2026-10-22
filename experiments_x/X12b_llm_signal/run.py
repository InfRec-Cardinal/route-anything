#!/usr/bin/env python3
"""
X12b_llm_signal -- "Route Anything" experiment X12, LIVE part (X12-b):
the value of a PRE-EXECUTION routing signal produced by an LLM that reads the task prompt and forecasts, as JSON,
the probability that a named cheap primary (Qwen3-Max; kimi-k2.5) solves the task, a difficulty label
{easy, medium, hard} and the expected number of lines of code.  Signal models: deepseek-v4-pro (all sampled tasks,
thinking disabled) and qwen3-max (200-task nested subsample).  Truth = the released CodeRouterBench score
(sufficiency H = 1{score >= theta}); the local sandbox is never used here.

Theory objects (labels as in main_body.tex, appendix_core.tex):
  eq:gate-costs          C_cascade(s) = a + (1-s) b,  C_direct = b,  C_info = k + s a + (1-s) b
  prop:voi (eq:band)     exact information strictly cheapest iff k/(b-a) < s < 1-k/a; nonempty iff k < a(b-a)/b
  eq:imperfect-voi       net value of an imperfect experiment Z:  f(s) - E f(s_Z) - k,   f(t) = min{a+(1-t)b, b}
  app:imperfect-voi      f(s) - E f(s_Z) <= f(s) - g(s) = min{s(b-a), a(1-s)}; pays only if k < a; a learned score used
                         through "s_hat >= a/b" can lose against no signal; any monotone transform used with the
                         transformed threshold attains the Blackwell value (re-thresholding)
  cor:info-vs-proposer   C_info - C_dec = b Delta p - a(1-s) at equal cost k

Definitions copied from ../X12a_voi_released_signals/run.py (which copied them from X3):
  f(s) = min{a+(1-s)b, b}; g(s) = s a + (1-s) b; sufficiency H = 1{score_primary >= theta};
  realized cascade cost = cost_P + (1-H) cost_B; direct = cost_B; perfect information = H cost_P + (1-H) cost_B;
  uninformed choice from probing: cascade iff s > a/b; gross VOI = realized mean cost of the uninformed choice minus
  realized mean cost of the signal usage (id_test); net VOI = gross - k; capture ratio = gross VOI / (f(s)-g(s));
  decided = realized 95% CI excludes 0; degenerate = bound <= 1e-9.
  Blackwell-optimal usage: attempt iff s_Z > a/b with s_Z the calibrated posterior (isotonic regression fitted on probing,
  pooled over states and anchored per state so that E[s_Z] = s on probing -- eq:imperfect-voi's premise);
  native usage of a probability signal: attempt iff raw p >= a/b (the paper's "s_hat >= a/b" rule).

Claims: C1 bound never exceeded (IDENTITY-like inequality on expectations, realized OUT-OF-SAMPLE);
        C2 LLM capture vs released routers (same tasks; X12-a values quoted) and vs a TF-IDF LR router trained on the
           probing prompts (OUT-OF-SAMPLE);
        C3 within state, capture follows decision usefulness (AUROC / switch share) more than ECE (OUT-OF-SAMPLE);
        C4 re-thresholding recovers value lost to miscalibration (OUT-OF-SAMPLE; shrunk variant SEMI-SYNTHETIC);
        C5 cor:info-vs-proposer sign agreement (OUT-OF-SAMPLE);
        C6 k < a per state and primary (measurement; LIVE cost of the signal).
API only through common/packy.py (tag X12b; deepseek-v4-pro workers 4, qwen3-max workers 4); every call cached; partial
signal tables persisted every 100 tasks.  numpy / pandas / scipy / scikit-learn / matplotlib.  Seed 20260917.  Dollars are
the main basis (CodeRouterBench csv cost_usd; signal priced at list prices, see PRICES); tokens and seconds for k < a.
"""
import os, sys, json, time, math, re, csv, collections, argparse
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.dirname(HERE)
sys.path.insert(0, EX)
from common import packy, sources  # noqa: E402  (import only; never modified)

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", type=int, default=0, help="development only: run N tasks per split through the signal models (outputs go to _dev/)")
ap.add_argument("--analyze", action="store_true", help="development only: with --smoke, continue into the analysis")
ap.add_argument("--collect", action="store_true", help="collect new forecasts; otherwise replay the saved per-task calls")
ARGS = ap.parse_args()
OUT = HERE if not ARGS.smoke else os.path.join(HERE, "_dev"); os.makedirs(OUT, exist_ok=True)

T0 = time.time()
SEED = 20260917
NBOOT = 1000                     # paired task resamples, 95% percentile CIs
NPERM = 20                       # permutations of the signal within cell (SEMI-SYNTHETIC uninformative signal)
NSPLIT = 10                      # half-splits of id_test for the cross-fitted posterior of the BINARY released routers (X12-a scheme)
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
ACR = os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")
ACR_RELEASE = os.environ.get("ACROUTER_RELEASE_DECISIONS", "data/agent-as-a-router/outputs/current/id_decisions.jsonl")
X12A = os.path.join(EX, "X12a_voi_released_signals")

TAG = "X12b"
SIG_MODEL = "deepseek-v4-pro"; SIG_MODEL2 = "qwen3-max"; WORKERS = 4
THINKING_OFF = {"thinking": {"type": "disabled"}}      # deepseek-v4-pro reasons by default (see infra_smoke report (c))
MAX_OUT_TOKENS = 200; MAX_PROMPT_CHARS = 5000; BATCH = 100
STRONG = "claude-opus-4-6"
PRIMARIES = ["Qwen3-Max", "kimi-k2.5"]                 # the two primaries the LLM is asked about
ALL_PRIMARIES = ["Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "MiniMax-M2.7", "glm-5", "gpt-5.4", "claude-sonnet-4-6"]
THETAS = [1.0, 0.5]; THETA_MAIN = 1.0
SOURCES = ["humaneval", "he", "mbpp", "bcb", "hex", "lcb", "ds1000"]
N_PER_SPLIT = 600                                      # main sample per split (deepseek-v4-pro signal on all 1,200 tasks)
N_QWEN_PER_SPLIT = int(os.environ.get("X12B_QWEN_PER_SPLIT", "200"))  # nested subsample for the qwen3-max second signal (relay latency ~33 s/call at cap 4 -> 400 tasks ~55 min)
ROUTERS = ["finetuned_router_qwen35_08b", "finetuned_router_qwen35_2b", "finetuned_router_qwen35_9b_v3", "finetuned_router_qwen35_27b_v3",
           "logreg", "tfidf_mlp", "routellm_mf", "routellm_sw", "dimension_best"]
FINETUNED = set(ROUTERS[:4]); SIGNALS_REL = ROUTERS + ["acrouter_release"]; ZDEFS = ["this_primary", "any_non_opus"]
# $/token list prices for the signal call.  deepseek-v4-pro: DeepSeek's official time-of-day prices in force since 2026-08-16
# (peak $1.32/M input, $3.96/M output; off-peak half; third-party trackers, see README); qwen3-max: models.json list price
# ($0.34/M in, $1.38/M out); self-hosted: the protocol's $0.054/M router price (X7 R1, X12-a k_tok).
PRICES = {"deepseek-v4-pro": {"peak": (1.32e-6, 3.96e-6), "offpeak": (0.66e-6, 1.98e-6)},
          "qwen3-max": {"list": (0.34e-6, 1.38e-6), "list_half": (0.17e-6, 0.69e-6)}, "self_hosted_per_token": 0.054e-6}
K_MAIN = "peak"                                          # main dollar price of the deepseek signal (conservative upper price)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'text.usetex': False,
                     'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65, 'lines.linewidth': 1.4})
BLUE, ORANGE, PURPLE, GREEN, GREY = '#23639b', '#ba5934', '#6a4c93', '#3b7a57', '#808080'

rng = np.random.default_rng(SEED)
NAN = float("nan")

# ------------------------------------------------------------------------------------------------ helpers (copied from X12-a)
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (NAN, NAN)
    lo, hi = np.percentile(x, [2.5, 97.5]); return float(lo), float(hi)

def decided(lo, hi):
    return bool(np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi))

def bm(v, idx):
    """bootstrap means of a per-task vector under a (B, n) index matrix"""
    return v[idx].mean(1)

def fdiv(n, d):
    n = np.asarray(n, float); d = np.asarray(d, float)
    return np.where(d != 0, n / np.where(d != 0, d, 1.0), NAN)

def auroc_binary(Z, H):
    n1 = int(H.sum()); n0 = len(H) - n1
    if n1 == 0 or n0 == 0: return NAN
    return float((1 + (Z & H).sum() / n1 - (Z & ~H).sum() / n0) / 2)

def ece_binary(Z, H):
    return float((Z != H).mean())

def auroc_cont(z, H):
    n1 = int(H.sum()); n0 = len(H) - n1
    if n1 == 0 or n0 == 0: return NAN
    r = stats.rankdata(z); return float((r[H].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def auroc_cont_boot(z, Hf, idx):
    zb = z[idx]; hb = Hf[idx]; r = stats.rankdata(zb, axis=1)
    n1 = hb.sum(1); n0 = idx.shape[1] - n1
    out = fdiv((r * hb).sum(1) - n1 * (n1 + 1) / 2, n1 * n0); out[(n1 == 0) | (n0 == 0)] = NAN; return out

def ece_bins(z, nb=10):
    edges = np.unique(np.quantile(z, np.linspace(0, 1, nb + 1)[1:-1])); return np.searchsorted(edges, z, side="right"), len(edges) + 1

def ece_cont(z, H, nb=10):
    b, nbin = ece_bins(z, nb); tot = 0.0
    for j in range(nbin):
        m = b == j
        if m.any(): tot += abs(H[m].sum() - z[m].sum())
    return float(tot / len(z))

def ece_cont_boot(z, Hf, idx, nb=10):
    b, nbin = ece_bins(z, nb); bb = b[idx]; zb = z[idx]; hb = Hf[idx]; tot = np.zeros(idx.shape[0])
    for j in range(nbin):
        m = bb == j; tot += np.abs((hb * m).sum(1) - (zb * m).sum(1))
    return tot / idx.shape[1]

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return NAN, int(m.sum())
    return float(stats.spearmanr(x[m], y[m]).statistic), int(m.sum())

def half_splits(labels, n_splits, rng_):
    labels = np.asarray(labels); out = []
    for _ in range(n_splits):
        A = np.zeros(len(labels), bool)
        for lab in np.unique(labels):
            pos = rng_.permutation(np.flatnonzero(labels == lab)); A[pos[:len(pos) // 2]] = True
        out.append(A)
    return out

def crossfit_binary(H, Z, splits, ab, s_prior):
    """X12-a: Blackwell use of a BINARY signal with cross-fitted, prior-anchored posteriors (attempt iff s_Z > a/b)."""
    attfrac = np.zeros(len(H)); n_anti = 0
    for A in splits:
        for cal, tgt in ((A, ~A), (~A, A)):
            c1 = cal & Z; c0 = cal & ~Z; base = H[cal].mean() if cal.any() else s_prior
            l1 = (H[c1].mean() - base) if c1.any() else 0.0
            l0 = (H[c0].mean() - base) if c0.any() else 0.0
            if l1 < l0: n_anti += 1
            post = np.clip(s_prior + np.where(Z[tgt], l1, l0), 0.0, 1.0)
            attfrac[tgt] += post > ab
    return attfrac / len(splits), n_anti

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o

def allocate(caps, total):
    """water-filling: equal shares per key, capped by availability, remainder redistributed (deterministic)"""
    alloc = {k: 0 for k in caps}; remaining = int(total); active = {k for k in caps if caps[k] > 0}
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            for k in sorted(active, key=lambda q: (-(caps[q] - alloc[q]), q)):
                if remaining == 0: break
                if alloc[k] < caps[k]: alloc[k] += 1; remaining -= 1
            break
        for k in sorted(active):
            take = min(share, caps[k] - alloc[k]); alloc[k] += take; remaining -= take
            if alloc[k] >= caps[k]: active.discard(k)
    return alloc

def largest_remainder(weights, total):
    w = np.asarray(weights, float); q = w / w.sum() * total; fl = np.floor(q).astype(int); rem = total - fl.sum()
    order = np.argsort(-(q - fl), kind="stable")
    for i in order[:rem]: fl[i] += 1
    return fl

results = {"experiment_id": "X12b_llm_signal", "seed": SEED, "nboot": NBOOT, "n_permutations": NPERM, "n_half_splits": NSPLIT, "tag": TAG,
           "cost_basis": "dollars (CodeRouterBench csv cost_usd for a, b; signal call priced at list prices, PRICES); tokens and seconds for the k < a condition",
           "prices": clean(PRICES), "k_main_price": K_MAIN}

# ================================================================================================ A. data and sample
print("[data] loading CodeRouterBench", flush=True)
df = pd.read_csv(f"{ACR}/coderouterbench/id_results_long.csv")
S = df.pivot(index="task_id", columns="model", values="score"); Cc = df.pivot(index="task_id", columns="model", values="cost_usd")
TT = df.pivot(index="task_id", columns="model", values="total_tokens").astype(float)
LT = df.pivot(index="task_id", columns="model", values="latency_ms").astype(float) / 1000.0
INTOK = df.pivot(index="task_id", columns="model", values="input_tokens").mean(axis=1)
meta = df.drop_duplicates("task_id").set_index("task_id")[["split", "dimension"]].loc[S.index]
meta["source"] = meta.index.str.split("_").str[1]; meta["cell"] = meta.dimension + "|" + meta.source; meta["all"] = "ALL"
assert S.shape == (9999, 8) and not S.isna().any().any()
allzero = (S.max(axis=1) <= 0)
pop = meta[meta.source.isin(SOURCES)].copy(); pop["allzero"] = allzero.loc[pop.index]
results["population"] = dict(sources=SOURCES, n_tasks_in_sources=int(len(pop)), n_allzero_excluded=int(pop.allzero.sum()),
                             allzero_by_source={s: int(v) for s, v in pop[pop.allzero].groupby("source").size().items()},
                             n_population=int((~pop.allzero).sum()),
                             per_source_split={f"{s}|{sp}": int(n) for (s, sp), n in pop[~pop.allzero].groupby(["source", "split"]).size().items()},
                             rule=("tasks on which all 8 released scores are 0 are excluded in EVERY source (the addendum's bcb rule applied uniformly): "
                                   "the smoke report shows the sandbox passes 36-65 % of the recorded responses on such tasks in bcb / ds1000 / lcb "
                                   "(released-harness failures) and the mbpp dim1 zeros are a hidden-function-name artefact no pre-execution signal can predict; "
                                   "dim9 mbpp is entirely all-zero. A sensitivity with the all-zero tasks kept is reported for s and the bound (offline)."))
pop = pop[~pop.allzero]

def sample_split(split, n_total):
    sub = pop[pop.split == split]
    caps = {s: int((sub.source == s).sum()) for s in SOURCES}
    alloc = allocate(caps, n_total); out = []
    for s in SOURCES:
        g = sub[sub.source == s]; cells = sorted(g.cell.unique()); sizes = [int((g.cell == c).sum()) for c in cells]
        per_cell = largest_remainder(sizes, alloc[s])
        for c, n in zip(cells, per_cell):
            ids = np.array(sorted(g.index[g.cell == c])); n = min(int(n), len(ids))
            if n > 0: out.extend(rng.choice(ids, n, replace=False).tolist())
    return sorted(out), alloc

prob_ids, alloc_p = sample_split("probing", N_PER_SPLIT); test_ids, alloc_t = sample_split("id_test", N_PER_SPLIT)
smp = meta.loc[prob_ids + test_ids].copy(); smp["in_sub"] = False
for split, ids in [("probing", prob_ids), ("id_test", test_ids)]:
    caps = {s: int((meta.loc[ids].source == s).sum()) for s in SOURCES}; al = allocate(caps, N_QWEN_PER_SPLIT)
    for s in SOURCES:
        pool_ = np.array(sorted([t for t in ids if meta.loc[t, "source"] == s]))
        if al[s] > 0: smp.loc[rng.choice(pool_, al[s], replace=False), "in_sub"] = True
results["sample"] = dict(n_probing=len(prob_ids), n_id_test=len(test_ids), alloc_probing=alloc_p, alloc_id_test=alloc_t, stratified_by="source (water-filling), then cell (largest remainder)",
                         n_sub_probing=int(smp[(smp.split == "probing")].in_sub.sum()), n_sub_id_test=int(smp[(smp.split == "id_test")].in_sub.sum()),
                         second_signal_sample=f"qwen3-max on a nested subsample of {N_QWEN_PER_SPLIT} + {N_QWEN_PER_SPLIT} tasks stratified by source (the relay's qwen3-max latency, bimodal 1.5 s / 62 s, mean 33 s at the 4-worker cap, made the full sample infeasible inside the 2 h script limit)",
                         cells_probing={c: int(n) for c, n in smp[smp.split == "probing"].groupby("cell").size().items()},
                         cells_id_test={c: int(n) for c, n in smp[smp.split == "id_test"].groupby("cell").size().items()})
if ARGS.smoke:   # development: a spread of ids across sources (every k-th sorted id)
    prob_ids = prob_ids[::max(1, len(prob_ids) // ARGS.smoke)][:ARGS.smoke]; test_ids = test_ids[::max(1, len(test_ids) // ARGS.smoke)][:ARGS.smoke]; smp = smp.loc[prob_ids + test_ids]
print(f"[sample] probing {alloc_p} | id_test {alloc_t}", flush=True)

# --- prompts
print("[prompts] reconstructing", flush=True); t_p = time.time()
PROMPTS = {}; plen = {}; langs = {}
for tid in smp.index:
    t = sources.load_task(tid)
    p = (t.get("prompt") or "")
    assert t["found"] and p.strip(), tid
    PROMPTS[tid] = p; plen[tid] = len(p); langs[tid] = t.get("language")
smp["prompt_chars"] = pd.Series(plen); smp["language"] = pd.Series(langs); smp["truncated"] = smp.prompt_chars > MAX_PROMPT_CHARS
print(f"[prompts] {len(PROMPTS)} prompts in {time.time() - t_p:.0f}s; truncated {int(smp.truncated.sum())}", flush=True)

# ================================================================================================ B. the LLM signal (LIVE)
SYSTEM = ("You are a senior software engineer who forecasts how likely specific AI coding models are to solve programming tasks. "
          "You never solve the task yourself; you only forecast. You answer with one JSON object and nothing else.")
USER_TMPL = """Forecast, for the programming task below, how likely each of two AI models is to produce a fully correct answer in a single attempt (one response, no tools, no code execution, no retries). "Fully correct" means the response passes every hidden test of the task's official benchmark harness; partially working code, a wrong function name or signature, or a wrong output format counts as incorrect.

Models to forecast:
(A) Qwen3-Max -- Alibaba's flagship general-purpose model (Qwen3 generation, non-reasoning mode).
(B) Kimi K2.5 -- Moonshot AI's open-weight mixture-of-experts model (K2 generation).

Consider the task's algorithmic difficulty, the precision the hidden tests demand (edge cases, exact output formats, exact names and signatures), how complete and unambiguous the specification is, and the length of the required code. Give honest, calibrated probabilities using the whole 0-1 range.

Return ONLY this JSON object (no markdown, no code fence, no commentary):
{{"p_qwen3_max": <probability 0-1 that Qwen3-Max is fully correct>, "p_kimi_k2_5": <probability 0-1 that Kimi K2.5 is fully correct>, "difficulty": "easy" | "medium" | "hard", "expected_lines": <integer: lines of code in a typical correct solution>}}

### Task (verbatim; do not solve it)
{task}"""

def truncate_mid(s, n):
    if len(s) <= n: return s
    h = n // 2; return s[:h] + f"\n... [{len(s) - n} characters omitted] ...\n" + s[-h:]

def messages_for(tid):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_TMPL.format(task=truncate_mid(PROMPTS[tid], MAX_PROMPT_CHARS))}]

_LAB = {"easy": "easy", "medium": "medium", "med": "medium", "moderate": "medium", "hard": "hard", "difficult": "hard", "very hard": "hard"}

def _num(v):
    try:
        if isinstance(v, str): v = v.strip().rstrip("%")
        x = float(v)
    except (TypeError, ValueError):
        return NAN
    if 1.0 < x <= 100.0: x /= 100.0
    return float(min(max(x, 0.0), 1.0))

def parse_signal(text):
    """robust JSON extraction: whole text -> first {...} containing p_qwen3_max -> per-field regexes"""
    out = dict(p_qwen=NAN, p_kimi=NAN, label=None, lines=NAN, parse_ok=False)
    if not text: return out
    cands = []
    t = text.strip()
    if t.startswith("```"): t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t, flags=re.S)
    cands.append(t)
    m = re.search(r"\{[^{}]*p_qwen3_max[^{}]*\}", text, re.S)
    if m: cands.append(m.group(0))
    m2 = re.search(r"\{.*\}", text, re.S)
    if m2: cands.append(m2.group(0))
    js = None
    for c in cands:
        try:
            v = json.loads(c)
            if isinstance(v, dict): js = v; break
        except ValueError:
            continue
    if js is not None:
        out["p_qwen"] = _num(js.get("p_qwen3_max")); out["p_kimi"] = _num(js.get("p_kimi_k2_5"))
        lab = str(js.get("difficulty", "")).strip().lower(); out["label"] = _LAB.get(lab)
        try: out["lines"] = float(int(round(float(js.get("expected_lines")))))
        except (TypeError, ValueError): out["lines"] = NAN
    else:
        mq = re.search(r"p_qwen3_max\"?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", text); mk = re.search(r"p_kimi_k2_5\"?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", text)
        md = re.search(r"difficulty\"?\s*[:=]\s*\"?\s*(easy|medium|hard)", text, re.I); ml = re.search(r"expected_lines\"?\s*[:=]\s*([0-9]+)", text)
        if mq: out["p_qwen"] = _num(mq.group(1) + mq.group(2))
        if mk: out["p_kimi"] = _num(mk.group(1) + mk.group(2))
        if md: out["label"] = md.group(1).lower()
        if ml: out["lines"] = float(ml.group(1))
    out["parse_ok"] = bool(np.isfinite(out["p_qwen"]) and np.isfinite(out["p_kimi"]) and out["label"] is not None)
    return out

client = packy.PackyClient(quiet=True, offline=not ARGS.collect)
RATE = {SIG_MODEL: client.units_per_token(SIG_MODEL), SIG_MODEL2: client.units_per_token(SIG_MODEL2)}
LEDGER_BEFORE = client.ledger_units()
try: RELAY_BEFORE = client.relay_usage()
except Exception: RELAY_BEFORE = NAN

def run_signal_calls(model, tids, extra, partial_path):
    if not ARGS.collect:
        with open(partial_path) as recorded:
            rows = {record["task_id"]: record for record in map(json.loads, recorded) if record}
        return {task_id: rows[task_id] for task_id in tids}
    rows = {}; t_start = time.time()
    for i in range(0, len(tids), BATCH):
        batch = tids[i:i + BATCH]
        res = client.map(model, [messages_for(t) for t in batch], workers=WORKERS, tag=TAG, max_tokens=MAX_OUT_TOKENS, temperature=0, extra=extra)
        for t, r in zip(batch, res): rows[t] = r
        with open(partial_path, "w") as fh:
            for t, r in rows.items():
                fh.write(json.dumps(dict(task_id=t, text=r.get("text"), finish_reason=r.get("finish_reason"), usage=r.get("usage"), latency_s=r.get("latency_s"),
                                         cached=r.get("cached"), error=r.get("error"), **parse_signal(r.get("text"))), ensure_ascii=False) + "\n")
        print(f"[{model}] {min(i + BATCH, len(tids))}/{len(tids)} ({time.time() - t_start:.0f}s; session units {client.usage_summary()['session_units_est']:.1f})", flush=True)
    # one extra pass for transport errors (the client already retried 3x)
    errs = [t for t, r in rows.items() if r.get("finish_reason") == "error"]
    if errs:
        res = client.map(model, [messages_for(t) for t in errs], workers=WORKERS, tag=TAG, max_tokens=MAX_OUT_TOKENS, temperature=0, extra=extra)
        for t, r in zip(errs, res): rows[t] = r
    return rows

SIG_COLS = ["task_id", "signal_model", "text", "finish_reason", "error", "cached", "latency_s", "prompt_tokens", "completion_tokens", "total_tokens", "k_units", "k_self", "p_qwen", "p_kimi", "label", "lines", "parse_ok"]

def tabulate(rows, model):
    price = PRICES[model]; rate = RATE[model]; out = []
    if not rows:
        return pd.DataFrame(columns=SIG_COLS + [f"k_{pn}" for pn in price]).astype({"cached": bool, "parse_ok": bool, "p_qwen": float, "p_kimi": float, "lines": float, "latency_s": float, "total_tokens": float, "prompt_tokens": float, "completion_tokens": float, "k_units": float, "k_self": float, **{f"k_{pn}": float for pn in price}}).set_index("task_id")
    for t, r in rows.items():
        u = r.get("usage") or {}; pi, po = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0); tot = int(u.get("total_tokens") or (pi + po))
        p = parse_signal(r.get("text"))
        d = dict(task_id=t, signal_model=model, text=r.get("text"), finish_reason=r.get("finish_reason"), error=r.get("error"), cached=bool(r.get("cached")), latency_s=r.get("latency_s"),
                 prompt_tokens=pi, completion_tokens=po, total_tokens=tot, k_units=tot * rate, k_self=tot * PRICES["self_hosted_per_token"], **p)
        for pn, (cin, cout) in price.items(): d[f"k_{pn}"] = pi * cin + po * cout
        out.append(d)
    return pd.DataFrame(out).set_index("task_id")

all_ids = list(smp.index); sub_ids = list(smp.index[smp.in_sub])
print(f"[signal] {SIG_MODEL} on {len(all_ids)} tasks (workers {WORKERS}); {SIG_MODEL2} on {len(sub_ids)} (workers {WORKERS}); the two models run concurrently", flush=True)
t_sig = time.time()
from concurrent.futures import ThreadPoolExecutor as _TPE
with _TPE(max_workers=2) as _ex:
    _f_ds = _ex.submit(run_signal_calls, SIG_MODEL, all_ids, THINKING_OFF, os.path.join(OUT, "signal_calls_deepseek_partial.jsonl"))
    _f_qw = _ex.submit(run_signal_calls, SIG_MODEL2, sub_ids, None, os.path.join(OUT, "signal_calls_qwen3max_partial.jsonl"))
    raw_ds = _f_ds.result(); raw_qw = _f_qw.result()
sig_ds = tabulate(raw_ds, SIG_MODEL); sig_qw = tabulate(raw_qw, SIG_MODEL2)
SIG_WALL = time.time() - t_sig
usage = client.usage_summary(); LEDGER_AFTER = client.ledger_units()
try: RELAY_AFTER = client.relay_usage()
except Exception: RELAY_AFTER = NAN
results["api"] = dict(signal_model=SIG_MODEL, signal_model_2=SIG_MODEL2, workers=WORKERS, thinking=dict(deepseek="disabled", qwen3_max="default (no thinking parameter sent)"), max_tokens=MAX_OUT_TOKENS, temperature=0,
                      max_prompt_chars=MAX_PROMPT_CHARS, ledger_units_per_token=RATE, session=clean(usage["session"]), session_units_est=usage["session_units_est"],
                      ledger_units_before=LEDGER_BEFORE, ledger_units_after=LEDGER_AFTER, ledger_units_this_run=LEDGER_AFTER - LEDGER_BEFORE,
                      ledger_note="ledger_units_this_run is the delta of the SHARED ledger (other experiments append concurrently); the experiment's own spend is ledger_units_tag_X12b",
                      ledger_units_tag_X12b=(lambda rows_: dict(total_units=float(sum(r_["units_est"] for r_ in rows_)), n_rows=len(rows_),
                                                              by_model={m_: dict(live_calls=sum(1 for r_ in rows_ if r_["model"] == m_ and not r_["cached"]), cached_calls=sum(1 for r_ in rows_ if r_["model"] == m_ and r_["cached"]),
                                                                                 live_tokens=int(sum(r_["prompt_tokens"] + r_["completion_tokens"] for r_ in rows_ if r_["model"] == m_ and not r_["cached"])),
                                                                                 units=float(sum(r_["units_est"] for r_ in rows_ if r_["model"] == m_))) for m_ in sorted({r_["model"] for r_ in rows_})}))(
                          [json.loads(l_) for l_ in (open(client.ledger_path) if os.path.exists(client.ledger_path) else []) if l_.strip() and json.loads(l_).get("tag") == TAG]),
                      relay_total_usage_before=RELAY_BEFORE, relay_total_usage_after=RELAY_AFTER, relay_delta_units_upper_bound=(RELAY_AFTER - RELAY_BEFORE),
                      wall_seconds_signal_phase=SIG_WALL, concurrency=clean(usage["concurrency"]),
                      per_model={m: dict(n_calls=int(len(d)), n_cached=int(d.cached.sum()), n_live=int((~d.cached).sum()), n_errors=int((d.finish_reason == "error").sum()),
                                         n_parse_ok=int(d.parse_ok.sum()), tokens_prompt=int(d.prompt_tokens.sum()), tokens_completion=int(d.completion_tokens.sum()), tokens_total=int(d.total_tokens.sum()),
                                         mean_tokens_per_call=float(d.total_tokens.mean()), mean_latency_s=float(d.latency_s.mean()), units_at_ledger_rate=float(d.k_units.sum()),
                                         units_at_conservative_0p05=float(d.total_tokens.sum() * 0.05)) for m, d in [(SIG_MODEL, sig_ds), (SIG_MODEL2, sig_qw)]})
print(f"[signal] done in {SIG_WALL:.0f}s; deepseek parse_ok {int(sig_ds.parse_ok.sum())}/{len(sig_ds)}, qwen parse_ok {int(sig_qw.parse_ok.sum())}/{len(sig_qw)}; "
      f"session units {usage['session_units_est']:.1f}; ledger +{LEDGER_AFTER - LEDGER_BEFORE:.1f}", flush=True)
if ARGS.smoke and not ARGS.analyze:
    for t in all_ids[:10]: print(t, "|", (raw_ds[t].get("text") or "")[:300].replace("\n", " "), "|", raw_ds[t].get("usage"), raw_ds[t].get("latency_s"))
    for t in sub_ids[:10]: print("QWEN", t, "|", (raw_qw[t].get("text") or "")[:300].replace("\n", " "), "|", raw_qw[t].get("usage"), raw_qw[t].get("latency_s"))
    print(json.dumps(clean(results["api"]), indent=1)); sys.exit(0)

# --- per-task signal table (imputation for unparsed rows: p = 0.5, label = medium, lines = median; flagged)
def finalize(sig):
    sig = sig.copy(); sig["imputed"] = ~sig.parse_ok
    med_lines = float(np.nanmedian(sig.lines)) if np.isfinite(sig.lines).any() else 20.0
    sig["p_qwen"] = sig.p_qwen.fillna(0.5); sig["p_kimi"] = sig.p_kimi.fillna(0.5); sig["label"] = sig.label.fillna("medium"); sig["lines"] = sig.lines.fillna(med_lines)
    sig["label_code"] = sig.label.map({"hard": 0.0, "medium": 1.0, "easy": 2.0})
    return sig
sig_ds = finalize(sig_ds); sig_qw = finalize(sig_qw)
task_tbl = smp.join(sig_ds[["p_qwen", "p_kimi", "label", "lines", "parse_ok", "imputed", "cached", "latency_s", "prompt_tokens", "completion_tokens", "total_tokens", "k_units", "k_peak", "k_offpeak", "k_self", "finish_reason"]])
task_tbl = task_tbl.join(sig_qw[["p_qwen", "p_kimi", "label", "lines", "parse_ok", "imputed", "cached", "latency_s", "prompt_tokens", "completion_tokens", "total_tokens", "k_units", "k_list", "k_list_half", "k_self", "finish_reason"]].add_prefix("qw_"))
for m in ALL_PRIMARIES + [STRONG]:
    task_tbl[f"score_{m}"] = S.loc[task_tbl.index, m]; task_tbl[f"cost_{m}"] = Cc.loc[task_tbl.index, m]
TASK_TBL_PATH = os.path.join(OUT, "signal_rows.csv"); task_tbl.to_csv(TASK_TBL_PATH)

# ================================================================================================ C. TF-IDF logistic-regression router trained on the probing prompts
print("[tfidf] training", flush=True)
P_IDX = np.array(prob_ids); T_IDX = np.array(test_ids)
txt_p = [PROMPTS[t] for t in P_IDX]; txt_t = [PROMPTS[t] for t in T_IDX]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True, dtype=np.float32); Xp = vec.fit_transform(txt_p); Xt = vec.transform(txt_t)
vec_u = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=20000, sublinear_tf=True, dtype=np.float32); Xp_u = vec_u.fit_transform(txt_p); Xt_u = vec_u.transform(txt_t)
TFIDF = {}; tfidf_doc = {}
for theta in THETAS:
    for m in PRIMARIES:
        y = (S.loc[P_IDX, m].to_numpy() >= theta).astype(int); yt = (S.loc[T_IDX, m].to_numpy() >= theta)
        skf = StratifiedKFold(max(2, min(5, int(np.bincount(y).min()))), shuffle=True, random_state=SEED)
        for v, (XP, XT, C) in {"tfidf_lr": (Xp, Xt, 1.0), "tfidf_lr_uni": (Xp_u, Xt_u, 0.3)}.items():
            oof = np.zeros(len(P_IDX))
            for tr, va in skf.split(XP, y):
                oof[va] = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(XP[tr], y[tr]).predict_proba(XP[va])[:, 1]
            full = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(XP, y); pt = full.predict_proba(XT)[:, 1]
            TFIDF[(theta, m, v)] = (oof, pt)
            tfidf_doc[f"{v}|{m}|theta{theta}"] = dict(n_features=int(XP.shape[1]), C=C, probing_prevalence=float(y.mean()), auroc_oof_probing=auroc_cont(oof, y.astype(bool)),
                                                       auroc_id_test=auroc_cont(pt, yt), ece_id_test=ece_cont(pt, yt))
results["tfidf_router"] = tfidf_doc
for m in PRIMARIES:
    oof, pt = TFIDF[(THETA_MAIN, m, "tfidf_lr")]; col = pd.Series(np.concatenate([oof, pt]), index=list(P_IDX) + list(T_IDX))
    task_tbl[f"tfidf_p_{'qwen' if m == 'Qwen3-Max' else 'kimi'}_theta1"] = col.reindex(task_tbl.index)
task_tbl.to_csv(TASK_TBL_PATH)
print("[tfidf] " + "; ".join(f"{k}: AUROC {v['auroc_id_test']:.3f}" for k, v in tfidf_doc.items() if k.endswith("theta1.0")), flush=True)

# --- released router decisions on the sampled id_test tasks (binary Z; X12-a scheme)
DEC = {}
for r in SIGNALS_REL:
    path = f"{ACR}/id/voter_decisions/{r}.jsonl" if r != "acrouter_release" else ACR_RELEASE
    d = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()]).drop_duplicates("task_id").set_index("task_id").chosen_model
    DEC[r] = d.reindex(T_IDX); assert not DEC[r].isna().any(), r
x12a_rows = pd.read_csv(os.path.join(X12A, "crb_signal_rows.csv")) if os.path.exists(os.path.join(X12A, "crb_signal_rows.csv")) else None

# ================================================================================================ D. per-state valuation
# arrays over the probing / id_test samples
SC_P = {m: S.loc[P_IDX, m].to_numpy() for m in ALL_PRIMARIES + [STRONG]}; SC_T = {m: S.loc[T_IDX, m].to_numpy() for m in ALL_PRIMARIES + [STRONG]}
CO_P = {m: Cc.loc[P_IDX, m].to_numpy() for m in ALL_PRIMARIES + [STRONG]}; CO_T = {m: Cc.loc[T_IDX, m].to_numpy() for m in ALL_PRIMARIES + [STRONG]}
TK_P = {m: TT.loc[P_IDX, m].to_numpy() for m in ALL_PRIMARIES + [STRONG]}; LA_P = {m: LT.loc[P_IDX, m].to_numpy() for m in ALL_PRIMARIES + [STRONG]}
Z_LLM = {"Qwen3-Max": (sig_ds.loc[P_IDX, "p_qwen"].to_numpy(), sig_ds.loc[T_IDX, "p_qwen"].to_numpy()), "kimi-k2.5": (sig_ds.loc[P_IDX, "p_kimi"].to_numpy(), sig_ds.loc[T_IDX, "p_kimi"].to_numpy())}
Z_LAB = (sig_ds.loc[P_IDX, "label_code"].to_numpy(), sig_ds.loc[T_IDX, "label_code"].to_numpy())
Z_LIN = (-sig_ds.loc[P_IDX, "lines"].to_numpy(), -sig_ds.loc[T_IDX, "lines"].to_numpy())
IN_SUB_P = smp.loc[P_IDX, "in_sub"].to_numpy(); IN_SUB_T = smp.loc[T_IDX, "in_sub"].to_numpy()
_rq = lambda c: (sig_qw[c].reindex(P_IDX).to_numpy(dtype=float), sig_qw[c].reindex(T_IDX).to_numpy(dtype=float))   # NaN outside the subsample
Z_QW = {"Qwen3-Max": _rq("p_qwen"), "kimi-k2.5": _rq("p_kimi")}
Z_QLAB = _rq("label_code"); _ql = _rq("lines"); Z_QLIN = (-_ql[0], -_ql[1])
KD = {pn: (sig_ds.loc[P_IDX, f"k_{pn}"].to_numpy(), sig_ds.loc[T_IDX, f"k_{pn}"].to_numpy()) for pn in ["peak", "offpeak", "self", "units"]}
KD["tok"] = (sig_ds.loc[P_IDX, "total_tokens"].to_numpy(), sig_ds.loc[T_IDX, "total_tokens"].to_numpy()); KD["sec"] = (sig_ds.loc[P_IDX, "latency_s"].to_numpy(), sig_ds.loc[T_IDX, "latency_s"].to_numpy())
KQ = {pn: _rq(f"k_{pn}") for pn in ["list", "list_half", "self", "units"]}; KQ["tok"] = _rq("total_tokens"); KQ["sec"] = _rq("latency_s")
cell_p = smp.loc[P_IDX, "cell"].to_numpy(); cell_t = smp.loc[T_IDX, "cell"].to_numpy()
PARTS = {"source": "source", "cell": "cell", "dim": "dimension", "all": "all"}
STATES = [(p, c) for p, col in PARTS.items() for c in sorted(smp[col].unique())]
# global permutations within cell (serve every partition since cell refines source, dim and all)
def perm_within(labels, rng_):
    out = np.arange(len(labels))
    for lab in np.unique(labels):
        pos = np.flatnonzero(labels == lab); out[pos] = rng_.permutation(pos)
    return out
PERM_P = [perm_within(cell_p, rng) for _ in range(NPERM)]; PERM_T = [perm_within(cell_t, rng) for _ in range(NPERM)]

# pooled isotonic calibrators fitted on the whole probing sample (one per theta, primary, signal)
def fit_iso(z, H):
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(z, H.astype(float))
ISO = {}
for theta in THETAS:
    for m in PRIMARIES:
        Hp_all = SC_P[m] >= theta
        ISO[(theta, m, "llm_p")] = fit_iso(Z_LLM[m][0], Hp_all); ISO[(theta, m, "llm_label")] = fit_iso(Z_LAB[0], Hp_all); ISO[(theta, m, "llm_lines")] = fit_iso(Z_LIN[0], Hp_all)
        ISO[(theta, m, "llm_p_shrunk")] = fit_iso(Z_LLM[m][0] / 10.0, Hp_all)
        for v in ["tfidf_lr", "tfidf_lr_uni"]: ISO[(theta, m, v)] = fit_iso(TFIDF[(theta, m, v)][0], Hp_all)
        ISO[(theta, m, "qwen_p")] = fit_iso(Z_QW[m][0][IN_SUB_P], Hp_all[IN_SUB_P]); ISO[(theta, m, "qwen_label")] = fit_iso(Z_QLAB[0][IN_SUB_P], Hp_all[IN_SUB_P]); ISO[(theta, m, "qwen_lines")] = fit_iso(Z_QLIN[0][IN_SUB_P], Hp_all[IN_SUB_P])
        ISO[(theta, m, "qwen_p_shrunk")] = fit_iso(Z_QW[m][0][IN_SUB_P] / 10.0, Hp_all[IN_SUB_P]); ISO[(theta, m, "llm_p_sub")] = fit_iso(Z_LLM[m][0][IN_SUB_P], Hp_all[IN_SUB_P])
        ISO[(theta, m, "llm_ensemble_mean")] = fit_iso((Z_LLM[m][0][IN_SUB_P] + Z_QW[m][0][IN_SUB_P]) / 2, Hp_all[IN_SUB_P])
        ISO[(theta, m, "llm_p_perm")] = [fit_iso(Z_LLM[m][0][pp], Hp_all) for pp in PERM_P]

def posteriors(iso, z_p_state, z_t_state, s_p, Hp_state):
    """anchored pooled isotonic (main), plain pooled isotonic, per-state isotonic (n >= 30, both classes; else anchored)"""
    pooled_t = iso.predict(z_t_state); shift = s_p - float(iso.predict(z_p_state).mean())
    anch = np.clip(pooled_t + shift, 0.0, 1.0)
    if len(z_p_state) >= 30 and 0 < Hp_state.sum() < len(Hp_state): st = fit_iso(z_p_state, Hp_state).predict(z_t_state)
    else: st = anch
    return anch, pooled_t, st, shift

def best_threshold_on_probing(z_p, casc_p, Cdir_p):
    """rank-preserving re-thresholding: t* minimising the PROBING mean cost of 'attempt iff z >= t' (IN-SAMPLE on probing)"""
    cand = np.unique(np.concatenate([z_p, [np.inf]])); best_t, best_c = None, np.inf
    for t in cand:
        c = float(np.where(z_p >= t, casc_p, Cdir_p).mean())
        if c < best_c - 1e-15: best_c, best_t = c, t
    return best_t

sig_rows, band_rows, ivp_rows, ivp2_rows, kcond_rows, skipped, cf_rows = [], [], [], [], [], [], []
BV = {}   # bootstrap vectors of gross_bw / gross_native etc. for paired comparisons and the C3 task bootstrap
t_st = time.time()
for (part, st) in STATES:
    col = PARTS[part]
    pid = np.flatnonzero(smp.loc[P_IDX, col].to_numpy() == st); tid = np.flatnonzero(smp.loc[T_IDX, col].to_numpy() == st)
    nP, nT = len(pid), len(tid)
    if nP < 5 or nT < 5: skipped.append(dict(part=part, state=st, n_prob=nP, n_test=nT)); continue
    small = bool(nP < 20 or nT < 20)
    ip = rng.integers(0, nP, (NBOOT, nP), dtype=np.int32); it = rng.integers(0, nT, (NBOOT, nT), dtype=np.int32); splits = half_splits(cell_t[tid], NSPLIT, rng)
    cB_p = CO_P[STRONG][pid]; cB_t = CO_T[STRONG][tid]; yB_t = SC_T[STRONG][tid]; b_p = float(cB_p.mean()); bB = bm(cB_p, ip); dirB = bm(cB_t, it); Cdir = cB_t
    kmean = {pn: float(np.concatenate([KD[pn][0][pid], KD[pn][1][tid]]).mean()) for pn in KD}      # signal cost per task, all sampled tasks of the state
    kt = {pn: KD[pn][1][tid] for pn in KD}; kt_mean = {pn: float(v.mean()) for pn, v in kt.items()}  # id_test only (enters net VOI)
    kqmean = {pn: float(np.nanmean(np.concatenate([KQ[pn][0][pid], KQ[pn][1][tid]]))) if np.isfinite(np.concatenate([KQ[pn][0][pid], KQ[pn][1][tid]])).any() else NAN for pn in KQ}
    kqt_mean = {pn: float(np.nanmean(KQ[pn][1][tid])) if np.isfinite(KQ[pn][1][tid]).any() else NAN for pn in KQ}
    ktok054 = float((INTOK.loc[T_IDX[tid]] * PRICES["self_hosted_per_token"]).mean())               # X12-a's prompt-token proxy
    # ---- k < a (all 7 primaries, three bases; probing primitives)
    for m in ALL_PRIMARIES:
        Hp = SC_P[m][pid] >= THETA_MAIN; s_ = float(Hp.mean()); a_ = float(CO_P[m][pid].mean()); at = float(TK_P[m][pid].mean()); asec = float(LA_P[m][pid].mean())
        kcond_rows.append(dict(part=part, state=st, primary=m, n_prob=nP, n_test=nT, small=small, s=s_, a_usd=a_, b_usd=b_p, a_over_b=a_ / b_p if b_p else NAN, a_tok=at, b_tok=float(TK_P[STRONG][pid].mean()),
                               a_sec=asec, b_sec=float(LA_P[STRONG][pid].mean()), k_peak=kmean["peak"], k_offpeak=kmean["offpeak"], k_self=kmean["self"], k_units=kmean["units"], k_tok=kmean["tok"], k_sec=kmean["sec"],
                               k_tok054=ktok054, a_1ms_usd=a_ * (1 - s_), band_nonempty_max_k_usd=a_ * (b_p - a_) / b_p if b_p > a_ else 0.0,
                               k_peak_lt_a=bool(kmean["peak"] < a_), k_offpeak_lt_a=bool(kmean["offpeak"] < a_), k_self_lt_a=bool(kmean["self"] < a_), k_tok_lt_a_tok=bool(kmean["tok"] < at), k_sec_lt_a_sec=bool(kmean["sec"] < asec),
                               k_peak_lt_a_1ms=bool(kmean["peak"] < a_ * (1 - s_)), k_offpeak_lt_a_1ms=bool(kmean["offpeak"] < a_ * (1 - s_)),
                               kq_list=kqmean["list"], kq_list_half=kqmean["list_half"], kq_self=kqmean["self"], kq_tok=kqmean["tok"], kq_sec=kqmean["sec"],
                               kq_list_lt_a=bool(kqmean["list"] < a_), kq_list_half_lt_a=bool(kqmean["list_half"] < a_), kq_tok_lt_a_tok=bool(kqmean["tok"] < at), kq_sec_lt_a_sec=bool(kqmean["sec"] < asec)))
    for theta in THETAS:
        Hp_all = {m: SC_P[m][pid] >= theta for m in ALL_PRIMARIES}; Ht_all = {m: SC_T[m][tid] >= theta for m in ALL_PRIMARIES}
        for m in PRIMARIES:
            Hp = Hp_all[m]; cP_p = CO_P[m][pid]; s_p = float(Hp.mean()); a_p = float(cP_p.mean()); ab = a_p / b_p
            sB = bm(Hp.astype(float), ip); aB = bm(cP_p, ip); bound_c = min(s_p * (b_p - a_p), a_p * (1 - s_p)); boundB = np.minimum(sB * (bB - aB), aB * (1 - sB))
            Ht = Ht_all[m]; Hf = Ht.astype(float); cP_t = CO_T[m][tid]; yP_t = SC_T[m][tid]
            casc = cP_t + (1 - Hf) * cB_t; perf = Hf * cP_t + (1 - Hf) * cB_t; dlt = casc - Cdir; Vcasc = Hf * yP_t + (1 - Hf) * yB_t
            casc_p = cP_p + (1 - Hp.astype(float)) * cB_p
            u0_att = bool(s_p > ab); Cu0 = casc if u0_att else Cdir; Vu0 = Vcasc if u0_att else yB_t; u0B = bm(Cu0, it); cascB = bm(casc, it); perfB = bm(perf, it)
            f_ins = min(casc.mean(), Cdir.mean()); bound_real = f_ins - perf.mean(); s_t = float(Ht.mean()); a_t = float(cP_t.mean()); b_t = float(cB_t.mean()); Hsum = bm(Hf, it) * nT
            bc = bound_c if bound_c > 1e-9 else NAN
            base = dict(part=part, state=st, theta=theta, primary=m, n_prob=nP, n_test=nT, small=small, s=s_p, a=a_p, b=b_p, a_over_b=ab, s_test=s_t, a_test=a_t, b_test=b_t,
                        k_peak=kt_mean["peak"], k_offpeak=kt_mean["offpeak"], k_self=kt_mean["self"], k_units=kt_mean["units"], k_tok=kt_mean["tok"], k_sec=kt_mean["sec"],
                        u0_attempt=u0_att, C_u0=float(Cu0.mean()), C_cascade=float(casc.mean()), C_direct=float(Cdir.mean()), C_perfect=float(perf.mean()), f_ins=f_ins,
                        bound_closed=bound_c, bound_closed_lo=ci(boundB)[0], bound_closed_hi=ci(boundB)[1], bound_real=bound_real, degenerate_prior=bool(bound_c <= 1e-9),
                        V_direct=float(yB_t.mean()), V_cascade=float(Vcasc.mean()), V_u0=float(Vu0.mean()), eta_B=float(yB_t.sum() / Cdir.sum()), eta_u0=float(Vu0.sum() / Cu0.sum()))
            # ---- band position of s and the realized cheapest option with EXACT information at cost k (prop:voi), k = the measured signal cost
            for kname, kval in [("0", 0.0), ("peak", kt_mean["peak"]), ("offpeak", kt_mean["offpeak"]), ("self", kt_mean["self"]), ("tok054", ktok054)]:
                lo_b, hi_b = kval / (b_p - a_p) if b_p > a_p else NAN, 1 - kval / a_p if a_p > 0 else NAN
                th = {"cascade": a_p + (1 - s_p) * b_p, "direct": b_p, "info": kval + s_p * a_p + (1 - s_p) * b_p}
                real = {"cascade": float(casc.mean()), "direct": float(Cdir.mean()), "info": float(perf.mean() + kval)}
                mB = (perfB + kval) - np.minimum(cascB, dirB); mlo, mhi = ci(mB)
                band_rows.append(dict(part=part, state=st, theta=theta, primary=m, k_name=kname, k=kval, n_prob=nP, n_test=nT, small=small, s=s_p, a=a_p, b=b_p, a_over_b=ab, band_lo=lo_b, band_hi=hi_b,
                                      band_nonempty=bool(np.isfinite(lo_b) and np.isfinite(hi_b) and lo_b < hi_b), u=(s_p - lo_b) / (hi_b - lo_b) if (np.isfinite(lo_b) and np.isfinite(hi_b) and hi_b > lo_b) else NAN,
                                      pred_info_in_band=bool(np.isfinite(lo_b) and np.isfinite(hi_b) and lo_b < s_p < hi_b), pred_argmin=min(th, key=th.get),
                                      C_cascade_real=real["cascade"], C_direct_real=real["direct"], C_info_real=real["info"], real_argmin=min(real, key=real.get),
                                      info_minus_best_uninformed=real["info"] - min(real["cascade"], real["direct"]), lo=mlo, hi=mhi, decided=decided(mlo, mhi)))
            # ---- cor:info-vs-proposer, exact-information form (X12-a): k' = a2 - a
            for m2 in ALL_PRIMARIES:
                if m2 == m: continue
                a2_p = float(CO_P[m2][pid].mean())
                if a2_p <= a_p: continue
                s2_p = float(Hp_all[m2].mean()); dp = s2_p - s_p; kp = a2_p - a_p
                pred = b_p * dp - a_p * (1 - s_p); predB = bB * (bm(Hp_all[m2].astype(float), ip) - sB) - aB * (1 - sB)
                H2 = Ht_all[m2].astype(float); casc2 = CO_T[m2][tid] + (1 - H2) * cB_t
                d_eq = (perf + kp) - casc2; dB = bm(d_eq, it); lo, hi = ci(dB)
                ivp_rows.append(dict(part=part, state=st, theta=theta, primary=m, upgrade=m2, n_prob=nP, n_test=nT, small=small, s=s_p, a=a_p, b=b_p, s_upgrade=s2_p, delta_p=dp, k_prime=kp,
                                     pred_info_minus_dec=pred, pred_lo=ci(predB)[0], pred_hi=ci(predB)[1], pred_info_better=bool(pred < 0),
                                     real_info_minus_dec=float(d_eq.mean()), real_lo=lo, real_hi=hi, real_info_better=bool(d_eq.mean() < 0), decided=decided(lo, hi)))
            # Opus direct as the "upgrade" (s2 = 1, a2 = b): C_dec = b; exact info priced at k' = b - a  ->  prediction (1-s)(b-a) > 0 always (degenerate instance)
            ivp_rows.append(dict(part=part, state=st, theta=theta, primary=m, upgrade="opus_direct", n_prob=nP, n_test=nT, small=small, s=s_p, a=a_p, b=b_p, s_upgrade=1.0, delta_p=1 - s_p, k_prime=b_p - a_p,
                                 pred_info_minus_dec=(1 - s_p) * (b_p - a_p), pred_lo=NAN, pred_hi=NAN, pred_info_better=False, real_info_minus_dec=float((perf + (b_p - a_p) - Cdir).mean()),
                                 real_lo=ci(perfB + (b_p - a_p) - dirB)[0], real_hi=ci(perfB + (b_p - a_p) - dirB)[1], real_info_better=bool((perf + (b_p - a_p) - Cdir).mean() < 0),
                                 decided=decided(*ci(perfB + (b_p - a_p) - dirB))))

            # ---- the signals
            def value(name, family, kind, z_t, att_bw, natives, extra_cols, calib=None, z_for_metrics=None, semi=False, k_own=None):
                """standard block: costs, gross / net VOI, capture, CIs, bound checks, AUROC / ECE.  att_bw: attempt indicator or frequency (Blackwell).
                k_own: (k_main, k_low) per-task mean signal cost of THIS signal on the state's id_test tasks (default: the deepseek signal's peak / off-peak cost)."""
                k_main_, k_low_ = k_own if k_own is not None else (kt_mean["peak"], kt_mean["offpeak"])
                af = np.asarray(att_bw, float); Cbw = Cdir + af * dlt; bwB = dirB + bm(af * dlt, it); g_bw = float(Cu0.mean() - Cbw.mean()); g_bwB = u0B - bwB
                Vbw = yB_t + af * (Vcasc - yB_t)
                zz = z_t if z_for_metrics is None else z_for_metrics
                if kind == "binary":
                    Zb = zz.astype(bool); au = auroc_binary(Zb, Ht); ec = ece_binary(Zb, Ht); zs = bm(Zb.astype(float), it) * nT; zh = bm((Zb & Ht).astype(float), it) * nT
                    auB = (1 + fdiv(zh, Hsum) - fdiv(zs - zh, nT - Hsum)) / 2; auB[(Hsum == 0) | (Hsum == nT)] = NAN; ecB = (zs + Hsum - 2 * zh) / nT
                else:
                    au = auroc_cont(zz, Ht); auB = auroc_cont_boot(zz, Hf, it)
                    if kind == "prob": ec = ece_cont(zz, Ht); ecB = ece_cont_boot(zz, Hf, it)
                    else: ec = NAN; ecB = np.full(NBOOT, NAN)
                row = dict(base, signal=name, signal_family=family, kind=kind, semi_synthetic=semi, auroc=au, auroc_lo=ci(auB)[0], auroc_hi=ci(auB)[1], ece=ec, ece_lo=ci(ecB)[0], ece_hi=ci(ecB)[1],
                           C_bw=float(Cbw.mean()), frac_attempt_bw=float(af.mean()), switch_share_bw=float(np.abs(af - float(u0_att)).mean()),
                           gross_bw=g_bw, gross_bw_lo=ci(g_bwB)[0], gross_bw_hi=ci(g_bwB)[1], capture_bw=g_bw / bc, capture_bw_lo=ci(g_bwB)[0] / bc, capture_bw_hi=ci(g_bwB)[1] / bc,
                           k_sig=k_main_, k_sig_low=k_low_, net_bw_peak=g_bw - k_main_, net_bw_offpeak=g_bw - k_low_, net_bw_self=g_bw - kt_mean["self"],
                           net_bw_peak_lo=ci(g_bwB - k_main_)[0], net_bw_peak_hi=ci(g_bwB - k_main_)[1],
                           bound_minus_k_peak=bound_c - k_main_, bound_minus_k_offpeak=bound_c - k_low_,
                           viol_bw_pt=bool(bound_c > 1e-9 and g_bw > bound_c + 1e-12), viol_bw_ci=bool(bound_c > 1e-9 and ci(g_bwB - boundB)[0] > 0), ratio_bw=g_bw / bc,
                           V_bw=float(Vbw.mean()), e_bw=float((af * Hf * (yP_t < yB_t)).mean()), eta_bw=float(Vbw.sum() / Cbw.sum()), gross_bw_ins=float(f_ins - Cbw.mean()))
                bvec = dict(g_bw=g_bwB.astype(np.float32), auroc=auB.astype(np.float32), ece=ecB.astype(np.float32), bound=boundB.astype(np.float32))
                for nn, natt in natives.items():
                    if natt is None:
                        for suf in ["", "_lo", "_hi"]: row[f"gross_{nn}{suf}"] = NAN; row[f"capture_{nn}{suf}"] = NAN
                        row[f"frac_attempt_{nn}"] = NAN; row[f"viol_{nn}_pt"] = False; row[f"viol_{nn}_ci"] = False; row[f"net_{nn}_peak"] = NAN; row[f"V_{nn}"] = NAN; row[f"e_{nn}"] = NAN; continue
                    nf = np.asarray(natt, float); Cn = Cdir + nf * dlt; nB = dirB + bm(nf * dlt, it); g_n = float(Cu0.mean() - Cn.mean()); g_nB = u0B - nB; Vn = yB_t + nf * (Vcasc - yB_t)
                    row.update({f"gross_{nn}": g_n, f"gross_{nn}_lo": ci(g_nB)[0], f"gross_{nn}_hi": ci(g_nB)[1], f"capture_{nn}": g_n / bc, f"capture_{nn}_lo": ci(g_nB)[0] / bc, f"capture_{nn}_hi": ci(g_nB)[1] / bc,
                                f"frac_attempt_{nn}": float(nf.mean()), f"viol_{nn}_pt": bool(bound_c > 1e-9 and g_n > bound_c + 1e-12), f"viol_{nn}_ci": bool(bound_c > 1e-9 and ci(g_nB - boundB)[0] > 0),
                                f"net_{nn}_peak": g_n - k_main_, f"V_{nn}": float(Vn.mean()), f"e_{nn}": float((nf * Hf * (yP_t < yB_t)).mean()),
                                f"bw_minus_{nn}": g_bw - g_n, f"bw_minus_{nn}_lo": ci(g_bwB - g_nB)[0], f"bw_minus_{nn}_hi": ci(g_bwB - g_nB)[1]})
                    bvec[f"g_{nn}"] = g_nB.astype(np.float32)
                row.update(extra_cols)
                if calib is not None:
                    for cn, af2 in calib.items():
                        af2 = np.asarray(af2, float); g2 = float(Cu0.mean() - (Cdir + af2 * dlt).mean()); g2B = u0B - (dirB + bm(af2 * dlt, it))
                        row[f"gross_bw_{cn}"] = g2; row[f"capture_bw_{cn}"] = g2 / bc; row[f"capture_bw_{cn}_lo"] = ci(g2B)[0] / bc; row[f"capture_bw_{cn}_hi"] = ci(g2B)[1] / bc; row[f"frac_attempt_bw_{cn}"] = float(af2.mean())
                BV[(part, st, theta, m, name)] = bvec; sig_rows.append(row)

            # (1) the LLM probability for this primary (deepseek-v4-pro)
            z_p, z_t = Z_LLM[m][0][pid], Z_LLM[m][1][tid]
            anch, pooled_t, st_t, shift = posteriors(ISO[(theta, m, "llm_p")], z_p, z_t, s_p, Hp)
            tstar = best_threshold_on_probing(z_p, casc_p, cB_p)
            value("llm_p", "llm", "prob", z_t, anch > ab, {"native_ab": z_t >= ab, "native_half": z_t >= 0.5, "rethr": z_t >= tstar},
                  dict(mean_p_test=float(z_t.mean()), mean_p_prob=float(z_p.mean()), mean_posterior_test=float(anch.mean()), anchor_shift=shift, ece_posterior_anchored=ece_cont(anch, Ht),
                       ece_posterior_pooled=ece_cont(pooled_t, Ht), t_star_probing=float(tstar) if np.isfinite(tstar) else NAN, n_imputed_test=int(sig_ds.loc[T_IDX[tid], "imputed"].sum()),
                       frac_posterior_below_ab=float((anch <= ab).mean()), cross_auroc_other_primary=auroc_cont(Z_LLM[[q for q in PRIMARIES if q != m][0]][1][tid], Ht)),
                  calib={"pooledcal": pooled_t > ab, "statecal": st_t > ab})
            # (2) the difficulty label as a coarse Z (native: attempt unless 'hard')
            zl_p, zl_t = Z_LAB[0][pid], Z_LAB[1][tid]
            anch_l, pooled_l, st_l, shift_l = posteriors(ISO[(theta, m, "llm_label")], zl_p, zl_t, s_p, Hp)
            post_by_label = {lab: float(anch_l[zl_t == code].mean()) if (zl_t == code).any() else NAN for lab, code in [("hard", 0.0), ("medium", 1.0), ("easy", 2.0)]}
            prob_by_label = {lab: float(Hp[zl_p == code].mean()) if (zl_p == code).any() else NAN for lab, code in [("hard", 0.0), ("medium", 1.0), ("easy", 2.0)]}
            value("llm_label", "llm", "label", zl_t, anch_l > ab, {"native_ab": None, "native_half": None, "native_label": zl_t >= 1.0},
                  dict(share_easy_test=float((zl_t == 2).mean()), share_medium_test=float((zl_t == 1).mean()), share_hard_test=float((zl_t == 0).mean()),
                       s_given_easy_prob=prob_by_label["easy"], s_given_medium_prob=prob_by_label["medium"], s_given_hard_prob=prob_by_label["hard"],
                       post_easy=post_by_label["easy"], post_medium=post_by_label["medium"], post_hard=post_by_label["hard"], anchor_shift=shift_l), calib={"pooledcal": pooled_l > ab, "statecal": st_l > ab})
            # (3) expected lines of code (negated: longer expected solutions = harder); Blackwell only
            zn_p, zn_t = Z_LIN[0][pid], Z_LIN[1][tid]
            anch_n, pooled_n, st_n, shift_n = posteriors(ISO[(theta, m, "llm_lines")], zn_p, zn_t, s_p, Hp)
            value("llm_lines", "llm", "score", zn_t, anch_n > ab, {"native_ab": None, "native_half": None}, dict(mean_expected_lines_test=float(-zn_t.mean()), anchor_shift=shift_n),
                  calib={"pooledcal": pooled_n > ab, "statecal": st_n > ab})
            # (4) the shrunk score p/10 (SEMI-SYNTHETIC; the paper's miscalibration example): Blackwell unchanged, native a/b rule collapses
            zs_p, zs_t = z_p / 10.0, z_t / 10.0
            anch_s, pooled_s, st_s, _ = posteriors(ISO[(theta, m, "llm_p_shrunk")], zs_p, zs_t, s_p, Hp)
            value("llm_p_shrunk", "llm_semi", "prob", zs_t, anch_s > ab, {"native_ab": zs_t >= ab, "native_half": zs_t >= 0.5}, dict(), semi=True)
            # (5) permuted LLM probability within cell (SEMI-SYNTHETIC: calibrated but uninformative); mean over NPERM permutations
            afp = np.zeros(nT); aup = []
            for r_, (pp, pt_) in enumerate(zip(PERM_P, PERM_T)):
                zp_perm = Z_LLM[m][0][pp][pid]; zt_perm = Z_LLM[m][1][pt_][tid]; iso_r = ISO[(theta, m, "llm_p_perm")][r_]
                post_r = np.clip(iso_r.predict(zt_perm) + (s_p - float(iso_r.predict(zp_perm).mean())), 0, 1); afp += post_r > ab; aup.append(auroc_cont(zt_perm, Ht))
            value("llm_p_perm", "llm_semi", "prob", z_t, afp / NPERM, {"native_ab": None, "native_half": None}, dict(auroc_permuted=float(np.nanmean(aup)) if np.isfinite(aup).any() else NAN), semi=True)
            # (6) TF-IDF logistic routers trained on the probing prompts
            for v in ["tfidf_lr", "tfidf_lr_uni"]:
                oof, pt_all = TFIDF[(theta, m, v)]; zv_p, zv_t = oof[pid], pt_all[tid]
                anch_v, pooled_v, st_v, shift_v = posteriors(ISO[(theta, m, v)], zv_p, zv_t, s_p, Hp)
                value(v, "tfidf", "prob", zv_t, anch_v > ab, {"native_ab": zv_t >= ab, "native_half": zv_t >= 0.5}, dict(anchor_shift=shift_v, ece_posterior_anchored=ece_cont(anch_v, Ht)),
                      calib={"pooledcal": pooled_v > ab, "statecal": st_v > ab}, k_own=(ktok054, ktok054))
            # (7) released routers (binary Z, decisions on id_test only; X12-a cross-fit scheme) -- like-for-like comparison on the same tasks
            for r in SIGNALS_REL:
                ch = DEC[r].to_numpy()[tid]
                for zdef in ZDEFS:
                    Z = (ch == m) if zdef == "this_primary" else (ch != STRONG)
                    af_r, n_anti = crossfit_binary(Ht, Z, splits, ab, s_p)
                    fam = "released_finetuned" if r in FINETUNED else "released_lookup" if r == "acrouter_release" else "released_classical"
                    value(f"{r}|{zdef}", fam, "binary", Z.astype(float), af_r, {"native_ab": None, "native_half": None, "native_router": Z.astype(float)},
                          dict(P_Z1=float(Z.mean()), s_given_Z1=float(Ht[Z].mean()) if Z.any() else NAN, s_given_Z0=float(Ht[~Z].mean()) if (~Z).any() else NAN, n_anti_monotone_calibrations=n_anti, router=r, zdef=zdef), k_own=(0.0, 0.0))
            # (8) the qwen3-max second signal on the nested subsample; deepseek re-valued on the same tasks (paired), plus the two-source ensemble
            if part in ("all", "source", "dim"):
                mp = IN_SUB_P[pid]; mt = IN_SUB_T[tid]
                if mp.sum() >= 5 and mt.sum() >= 5:
                    pid_s = pid[mp]; tid_s = tid[mt]; nPs, nTs = len(pid_s), len(tid_s); it_s = rng.integers(0, nTs, (NBOOT, nTs), dtype=np.int32)
                    Ht_s = Ht[mt]; Hf_s = Ht_s.astype(float); casc_s = casc[mt]; Cdir_s = Cdir[mt]; dlt_s = dlt[mt]; Cu0_s = Cu0[mt]; perf_s = perf[mt]; yP_s = yP_t[mt]; yB_s = yB_t[mt]; Vcasc_s = Vcasc[mt]
                    u0B_s = bm(Cu0_s, it_s); dirB_s = bm(Cdir_s, it_s); Hp_s = Hp[mp]; casc_p_s = casc_p[mp]; cB_p_s = cB_p[mp]
                    def value_sub(name, family, kind, z_p_s, z_t_s, iso, natives, extra_cols, k_own, semi=False):
                        pooled_ = iso.predict(z_t_s); shift_ = s_p - float(iso.predict(z_p_s).mean()); anch_ = np.clip(pooled_ + shift_, 0, 1); af = (anch_ > ab).astype(float)
                        C_ = Cdir_s + af * dlt_s; gB = u0B_s - bm(C_, it_s); g = float(Cu0_s.mean() - C_.mean()); Vbw_ = yB_s + af * (Vcasc_s - yB_s)
                        if kind == "prob": au, ec = auroc_cont(z_t_s, Ht_s), ece_cont(z_t_s, Ht_s); auB = auroc_cont_boot(z_t_s, Hf_s, it_s); ecB = ece_cont_boot(z_t_s, Hf_s, it_s)
                        else: au, ec = auroc_cont(z_t_s, Ht_s), NAN; auB = auroc_cont_boot(z_t_s, Hf_s, it_s); ecB = np.full(NBOOT, NAN)
                        row = dict(base, signal=name, signal_family=family, kind=kind, semi_synthetic=semi, subsample=True, n_prob=nPs, n_test=nTs, small=bool(nPs < 20 or nTs < 20),
                                   auroc=au, auroc_lo=ci(auB)[0], auroc_hi=ci(auB)[1], ece=ec, ece_lo=ci(ecB)[0], ece_hi=ci(ecB)[1],
                                   C_u0=float(Cu0_s.mean()), C_cascade=float(casc_s.mean()), C_direct=float(Cdir_s.mean()), C_perfect=float(perf_s.mean()), C_bw=float(C_.mean()),
                                   frac_attempt_bw=float(af.mean()), switch_share_bw=float(np.abs(af - float(u0_att)).mean()), gross_bw=g, gross_bw_lo=ci(gB)[0], gross_bw_hi=ci(gB)[1],
                                   capture_bw=g / bc, capture_bw_lo=ci(gB)[0] / bc, capture_bw_hi=ci(gB)[1] / bc, k_sig=k_own[0], k_sig_low=k_own[1], net_bw_peak=g - k_own[0], net_bw_offpeak=g - k_own[1],
                                   net_bw_peak_lo=ci(gB - k_own[0])[0], net_bw_peak_hi=ci(gB - k_own[0])[1], viol_bw_pt=bool(bound_c > 1e-9 and g > bound_c + 1e-12), viol_bw_ci=bool(bound_c > 1e-9 and ci(gB - boundB)[0] > 0), ratio_bw=g / bc,
                                   V_bw=float(Vbw_.mean()), e_bw=float((af * Hf_s * (yP_s < yB_s)).mean()), eta_bw=float(Vbw_.sum() / C_.sum()), mean_p_test=float(z_t_s.mean()), mean_posterior_test=float(anch_.mean()), anchor_shift=shift_,
                                   ece_posterior_anchored=ece_cont(anch_, Ht_s) if kind == "prob" else NAN, V_direct=float(yB_s.mean()), V_u0=float((Vcasc_s if u0_att else yB_s).mean()))
                        bvec = dict(g_bw=gB.astype(np.float32))
                        for nn, natt in natives.items():
                            if natt is None:
                                for suf in ["", "_lo", "_hi"]: row[f"gross_{nn}{suf}"] = NAN; row[f"capture_{nn}{suf}"] = NAN
                                row[f"frac_attempt_{nn}"] = NAN; row[f"viol_{nn}_pt"] = False; row[f"viol_{nn}_ci"] = False; continue
                            nf = np.asarray(natt, float); Cn = Cdir_s + nf * dlt_s; nB = u0B_s - bm(Cn, it_s); gn = float(Cu0_s.mean() - Cn.mean())
                            row.update({f"gross_{nn}": gn, f"gross_{nn}_lo": ci(nB)[0], f"gross_{nn}_hi": ci(nB)[1], f"capture_{nn}": gn / bc, f"capture_{nn}_lo": ci(nB)[0] / bc, f"capture_{nn}_hi": ci(nB)[1] / bc, f"frac_attempt_{nn}": float(nf.mean()),
                                        f"viol_{nn}_pt": bool(bound_c > 1e-9 and gn > bound_c + 1e-12), f"viol_{nn}_ci": bool(bound_c > 1e-9 and ci(nB - boundB)[0] > 0), f"net_{nn}_peak": gn - k_own[0],
                                        f"bw_minus_{nn}": g - gn, f"bw_minus_{nn}_lo": ci(gB - nB)[0], f"bw_minus_{nn}_hi": ci(gB - nB)[1]})
                            bvec[f"g_{nn}"] = nB.astype(np.float32)
                        row.update(extra_cols); BV[(part, st, theta, m, name)] = bvec; sig_rows.append(row)
                    kq = (kqt_mean["list"], kqt_mean["list_half"]); kd = (float(kt["peak"][mt].mean()), float(kt["offpeak"][mt].mean()))
                    zq_p, zq_t = Z_QW[m][0][pid_s], Z_QW[m][1][tid_s]; zd_p, zd_t = Z_LLM[m][0][pid_s], Z_LLM[m][1][tid_s]
                    value_sub("qwen_p", "llm2", "prob", zq_p, zq_t, ISO[(theta, m, "qwen_p")], {"native_ab": zq_t >= ab, "native_half": zq_t >= 0.5, "rethr": zq_t >= best_threshold_on_probing(zq_p, casc_p_s, cB_p_s)},
                              dict(spearman_with_deepseek_p=spearman(zq_t, zd_t)[0], cross_auroc_other_primary=auroc_cont(Z_QW[[q for q in PRIMARIES if q != m][0]][1][tid_s], Ht_s), n_imputed_test=int(sig_qw.loc[T_IDX[tid_s], "imputed"].sum()),
                                   k_tok_own=kqt_mean["tok"], k_sec_own=kqt_mean["sec"]), kq)
                    value_sub("llm_p_sub", "llm", "prob", zd_p, zd_t, ISO[(theta, m, "llm_p_sub")], {"native_ab": zd_t >= ab, "native_half": zd_t >= 0.5, "rethr": zd_t >= best_threshold_on_probing(zd_p, casc_p_s, cB_p_s)}, dict(), kd)
                    zql_p, zql_t = Z_QLAB[0][pid_s], Z_QLAB[1][tid_s]
                    value_sub("qwen_label", "llm2", "label", zql_p, zql_t, ISO[(theta, m, "qwen_label")], {"native_ab": None, "native_half": None, "native_label": zql_t >= 1.0},
                              dict(share_easy_test=float((zql_t == 2).mean()), share_medium_test=float((zql_t == 1).mean()), share_hard_test=float((zql_t == 0).mean())), kq)
                    value_sub("qwen_lines", "llm2", "score", Z_QLIN[0][pid_s], Z_QLIN[1][tid_s], ISO[(theta, m, "qwen_lines")], {"native_ab": None, "native_half": None}, dict(mean_expected_lines_test=float(-Z_QLIN[1][tid_s].mean())), kq)
                    value_sub("qwen_p_shrunk", "llm_semi", "prob", zq_p / 10.0, zq_t / 10.0, ISO[(theta, m, "qwen_p_shrunk")], {"native_ab": zq_t / 10.0 >= ab, "native_half": zq_t / 10.0 >= 0.5}, dict(), kq, semi=True)
                    value_sub("llm_ensemble_mean", "llm", "prob", (zd_p + zq_p) / 2, (zd_t + zq_t) / 2, ISO[(theta, m, "llm_ensemble_mean")], {"native_ab": (zd_t + zq_t) / 2 >= ab, "native_half": None}, dict(), (kd[0] + kq[0], kd[1] + kq[1]))
            # ---- counterfactual price ratio (SEMI-SYNTHETIC): rescale cost_B by lam so that a/b = r; the posterior is unchanged, only the threshold and the payoffs move
            if theta == THETA_MAIN and part in ("all", "source"):
                for r_target in [0.05, 0.1, 0.2, 0.3, 0.5]:
                    lam = (a_p / r_target) / b_p; cB_p2 = cB_p * lam; cB_t2 = cB_t * lam; b2 = float(cB_p2.mean()); ab2 = a_p / b2
                    bound2 = min(s_p * (b2 - a_p), a_p * (1 - s_p)); casc2_ = cP_t + (1 - Hf) * cB_t2; dlt2 = casc2_ - cB_t2; u0_2 = bool(s_p > ab2); Cu0_2 = casc2_ if u0_2 else cB_t2
                    for sname, post_, raw_ in [("llm_p", anch, z_t), ("tfidf_lr", posteriors(ISO[(theta, m, "tfidf_lr")], TFIDF[(theta, m, "tfidf_lr")][0][pid], TFIDF[(theta, m, "tfidf_lr")][1][tid], s_p, Hp)[0], TFIDF[(theta, m, "tfidf_lr")][1][tid])]:
                        for uname, att in [("bw", post_ > ab2), ("native_ab", raw_ >= ab2)]:
                            af2 = att.astype(float); C2_ = cB_t2 + af2 * dlt2; g2 = float(Cu0_2.mean() - C2_.mean()); g2B = bm(Cu0_2, it) - bm(C2_, it)
                            cf_rows.append(dict(part=part, state=st, primary=m, signal=sname, usage=uname, r_target=r_target, lam=lam, s=s_p, a=a_p, b_cf=b2, bound_cf=bound2, u0_attempt_cf=u0_2, frac_attempt=float(af2.mean()),
                                                switch_share=float(np.abs(af2 - float(u0_2)).mean()), gross=g2, gross_lo=ci(g2B)[0], gross_hi=ci(g2B)[1], capture=g2 / bound2 if bound2 > 1e-9 else NAN,
                                                capture_lo=ci(g2B)[0] / bound2 if bound2 > 1e-9 else NAN, capture_hi=ci(g2B)[1] / bound2 if bound2 > 1e-9 else NAN, n_test=nT))
            # ---- cor:info-vs-proposer with the ACTUAL signal: [k_sig + C_BW] vs the upgraded cascade (probing prediction vs id_test realization)
            post_p_anch = np.clip(ISO[(theta, m, "llm_p")].predict(z_p) + shift, 0, 1); att_p = post_p_anch > ab
            C_bw_prob = float(np.where(att_p, casc_p, cB_p).mean()); k_prob = float(KD[K_MAIN][0][pid].mean())
            g_bw_row = [r_ for r_ in sig_rows if r_["signal"] == "llm_p" and r_["part"] == part and r_["state"] == st and r_["theta"] == theta and r_["primary"] == m][-1]
            for m2 in ALL_PRIMARIES:
                if m2 == m: continue
                a2_p = float(CO_P[m2][pid].mean())
                if a2_p <= a_p: continue
                s2_p = float(Hp_all[m2].mean()); casc2_p = float((CO_P[m2][pid] + (1 - Hp_all[m2].astype(float)) * cB_p).mean())
                H2 = Ht_all[m2].astype(float); casc2 = CO_T[m2][tid] + (1 - H2) * cB_t
                d_pred = (k_prob + C_bw_prob) - casc2_p; d_real_vec = (kt["peak"] + (Cdir + (anch > ab) * dlt)) - casc2; dB = bm(d_real_vec, it); lo, hi = ci(dB)
                ivp2_rows.append(dict(part=part, state=st, theta=theta, primary=m, upgrade=m2, n_prob=nP, n_test=nT, small=small, s=s_p, a=a_p, b=b_p, s_upgrade=s2_p, delta_p=s2_p - s_p, k_prime=a2_p - a_p, k_sig=k_prob,
                                      pred_signal_minus_upgrade=d_pred, pred_signal_better=bool(d_pred < 0), real_signal_minus_upgrade=float(d_real_vec.mean()), real_lo=lo, real_hi=hi,
                                      real_signal_better=bool(d_real_vec.mean() < 0), decided=decided(lo, hi), capture_bw_llm=g_bw_row["capture_bw"]))
    print(f"[state] {part}/{st} nP={nP} nT={nT} done ({time.time() - t_st:.0f}s)", flush=True)

sig = pd.DataFrame(sig_rows); sig["subsample"] = sig["subsample"].fillna(False).astype(bool) if "subsample" in sig else False
band = pd.DataFrame(band_rows); ivp = pd.DataFrame(ivp_rows); ivp2 = pd.DataFrame(ivp2_rows); kcond = pd.DataFrame(kcond_rows)
ivp["next_stronger"] = False
_r = ivp[ivp.upgrade != "opus_direct"]; _rank = np.where(_r.delta_p > 0, 0.0, 1e9) + _r.k_prime.to_numpy()
pick = pd.Series(_rank, index=_r.index).groupby([_r.part, _r.state, _r.theta, _r.primary]).idxmin(); ivp.loc[pick.to_numpy(), "next_stronger"] = True
cf = pd.DataFrame(cf_rows)
for name, d in [("signal_state_rows.csv", sig), ("band_rows.csv", band), ("info_vs_proposer_rows.csv", ivp), ("info_vs_proposer_actual_signal_rows.csv", ivp2), ("k_condition_rows.csv", kcond), ("counterfactual_price_ratio_rows.csv", cf)]:
    d.to_csv(os.path.join(OUT, name), index=False)
results["n_rows"] = dict(signal_state_rows=int(len(sig)), band_rows=int(len(band)), counterfactual_rows=int(len(cf)), info_vs_proposer_rows=int(len(ivp)), info_vs_proposer_actual_signal_rows=int(len(ivp2)), k_condition_rows=int(len(kcond)), skipped_states=skipped)
print(f"[rows] {len(sig)} signal rows in {time.time() - t_st:.0f}s", flush=True)

# ================================================================================================ E. signal diagnostics
main = sig[(sig.theta == THETA_MAIN) & (~sig.subsample)]
d_ = sig_ds.loc[task_tbl.index]
diag = dict(n_tasks=int(len(d_)), parse_ok=int(d_.parse_ok.sum()), imputed=int(d_.imputed.sum()), mean_p_qwen=float(d_.p_qwen.mean()), mean_p_kimi=float(d_.p_kimi.mean()),
            quantiles_p_qwen={q: float(np.quantile(d_.p_qwen, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}, share_round_to_0p05_p_qwen=float((np.abs(d_.p_qwen * 20 - np.round(d_.p_qwen * 20)) < 1e-9).mean()),
            corr_p_qwen_p_kimi_spearman=spearman(d_.p_qwen, d_.p_kimi)[0], label_shares={k: float(v) for k, v in d_.label.value_counts(normalize=True).items()},
            expected_lines_quantiles={q: float(np.quantile(d_.lines, q)) for q in [0.05, 0.5, 0.95]}, mean_prompt_chars=float(smp.prompt_chars.mean()), n_truncated=int(smp.truncated.sum()),
            tokens_per_call=dict(mean=float(d_.total_tokens.mean()), min=int(d_.total_tokens.min()), max=int(d_.total_tokens.max())), latency_s=dict(mean=float(d_.latency_s.mean()), p95=float(np.quantile(d_.latency_s, .95))),
            k_per_task_usd=dict(peak=float(d_.k_peak.mean()), offpeak=float(d_.k_offpeak.mean()), self_hosted=float(d_.k_self.mean()), cv_peak=float(d_.k_peak.std() / d_.k_peak.mean())))
for m in PRIMARIES:
    Ht_all_main = SC_T[m] >= THETA_MAIN; Hp_all_main = SC_P[m] >= THETA_MAIN; zp, zt = Z_LLM[m]
    diag[f"pooled_{m}"] = dict(auroc_probing=auroc_cont(zp, Hp_all_main), auroc_id_test=auroc_cont(zt, Ht_all_main), ece_id_test_raw=ece_cont(zt, Ht_all_main),
                               ece_id_test_isotonic=ece_cont(ISO[(THETA_MAIN, m, "llm_p")].predict(zt), Ht_all_main), s_probing=float(Hp_all_main.mean()), mean_p=float(zt.mean()),
                               auroc_of_other_primarys_p=auroc_cont(Z_LLM[[q for q in PRIMARIES if q != m][0]][1], Ht_all_main), auroc_label=auroc_cont(Z_LAB[1], Ht_all_main), auroc_neg_lines=auroc_cont(Z_LIN[1], Ht_all_main),
                               auroc_tfidf_lr=auroc_cont(TFIDF[(THETA_MAIN, m, "tfidf_lr")][1], Ht_all_main),
                               auroc_qwen3max_p_subsample=auroc_cont(Z_QW[m][1][IN_SUB_T], Ht_all_main[IN_SUB_T]), auroc_deepseek_p_same_subsample=auroc_cont(zt[IN_SUB_T], Ht_all_main[IN_SUB_T]),
                               ece_qwen3max_p_raw_subsample=ece_cont(Z_QW[m][1][IN_SUB_T], Ht_all_main[IN_SUB_T]), auroc_ensemble_mean_subsample=auroc_cont((zt[IN_SUB_T] + Z_QW[m][1][IN_SUB_T]) / 2, Ht_all_main[IN_SUB_T]),
                               auroc_qwen3max_label_subsample=auroc_cont(Z_QLAB[1][IN_SUB_T], Ht_all_main[IN_SUB_T]), auroc_qwen3max_neg_lines_subsample=auroc_cont(Z_QLIN[1][IN_SUB_T], Ht_all_main[IN_SUB_T]),
                               s_by_label_probing={lab: float(Hp_all_main[Z_LAB[0] == code].mean()) if (Z_LAB[0] == code).any() else NAN for lab, code in [("easy", 2.0), ("medium", 1.0), ("hard", 0.0)]},
                               per_source_auroc={s: auroc_cont(zt[smp.loc[T_IDX, "source"].to_numpy() == s], Ht_all_main[smp.loc[T_IDX, "source"].to_numpy() == s]) for s in SOURCES})
q_ = sig_qw.loc[sub_ids]; d_sub = d_.loc[sub_ids]
diag["qwen3_max_signal"] = dict(n=int(len(q_)), parse_ok=int(q_.parse_ok.sum()), imputed=int(q_.imputed.sum()), mean_p_qwen=float(q_.p_qwen.mean()), mean_p_kimi=float(q_.p_kimi.mean()),
                                quantiles_p_qwen={q: float(np.quantile(q_.p_qwen, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}, label_shares={k: float(v) for k, v in q_.label.value_counts(normalize=True).items()},
                                tokens_per_call=dict(mean=float(q_.total_tokens.mean()), min=int(q_.total_tokens.min()), max=int(q_.total_tokens.max())), latency_s=dict(mean=float(q_.latency_s.mean()), p95=float(np.quantile(q_.latency_s, .95))),
                                k_per_task_usd=dict(list=float(q_.k_list.mean()), list_half=float(q_.k_list_half.mean()), self_hosted=float(q_.k_self.mean())),
                                spearman_with_deepseek_p_qwen=spearman(q_.p_qwen, d_sub.p_qwen)[0], spearman_with_deepseek_p_kimi=spearman(q_.p_kimi, d_sub.p_kimi)[0], label_agreement_with_deepseek=float((q_.label == d_sub.label).mean()),
                                spearman_lines_with_deepseek=spearman(q_.lines, d_sub.lines)[0])
# all-zero sensitivity (offline): s per source with and without the all-zero tasks, Qwen3-Max theta=1, probing population
az = {}
for s in SOURCES:
    full = meta[(meta.source == s) & (meta.split == "probing")]; kept = full[~allzero.loc[full.index]]
    az[s] = {m: dict(s_excl_allzero=float((S.loc[kept.index, m] >= 1).mean()), s_incl_allzero=float((S.loc[full.index, m] >= 1).mean()), n_excl=int(len(kept)), n_incl=int(len(full))) for m in PRIMARIES}
diag["allzero_sensitivity_probing_population"] = az
hx = meta[(meta.source == "hex") & (meta.split == "probing")]; hx_lang = hx.index.str.split("_").str[2]
diag["hex_by_language_probing_population"] = {lang: {m: dict(n=int((hx_lang == lang).sum()), s_theta1=float((S.loc[hx.index[hx_lang == lang], m] >= 1).mean()), s_theta0p5=float((S.loc[hx.index[hx_lang == lang], m] >= 0.5).mean()),
                                                              mean_score=float(S.loc[hx.index[hx_lang == lang], m].mean())) for m in PRIMARIES} for lang in sorted(set(hx_lang))}
diag["hex_language_note"] = "the released HumanEval-X scores give partial credit 0.8 on most non-python tasks, so at theta=1 the non-python languages are 'insufficient' and a prompt router that recognises the language predicts H; at theta=0.5 they are sufficient"
results["signal_diagnostics"] = diag

# ================================================================================================ F. claims
REAL = main[~main.semi_synthetic]
# ---- C1: bound
def c1_summary(sub):
    ok = sub[~sub.degenerate_prior]
    nat_cols = [c for c in ["native_ab", "native_half", "native_label", "native_router", "rethr"] if f"viol_{c}_pt" in sub]
    return dict(n_rows=int(len(sub)), n_degenerate_excluded=int(sub.degenerate_prior.sum()), n_rows_tested=int(len(ok)),
                bw=dict(n_point_violations=int(ok.viol_bw_pt.sum()), n_ci_violations=int(ok.viol_bw_ci.sum()), max_ratio=float(ok.ratio_bw.max()), n_ratio_above_0p9=int((ok.ratio_bw > 0.9).sum())),
                native={c: dict(n_point_violations=int(ok[f"viol_{c}_pt"].fillna(False).astype(bool).sum()), n_ci_violations=int(ok[f"viol_{c}_ci"].fillna(False).astype(bool).sum()),
                                max_ratio=float(np.nanmax(ok[f"capture_{c}"])) if np.isfinite(ok[f"capture_{c}"]).any() else NAN) for c in nat_cols},
                calibrator_variants={cn: dict(n_point_violations=int((ok[f"capture_bw_{cn}"] > 1 + 1e-9).sum()), max_ratio=float(np.nanmax(ok[f"capture_bw_{cn}"]))) for cn in ["pooledcal", "statecal"] if f"capture_bw_{cn}" in ok},
                point_violations=ok[ok.viol_bw_pt | ok[[f"viol_{c}_pt" for c in nat_cols]].fillna(False).astype(bool).any(axis=1)][["part", "state", "theta", "primary", "signal", "n_test", "small", "s", "a_over_b", "bound_closed", "gross_bw", "gross_bw_lo", "gross_bw_hi", "viol_bw_ci"] + [f"gross_{c}" for c in nat_cols]].to_dict("records"))
C1 = {"all_rows": c1_summary(sig), "theta1": c1_summary(sig[sig.theta == 1.0]), "theta0.5": c1_summary(sig[sig.theta == 0.5]), "llm_p_only": c1_summary(sig[sig.signal == "llm_p"]),
      "frac_tasks_primary_costlier_than_reference": {m: float((Cc.loc[T_IDX, m] > Cc.loc[T_IDX, STRONG]).mean()) for m in PRIMARIES}}
results["C1_bound"] = C1

# ---- C2: LLM vs released routers (same tasks) and vs TF-IDF (paired bootstrap)
def paired(part, st, theta, m, s1, s2, key1="g_bw", key2="g_bw"):
    a_ = BV.get((part, st, theta, m, s1)); b_ = BV.get((part, st, theta, m, s2))
    if a_ is None or b_ is None or key1 not in a_ or key2 not in b_: return dict(diff=NAN, lo=NAN, hi=NAN)
    d = a_[key1].astype(float) - b_[key2].astype(float); return dict(diff=float(d.mean()), lo=ci(d)[0], hi=ci(d)[1])
def row_of(part, st, theta, m, s_):
    r_ = sig[(sig.part == part) & (sig.state == st) & (sig.theta == theta) & (sig.primary == m) & (sig.signal == s_)]
    return r_.iloc[0] if len(r_) else None
def agg_rows(theta, part, s_):
    return sig[(sig.theta == theta) & (sig.part == part) & (sig.signal == s_)]
def agg_capture(s_, theta, part="source", m=None):
    """task-weighted gross VOI / task-weighted bound over the states of a partition (both primaries unless m given); CI from the per-state paired bootstrap vectors"""
    rows = agg_rows(theta, part, s_); rows = rows if m is None else rows[rows.primary == m]; rows = rows[~rows.degenerate_prior]
    if not len(rows): return dict(n_states=0)
    W = rows.n_test.to_numpy(float); g = rows.gross_bw.to_numpy(); bnd = rows.bound_closed.to_numpy()
    gB = np.stack([BV[(r.part, r.state, r.theta, r.primary, r.signal)]["g_bw"].astype(float) for _, r in rows.iterrows()])
    bB = np.stack([BV[(r.part, r.state, r.theta, r.primary, r.signal)]["bound"].astype(float) if "bound" in BV[(r.part, r.state, r.theta, r.primary, r.signal)] else np.full(NBOOT, b_) for (_, r), b_ in zip(rows.iterrows(), bnd)])
    agg_g = float((W * g).sum() / W.sum()); agg_b = float((W * bnd).sum() / W.sum())
    with np.errstate(divide="ignore", invalid="ignore"): capB = fdiv((W[:, None] * gB).sum(0), (W[:, None] * bB).sum(0))
    return dict(n_states=int(len(rows)), n_tasks=int(W.sum()), gross_taskweighted=agg_g, bound_taskweighted=agg_b, capture=agg_g / agg_b if agg_b > 1e-12 else NAN, capture_lo=ci(capB)[0], capture_hi=ci(capB)[1],
                gross_lo=ci((W[:, None] * gB).sum(0) / W.sum())[0], gross_hi=ci((W[:, None] * gB).sum(0) / W.sum())[1], n_states_ci_above_0=int((rows.gross_bw_lo > 0).sum()), n_states_ci_below_0=int((rows.gross_bw_hi < 0).sum()),
                mean_k_sig=float((W * rows.k_sig.to_numpy()).sum() / W.sum()), net_taskweighted=float((W * (g - rows.k_sig.to_numpy())).sum() / W.sum()))
def agg_paired(sA, sB, theta, part="source", m=None, keyA="g_bw", keyB="g_bw"):
    """task-weighted paired difference gross(A) - gross(B) over the states of a partition (same resamples per state), also as a capture-ratio difference"""
    ra = agg_rows(theta, part, sA); rb = agg_rows(theta, part, sB)
    if m is not None: ra = ra[ra.primary == m]; rb = rb[rb.primary == m]
    ra = ra.set_index(["state", "primary"]); rb = rb.set_index(["state", "primary"]); idx = [i for i in ra.index if i in rb.index and not ra.loc[i, "degenerate_prior"]]
    if not idx: return dict(n_states=0)
    W = np.array([ra.loc[i, "n_test"] for i in idx], float); bnd = np.array([ra.loc[i, "bound_closed"] for i in idx])
    dB = np.stack([BV[(part, i[0], theta, i[1], sA)][keyA].astype(float) - BV[(part, i[0], theta, i[1], sB)][keyB].astype(float) for i in idx])
    d = np.array([ra.loc[i, "gross_bw" if keyA == "g_bw" else "gross_" + keyA[2:]] - rb.loc[i, "gross_bw" if keyB == "g_bw" else "gross_" + keyB[2:]] for i in idx])
    aggd = float((W * d).sum() / W.sum()); aggb = float((W * bnd).sum() / W.sum()); dBw = (W[:, None] * dB).sum(0) / W.sum()
    return dict(n_states=int(len(idx)), n_tasks=int(W.sum()), diff_gross=aggd, diff_lo=ci(dBw)[0], diff_hi=ci(dBw)[1], diff_capture=aggd / aggb if aggb > 1e-12 else NAN, diff_capture_lo=ci(dBw)[0] / aggb if aggb > 1e-12 else NAN,
                diff_capture_hi=ci(dBw)[1] / aggb if aggb > 1e-12 else NAN, decided=decided(ci(dBw)[0], ci(dBw)[1]), n_states_A_ci_above_B=int(sum(ci(dB[j])[0] > 0 for j in range(len(idx)))), n_states_B_ci_above_A=int(sum(ci(dB[j])[1] < 0 for j in range(len(idx)))))
C2 = {"comparisons": [], "x12a_reference": {}, "aggregate_source_theta1": {}, "aggregate_source_theta0.5": {}}
MAIN_SIGNALS = ["llm_p", "llm_label", "llm_lines", "tfidf_lr", "tfidf_lr_uni", "llm_p_shrunk", "llm_p_perm"] + [f"{r}|{z}" for r in SIGNALS_REL for z in ZDEFS]
SUB_SIGNALS = ["qwen_p", "llm_p_sub", "qwen_label", "qwen_lines", "llm_ensemble_mean", "qwen_p_shrunk"]
for theta in THETAS:
    key = f"aggregate_source_theta{theta:g}" if theta != 1.0 else "aggregate_source_theta1"
    C2[key] = {"capture": {s_: agg_capture(s_, theta) for s_ in MAIN_SIGNALS + SUB_SIGNALS},
               "capture_by_primary": {m: {s_: agg_capture(s_, theta, m=m) for s_ in ["llm_p", "tfidf_lr", "qwen_p", "llm_p_sub", "finetuned_router_qwen35_27b_v3|this_primary", "finetuned_router_qwen35_27b_v3|any_non_opus"]} for m in PRIMARIES},
               "llm_minus_tfidf_lr": agg_paired("llm_p", "tfidf_lr", theta), "llm_minus_tfidf_lr_uni": agg_paired("llm_p", "tfidf_lr_uni", theta),
               "llm_minus_released": {f"{r}|{z}": agg_paired("llm_p", f"{r}|{z}", theta) for r in SIGNALS_REL for z in ZDEFS},
               "llm_minus_label": agg_paired("llm_p", "llm_label", theta), "llm_minus_lines": agg_paired("llm_p", "llm_lines", theta), "llm_minus_perm": agg_paired("llm_p", "llm_p_perm", theta),
               "deepseek_minus_qwen3max_subsample": agg_paired("llm_p_sub", "qwen_p", theta), "ensemble_minus_deepseek_subsample": agg_paired("llm_ensemble_mean", "llm_p_sub", theta),
               "bw_minus_native_ab_llm": agg_paired("llm_p", "llm_p", theta, keyA="g_bw", keyB="g_native_ab"), "bw_minus_native_ab_shrunk": agg_paired("llm_p_shrunk", "llm_p_shrunk", theta, keyA="g_bw", keyB="g_native_ab"),
               "bw_minus_rethr_llm": agg_paired("llm_p", "llm_p", theta, keyA="g_bw", keyB="g_rethr"), "rethr_minus_native_ab_llm": agg_paired("llm_p", "llm_p", theta, keyA="g_rethr", keyB="g_native_ab"),
               "bw_minus_native_ab_tfidf": agg_paired("tfidf_lr", "tfidf_lr", theta, keyA="g_bw", keyB="g_native_ab"), "bw_minus_native_half_tfidf": agg_paired("tfidf_lr", "tfidf_lr", theta, keyA="g_bw", keyB="g_native_half"),
               "bw_minus_native_router_ft27b": agg_paired("finetuned_router_qwen35_27b_v3|this_primary", "finetuned_router_qwen35_27b_v3|this_primary", theta, keyA="g_bw", keyB="g_native_router"),
               "by_primary": {m: dict(llm_minus_tfidf_lr=agg_paired("llm_p", "tfidf_lr", theta, m=m), llm_minus_ft27b=agg_paired("llm_p", "finetuned_router_qwen35_27b_v3|this_primary", theta, m=m),
                                      deepseek_minus_qwen3max_subsample=agg_paired("llm_p_sub", "qwen_p", theta, m=m), bw_minus_native_ab_llm=agg_paired("llm_p", "llm_p", theta, m=m, keyA="g_bw", keyB="g_native_ab"),
                                      bw_minus_native_ab_shrunk=agg_paired("llm_p_shrunk", "llm_p_shrunk", theta, m=m, keyA="g_bw", keyB="g_native_ab")) for m in PRIMARIES}}
C2["aggregate_cell_theta1_capture"] = {s_: agg_capture(s_, 1.0, part="cell") for s_ in ["llm_p", "tfidf_lr", "finetuned_router_qwen35_27b_v3|this_primary", "finetuned_router_qwen35_27b_v3|any_non_opus", "llm_p_perm"]}
C2["aggregate_dim_theta1_capture"] = {s_: agg_capture(s_, 1.0, part="dim") for s_ in ["llm_p", "tfidf_lr", "finetuned_router_qwen35_27b_v3|this_primary", "finetuned_router_qwen35_27b_v3|any_non_opus", "llm_p_perm"]}
for theta in THETAS:
    for m in PRIMARIES:
        for (part, st) in [("all", "ALL")] + [("source", s) for s in SOURCES]:
            llm = row_of(part, st, theta, m, "llm_p")
            if llm is None or not np.isfinite(llm.capture_bw): continue
            rel = sig[(sig.part == part) & (sig.state == st) & (sig.theta == theta) & (sig.primary == m) & sig.signal_family.str.startswith("released")]
            best = rel.loc[rel.capture_bw.idxmax()] if len(rel) and np.isfinite(rel.capture_bw).any() else None
            ft27 = row_of(part, st, theta, m, "finetuned_router_qwen35_27b_v3|this_primary"); tf = row_of(part, st, theta, m, "tfidf_lr"); tfu = row_of(part, st, theta, m, "tfidf_lr_uni")
            bc_ = llm.bound_closed
            ent = dict(part=part, state=st, theta=theta, primary=m, n_test=int(llm.n_test), s=llm.s, a_over_b=llm.a_over_b, bound=bc_,
                       llm=dict(capture=llm.capture_bw, lo=llm.capture_bw_lo, hi=llm.capture_bw_hi, auroc=llm.auroc, ece=llm.ece, net_peak=llm.net_bw_peak, net_offpeak=llm.net_bw_offpeak),
                       tfidf=dict(capture=tf.capture_bw, lo=tf.capture_bw_lo, hi=tf.capture_bw_hi, auroc=tf.auroc, ece=tf.ece, native_half_capture=tf.capture_native_half) if tf is not None else None,
                       tfidf_uni=dict(capture=tfu.capture_bw, auroc=tfu.auroc) if tfu is not None else None,
                       best_released=dict(signal=best.signal, capture=best.capture_bw, lo=best.capture_bw_lo, hi=best.capture_bw_hi, native_capture=best.capture_native_router, auroc=best.auroc) if best is not None else None,
                       ft27b_this_primary=dict(capture=ft27.capture_bw, lo=ft27.capture_bw_lo, hi=ft27.capture_bw_hi, native_capture=ft27.capture_native_router, auroc=ft27.auroc) if ft27 is not None else None)
            for nm, other in [("tfidf_lr", "tfidf_lr"), ("tfidf_lr_uni", "tfidf_lr_uni"), ("ft27b", "finetuned_router_qwen35_27b_v3|this_primary")] + ([("best_released", best.signal)] if best is not None else []):
                pdiff = paired(part, st, theta, m, "llm_p", other); ent[f"llm_minus_{nm}"] = dict(diff_capture=pdiff["diff"] / bc_ if bc_ > 1e-9 else NAN, lo_capture=pdiff["lo"] / bc_ if bc_ > 1e-9 else NAN, hi_capture=pdiff["hi"] / bc_ if bc_ > 1e-9 else NAN, decided=decided(pdiff["lo"], pdiff["hi"]))
            C2["comparisons"].append(ent)
if x12a_rows is not None:
    xa = x12a_rows[(x12a_rows.part == "all") & (x12a_rows.theta == 1.0) & (x12a_rows.primary.isin(PRIMARIES))]
    C2["x12a_reference"] = {f"{r.primary}|{r.signal}|{r.zdef}": dict(capture_bw=r.capture_bw, lo=r.capture_bw_lo, hi=r.capture_bw_hi, capture_native=r.capture_native, auroc=r.auroc, ece=r.ece, n_test=int(r.n_test)) for _, r in xa.iterrows()}
    C2["x12a_note"] = "X12-a values are on the full id_test (2,919 tasks, all 21 cells incl. non-exec sources); the rows above recompute the same released routers on X12-b's 600 sampled id_test tasks"
# qwen3-max second signal vs deepseek on the same tasks (paired), pooled and per source
C2["second_signal_qwen3max"] = []
for theta in THETAS:
    for m in PRIMARIES:
        for (part, st) in [("all", "ALL")] + [("source", s) for s in SOURCES]:
            q = row_of(part, st, theta, m, "qwen_p"); d = row_of(part, st, theta, m, "llm_p_sub"); e = row_of(part, st, theta, m, "llm_ensemble_mean")
            if q is None or d is None or not np.isfinite(q.capture_bw): continue
            pdiff = paired(part, st, theta, m, "llm_p_sub", "qwen_p"); pdiff_e = paired(part, st, theta, m, "llm_ensemble_mean", "llm_p_sub"); bc_ = q.bound_closed
            C2["second_signal_qwen3max"].append(dict(part=part, state=st, theta=theta, primary=m, n_test=int(q.n_test),
                                                    qwen3max=dict(capture=q.capture_bw, lo=q.capture_bw_lo, hi=q.capture_bw_hi, auroc=q.auroc, ece=q.ece, k_list_usd=q.k_sig, net=q.net_bw_peak, native_ab_capture=q.capture_native_ab, spearman_with_deepseek=q.spearman_with_deepseek_p),
                                                    deepseek_same_tasks=dict(capture=d.capture_bw, lo=d.capture_bw_lo, hi=d.capture_bw_hi, auroc=d.auroc, ece=d.ece, k_peak_usd=d.k_sig, net=d.net_bw_peak, native_ab_capture=d.capture_native_ab),
                                                    ensemble_mean=dict(capture=e.capture_bw, lo=e.capture_bw_lo, hi=e.capture_bw_hi, auroc=e.auroc) if e is not None else None,
                                                    deepseek_minus_qwen3max_capture=dict(diff=pdiff["diff"] / bc_ if bc_ > 1e-9 else NAN, lo=pdiff["lo"] / bc_ if bc_ > 1e-9 else NAN, hi=pdiff["hi"] / bc_ if bc_ > 1e-9 else NAN, decided=decided(pdiff["lo"], pdiff["hi"])),
                                                    ensemble_minus_deepseek_same_tasks_capture=dict(diff=pdiff_e["diff"] / bc_ if bc_ > 1e-9 else NAN, lo=pdiff_e["lo"] / bc_ if bc_ > 1e-9 else NAN, hi=pdiff_e["hi"] / bc_ if bc_ > 1e-9 else NAN, decided=decided(pdiff_e["lo"], pdiff_e["hi"]))))
results["C2_llm_vs_routers"] = C2

# ---- C3: within (state, primary): capture vs AUROC / switch share vs ECE across real signal variants
def within_block(sub, cap_col, x_col="auroc"):
    rs = []; groups = []
    for key, g in sub.groupby(["part", "state", "primary"]):
        g = g[np.isfinite(g[cap_col]) & np.isfinite(g[x_col]) & np.isfinite(g.ece)]
        if len(g) >= 5 and g[cap_col].nunique() > 1 and g[x_col].nunique() > 1 and g.ece.nunique() > 1:
            rs.append((spearman(g[cap_col], g[x_col])[0], spearman(g[cap_col], g.ece)[0])); groups.append("|".join(map(str, key)) + f" (n={len(g)})")
    if not rs: return dict(n_groups=0)
    rs = np.array(rs); ad = np.abs(rs[:, 0]) - np.abs(rs[:, 1]); d = rs[:, 0] - rs[:, 1]; rr = np.random.default_rng(SEED + 2); aboots = []; boots = []
    for _ in range(NBOOT):
        i = rr.integers(0, len(ad), len(ad)); aboots.append(float(np.nanmean(ad[i]))); boots.append(float(np.nanmean(d[i])))
    return dict(n_groups=int(len(rs)), x_col=x_col, mean_spearman_x=float(np.nanmean(rs[:, 0])), mean_spearman_ece=float(np.nanmean(rs[:, 1])), mean_abs_diff=float(np.nanmean(ad)), abs_diff_ci_group_boot=ci(aboots),
                mean_diff=float(np.nanmean(d)), diff_ci_group_boot=ci(boots), frac_groups_abs_x_gt_abs_ece=float(np.nanmean(np.abs(rs[:, 0]) > np.abs(rs[:, 1]))), groups=groups[:60])
def pooled_corr(sub, cap_col):
    ok = sub[np.isfinite(sub[cap_col]) & np.isfinite(sub.auroc) & np.isfinite(sub.ece)]
    return dict(n_rows=int(len(ok)), spearman_auroc=spearman(ok[cap_col], ok.auroc)[0], spearman_ece=spearman(ok[cap_col], ok.ece)[0], spearman_switch_share=spearman(ok[cap_col], ok.switch_share_bw)[0])
REALNS = REAL[~REAL.small]
C3 = {"within_state_primary_bw": within_block(REALNS, "capture_bw", "auroc"), "within_state_primary_bw_switch": within_block(REALNS, "capture_bw", "switch_share_bw"),
      "within_state_primary_native": within_block(REALNS.assign(capture_native=np.where(REALNS.signal_family.str.startswith("released"), REALNS.capture_native_router, np.where(REALNS.signal == "llm_label", REALNS.capture_native_label, REALNS.capture_native_ab))), "capture_native", "auroc"),
      "within_llm_tfidf_only_bw": within_block(REALNS[REALNS.signal_family.isin(["llm", "llm2", "tfidf"])], "capture_bw", "auroc"),
      "within_llm_tfidf_only_bw_switch": within_block(REALNS[REALNS.signal_family.isin(["llm", "llm2", "tfidf"])], "capture_bw", "switch_share_bw"),
      "pooled_rows_bw": pooled_corr(REALNS, "capture_bw"), "pooled_rows_llm_p_only_bw": pooled_corr(REALNS[REALNS.signal == "llm_p"], "capture_bw"),
      "note": "variants per (state, primary) on the full 600/600 sample: llm_p, llm_label, llm_lines, tfidf_lr, tfidf_lr_uni + 10 released routers x 2 Z-definitions (binary, cross-fitted); the qwen3-max family lives on the 200/200 subsample and is excluded here; semi-synthetic variants excluded; small states (n < 20) excluded"}
# the shrunk check: same AUROC and Blackwell capture, different ECE and native capture
sh = main[main.signal == "llm_p_shrunk"].set_index(["part", "state", "primary"]); bs = main[main.signal == "llm_p"].set_index(["part", "state", "primary"]).loc[sh.index]
C3["shrunk_distortion_check"] = dict(n_states=int(len(sh)), max_abs_diff_auroc=float((sh.auroc - bs.auroc).abs().max()), max_abs_diff_capture_bw=float(np.nanmax((sh.capture_bw - bs.capture_bw).abs())),
                                     mean_ece_real=float(bs.ece.mean()), mean_ece_shrunk=float(sh.ece.mean()), mean_capture_native_ab_real=float(np.nanmean(bs.capture_native_ab)), mean_capture_native_ab_shrunk=float(np.nanmean(sh.capture_native_ab)),
                                     n_states_shrunk_native_ab_ci_below_0=int((sh.capture_native_ab_hi < 0).sum()), mean_frac_attempt_native_ab_shrunk=float(sh.frac_attempt_native_ab.mean()), mean_frac_attempt_native_ab_real=float(bs.frac_attempt_native_ab.mean()))
results["C3_decision_usefulness_vs_ece"] = C3

# ---- C4: re-thresholding (native a/b rule vs Blackwell / probing-tuned threshold)
def c4_block(sub):
    ok = sub[np.isfinite(sub.capture_bw) & np.isfinite(sub.capture_native_ab)]
    return dict(n_rows=int(len(ok)), mean_capture_native_ab=float(ok.capture_native_ab.mean()), mean_capture_bw=float(ok.capture_bw.mean()), mean_capture_rethr=float(np.nanmean(ok.capture_rethr)) if "capture_rethr" in ok and np.isfinite(ok.capture_rethr).any() else NAN,
                n_bw_minus_native_ab_ci_above_0=int((ok.bw_minus_native_ab_lo > 0).sum()), n_bw_minus_native_ab_ci_below_0=int((ok.bw_minus_native_ab_hi < 0).sum()), n_native_ab_ci_below_0=int((ok.capture_native_ab_hi < 0).sum()),
                n_bw_ci_above_0=int((ok.capture_bw_lo > 0).sum()), n_bw_ci_below_0=int((ok.capture_bw_hi < 0).sum()), mean_frac_attempt_native_ab=float(ok.frac_attempt_native_ab.mean()), mean_frac_attempt_bw=float(ok.frac_attempt_bw.mean()))
C4 = {"llm_p_states_theta1": c4_block(main[(main.signal == "llm_p") & (~main.small)]), "llm_p_shrunk_states_theta1": c4_block(main[(main.signal == "llm_p_shrunk") & (~main.small)]),
      "qwen_p_states_theta1": c4_block(sig[(sig.signal == "qwen_p") & (sig.theta == 1.0) & (~sig.small) & (sig.part != "all")]), "qwen_p_shrunk_states_theta1": c4_block(sig[(sig.signal == "qwen_p_shrunk") & (sig.theta == 1.0) & (~sig.small) & (sig.part != "all")]),
      "tfidf_lr_states_theta1": c4_block(main[(main.signal == "tfidf_lr") & (~main.small)]), "pooled": []}
for theta in THETAS:
    for m in PRIMARIES:
        for s_ in ["llm_p", "llm_p_shrunk", "qwen_p", "qwen_p_shrunk", "llm_ensemble_mean", "tfidf_lr"]:
            r = row_of("all", "ALL", theta, m, s_)
            if r is None: continue
            C4["pooled"].append(dict(theta=theta, primary=m, signal=s_, s=r.s, a_over_b=r.a_over_b, ece_raw=r.ece, auroc=r.auroc, mean_p=r.get("mean_p_test", NAN), frac_attempt_native_ab=r.frac_attempt_native_ab, frac_attempt_bw=r.frac_attempt_bw,
                                     capture_native_ab=r.capture_native_ab, capture_native_ab_lo=r.capture_native_ab_lo, capture_native_ab_hi=r.capture_native_ab_hi, capture_native_half=r.capture_native_half,
                                     capture_bw=r.capture_bw, capture_bw_lo=r.capture_bw_lo, capture_bw_hi=r.capture_bw_hi, capture_rethr=r.get("capture_rethr", NAN), t_star=r.get("t_star_probing", NAN),
                                     bw_minus_native_ab=r.bw_minus_native_ab / r.bound_closed if r.bound_closed > 1e-9 else NAN, bw_minus_native_ab_lo=r.bw_minus_native_ab_lo / r.bound_closed if r.bound_closed > 1e-9 else NAN,
                                     bw_minus_native_ab_hi=r.bw_minus_native_ab_hi / r.bound_closed if r.bound_closed > 1e-9 else NAN, capture_bw_pooledcal=r.get("capture_bw_pooledcal", NAN), capture_bw_statecal=r.get("capture_bw_statecal", NAN)))
results["C4_rethresholding"] = C4

# ---- C5: cor:info-vs-proposer
def c5_summary(sub, pc="pred_info_better", rc="real_info_better"):
    dec = sub[sub.decided]
    out = dict(n_rows=int(len(sub)), n_decided=int(len(dec)), agreement_all=float((sub[pc] == sub[rc]).mean()) if len(sub) else NAN, agreement_decided=float((dec[pc] == dec[rc]).mean()) if len(dec) else NAN,
               baseline_always_info=float(dec[rc].mean()) if len(dec) else NAN, baseline_always_upgrade=float((~dec[rc]).mean()) if len(dec) else NAN,
               baseline_info_iff_dp_le_0=float(((dec.delta_p <= 0) == dec[rc]).mean()) if len(dec) else NAN,
               confusion_decided={"info/info": int((dec[pc] & dec[rc]).sum()), "upg/upg": int((~dec[pc] & ~dec[rc]).sum()), "info/upg": int((dec[pc] & ~dec[rc]).sum()), "upg/info": int((~dec[pc] & dec[rc]).sum())})
    dp = dec[dec.delta_p > 0]; out["decided_delta_p_positive"] = dict(n=int(len(dp)), agreement=float((dp[pc] == dp[rc]).mean()) if len(dp) else NAN, baseline_always_info=float(dp[rc].mean()) if len(dp) else NAN)
    if "next_stronger" in dec:
        ns = dec[dec.next_stronger]; out["decided_next_stronger"] = dict(n=int(len(ns)), agreement=float((ns[pc] == ns[rc]).mean()) if len(ns) else NAN, baseline_always_info=float(ns[rc].mean()) if len(ns) else NAN)
    return out
iv_main = ivp[(ivp.theta == 1.0) & (ivp.part != "all") & (ivp.upgrade != "opus_direct") & (~ivp.small)]
C5 = {"exact_info_theta1": c5_summary(iv_main), "exact_info_theta0.5": c5_summary(ivp[(ivp.theta == 0.5) & (ivp.part != "all") & (ivp.upgrade != "opus_direct") & (~ivp.small)]),
      "exact_info_theta1_incl_small": c5_summary(ivp[(ivp.theta == 1.0) & (ivp.part != "all") & (ivp.upgrade != "opus_direct")]),
      "actual_signal_theta1": c5_summary(ivp2[(ivp2.theta == 1.0) & (ivp2.part != "all") & (~ivp2.small)], "pred_signal_better", "real_signal_better"),
      "actual_signal_theta1_pooled_rows": ivp2[(ivp2.theta == 1.0) & (ivp2.part == "all")][["primary", "upgrade", "s", "s_upgrade", "delta_p", "k_prime", "k_sig", "pred_signal_minus_upgrade", "real_signal_minus_upgrade", "real_lo", "real_hi", "decided"]].to_dict("records"),
      "opus_direct_rows_pooled": ivp[(ivp.part == "all") & (ivp.upgrade == "opus_direct")][["theta", "primary", "s", "a", "b", "pred_info_minus_dec", "real_info_minus_dec", "real_lo", "real_hi"]].to_dict("records"),
      "pooled_rows_theta1": ivp[(ivp.part == "all") & (ivp.theta == 1.0) & (ivp.upgrade != "opus_direct")][["primary", "upgrade", "s", "a", "b", "s_upgrade", "delta_p", "k_prime", "pred_info_minus_dec", "real_info_minus_dec", "real_lo", "real_hi", "decided", "next_stronger"]].to_dict("records")}
results["C5_info_vs_proposer"] = C5

# ---- C6: k < a
kc = kcond[~kcond.small]
def kc_block(sub):
    out = {}
    for m in ALL_PRIMARIES:
        g = sub[sub.primary == m]
        out[m] = dict(n_states=int(len(g)), mean_a_usd=float(g.a_usd.mean()), mean_k_peak=float(g.k_peak.mean()), mean_k_offpeak=float(g.k_offpeak.mean()), mean_k_self=float(g.k_self.mean()),
                      n_k_peak_ge_a=int((~g.k_peak_lt_a).sum()), n_k_offpeak_ge_a=int((~g.k_offpeak_lt_a).sum()), n_k_self_ge_a=int((~g.k_self_lt_a).sum()),
                      n_k_peak_ge_a_1ms=int((~g.k_peak_lt_a_1ms).sum()), n_k_offpeak_ge_a_1ms=int((~g.k_offpeak_lt_a_1ms).sum()),
                      n_band_empty_at_k_peak=int((g.k_peak >= g.band_nonempty_max_k_usd).sum()), n_band_empty_at_k_offpeak=int((g.k_offpeak >= g.band_nonempty_max_k_usd).sum()),
                      n_k_tok_ge_a_tok=int((~g.k_tok_lt_a_tok).sum()), n_k_sec_ge_a_sec=int((~g.k_sec_lt_a_sec).sum()), mean_k_over_a_peak=float((g.k_peak / g.a_usd).mean()), mean_k_over_a_tok=float((g.k_tok / g.a_tok).mean()), mean_k_over_a_sec=float((g.k_sec / g.a_sec).mean()),
                      mean_k_over_b_peak=float((g.k_peak / g.b_usd).mean()))
    return out
C6 = {"non_small_states": kc_block(kc), "source_states": kc_block(kcond[kcond.part == "source"]), "pooled": kcond[kcond.part == "all"][["primary", "s", "a_usd", "b_usd", "a_over_b", "k_peak", "k_offpeak", "k_self", "k_units", "a_tok", "k_tok", "a_sec", "k_sec", "k_peak_lt_a", "k_offpeak_lt_a", "k_tok_lt_a_tok", "k_sec_lt_a_sec", "kq_list", "kq_list_half", "kq_tok", "kq_sec", "kq_list_lt_a", "kq_list_half_lt_a", "kq_tok_lt_a_tok", "kq_sec_lt_a_sec"]].to_dict("records"),
      "per_source_qwen3max": kcond[(kcond.part == "source") & (kcond.primary == "Qwen3-Max")][["state", "s", "a_usd", "b_usd", "k_peak", "k_offpeak", "k_self", "a_tok", "k_tok", "a_sec", "k_sec", "k_peak_lt_a", "k_offpeak_lt_a", "k_tok_lt_a_tok", "k_sec_lt_a_sec", "kq_list", "kq_list_half", "kq_tok", "kq_sec", "kq_list_lt_a", "kq_tok_lt_a_tok", "kq_sec_lt_a_sec"]].to_dict("records")}
# qwen3-max signal cost vs a
C6["qwen3max_signal_pooled"] = dict(k_list_usd=float(q_.k_list.mean()), k_list_half_usd=float(q_.k_list_half.mean()), k_self_usd=float(q_.k_self.mean()), k_tok=float(q_.total_tokens.mean()), k_sec=float(q_.latency_s.mean()))
C6["qwen3max_non_small_states"] = {m: dict(n_states=int((kc.primary == m).sum()), n_kq_list_ge_a=int((~kc[kc.primary == m].kq_list_lt_a).sum()), n_kq_list_half_ge_a=int((~kc[kc.primary == m].kq_list_half_lt_a).sum()),
                                           n_kq_tok_ge_a_tok=int((~kc[kc.primary == m].kq_tok_lt_a_tok).sum()), n_kq_sec_ge_a_sec=int((~kc[kc.primary == m].kq_sec_lt_a_sec).sum()), mean_kq_over_a_list=float((kc[kc.primary == m].kq_list / kc[kc.primary == m].a_usd).mean())) for m in ALL_PRIMARIES}
results["C6_k_vs_a"] = C6

# ---- band positions (prop:voi) with the measured k
bnd = band[(band.theta == 1.0) & (band.part == "source") & (~band.small)]
def band_agree(sub):
    d = sub[sub.decided]; return dict(n=int(len(sub)), n_decided=int(len(d)), n_s_in_band=int(sub.pred_info_in_band.sum()), n_band_nonempty=int(sub.band_nonempty.sum()),
                                      agreement_decided=float((d.pred_info_in_band == (d.real_argmin == "info")).mean()) if len(d) else NAN, argmin_agreement_decided=float((d.pred_argmin == d.real_argmin).mean()) if len(d) else NAN,
                                      baseline_nonempty_band=float((d.band_nonempty == (d.real_argmin == "info")).mean()) if len(d) else NAN)
results["band_prop_voi"] = {f"source|k={kn}": band_agree(bnd[bnd.k_name == kn]) for kn in ["0", "peak", "offpeak", "self", "tok054"]}
results["band_prop_voi"]["cell_theta1"] = {f"k={kn}": band_agree(band[(band.theta == 1.0) & (band.part == "cell") & (~band.small) & (band.k_name == kn)]) for kn in ["peak", "offpeak"]}
results["band_pooled"] = band[band.part == "all"][["theta", "primary", "k_name", "k", "s", "a_over_b", "band_lo", "band_hi", "band_nonempty", "u", "pred_argmin", "real_argmin", "info_minus_best_uninformed", "lo", "hi", "decided"]].to_dict("records")
results["band_source_rows_peak"] = band[(band.part == "source") & (band.k_name == "peak") & (band.theta == 1.0)][["state", "primary", "n_test", "s", "a_over_b", "band_lo", "band_hi", "band_nonempty", "u", "pred_argmin", "real_argmin", "decided"]].to_dict("records")

# ---- counterfactual price ratio summary (SEMI-SYNTHETIC)
cfs = cf[(cf.part == "source")]; cfp = cf[cf.part == "all"]
def cf_agg(sub):
    out = {}
    for (sname, uname, r_), g in sub.groupby(["signal", "usage", "r_target"]):
        g = g[np.isfinite(g.capture)]; W = g.n_test.to_numpy(float)
        out[f"{sname}|{uname}|r={r_:g}"] = dict(n_states=int(len(g)), capture_taskweighted=float((W * g.gross).sum() / (W * g.bound_cf).sum()) if (W * g.bound_cf).sum() > 0 else NAN, mean_switch_share=float(g.switch_share.mean()),
                                              n_states_ci_above_0=int((g.gross_lo > 0).sum()), n_states_ci_below_0=int((g.gross_hi < 0).sum()), mean_frac_attempt=float(g.frac_attempt.mean()))
    return out
results["counterfactual_price_ratio"] = dict(note="SEMI-SYNTHETIC: cost_B rescaled per state so that a/b equals r (probing means); posteriors unchanged; realized on id_test with the rescaled reference cost",
                                              source_states_theta1=cf_agg(cfs), pooled_theta1={f"{r.signal}|{r.usage}|{r.primary}|r={r.r_target:g}": dict(capture=r.capture, lo=r.capture_lo, hi=r.capture_hi, switch_share=r.switch_share, frac_attempt=r.frac_attempt, u0_attempt=bool(r.u0_attempt_cf)) for _, r in cfp.iterrows()})
# ---- permutation (uninformative signal) summary
pm = main[main.signal == "llm_p_perm"]; pm_ok = pm[np.isfinite(pm.capture_bw)]
results["permuted_signal"] = dict(n_rows=int(len(pm_ok)), mean_capture=float(pm_ok.capture_bw.mean()), median_abs_capture=float(pm_ok.capture_bw.abs().median()), n_ci_above_0=int((pm_ok.gross_bw_lo > 0).sum()), n_ci_below_0=int((pm_ok.gross_bw_hi < 0).sum()),
                                  mean_auroc_permuted=float(pm_ok.auroc_permuted.mean()), mean_capture_real_same_rows=float(main[main.signal == "llm_p"].set_index(["part", "state", "primary"]).loc[pm_ok.set_index(["part", "state", "primary"]).index].capture_bw.mean()))

# ---- hypothesis violations
llm_main = main[(main.signal == "llm_p") & (main.part != "all")]
hv = dict(posterior_mean_vs_prior=dict(mean_abs_anchor_shift=float(llm_main.anchor_shift.abs().mean()), max_abs_anchor_shift=float(llm_main.anchor_shift.abs().max()), mean_abs_posterior_mean_minus_s_test=float((llm_main.mean_posterior_test - llm_main.s_test).abs().mean())),
          prior_drift_probing_to_id_test=dict(mean_abs_s_minus_s_test=float((llm_main.s - llm_main.s_test).abs().mean()), max_abs=float((llm_main.s - llm_main.s_test).abs().max()), n_states_u0_flips=int(((llm_main.s > llm_main.a_over_b) != (llm_main.s_test > llm_main.a_test / llm_main.b_test)).sum()), n_states=int(len(llm_main))),
          calibration=dict(mean_ece_raw=float(llm_main.ece.mean()), mean_ece_posterior_anchored=float(llm_main.ece_posterior_anchored.mean()), mean_ece_posterior_pooled=float(llm_main.ece_posterior_pooled.mean()),
                           qwen3max_mean_ece_raw_subsample=float(sig[(sig.signal == "qwen_p") & (sig.theta == 1.0) & (sig.part == "source")].ece.mean())),
          lcb_data_gap="28 lcb recorded responses are empty although the released score is 1 (infra Addendum 2); truth here is the released score, so these tasks are kept",
          cost_ordering_0_lt_a_lt_b=C1["frac_tasks_primary_costlier_than_reference"],
          value_preservation={f"theta{t}": dict(n_rows=int(len(g)), frac_V_bw_below_V_direct=float((g.V_bw < g.V_direct - 1e-12).mean()), mean_e_bw=float(g.e_bw.mean())) for t, g in sig[sig.signal == "llm_p"].groupby("theta")},
          constant_k=dict(cv_k_peak_across_tasks=diag["k_per_task_usd"]["cv_peak"], note="k varies with prompt length; the state mean enters net VOI"),
          truth_caveats="lcb (private tests; sandbox agreement 0.73-0.92) and ds1000 (0.65) rely on the released score as truth; hex non-python and dim9 scores are partial-credit / proxy scores (theta=1 treats <1 as insufficient)",
          signal_model_thinking="deepseek-v4-pro with thinking disabled (cost); the forecast is a single forward pass")
results["hypothesis_violations"] = hv

# ---- claims
claims = []
c1 = C1["all_rows"]; nat_pt = sum(v["n_point_violations"] for v in c1["native"].values()); nat_ci = sum(v["n_ci_violations"] for v in c1["native"].values())
claims.append(dict(id="C1", statement="No realized gross VOI of any usage of the LLM signal (or of any other signal valued here) exceeds the perfect-information bound min{s(b-a), a(1-s)} from probing; net VOI <= bound - k",
                   kind="IDENTITY-like inequality on expectations, realized OUT-OF-SAMPLE (probing primitives, id_test outcomes; LIVE signal)",
                   result=dict(n_rows_tested=c1["n_rows_tested"], bw_point_violations=c1["bw"]["n_point_violations"], bw_ci_violations=c1["bw"]["n_ci_violations"], bw_max_ratio=c1["bw"]["max_ratio"],
                               native_point_violations=nat_pt, native_ci_violations=nat_ci, llm_p_only=C1["llm_p_only"]["bw"], point_violation_rows=len(c1["point_violations"])),
                   verdict=("PASS" if c1["bw"]["n_ci_violations"] + nat_ci == 0 and c1["bw"]["n_point_violations"] + nat_pt == 0 else "PASS (qualified: point violations without CI support)" if c1["bw"]["n_ci_violations"] + nat_ci == 0 else "FAIL")))
A1 = C2["aggregate_source_theta1"]; a_tf = A1["llm_minus_tfidf_lr"]; a_rel = A1["llm_minus_released"]; a_cap = A1["capture"]
best_rel = max([k for k in a_rel if a_rel[k].get("n_states", 0) > 0], key=lambda k: a_cap[k]["capture"] if np.isfinite(a_cap[k].get("capture", NAN)) else -np.inf) if a_rel else None
def _cmp_verdict():
    if a_tf.get("n_states", 0) == 0: return "NOT-INSTANTIABLE"
    tf_up = a_tf["diff_lo"] > 0; tf_dn = a_tf["diff_hi"] < 0
    rel_not_below = (best_rel is None) or (a_rel[best_rel]["diff_hi"] >= 0)
    if tf_up and rel_not_below: return "PASS"
    if tf_up: return "PASS (qualified: beats TF-IDF; CI-below the best released router)"
    if tf_dn: return "FAIL"
    return "INCONCLUSIVE"
claims.append(dict(id="C2", statement="Valued state by state (7 sources, theta=1, both primaries, task-weighted), the LLM signal captures more of the perfect-information value than a TF-IDF logistic router trained on the same 600 probing prompts, and is not below the released routers valued on the same id_test tasks",
                   kind="OUT-OF-SAMPLE (LIVE signal; paired task bootstrap on the same resamples, aggregated over disjoint source states)",
                   result=dict(llm_capture_source_agg=a_cap["llm_p"], tfidf_capture_source_agg=a_cap["tfidf_lr"], llm_minus_tfidf=a_tf, llm_minus_tfidf_uni=A1["llm_minus_tfidf_lr_uni"], best_released=best_rel,
                               best_released_capture=a_cap.get(best_rel), llm_minus_best_released=a_rel.get(best_rel), llm_minus_ft27b_this_primary=a_rel.get("finetuned_router_qwen35_27b_v3|this_primary"),
                               llm_minus_perm=A1["llm_minus_perm"], by_primary=A1["by_primary"], pooled_state=[c for c in C2["comparisons"] if c["part"] == "all" and c["theta"] == 1.0],
                               second_signal_agg=dict(qwen3max=a_cap["qwen_p"], deepseek_same_tasks=a_cap["llm_p_sub"], deepseek_minus_qwen3max=A1["deepseek_minus_qwen3max_subsample"], ensemble=a_cap["llm_ensemble_mean"], ensemble_minus_deepseek=A1["ensemble_minus_deepseek_subsample"]),
                               x12a_ft27b_full_id_test={k: v for k, v in C2["x12a_reference"].items() if "27b" in k}),
                   verdict=_cmp_verdict(), verdict_rule="task-weighted paired aggregate over the 7 source states (theta=1, both primaries): PASS if the CI of gross_bw(llm_p) - gross_bw(tfidf_lr) is above 0 and llm_p is not CI-below the best released router; FAIL if the TF-IDF difference is CI-below 0; else INCONCLUSIVE"))
w = C3["within_state_primary_bw"]; wsw = C3["within_state_primary_bw_switch"]
claims.append(dict(id="C3", statement="Within (state, primary), the Blackwell capture ratio follows decision usefulness (AUROC; switch share) more than calibration (ECE) across signal variants",
                   kind="OUT-OF-SAMPLE", result=dict(auroc_vs_ece=w, switch_share_vs_ece=wsw, native_usage=C3["within_state_primary_native"], llm_tfidf_only=C3["within_llm_tfidf_only_bw"], shrunk_check=C3["shrunk_distortion_check"], pooled_rows=C3["pooled_rows_bw"]),
                   verdict=("PASS" if w.get("n_groups", 0) >= 5 and w["abs_diff_ci_group_boot"][0] > 0 else "FAIL" if w.get("n_groups", 0) >= 5 and w["abs_diff_ci_group_boot"][1] < 0 else "INCONCLUSIVE"),
                   verdict_rule="group bootstrap CI of mean(|rho(capture_bw, AUROC)| - |rho(capture_bw, ECE)|) over (state, primary) groups with >= 5 real variants: PASS if above 0, FAIL if below 0"))
p4 = [r for r in C4["pooled"] if r["theta"] == 1.0]
st4 = sig[(sig.signal.isin(["llm_p", "llm_p_shrunk"])) & (sig.theta == 1.0) & (sig.part == "source") & (~sig.small)]
_real = st4[st4.signal == "llm_p"].set_index(["state", "primary"]); _shr = st4[st4.signal == "llm_p_shrunk"].set_index(["state", "primary"]).loc[_real.index]
inv = float(np.nanmax((_real.capture_bw - _shr.capture_bw).abs())) if len(_real) else NAN
bites = _shr[(_shr.frac_attempt_native_ab < _real.frac_attempt_native_ab - 1e-9)]
C4["invariance_and_recovery_source_theta1"] = dict(n_states=int(len(_real)), max_abs_diff_capture_bw_real_vs_shrunk=inv, n_states_shrunk_native_bites=int(len(bites)), n_bites_bw_ci_above_native=int((bites.bw_minus_native_ab_lo > 0).sum()),
                                                   n_real_bw_ci_below_native=int((_real.bw_minus_native_ab_hi < 0).sum()), n_real_bw_ci_above_native=int((_real.bw_minus_native_ab_lo > 0).sum()),
                                                   agg_bw_minus_native_shrunk=C2["aggregate_source_theta1"]["bw_minus_native_ab_shrunk"], agg_bw_minus_native_real=C2["aggregate_source_theta1"]["bw_minus_native_ab_llm"],
                                                   agg_rethr_minus_native_real=C2["aggregate_source_theta1"]["rethr_minus_native_ab_llm"], agg_bw_minus_rethr_real=C2["aggregate_source_theta1"]["bw_minus_rethr_llm"])
R4 = C4["invariance_and_recovery_source_theta1"]
ok_inv = np.isfinite(inv) and inv < 1e-6; ok_shr = R4["agg_bw_minus_native_shrunk"].get("diff_lo", NAN) > 0; ok_real = R4["n_real_bw_ci_below_native"] == 0
claims.append(dict(id="C4", statement="Re-thresholding recovers the value a miscalibrated score loses under the native rule 'attempt iff s_hat >= a/b': the Blackwell usage (calibrated posterior at a/b) is invariant to the monotone shrinkage p/10, beats the native rule on the shrunk score (paired, task-weighted over source states), and is never CI-below the native rule on the real LLM score",
                   kind="OUT-OF-SAMPLE (LIVE signal); shrunk variant SEMI-SYNTHETIC", result=dict(source_states=R4, pooled_theta1=p4, states=C4["llm_p_states_theta1"], states_shrunk=C4["llm_p_shrunk_states_theta1"], states_tfidf=C4["tfidf_lr_states_theta1"], states_qwen=C4["qwen_p_states_theta1"]),
                   verdict=("PASS" if ok_inv and ok_shr and ok_real else "PASS (qualified: invariance holds and Blackwell never CI-below native, but the shrunk-score recovery is not CI-decided)" if ok_inv and ok_real else "FAIL"),
                   verdict_rule="source states, theta=1, both primaries: (i) max |capture_bw(real) - capture_bw(p/10)| < 1e-6; (ii) task-weighted paired CI of gross_bw - gross_native_ab for the shrunk score above 0; (iii) the real score's Blackwell usage is CI-below its native a/b usage in no state"))
p5 = C5["exact_info_theta1"]; p5b = C5["actual_signal_theta1"]
claims.append(dict(id="C5", statement="cor:info-vs-proposer: buy routing information rather than upgrade the primary (Qwen3-Max / kimi-k2.5 -> a costlier recorded primary) iff b*Delta p < a(1-s); sign agreement on decided rows; also the actual-signal version [k + C_BW] vs the upgraded cascade",
                   kind="OUT-OF-SAMPLE", result=dict(exact_info=p5, actual_signal=p5b, theta0_5=C5["exact_info_theta0.5"]["agreement_decided"], opus_direct_pooled=C5["opus_direct_rows_pooled"]),
                   verdict=("PASS" if p5["n_decided"] >= 10 and p5["agreement_decided"] >= 0.85 and p5["agreement_decided"] > max(p5["baseline_always_info"], p5["baseline_always_upgrade"], p5["baseline_info_iff_dp_le_0"])
                            else "PASS (qualified: >= 0.85 but not above the best naive baseline)" if p5["n_decided"] >= 10 and p5["agreement_decided"] >= 0.85 else "INCONCLUSIVE" if p5["n_decided"] < 10 else "FAIL")))
k6 = C6["non_small_states"]
claims.append(dict(id="C6", statement="The LLM signal is affordable, k < a, in every state for the primaries it forecasts (Qwen3-Max, kimi-k2.5); for cheaper primaries (qwen3.5-plus) it may cost more than the attempt",
                   kind="LIVE measurement (signal tokens x list price; tokens; seconds) against probing a",
                   result={m: dict(n_states=k6[m]["n_states"], n_k_peak_ge_a=k6[m]["n_k_peak_ge_a"], n_k_offpeak_ge_a=k6[m]["n_k_offpeak_ge_a"], n_k_self_ge_a=k6[m]["n_k_self_ge_a"], n_k_tok_ge_a_tok=k6[m]["n_k_tok_ge_a_tok"], n_k_sec_ge_a_sec=k6[m]["n_k_sec_ge_a_sec"],
                                   mean_k_over_a_peak=k6[m]["mean_k_over_a_peak"], mean_a_usd=k6[m]["mean_a_usd"], mean_k_peak=k6[m]["mean_k_peak"]) for m in ALL_PRIMARIES},
                   verdict=("PASS" if all(k6[m]["n_k_peak_ge_a"] == 0 for m in PRIMARIES) else "FAIL (at the peak list price)" if any(k6[m]["n_k_offpeak_ge_a"] > 0 for m in PRIMARIES) else "PASS (qualified: only at off-peak price)"),
                   verdict_rule="PASS iff k_peak < a in every non-small state for Qwen3-Max and kimi-k2.5; qwen3.5-plus reported"))
results["claims"] = claims

# ================================================================================================ G. figures
FAM = {"llm": (BLUE, "o", "LLM signal (deepseek-v4-pro): p, label, lines"), "tfidf": (GREEN, "^", "TF-IDF LR on probing prompts"), "released_finetuned": (PURPLE, "s", "released fine-tuned Qwen routers"),
       "released_classical": (ORANGE, "D", "released classical routers"), "released_lookup": (ORANGE, "x", "ACRouter release lookup")}
def scatter_fam(ax, sub, xcol, ycol):
    for fam, (colr, mk, lab) in FAM.items():
        g = sub[sub.signal_family == fam]
        if len(g): ax.scatter(g[xcol], g[ycol], s=10, color=colr, marker=mk, alpha=.65, linewidths=(.8 if mk == "x" else 0), label=lab)
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), layout="constrained")
for ax, (xc, xl) in zip(axes, [("auroc", "AUROC of Z against sufficiency (id_test)"), ("switch_share_bw", "switch share of the Blackwell rule"), ("ece", "ECE of Z (id_test)")]):
    scatter_fam(ax, REALNS[np.isfinite(REALNS[xc])], xc, "capture_bw"); ax.axhline(0, color=GREY, lw=.6); ax.axhline(1, color=GREY, lw=.6, ls=":"); ax.set_yscale("symlog", linthresh=0.1)
    r_ = spearman(REALNS.capture_bw, REALNS[xc])[0]; ax.text(0.02, 0.97, fr"$\rho={r_:.2f}$", transform=ax.transAxes, va="top", fontsize=7.5); ax.set_xlabel(xl)
    if xc == "auroc": ax.axvline(0.5, color=GREY, lw=.6, ls=":")
axes[0].set_ylabel("Blackwell capture ratio (symlog)")
hh, ll = axes[0].get_legend_handles_labels(); fig.legend(hh, ll, loc="outside lower center", ncol=3, frameon=False, fontsize=6.5, handletextpad=.2, columnspacing=1.0)
fig.savefig(os.path.join(FIG, "fig1_capture_vs_auroc_ece.pdf"), metadata={"Creator": "X12b run.py"}); plt.close(fig)
# Fig 2: band positions per source state with k = measured signal cost (peak); realized cheapest
CH = {"cascade": (BLUE, "o", "cascade cheapest"), "direct": (ORANGE, "s", "direct cheapest"), "info": (PURPLE, "D", "exact information cheapest")}
fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.2), layout="constrained")
for ax, kn in zip(axes, ["peak", "offpeak"]):
    sub = band[(band.part == "source") & (band.theta == 1.0) & (band.k_name == kn)].sort_values(["primary", "a_over_b"]).reset_index(drop=True)
    for y, r in sub.iterrows():
        if r.band_nonempty: ax.plot([max(r.band_lo, 0), min(r.band_hi, 1)], [y, y], color=GREY, lw=2.2, alpha=.35, solid_capstyle="butt")
        ax.plot([r.a_over_b], [y], marker="|", color=GREY, ms=6, mew=.8); colr, mk, _ = CH[r.real_argmin]; ax.plot([r.s], [y], marker=mk, color=colr, ms=4, mew=0, ls="none", alpha=.9 if r.decided else .45)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels([f"{r.state} | {r.primary}" for _, r in sub.iterrows()], fontsize=6); ax.set_xlim(0, 1); ax.invert_yaxis(); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(fr"$s$ (probing); band at measured $k$ ({'peak' if kn == 'peak' else 'off-peak'}); tick $a/b$")
h = [plt.Line2D([], [], marker=mk, color=colr, ls="none", ms=4, label=lab) for colr, mk, lab in CH.values()]; fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, "fig2_band_positions.pdf"), metadata={"Creator": "X12b run.py"}); plt.close(fig)
# Fig 3: ROC of the LLM signal vs the TF-IDF router (pooled id_test, theta=1), one panel per primary
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), layout="constrained")
src_t = smp.loc[T_IDX, "source"].to_numpy()
for ax, m in zip(axes, PRIMARIES):
    H = SC_T[m] >= THETA_MAIN
    for z, lab, colr, ls in [(Z_LLM[m][1], f"deepseek-v4-pro p (AUROC {auroc_cont(Z_LLM[m][1], H):.2f})", BLUE, "-"), (TFIDF[(THETA_MAIN, m, 'tfidf_lr')][1], f"TF-IDF LR (AUROC {auroc_cont(TFIDF[(THETA_MAIN, m, 'tfidf_lr')][1], H):.2f})", GREEN, "-"),
                              (Z_LIN[1], f"-expected lines (AUROC {auroc_cont(Z_LIN[1], H):.2f})", GREY, "--")]:
        fpr, tpr, _ = roc_curve(H, z); ax.plot(fpr, tpr, color=colr, ls=ls, label=lab)
    fpr, tpr, _ = roc_curve(H, Z_LAB[1]); ax.plot(fpr, tpr, color=ORANGE, marker="o", ms=3, lw=1, label=f"difficulty label (AUROC {auroc_cont(Z_LAB[1], H):.2f})")
    zq = Z_QW[m][1][IN_SUB_T]; Hq = H[IN_SUB_T]
    if 0 < Hq.sum() < len(Hq): fpr, tpr, _ = roc_curve(Hq, zq); ax.plot(fpr, tpr, color=PURPLE, ls=":", label=f"qwen3-max p, {int(IN_SUB_T.sum())}-task subsample (AUROC {auroc_cont(zq, Hq):.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=.5, ls=":"); ax.set_xlabel(f"false positive rate (attempt on an insufficient task)\nprimary {m}, $\\theta=1$, pooled id_test (n={len(H)})"); ax.legend(loc="lower right", frameon=False, fontsize=6)
axes[0].set_ylabel("true positive rate (attempt on a sufficient task)")
fig.savefig(os.path.join(FIG, "fig3_roc_llm_vs_tfidf.pdf"), metadata={"Creator": "X12b run.py"}); plt.close(fig)
# Fig 4: reliability of the raw LLM probability and of the isotonic posterior (pooled, theta=1)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.7), layout="constrained")
for ax, m in zip(axes, PRIMARIES):
    H = (SC_T[m] >= THETA_MAIN).astype(float); z = Z_LLM[m][1]; post = ISO[(THETA_MAIN, m, "llm_p")].predict(z)
    for zz, lab, colr in [(z, "raw LLM probability", BLUE), (post, "isotonic posterior (fitted on probing)", PURPLE)]:
        bidx, nb = ece_bins(zz, 10); xs = [zz[bidx == j].mean() for j in range(nb) if (bidx == j).any()]; ys = [H[bidx == j].mean() for j in range(nb) if (bidx == j).any()]
        ax.plot(xs, ys, marker="o", ms=3, color=colr, label=f"{lab} (ECE {ece_cont(zz, H > 0):.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=.5, ls=":"); ax.set_xlabel(f"forecast (primary {m}; grey: histogram of the raw probability)"); ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ax2 = ax.twinx(); ax2.hist(z, bins=20, range=(0, 1), color=GREY, alpha=.2); ax2.set_yticks([]); ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
axes[0].set_ylabel("observed sufficiency rate (id_test)")
fig.savefig(os.path.join(FIG, "fig4_reliability.pdf"), metadata={"Creator": "X12b run.py"}); plt.close(fig)
# Fig 5: native a/b rule vs Blackwell (re-thresholding), per source state, real and shrunk signal
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.7), layout="constrained")
for ax, m in zip(axes, PRIMARIES):
    sub = main[(main.part == "source") & (main.primary == m) & (main.signal == "llm_p")].set_index("state").reindex(SOURCES); shr = main[(main.part == "source") & (main.primary == m) & (main.signal == "llm_p_shrunk")].set_index("state").reindex(SOURCES)
    xs = np.arange(len(SOURCES)); w_ = .27
    ax.bar(xs - w_, sub.capture_native_ab, w_, color=ORANGE, label=r"native: attempt iff $p\geq a/b$")
    ax.bar(xs, sub.capture_bw, w_, color=BLUE, yerr=[np.maximum(sub.capture_bw - sub.capture_bw_lo, 0), np.maximum(sub.capture_bw_hi - sub.capture_bw, 0)], error_kw=dict(lw=.6), label="Blackwell: calibrated posterior at $a/b$")
    ax.bar(xs + w_, shr.capture_native_ab, w_, color=ORANGE, alpha=.4, hatch="///", label=r"native rule on the shrunk score $p/10$")
    ax.axhline(0, color=GREY, lw=.6); ax.axhline(1, color=GREY, lw=.6, ls=":"); ax.set_xticks(xs); ax.set_xticklabels(SOURCES, rotation=30, ha="right"); ax.set_xlabel(f"source state (primary {m}, $\\theta=1$)")
    ax.set_yscale("symlog", linthresh=1.0)
axes[0].set_ylabel("capture ratio (symlog)"); axes[0].legend(loc="lower left", frameon=False, fontsize=6)
fig.savefig(os.path.join(FIG, "fig5_rethresholding.pdf"), metadata={"Creator": "X12b run.py"}); plt.close(fig)
# Fig 6: k vs a per source state (dollars, tokens, seconds)
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.6), layout="constrained")
kk = kcond[kcond.part == "source"]
MK = {"Qwen3-Max": ("o", BLUE), "kimi-k2.5": ("s", PURPLE), "qwen3.5-plus": ("^", ORANGE), "gpt-5.4": ("D", GREEN), "claude-sonnet-4-6": ("v", GREY), "glm-5": ("<", GREY), "MiniMax-M2.7": (">", GREY)}
for ax, (acol, kcol, xl) in zip(axes, [("a_usd", "k_peak", "attempt cost $a$ (\\$/task, probing)"), ("a_tok", "k_tok", "attempt tokens $a$"), ("a_sec", "k_sec", "attempt latency $a$ (s)")]):
    for m, (mk, colr) in MK.items():
        g = kk[kk.primary == m]; ax.scatter(g[acol], g[kcol], s=14, marker=mk, color=colr, alpha=.8, linewidths=0, label=m)
        if acol == "a_usd": ax.scatter(g[acol], g.k_offpeak, s=14, marker=mk, facecolors="none", edgecolors=colr, linewidths=.6, alpha=.8)
    lim = [min(kk[acol].min(), kk[kcol].min()) * .7, max(kk[acol].max(), kk[kcol].max()) * 1.4]; ax.plot(lim, lim, color=GREY, lw=.6, ls=":"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel("signal cost $k$ (\\$/task)\nfilled: peak, hollow: off-peak"); axes[1].set_ylabel("signal tokens $k$"); axes[2].set_ylabel("signal latency $k$ (s)")
hh, ll = axes[0].get_legend_handles_labels(); fig.legend(hh, ll, loc="outside lower center", ncol=7, frameon=False, fontsize=6)
fig.savefig(os.path.join(FIG, "fig6_k_vs_a.pdf"), metadata={"Creator": "X12b run.py"}); plt.close(fig)
# Fig 7: info vs proposer
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.7), layout="constrained")
iv = ivp[(ivp.part != "all") & (ivp.theta == 1.0) & (ivp.upgrade != "opus_direct") & (~ivp.small)]
ax = axes[0]; ax.scatter(iv[iv.decided].pred_info_minus_dec * 1e3, iv[iv.decided].real_info_minus_dec * 1e3, s=8, color=BLUE, alpha=.6, linewidths=0, label="decided")
ax.scatter(iv[~iv.decided].pred_info_minus_dec * 1e3, iv[~iv.decided].real_info_minus_dec * 1e3, s=8, facecolors="none", edgecolors=BLUE, linewidths=.5, alpha=.5, label="inconclusive")
ax.set_xscale("symlog", linthresh=0.1); ax.set_yscale("symlog", linthresh=0.1); ax.axhline(0, color=GREY, lw=.6); ax.axvline(0, color=GREY, lw=.6)
ax.set_xlabel("predicted $b\\,\\Delta p-a(1-s)$ (m\\$/task, probing)\nexact information at $k'=a_2-a$"); ax.set_ylabel(r"realized $C_{\rm info}(k')-C_{\rm dec}$ (m\$/task)"); ax.legend(loc="upper left", frameon=False, fontsize=6)
iv2 = ivp2[(ivp2.part != "all") & (ivp2.theta == 1.0) & (~ivp2.small)]
ax = axes[1]; ax.scatter(iv2[iv2.decided].pred_signal_minus_upgrade * 1e3, iv2[iv2.decided].real_signal_minus_upgrade * 1e3, s=8, color=PURPLE, alpha=.6, linewidths=0, label="decided")
ax.scatter(iv2[~iv2.decided].pred_signal_minus_upgrade * 1e3, iv2[~iv2.decided].real_signal_minus_upgrade * 1e3, s=8, facecolors="none", edgecolors=PURPLE, linewidths=.5, alpha=.5, label="inconclusive")
ax.set_xscale("symlog", linthresh=0.1); ax.set_yscale("symlog", linthresh=0.1); ax.axhline(0, color=GREY, lw=.6); ax.axvline(0, color=GREY, lw=.6)
ax.set_xlabel("predicted $[k+C_{\\rm BW}]-C_{\\rm cascade}(m_2)$ (m\\$/task, probing)\nthe actual LLM signal at its measured $k$"); ax.set_ylabel("realized (m\\$/task)"); ax.legend(loc="upper left", frameon=False, fontsize=6)
fig.savefig(os.path.join(FIG, "fig7_info_vs_proposer.pdf"), metadata={"Creator": "X12b run.py"}); plt.close(fig)

# ================================================================================================ H. audit (pure Python: csv/json/loops, no numpy/pandas)
def audit():
    out = {}
    sc = collections.defaultdict(dict); co = collections.defaultdict(dict)
    with open(f"{ACR}/coderouterbench/id_results_long.csv") as fh:
        for r in csv.DictReader(fh): sc[r["task_id"]][r["model"]] = float(r["score"]); co[r["task_id"]][r["model"]] = float(r["cost_usd"])
    rows = {}
    with open(os.path.join(OUT, "signal_rows.csv")) as fh:
        for r in csv.DictReader(fh): rows[r["task_id"]] = r
    pr = [t for t, r in rows.items() if r["split"] == "probing"]; te = [t for t, r in rows.items() if r["split"] == "id_test"]
    m = "Qwen3-Max"
    # (1) probing primitives and the closed-form bound, pooled
    s = sum(1.0 for t in pr if sc[t][m] >= 1.0) / len(pr); a = sum(co[t][m] for t in pr) / len(pr); b = sum(co[t][STRONG] for t in pr) / len(pr); bound = min(s * (b - a), a * (1 - s))
    rr = row_of("all", "ALL", 1.0, m, "llm_p"); out["pooled_bound_qwen_theta1"] = dict(run_py=float(rr.bound_closed), independent=bound, abs_diff=abs(float(rr.bound_closed) - bound))
    # (2) pooled Blackwell gross VOI of llm_p: PAVA isotonic re-implemented on the probing (p, H) pairs, threshold a/b, realized on id_test
    pts = sorted((float(rows[t]["p_qwen"]), 1.0 if sc[t][m] >= 1.0 else 0.0) for t in pr)
    ux = []; uy = []; uw = []                      # sklearn _make_unique: average y over tied x, weight = count
    for x, y in pts:
        if ux and ux[-1] == x: uy[-1] = (uy[-1] * uw[-1] + y) / (uw[-1] + 1); uw[-1] += 1
        else: ux.append(x); uy.append(y); uw.append(1)
    stack = []                                       # weighted PAVA over the unique points
    for x, y, w_ in zip(ux, uy, uw):
        stack.append([x, x, y, w_])
        while len(stack) > 1 and stack[-2][2] > stack[-1][2]:
            x0, x1, y1, w1 = stack.pop(); x00, x01, y0, w0 = stack.pop(); stack.append([x00, x1, (y0 * w0 + y1 * w1) / (w0 + w1), w0 + w1])
    fit_y = []                                       # fitted value at each unique x
    for x in ux:
        for lo_, hi_, yv, _ in stack:
            if lo_ <= x <= hi_: fit_y.append(yv); break
    def iso_pred(x):                                 # sklearn predict: clip to [x_min, x_max], then linear interpolation between unique x
        if x <= ux[0]: return fit_y[0]
        if x >= ux[-1]: return fit_y[-1]
        j = 0
        while ux[j + 1] < x: j += 1
        x0, x1 = ux[j], ux[j + 1]; return fit_y[j] + (fit_y[j + 1] - fit_y[j]) * (x - x0) / (x1 - x0)
    ab = a / b; u0 = s > ab; cu0 = cbw = 0.0
    for t in te:
        h = sc[t][m] >= 1.0; cp = co[t][m]; cb = co[t][STRONG]; c_casc = cp + (0.0 if h else cb); cu0 += (c_casc if u0 else cb); cbw += (c_casc if iso_pred(float(rows[t]["p_qwen"])) > ab else cb)
    g = (cu0 - cbw) / len(te); out["pooled_gross_bw_llm_p_qwen_theta1"] = dict(run_py=float(rr.gross_bw), independent=g, abs_diff=abs(float(rr.gross_bw) - g), note="isotonic regression re-implemented in pure Python (unique-x averaging, weighted PAVA, linear interpolation with clipping, as sklearn)")
    # (2b) state-level anchored pooled isotonic: hex | kimi-k2.5, llm_p (a non-degenerate row: capture > 0 with CI above 0)
    def bw_gross(pcol, m_, state_filter):
        pts_ = sorted((float(rows[t][pcol]), 1.0 if sc[t][m_] >= 1.0 else 0.0) for t in pr)
        ux_ = []; uy_ = []; uw_ = []
        for x, y in pts_:
            if ux_ and ux_[-1] == x: uy_[-1] = (uy_[-1] * uw_[-1] + y) / (uw_[-1] + 1); uw_[-1] += 1
            else: ux_.append(x); uy_.append(y); uw_.append(1)
        st_ = []
        for x, y, w_ in zip(ux_, uy_, uw_):
            st_.append([x, x, y, w_])
            while len(st_) > 1 and st_[-2][2] > st_[-1][2]:
                x0, x1, y1, w1 = st_.pop(); x00, x01, y0, w0 = st_.pop(); st_.append([x00, x1, (y0 * w0 + y1 * w1) / (w0 + w1), w0 + w1])
        fy_ = []
        for x in ux_:
            for lo_, hi_, yv, _ in st_:
                if lo_ <= x <= hi_: fy_.append(yv); break
        def pred(x):
            if x <= ux_[0]: return fy_[0]
            if x >= ux_[-1]: return fy_[-1]
            j = 0
            while ux_[j + 1] < x: j += 1
            return fy_[j] + (fy_[j + 1] - fy_[j]) * (x - ux_[j]) / (ux_[j + 1] - ux_[j])
        prs = [t for t in pr if state_filter(t)]; tes = [t for t in te if state_filter(t)]
        s_ = sum(1.0 for t in prs if sc[t][m_] >= 1.0) / len(prs); a_ = sum(co[t][m_] for t in prs) / len(prs); b_ = sum(co[t][STRONG] for t in prs) / len(prs)
        shift_ = s_ - sum(pred(float(rows[t][pcol])) for t in prs) / len(prs); ab_ = a_ / b_; u0_ = s_ > ab_; cu0_ = cbw_ = 0.0
        for t in tes:
            h = sc[t][m_] >= 1.0; cp = co[t][m_]; cb = co[t][STRONG]; c_casc = cp + (0.0 if h else cb); cu0_ += (c_casc if u0_ else cb)
            post_ = min(max(pred(float(rows[t][pcol])) + shift_, 0.0), 1.0); cbw_ += (c_casc if post_ > ab_ else cb)
        return (cu0_ - cbw_) / len(tes), min(s_ * (b_ - a_), a_ * (1 - s_))
    g_hex, bnd_hex = bw_gross("p_kimi", "kimi-k2.5", lambda t: rows[t]["source"] == "hex")
    rr2 = row_of("source", "hex", 1.0, "kimi-k2.5", "llm_p")
    out["hex_kimi_gross_bw_llm_p_theta1"] = dict(run_py=float(rr2.gross_bw), independent=g_hex, abs_diff=abs(float(rr2.gross_bw) - g_hex), capture_run_py=float(rr2.capture_bw), capture_independent=g_hex / bnd_hex)
    # (2c) pooled TF-IDF LR Blackwell gross VOI for Qwen3-Max (capture 0.13, CI above 0), from the probabilities stored in signal_rows.csv
    g_tf, bnd_tf = bw_gross("tfidf_p_qwen_theta1", "Qwen3-Max", lambda t: True)
    rr3 = row_of("all", "ALL", 1.0, "Qwen3-Max", "tfidf_lr")
    out["pooled_gross_bw_tfidf_lr_qwen_theta1"] = dict(run_py=float(rr3.gross_bw), independent=g_tf, abs_diff=abs(float(rr3.gross_bw) - g_tf), capture_run_py=float(rr3.capture_bw), capture_independent=g_tf / bnd_tf)
    # (3) pooled AUROC of p_qwen by the Mann-Whitney count
    pos = [float(rows[t]["p_qwen"]) for t in te if sc[t][m] >= 1.0]; neg = [float(rows[t]["p_qwen"]) for t in te if sc[t][m] < 1.0]
    au = sum((1.0 if x > y else 0.5 if x == y else 0.0) for x in pos for y in neg) / (len(pos) * len(neg))
    out["pooled_auroc_llm_p_qwen_theta1"] = dict(run_py=float(rr.auroc), independent=au, abs_diff=abs(float(rr.auroc) - au))
    out["max_abs_diff"] = max(v["abs_diff"] for v in out.values() if isinstance(v, dict) and "abs_diff" in v)
    return out
results["audit"] = audit()

# ================================================================================================ I. README tables (markdown) and outputs
def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))
lines = ["# readme tables (generated by run.py)", ""]
lines += ["## Pooled state, theta = 1 (signal rows; net VOI at the signal's own price: deepseek peak / off-peak, qwen3-max list / half list, TF-IDF prompt-token proxy, released routers k = 0)", "",
          "| primary | signal | s | a/b | AUROC | ECE | native capture [CI] | Blackwell capture [CI] | attempt share BW | net VOI high/low price (m$/task) |", "|---|---|---|---|---|---|---|---|---|---|"]
for m in PRIMARIES:
    for s_ in ["llm_p", "llm_label", "llm_lines", "tfidf_lr", "tfidf_lr_uni", "llm_p_shrunk", "llm_p_perm", "llm_p_sub", "qwen_p", "qwen_label", "qwen_lines", "llm_ensemble_mean", "qwen_p_shrunk", "finetuned_router_qwen35_27b_v3|this_primary", "finetuned_router_qwen35_9b_v3|this_primary", "finetuned_router_qwen35_2b|this_primary", "finetuned_router_qwen35_08b|this_primary", "logreg|this_primary", "tfidf_mlp|this_primary", "routellm_mf|this_primary", "routellm_sw|this_primary", "dimension_best|this_primary", "acrouter_release|this_primary", "finetuned_router_qwen35_27b_v3|any_non_opus", "logreg|any_non_opus"]:
        r = row_of("all", "ALL", 1.0, m, s_)
        if r is None: continue
        rel = s_.startswith(("finetuned", "logreg", "routellm", "tfidf_mlp", "dimension", "acrouter")); lab = s_ in ("llm_label", "qwen_label")
        natc = r.capture_native_router if rel else (r.capture_native_label if lab else r.capture_native_ab)
        natlo = r.get("capture_native_router_lo") if rel else (r.get("capture_native_label_lo") if lab else r.get("capture_native_ab_lo"))
        nathi = r.get("capture_native_router_hi") if rel else (r.get("capture_native_label_hi") if lab else r.get("capture_native_ab_hi"))
        lines.append(f"| {m} | {s_}{' (subsample n=' + str(int(r.n_test)) + ')' if bool(r.subsample) else ''} | {fmt(r.s)} | {fmt(r.a_over_b)} | {fmt(r.auroc)} | {fmt(r.ece)} | {fmt(natc, 2)} [{fmt(natlo, 2)}, {fmt(nathi, 2)}] | {fmt(r.capture_bw, 2)} [{fmt(r.capture_bw_lo, 2)}, {fmt(r.capture_bw_hi, 2)}] | {fmt(r.frac_attempt_bw, 2)} | {fmt(r.net_bw_peak * 1e3, 3)} / {fmt(r.net_bw_offpeak * 1e3, 3)} m$ |")
for sname in ["llm_p", "qwen_p", "tfidf_lr"]:
    lines += ["", f"## Per-source states, theta = 1, {sname}", "", "| primary | source | n_prob/n_test | s | a/b | AUROC | ECE | native a/b capture | Blackwell capture [CI] | switch share | bound (m$) | k (m$) | net (m$) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in PRIMARIES:
        for s in SOURCES:
            r = row_of("source", s, 1.0, m, sname)
            if r is None: continue
            lines.append(f"| {m} | {s} | {int(r.n_prob)}/{int(r.n_test)} | {fmt(r.s)} | {fmt(r.a_over_b)} | {fmt(r.auroc)} | {fmt(r.ece)} | {fmt(r.capture_native_ab, 2)} | {fmt(r.capture_bw, 2)} [{fmt(r.capture_bw_lo, 2)}, {fmt(r.capture_bw_hi, 2)}] | {fmt(r.switch_share_bw, 2)} | {fmt(r.bound_closed * 1e3, 3)} | {fmt(r.k_sig * 1e3, 3)} | {fmt(r.net_bw_peak * 1e3, 3)} |")
lines += ["", "## Pooled state, theta = 0.5 (llm_p, qwen_p, tfidf_lr, ft-27B)", "", "| primary | signal | s | a/b | AUROC | ECE | native a/b capture [CI] | Blackwell capture [CI] | attempt share BW |", "|---|---|---|---|---|---|---|---|---|"]
for m in PRIMARIES:
    for s_ in ["llm_p", "llm_label", "tfidf_lr", "llm_p_sub", "qwen_p", "finetuned_router_qwen35_27b_v3|this_primary"]:
        r = row_of("all", "ALL", 0.5, m, s_)
        if r is None: continue
        rel = s_.startswith("finetuned"); natc = r.capture_native_router if rel else (r.capture_native_label if s_ == "llm_label" else r.capture_native_ab)
        natlo = r.get("capture_native_router_lo") if rel else (r.get("capture_native_label_lo") if s_ == "llm_label" else r.get("capture_native_ab_lo")); nathi = r.get("capture_native_router_hi") if rel else (r.get("capture_native_label_hi") if s_ == "llm_label" else r.get("capture_native_ab_hi"))
        lines.append(f"| {m} | {s_}{' (subsample n=' + str(int(r.n_test)) + ')' if bool(r.subsample) else ''} | {fmt(r.s)} | {fmt(r.a_over_b)} | {fmt(r.auroc)} | {fmt(r.ece)} | {fmt(natc, 2)} [{fmt(natlo, 2)}, {fmt(nathi, 2)}] | {fmt(r.capture_bw, 2)} [{fmt(r.capture_bw_lo, 2)}, {fmt(r.capture_bw_hi, 2)}] | {fmt(r.frac_attempt_bw, 2)} |")
lines += ["", "## Per-source states, theta = 0.5, llm_p and tfidf_lr", "", "| primary | source | signal | s | a/b | AUROC | ECE | native a/b capture | Blackwell capture [CI] | switch share |", "|---|---|---|---|---|---|---|---|---|---|"]
for m in PRIMARIES:
    for s in SOURCES:
        for sname in ["llm_p", "tfidf_lr"]:
            r = row_of("source", s, 0.5, m, sname)
            if r is None: continue
            lines.append(f"| {m} | {s} | {sname} | {fmt(r.s)} | {fmt(r.a_over_b)} | {fmt(r.auroc)} | {fmt(r.ece)} | {fmt(r.capture_native_ab, 2)} | {fmt(r.capture_bw, 2)} [{fmt(r.capture_bw_lo, 2)}, {fmt(r.capture_bw_hi, 2)}] | {fmt(r.switch_share_bw, 2)} |")
lines += ["", "## Counterfactual price ratio (SEMI-SYNTHETIC; theta = 1; task-weighted over the 7 source states)", "", "| signal | usage | r = a/b | capture (task-weighted) | mean switch share | states CI>0 / CI<0 |", "|---|---|---|---|---|---|"]
for k_, v_ in results["counterfactual_price_ratio"]["source_states_theta1"].items():
    sname, uname, rr_ = k_.split("|"); lines.append(f"| {sname} | {uname} | {rr_[2:]} | {fmt(v_['capture_taskweighted'], 3)} | {fmt(v_['mean_switch_share'], 3)} | {v_['n_states_ci_above_0']} / {v_['n_states_ci_below_0']} |")
lines += ["", "## k vs a per source (Qwen3-Max primary; probing means; deepseek signal, then qwen3-max signal)", "", "| source | s | a ($) | b ($) | k peak ($) | k off-peak ($) | k self-hosted ($) | a tokens | k tokens | a s | k s | k<a peak / off-peak / tok / s | qwen3-max k list ($) / tok / s | k<a list / tok / s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in C6["per_source_qwen3max"]:
    lines.append(f"| {r['state']} | {fmt(r['s'])} | {r['a_usd']:.6f} | {r['b_usd']:.5f} | {r['k_peak']:.6f} | {r['k_offpeak']:.6f} | {r['k_self']:.7f} | {r['a_tok']:.0f} | {r['k_tok']:.0f} | {r['a_sec']:.1f} | {r['k_sec']:.1f} | {r['k_peak_lt_a']} / {r['k_offpeak_lt_a']} / {r['k_tok_lt_a_tok']} / {r['k_sec_lt_a_sec']} | {r['kq_list']:.6f} / {r['kq_tok']:.0f} / {r['kq_sec']:.1f} | {r['kq_list_lt_a']} / {r['kq_tok_lt_a_tok']} / {r['kq_sec_lt_a_sec']} |")
lines += ["", "## k vs a pooled, all primaries", "", "| primary | s | a ($) | a/b | k peak | k off-peak | k self | a tok | k tok | a s | k s | k<a peak/off/tok/s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in C6["pooled"]:
    lines.append(f"| {r['primary']} | {fmt(r['s'])} | {r['a_usd']:.6f} | {fmt(r['a_over_b'])} | {r['k_peak']:.6f} | {r['k_offpeak']:.6f} | {r['k_self']:.7f} | {r['a_tok']:.0f} | {r['k_tok']:.0f} | {r['a_sec']:.1f} | {r['k_sec']:.1f} | {r['k_peak_lt_a']}/{r['k_offpeak_lt_a']}/{r['k_tok_lt_a_tok']}/{r['k_sec_lt_a_sec']} |")
lines += ["", "## Claims", "", "| # | verdict | key numbers |", "|---|---|---|"]
for c in claims: lines.append(f"| {c['id']} | {c['verdict']} | {json.dumps(clean(c['result']))[:400]} |")
with open(os.path.join(OUT, "readme_tables.md"), "w") as fh: fh.write("\n".join(lines) + "\n")

results["runtime_seconds"] = time.time() - T0
with open(os.path.join(OUT, "results.json"), "w") as fh: json.dump(clean(results), fh, indent=1)
print("\n=== claims ===")
for c in claims: print(f"{c['id']}: {c['verdict']}  | {json.dumps(clean(c['result']))[:300]}")
print(f"audit max abs diff: {results['audit']['max_abs_diff']:.3e}")
print(f"API: {json.dumps(clean({k: v for k, v in results['api'].items() if k in ('session_units_est', 'ledger_units_this_run', 'relay_delta_units_upper_bound', 'wall_seconds_signal_phase')}))}")
print(f"runtime {results['runtime_seconds']:.0f} s")

