#!/usr/bin/env python3
"""Offline replay of Table 14: fresh retries versus disjoint-test feedback.
No model calls, generated-code execution, downloads, or task-level output files.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(earlier_dir, extension_dir, x7b_attempts, n_bootstrap=2000):
    R7, OUT, X7B = str(earlier_dir), str(extension_dir), str(x7b_attempts)
    SEED, B = 20260926, n_bootstrap
    OLD = ["deepseek-flash", "deepseek-v4-pro", "kimi-k2.5", "qwen3-max", "qwen3.5-plus"]; NEW = ["glm-5", "MiniMax-M2.7"]
    L = read_jsonl
    splits = {r["task_id"]: r for r in L(f"{R7}/splits.jsonl")}
    elig = {t for t, s in splits.items() if s["eligible"]}
    roots_old = L(f"{R7}/first_attempts.jsonl"); roots_new = [r for r in L(f"{OUT}/first_attempts_new.jsonl") if not r.get("api_error")]
    x7 = [json.loads(l) for l in open(X7B)]; x7roots = [r for r in x7 if r.get("branch") == "root"]
    den = dict(x7b_exec_clean_tasks=len({r["task_id"] for r in x7roots}), x7b_root_chains=len(x7roots),
               x7b_models=sorted({r["model"] for r in x7roots}), splits_rows=len(splits), eligible_tasks=len(elig),
               ineligible_by_reason=dict(Counter(s["reason"] for s in splits.values() if not s["eligible"])),
               r7_first_attempt_rows=len(roots_old), r7_first_attempt_tasks=len({r["task_id"] for r in roots_old}),
               new_first_attempt_rows={m: sum(1 for r in roots_new if r["model"] == m) for m in NEW})
    # retries: R7 archived live retries (sample 1) + today's
    old_calls = {}
    for r in L(f"{R7}/calls.jsonl"):
        if not r["api_error"] and r["arm"] in ("fresh", "fb_F"):
            old_calls[(r["task_id"], r["model"], r["arm"], r["seed"])] = r
    new_calls = {}
    for r in L(f"{OUT}/calls_new.jsonl"):
        if not r["api_error"]:
            new_calls[(r["task_id"], r["model"], r["arm"], r["seed"])] = r
    SEEDS = {"fresh": [2, 3, 4], "fb_F": [None, 3, 4]}
    fails = [r for r in roots_old + roots_new if not r["pass_S"] and r["task_id"] in elig]
    rows = []
    for f in fails:
        k = (f["task_id"], f["model"])
        rec = dict(task_id=f["task_id"], model=f["model"], pass_F_first=bool(f["pass_F"]), truncated_first=(f.get("finish_reason") == "length"))
        for arm in ("fresh", "fb_F"):
            vals = []
            for i, s in enumerate(SEEDS[arm]):
                r = old_calls.get(k + (arm, s)) if (f["model"] in OLD and i == 0) else new_calls.get(k + (arm, s))
                vals.append(None if r is None else float(r["pass_S"]))
            rec[arm] = vals
        rows.append(rec)
    def arr(sel, arm, samples):
        out = []
        for r in sel:
            v = [r[arm][i] for i in samples if r[arm][i] is not None]
            out.append(np.mean(v) if v else np.nan)
        return np.array(out)
    tasks_all = sorted({r["task_id"] for r in rows})
    rng = np.random.default_rng(SEED)
    BOOT = [rng.choice(len(tasks_all), len(tasks_all), replace=True) for _ in range(B)]
    def cluster_ci(sel, samples):
        fr, fb = arr(sel, "fresh", samples), arr(sel, "fb_F", samples)
        ok = np.isfinite(fr) & np.isfinite(fb)
        sel = [r for r, o in zip(sel, ok) if o]; fr, fb = fr[ok], fb[ok]
        tix = np.array([tasks_all.index(r["task_id"]) for r in sel])
        cnt_by_task = np.zeros(len(tasks_all)); fr_t = np.zeros(len(tasks_all)); fb_t = np.zeros(len(tasks_all))
        np.add.at(cnt_by_task, tix, 1); np.add.at(fr_t, tix, fr); np.add.at(fb_t, tix, fb)
        stats = []
        for b in BOOT:
            w = np.bincount(b, minlength=len(tasks_all)).astype(float)
            n = w @ cnt_by_task
            stats.append(((w @ fr_t) / n, (w @ fb_t) / n) if n > 0 else (np.nan, np.nan))
        st = np.array(stats); d = st[:, 1] - st[:, 0]
        q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
        return dict(n_failures=len(sel), n_tasks=int(len(set(tix))), fresh=[float(fr.mean())] + q(st[:, 0]), fb_F=[float(fb.mean())] + q(st[:, 1]),
                    diff=[float(fb.mean() - fr.mean())] + q(d), diff_pp=100 * float(fb.mean() - fr.mean()), se_diff=float(np.nanstd(d, ddof=1)),
                    n_attempts=int(sum(sum(v is not None for v in (r["fresh"][i] for i in samples)) + sum(v is not None for v in (r["fb_F"][i] for i in samples)) for r in sel)))
    res = dict(settings=dict(seed=SEED, B=B, bootstrap="task-cluster bootstrap over the tasks with >= 1 S-failure (all failures of a drawn task enter)",
                             samples=dict(fresh="seeds 2 (earlier call for the five initial models; extension for glm-5/MiniMax), 3, 4",
                                          fb_F="seed none (earlier call for the five initial models; extension for glm-5/MiniMax), 3, 4"),
                             unit="per failure: mean recovery on S over its samples; pooled = mean over failures"), denominators=den)
    per = {}
    for m in OLD + NEW:
        sel = [r for r in rows if r["model"] == m]
        per[m] = dict(all3=cluster_ci(sel, [0, 1, 2]), sample1=cluster_ci(sel, [0]), sample2=cluster_ci(sel, [1]), sample3=cluster_ci(sel, [2]),
                      first_attempt_pass_S=float(np.mean([r["pass_S"] for r in roots_old + roots_new if r["model"] == m])),
                      n_first=sum(1 for r in roots_old + roots_new if r["model"] == m))
    res["per_model"] = per
    pools = {"five_R7_models": OLD, "seven_models": OLD + NEW, "four_R7_models_no_flash": [m for m in OLD if m != "deepseek-flash"],
             "six_models_no_flash": [m for m in OLD + NEW if m != "deepseek-flash"], "two_new_models": NEW}
    res["pooled"] = {}
    for nm, ms in pools.items():
        sel = [r for r in rows if r["model"] in ms]
        res["pooled"][nm] = dict(all3=cluster_ci(sel, [0, 1, 2]), sample1=cluster_ci(sel, [0]), samples2_3=cluster_ci(sel, [1, 2]))
    sel = [r for r in rows if not r["truncated_first"]]
    res["pooled"]["seven_models_excl_truncated_first"] = dict(all3=cluster_ci(sel, [0, 1, 2]), sample1=cluster_ci(sel, [0]), samples2_3=cluster_ci(sel, [1, 2]))
    res["n_truncated_first_among_failures"] = {m: sum(1 for r in rows if r["model"] == m and r["truncated_first"]) for m in OLD + NEW}
    # heterogeneity across models (all 3 samples): Cochran's Q with bootstrap SEs
    for nm, ms in (("five_R7_models", OLD), ("seven_models", OLD + NEW)):
        d = np.array([per[m]["all3"]["diff"][0] for m in ms]); se = np.array([per[m]["all3"]["se_diff"] for m in ms])
        w = 1 / np.maximum(se, 1e-6) ** 2; mu = (w @ d) / w.sum(); Q = float(w @ (d - mu) ** 2); df = len(ms) - 1
        res.setdefault("heterogeneity", {})[nm] = dict(Q=Q, df=df, p=float(chi2.sf(Q, df)), I2=float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0,
                                                       fixed_effect=float(mu), per_model_diff=dict(zip(ms, d.tolist())))
    # feedback content split (all samples)
    for kind, flag in (("F_fail_feedback", False), ("all_F_pass_feedback", True)):
        sel = [r for r in rows if r["pass_F_first"] == flag]
        res.setdefault("by_feedback_kind", {})[kind] = cluster_ci(sel, [0, 1, 2])
    new_all = list(new_calls.values()) + [r for r in roots_new]
    res["spend_part2"] = dict(new_live_calls=len(new_calls) + len(roots_new), retry_calls_today=len(new_calls), first_attempt_calls_today=len(roots_new),
                              usd_list=float(sum(r.get("cost_usd") or 0 for r in new_all if not r.get("cached"))),
                              tokens=int(sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in new_all)),
                              truncated_today=int(sum(1 for r in new_all if r.get("finish_reason") == "length")))
    res["missing"] = dict(fresh=int(sum(v is None for r in rows for v in r["fresh"])), fb_F=int(sum(v is None for r in rows for v in r["fb_F"])))
    return res


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--earlier-dir', type=Path, required=True,
                        help='Directory containing splits.jsonl, first_attempts.jsonl, calls.jsonl')
    parser.add_argument('--extension-dir', type=Path, required=True,
                        help='Directory containing first_attempts_new.jsonl and calls_new.jsonl')
    parser.add_argument('--x7b-attempts', type=Path,
                        default=root / 'experiments_x/X7b_inloop_live/attempts_view_a_6144.jsonl',
                        help='Earlier X7b archive used only for denominator diagnostics')
    parser.add_argument('--bootstrap', type=int, default=2000,
                        help='Task-cluster bootstrap replicates; paper uses 2000')
    parser.add_argument('--output', type=Path, default=root / 'runs/disjoint_retry/results.json')
    args = parser.parse_args()
    result = analyze(args.earlier_dir, args.extension_dir, args.x7b_attempts, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    pooled = result['pooled']['seven_models']['all3']
    print(json.dumps({'failures': pooled['n_failures'], 'fresh': pooled['fresh'],
                      'feedback': pooled['fb_F'], 'difference': pooled['diff'],
                      'missing': result['missing'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
