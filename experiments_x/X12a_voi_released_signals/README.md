# X12a_voi_released_signals — value of routing information carried by released and self-trained routing signals (X12, offline part)

Theory objects tested (names as in the paper): eq:gate-costs, prop:voi / eq:band, eq:imperfect-voi, the perfect-information bound of app:imperfect-voi (`f(s) − E f(s_Z) ≤ f(s) − g(s) = min{s(b−a), a(1−s)}`), the "pays only if k < a" clause, the miscalibration example of app:imperfect-voi, and cor:info-vs-proposer (`C_info − C_dec = bΔp − a(1−s)`). Definitions of f, g, the sufficiency event, realized cascade / direct / perfect-information costs, the capture ratio and "decided" cells are copied from `../experiments/X3_threshold_voi/run.py`.

Everything below is produced by `run.py` (seed 20260917; from-scratch runtime 118 s, single process, no API calls). Files: `run.py`, `results.json` (all numbers with CIs, `claims`), `crb_signal_rows.csv` (8,680 CodeRouterBench rows), `rb_signal_rows.csv` (912 RouterBench rows), `band_rows.csv`, `info_vs_proposer_rows.csv`, `k_condition_rows.csv`, `figures/fig1..fig6*.pdf`, `summary_for_paper.md`. **X12-b (live-judge signals) is a later experiment and is NOT part of this directory** (pending).

## Setup

**Data (READ ONLY).** CodeRouterBench (CRB) `id_results_long.csv`: 9,999 tasks × 8 models, `score ∈ [0,1]`, `cost_usd` (token-log pricing). Reference B = `claude-opus-4-6`. Primaries m: all seven non-Opus models (Qwen3-Max, kimi-k2.5, qwen3.5-plus, MiniMax-M2.7, glm-5, gpt-5.4, claude-sonnet-4-6). Sufficiency `H = 1{score_m ≥ θ}`, θ = 1 (main) and 0.5. Splits: **probing** (7,080 tasks) gives every primitive (s, a, b, a/b, the bound, the uninformed choice); **id_test** (2,919) gives every realized quantity. States x: 9 dimensions, 21 dimension × source cells, and the pooled set. RouterBench (RB) `routerbench_0shot.pkl`: 36,497 prompts × 11 models, reference `gpt-4-1106-preview`, primaries {mixtral-8x7b, Yi-34B, claude-instant-v1, gpt-3.5-turbo}, sufficiency `score ≥ 0.5`, costs from the pickle; 70 / 30 split stratified by `eval_name` (25,547 fit / 10,950 held-out, seed 20260917); states: 8 source families (57 MMLU subsets merged, 3 MT-Bench subsets merged, the Chinese/RAG subsets merged), the 67 `eval_name`s with ≥ 100 prompts, and the pooled set. Cost basis: dollars throughout.

**Signals Z on CRB — exact fields found.** All nine files in `$ACR/id/voter_decisions/` contain exactly two string fields per line, `{"chosen_model", "task_id"}`, and nothing else: no score, probability, logit, margin or router-token field in any of them (`results.json → signal_fields_crb`). The four `finetuned_router_qwen35_*` files have 2,919 rows (= id_test); `dimension_best`, `logreg`, `tfidf_mlp`, `routellm_mf`, `routellm_sw` have 3,251 rows: the 2,919 id_test tasks plus 332 `dim10_*` (agentic) tasks that are not in the 9,999-task outcome matrix and are dropped. No file has a decision on a probing task. The ACRouter release policy (`outputs/current/id_decisions.jsonl`, 2,919 rows) has `{chosen_model, dimension, matched_key, matched_mode, task_id, voter}`, all strings, no score; it is added as a tenth CRB signal ("lookup"). **Consequently Z is the binary event "router chose a model ≠ Opus"** (`any_non_opus`, the protocol's definition) and, for C5 and as a second definition, "router chose the primary m itself" (`this_primary`, X3's definition).

**Posterior s_Z for a binary released signal.** Isotonic calibration on probing is impossible (no decisions there). Instead: 10 random half-splits of the id_test state (stratified by cell21); on the calibration half the lifts `P(H|Z=z) − P(H)` are estimated and the posterior on the other half is `s_Z = s_probing + lift_Z` clipped to [0, 1], so that E[s_Z] ≈ s (the premise of eq:imperfect-voi); Blackwell-optimal usage = attempt iff `s_Z > a/b` (a, b from probing); the attempt indicator is averaged over the 20 (split, half) assignments. Realized cost per task `C_direct + attempt·(C_cascade − C_direct)`. An in-sample variant (posterior and a/b from the whole id_test state; X3's construction) is reported as `capture_bw_ins`. **Native usage** = attempt iff Z = 1.

**Signals Z on RB.** Logistic regression on TF-IDF of the prompt (1–2-grams, `min_df=2`, 200k features, sublinear tf; `C=1`, liblinear) trained on the 70 % fit split to predict sufficiency of each primary; held-out AUROC 0.688–0.712, ECE 0.007–0.023. Second real variant `lr_tfidf_uni` (unigrams, 15,532 features, `C=0.3`; AUROC 0.681–0.705). Third, SEMI-SYNTHETIC variant `lr_tfidf_shrunk = lr_tfidf / 10`, the paper's shrunken-score example. `s_Z` = isotonic regression fitted on the fit split using 5-fold out-of-fold LR probabilities (so the calibrator never sees in-sample scores), applied to the held-out split; Blackwell usage = attempt iff `s_Z > a/b`; native usage = route to the primary iff `P ≥ 0.5` (the router's own threshold); also the raw score thresholded at a/b (`native_ab`).

**Uninformed reference, VOI, bound.** Uninformed choice from the fit split: cascade iff `a + (1−s)b < b`, i.e. `s > a/b`. Gross VOI = realized mean cost of the uninformed choice − realized mean cost of the signal usage (held-out); net VOI = gross − k. Bound = `min{s(b−a), a(1−s)}` with fit-split means; capture ratio = gross VOI / bound (rows with bound ≤ 1e-9, i.e. s ∈ {0, 1} in the fit split, are "degenerate" and excluded from ratios). k: 0 (main; the files carry no router tokens), k ∈ {0.001, 0.01}·b, and `k_tok` = prompt tokens × $0.054/M (CRB: mean `input_tokens` over the 8 models, mean $1.4e-5/task; RB: chars/4 proxy, mean $7e-6). Bootstrap: 1,000 paired task resamples per state, 95 % percentile CIs; two usages are always compared on the same resample. Permutation test: 20 permutations of Z within state (SEMI-SYNTHETIC). Sign-agreement rates are computed on decided rows (realized CI excludes 0) and shown next to naive baselines.

## Results

### Pooled state, CRB, θ = 1, primary Qwen3-Max (s = 0.214, a = $0.000532, b = $0.01237, a/b = 0.043; bound f − g = $0.000418/task; uninformed choice = cascade, $0.00801/task)

| Signal | Z | P(Z=1) | P(H\|Z=1) | P(H\|Z=0) | AUROC | ECE | native capture (OOS) [CI] | Blackwell capture (cross-fit) [CI] | in-sample BW |
|---|---|---|---|---|---|---|---|---|---|
| ft-0.8B | this_primary | 0.49 | 0.39 | 0.055 | 0.745 | 0.33 | 0.42 [0.35, 0.49] | 0.08 [0.07, 0.10] | 0.00 |
| ft-2B | this_primary | 0.50 | 0.40 | 0.042 | 0.760 | 0.32 | 0.47 [0.40, 0.53] | 0.42 [0.36, 0.48] | 0.50 |
| ft-9B | this_primary | 0.49 | 0.42 | 0.024 | 0.788 | 0.30 | 0.55 [0.48, 0.61] | 0.55 [0.48, 0.61] | 0.58 |
| ft-27B | this_primary | 0.51 | 0.41 | 0.018 | 0.787 | 0.31 | 0.56 [0.49, 0.61] | 0.56 [0.49, 0.61] | 0.59 |
| logreg | this_primary | 0.19 | 0.52 | 0.151 | 0.663 | 0.21 | −1.22 [−1.59, −0.87] | 0.00 | 0.00 |
| tfidf_mlp | this_primary | 0.19 | 0.50 | 0.157 | 0.649 | 0.22 | −1.74 [−2.18, −1.34] | 0.00 | 0.00 |
| routellm_mf | this_primary | 0.26 | 0.42 | 0.152 | 0.649 | 0.26 | −1.03 [−1.36, −0.72] | 0.00 | 0.00 |
| routellm_sw | this_primary | 0.24 | 0.45 | 0.148 | 0.661 | 0.24 | −1.28 [−1.68, −0.95] | 0.00 | 0.00 |
| dimension_best | this_primary | 0.10 | 0.78 | 0.155 | 0.671 | 0.16 | −1.47 [−1.85, −1.10] | 0.00 | 0.00 |
| ACRouter release | this_primary | 0.11 | 0.35 | 0.205 | 0.540 | 0.25 | −7.17 [−8.07, −6.29] | 0.00 | 0.00 |
| ft-27B | any_non_opus | 0.73 | 0.30 | 0.012 | 0.663 | 0.52 | 0.34 [0.28, 0.39] | 0.34 [0.28, 0.39] | 0.36 |
| ft-9B | any_non_opus | 0.73 | 0.29 | 0.019 | 0.657 | 0.52 | 0.32 [0.27, 0.37] | 0.32 [0.27, 0.37] | 0.35 |
| ft-2B | any_non_opus | 0.61 | 0.33 | 0.050 | 0.694 | 0.43 | 0.34 [0.27, 0.40] | 0.13 [0.10, 0.16] | 0.00 |
| ft-0.8B | any_non_opus | 0.58 | 0.33 | 0.064 | 0.689 | 0.42 | 0.30 [0.22, 0.36] | 0.00 | 0.00 |
| logreg | any_non_opus | 0.84 | 0.26 | 0.022 | 0.590 | 0.63 | 0.21 [0.17, 0.24] | 0.21 [0.17, 0.24] | 0.22 |
| routellm_mf | any_non_opus | 0.84 | 0.26 | 0.025 | 0.593 | 0.62 | 0.22 [0.18, 0.25] | 0.22 [0.18, 0.25] | 0.23 |
| tfidf_mlp | any_non_opus | 0.85 | 0.25 | 0.040 | 0.581 | 0.64 | 0.16 [0.11, 0.20] | 0.12 [0.08, 0.16] | 0.17 |
| routellm_sw | any_non_opus | 0.84 | 0.25 | 0.054 | 0.579 | 0.63 | 0.14 [0.08, 0.19] | 0.03 [0.02, 0.05] | 0.00 |
| dimension_best | any_non_opus | 0.65 | 0.20 | 0.260 | 0.460 | 0.61 | −0.84 [−1.00, −0.70] | 0.00 | 0.00 |
| ACRouter release | any_non_opus | 0.39 | 0.32 | 0.157 | 0.613 | 0.36 | −0.45 [−0.61, −0.28] | 0.00 | 0.00 |

Net VOI at k = 0.01·b = $0.00012/task: ft-27B/ft-9B (this_primary) +$0.0001, all classical rows ≤ 0; at k_tok = $0.000014: the fine-tuned family and the informative any_non_opus rows stay positive (`results.json → pooled_crb_theta1`). ECE of a binary forecast is its error rate as a classifier of H, `P(Z ≠ H)`; AUROC = (1 + TPR − FPR)/2.

**Dimension information versus task information.** The pooled Blackwell capture of the fine-tuned routers (0.55–0.56, this_primary) shrinks to 0.06–0.08 when the same signal is valued inside the 9 dimensions (task-weighted; `results.json → dimension_conditional_capture_qwen_theta1`): only multi_language (0.80 / 0.66 for ft-27B / ft-9B) and test_generation (0.17 / 0.30) keep a capture with CI above 0, five dimensions have exactly zero Blackwell value (both posteriors on the same side of a/b), and code_completion is negative. Most of what the released routers know about Qwen3-Max sufficiency is which dimension the task belongs to, which a state-aware uninformed policy already has.

### Pooled state, RB (held-out 30 %; θ = 0.5)

| Primary | s | a/b | AUROC | ECE (raw) | native (P ≥ 0.5) capture [CI] | raw score at a/b | Blackwell capture [CI] | permuted Z | attempt share BW |
|---|---|---|---|---|---|---|---|---|---|
| mixtral-8x7b | 0.566 | 0.041 | 0.711 | 0.023 | −4.87 [−5.26, −4.47] | 0.000 | −0.018 [−0.063, 0.011] | 0.000 | 0.989 |
| Yi-34B | 0.676 | 0.056 | 0.712 | 0.018 | −1.23 [−1.43, −1.03] | 0.009 | 0.009 [0.002, 0.016] | 0.000 | 0.989 |
| claude-instant | 0.635 | 0.071 | 0.688 | 0.007 | −1.51 [−1.68, −1.34] | 0.005 | 0.010 [0.004, 0.016] | 0.000 | 0.984 |
| gpt-3.5-turbo | 0.662 | 0.074 | 0.700 | 0.019 | −1.80 [−1.97, −1.63] | 0.000 | −0.031 [−0.095, 0.010] | 0.000 | 0.992 |

The router's own threshold 0.5 is far above the price ratio a/b ≈ 0.04–0.07, so "route to the primary iff P ≥ 0.5" sends 17–33 % of the tasks — with sufficiency probability 0.3–0.5 — straight to GPT-4 and costs 1.2–4.9 bounds more than the uninformed cascade. Used optimally the same TF-IDF router is worth at most 1 % of the perfect-information value: the isotonic posterior falls below a/b on ~1 % of held-out prompts. The shrunken variant (`lr_tfidf/10`) has identical AUROC and identical Blackwell capture (max |Δ| = 0.0 over 300 states) but ECE 0.15 → 0.58 and, used through the rule `ŝ ≥ a/b`, capture 0.004 → −15.2 (242 of 300 states with CI below 0): the app:imperfect-voi example on real data.

### prop:voi band, exact information at cost k (OUT-OF-SAMPLE; `band_rows.csv`, Figure 2)

Prediction: information cheapest iff `k/(b−a) < s < 1−k/a` (fit split); measurement: realized argmin of {cascade, direct, perfect information + k} on the held-out split. CRB θ = 1, dim9 + cell21 (210 rows): k = 0.001b → 178 decided, agreement 0.893 (3-way argmin 0.944) vs "info iff band non-empty" baseline 0.607; k = 0.01b → 171 decided, 0.877 (0.947) vs 0.532; k_tok → 177 decided, 0.904 (0.955) vs 0.605. RB (300 rows): 0.991–0.997 vs 0.953–0.983. Every pooled CRB and RB state has s inside its band at k = 0.01b (positions u = 0.15–0.27 on CRB, 0.74–0.82 on RB) and exact information is realized-cheapest in all of them.

### k < a (necessary condition; `k_condition_rows.csv`)

At k_tok (mean $1.4e-5 on CRB, $7e-6 on RB) and at k = 0.01·b, **0 of 217 CRB states (θ = 1) and 0 of 304 RB states have k ≥ a** (mean a = $0.0045 / $0.00013). The sharper cascade condition `k < a(1−s)` fails in 9 CRB states, all with s ≥ 0.97 (code_completion|he, code_generation|humaneval), and in 6 RB states. The band is empty in 18 CRB states, in every case because a ≥ b there (glm-5, MiniMax-M2.7, claude-sonnet-4-6 cost more than Opus in that state), not because k is large.

### cor:info-vs-proposer (OUT-OF-SAMPLE; `info_vs_proposer_rows.csv`, Figure 5)

For each (state, primary m) and each costlier model m2 (k′ = a₂ − a, Δp = s₂ − s from the fit split), prediction `bΔp − a(1−s) < 0` ⇔ buy exact information at cost k′ rather than upgrade; measurement: realized `[C_perfect + k′] − C_cascade(m2)` on the held-out split. Pooled main (CRB θ = 1 + RB): 1,981 rows, 967 decided, **agreement 0.966** vs always-info 0.818, always-upgrade 0.182, "info iff Δp ≤ 0" 0.859; on the 276 decided rows with Δp > 0: 0.935 vs 0.428; on the 230 decided "next-stronger" rows (cheapest costlier model with Δp > 0): 0.970 vs 0.765. CRB θ = 1: 0.998 (422 decided, one miss: kimi → gpt-5.4 pooled, predicted +$0.00004, realized +$0.00135); CRB θ = 0.5: 0.990; RB: 0.941 (545 decided; 19 info/upgrade and 13 upgrade/info errors). With free information (k = 0) the rule is nearly trivial (0.992 vs always-info 0.962).

## Claims

| # | Statement | Kind | Result | Verdict |
|---|---|---|---|---|
| C1 | No realized gross VOI exceeds the perfect-information bound min{s(b−a), a(1−s)}; net VOI ≤ bound − k | IDENTITY-like inequality on expectations, realized OUT-OF-SAMPLE | 7,146 non-degenerate (state, θ, primary, signal, Z) rows. **Blackwell usage: 0 point violations, largest ratio 0.984** (bug_fixing, θ = 0.5, gpt-5.4, ft-2B; θ = 1: 0.948, multi_language\|hex, Qwen3-Max, ft-27B). Native usage: 52 point violations, **0 CI-supported**, largest ratio 2.19; all 52 in eight small or degenerate groups (code_completion\|he n = 32, code_generation\|mbpp n = 69, canitedit n = 4, RB Chinese_character_riddles n = 30, bug_fixing\|debugbench MiniMax with bound $0.00008) | **PASS (qualified)** |
| C2 | Capture ratio correlates more with AUROC than with ECE across (x, m, Z) | OUT-OF-SAMPLE | Pre-registered pooled test (3,210 rows): \|ρ(capture_BW, AUROC)\| = 0.026 vs \|ρ(capture_BW, ECE)\| = 0.027, difference −0.001, task-bootstrap CI [−0.073, 0.087]. CRB alone 0.081 vs −0.052 (Δ\|ρ\| +0.029 [−0.054, 0.098]); RB alone 0.131 vs 0.107 (+0.023 [−0.136, 0.128]). Within-(state, primary) refinement (71 CRB groups, ≥ 5 signal variants): mean ρ 0.174 vs −0.074, Δ\|ρ\| +0.245, group-bootstrap CI [0.168, 0.320], 79 % of groups favour AUROC. Native usage: \|ρ_ECE\| = 0.275 > \|ρ_AUROC\| = 0.065 (Δ −0.21 [−0.243, −0.133]) | **INCONCLUSIVE** (pooled); PASS within state; opposite sign for native usage |
| C3 | A calibrated but uninformative Z (permuted within state) has net VOI ≈ −k | SEMI-SYNTHETIC | 3,436 rows: median \|capture\| = 0.000 (2,419 rows exactly 0), mean −0.022, 95th pct of \|capture\| 0.17; mean gross −$1.4e-6/task; net/k at k = 0.01b = −1.011; gross CI above 0 in 35 rows (1.0 %, nominal one-sided rate 2.5 %), 30 of them with \|s − a/b\| < 0.05 (11 distinct (state, primary)); mean AUROC of permuted Z 0.501 | **PASS** |
| C4 | cor:info-vs-proposer sign: buy information iff bΔp < a(1−s) | OUT-OF-SAMPLE | 967 decided rows, agreement 0.966 vs baselines 0.818 / 0.182 / 0.859; Δp > 0 subset 0.935 vs 0.428; next-stronger 0.970 vs 0.765; CRB 0.998, RB 0.941 | **PASS** |
| C5 | Negative capture of the classical routers reproduced and explained | IN-SAMPLE | X3's numbers reproduced to ≤ 4.5e-4 (−1.304, −1.859, −1.106, −1.375, −1.571; fine-tuned 0.452 / 0.499 / 0.584 / 0.594). All five have P(H = 1 \| Z = 0) = 0.148–0.157 > a/b = 0.045 (fine-tuned: 0.018–0.055): their rule sends to Opus tasks the uninformed cascade would have attempted. The Blackwell-optimal use of the same five Z is exactly 0 (both posteriors above a/b ⇒ attempt always) — their loss is a usage error, not an information deficit. ACRouter release lookup: −7.7 (P(H\|Z=0) = 0.205). Cell-mean identity C_native − C_cascade = P(Z=0)(s₀b − a) predicts +$0.0010–0.0013/task vs realized +$0.0004–0.0007: Opus is cheaper on the Z = 0, H = 1 tasks (b\|Z=0,H=1 = $0.0070–0.0087 vs b = $0.0124) | **PASS** |

Verdict rules are stored with each claim in `results.json → claims`. C2's rule is the absolute-value comparison on the pooled task bootstrap (PASS if the CI of |ρ_AUROC| − |ρ_ECE| is above 0, FAIL if below, else INCONCLUSIVE); the within-state and native-usage results are reported next to it and did not decide the verdict. C3's rule was changed once before this README was written: the first draft demanded zero rows with a CI above 0, which is unattainable by chance with > 3,000 simultaneous intervals; the rule now allows the nominal one-sided rate (2.5 %), and additionally requires median |capture| ≤ 0.02 and mean net/k within 10 % of −1.

**What C2 actually shows.** For a binary signal the Bayes rule has decision value only when the two posteriors straddle a/b. On CRB (θ = 1, dim9 + cell21) 2,413 of 4,200 rows have exactly zero Blackwell capture and 1,360 have a zero bound (s ∈ {0, 1} in probing); among the rest, capture is positive with CI above 0 in 38 % of the straddling rows and in 0.3 % of the non-straddling rows, and ρ(capture, straddle indicator) = 0.38 against ρ(capture, AUROC) = 0.08 on the same rows (`results.json → C2_spearman.straddle`). AUROC and ECE are both blind to where a/b sits, so neither predicts the capture ratio across states; holding the state fixed, AUROC does and ECE does not (within-state result), and the shrunken-score check shows the Blackwell capture is invariant to a monotone miscalibration while the native rule is destroyed by it — the substantive content of "decision usefulness, not calibration", which the pooled diagnostic did not detect.

## Hypothesis violations measured

* **0 < a < b task by task** (needed for the pointwise dominance C_Z ≥ C_perfect and for X3's in-sample "realized bound"): fails for glm-5 (40.2 % of id_test tasks cost more than Opus), MiniMax-M2.7 (20.6 %), claude-sonnet-4-6 (18.7 %), qwen3.5-plus (7.9 %), gpt-5.4 (7.0 %); holds for Qwen3-Max (0 %), kimi-k2.5 (1 task) and all four RB primaries. The in-sample realized bound is exceeded by the Blackwell usage in 280 CRB θ = 1 rows, all with glm-5 (180), MiniMax (80) or sonnet (20) as primary, and never on RB or with the other four primaries; the closed-form bound with fit-split means is never exceeded by the Blackwell usage.
* **E[s_Z] = s.** Cross-fit anchoring enforces it up to clipping on CRB; on RB the isotonic posterior's held-out mean differs from the fit-split s by 0.016 on average. Probing → id_test prior drift: mean |Δs| = 0.019, max 0.19 (canitedit, n = 4); the uninformed choice flips in-sample in 5 of 210 (state, primary) pairs.
* **Monotone signal.** In 9.1 % of the half-split calibrations the binary lift is anti-monotone (P(H|Z=1) < P(H|Z=0)); forcing monotonicity changes gross VOI by ≤ $0.0019/task in any row.
* **Calibration out of sample (RB).** Raw LR ECE 0.157 (mean over states) → 0.097 after isotonic calibration on the fit split; state-level miscalibration remains because the calibrator is fitted on the pooled fit split.
* **Value preservation.** θ = 1: V_BW ≥ V_direct in every CRB row and erroneous acceptance e = 0. θ = 0.5: V_BW < V_direct in 12.6 % of rows, mean e_BW = 0.060; the θ = 0.5 rows are therefore outside eq:gate-costs' value-preserving premise and are reported for completeness only. RB: 1.0 % of rows, mean e = 0.008.
* **Constant k.** The released files carry no router tokens; k = 0 is the main case and k_tok is a prompt-length proxy at the protocol's self-hosted price.

## Audit (independent recomputation, pure Python `csv`/`json`/loops, no numpy/pandas)

| Number | run.py | independent | abs diff |
|---|---|---|---|
| in-sample native capture, ft-27B, pooled, Qwen3-Max, θ = 1 (X3 definition) | 0.594386 | 0.594386 | 1.7e-14 |
| same for logreg | −1.303785 | −1.303785 | 9.6e-14 |
| RB pooled mixtral closed-form bound ($/task) | 5.847043e-05 | 5.847043e-05 | 1.4e-20 |
| RB pooled mixtral gross native VOI ($/task) | −2.849168e-04 | −2.849168e-04 | 4.3e-19 |
| pooled Spearman(capture_BW, AUROC), hand-written average ranks | 0.025537 | 0.025537 | 3.5e-18 |

## Deviations from the protocol

1. The released decision files contain no scores, so Z is binary and s_Z cannot be calibrated on probing (no decisions exist there); calibration is cross-fitted on id_test with the probing prior as anchor (plus an in-sample variant). This is the only place where id_test enters a "primitive"; the uninformed choice, a/b, s and the bound are probing-only.
2. All seven non-Opus primaries are used (the protocol names Qwen routers' targets only), and the ACRouter release lookup policy is added as a tenth CRB signal.
3. RB "native usage" of a probability router is defined as its own threshold 0.5; the a/b-thresholded raw score is reported alongside.
4. Router cost k = 0 in the main analysis (no router tokens in the data); k ∈ {0.001, 0.01}·b and k_tok are sensitivities.
5. The C3 verdict rule was relaxed from "zero CI-above-0 rows" to the nominal false-positive rate (see Claims).
6. C2 is decided on |ρ| (the protocol's "stronger correlation"); the signed difference is also stored.
7. X12-b (live-judge signals as Z) is pending; SCOPE-Router scores are not available locally (NOT-INSTANTIABLE here).

## Limitations

* Binary Z makes the Blackwell value a two-point posterior problem: capture is a step function of whether the posteriors straddle a/b, so most rows have identically zero capture and the pooled correlation test (C2) has little power; a continuous released score would be needed to test the AUROC/ECE claim as intended.
* Cells are small in places (canitedit n = 4, he n = 32, swebench n = 19, RB mtbench n = 24); the C1 native-usage point violations and most C3 false positives live there.
* On RB the price ratio a/b ≈ 0.04–0.07 is so low that a TF-IDF router almost never lowers the posterior below it; the Blackwell capture ≤ 1.6 % is a statement about this router at these prices, not about RouterBench routers in general.
* θ = 0.5 rows violate the value-preserving premise; latency and token bases are not used (Sec. 4 is stated in dollars); the 148 zero-cost Qwen3-Max rows are kept as recorded.
* The k_tok proxy prices only the prompt; the released routers' true inference cost is unknown.

