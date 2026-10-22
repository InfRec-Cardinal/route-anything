# X4_memory_tail — Sec. 5 of "Route Anything" (record process, thm:memory, retention, regret, thm:tail, arm reveal) on CodeRouterBench

Real-data, semi-synthetic test of the memory results of the paper (`main_body.tex` Sec. 5; `appendix_learning.tex` app:memory,
app:memory-overhead, app:retention, app:memory-regret, proof of thm:tail, app:arm-coverage).
Script: `run.py` (numpy / pandas / scipy / matplotlib only; fixed seeds; runtime 49 s on this Mac). Outputs: `results.json`
(every number below, with CIs), `figures/fig1..fig7.pdf` (vector PDF, paper style: Times + `text.usetex=True`, which works on this
machine; a mathtext fallback is coded but was not needed), this README.

## 1. Data and what is real

`$ACROUTER_DATA/coderouterbench/id_results_long.csv`: 9,999 tasks x 8 models, `score` in [0,1],
`cost_usd` (token-price cost estimates), `latency_ms`. Splits: probing 7,080 / id_test 2,919. Cross-checks: always-Opus on id_test = mean score
0.4383, total $34.02; DimensionBest fitted on probing scores 0.5168 on id_test (the released `dimension_best.jsonl` chooses glm-5 on
algorithm/bug_fixing and scores 0.4986). 148 cost entries and 3,150 latency entries are 0 and are kept as recorded.

Every value V, dollar D and second that enters a measurement is a real (task, model) entry of this matrix. Nothing about outcomes or
costs is generated. The constructed elements are listed in Sec. 3.

## 2. Instantiation of the paper's objects

* **Cells** j: 21 = dimension x source-benchmark prefix (`task_id.split('_')[1]`); id_test sizes 4 (code_refactoring|canitedit) to 327.
  Also J = 9 dimension cells (Block 1f). nu_j = the id_test cell frequencies (the population of the i.i.d. stream).
* **Stream**: i.i.d. draws with replacement from the 2,919 id_test tasks, 50 seeds x 20,000 epochs (Blocks 1, 1f, 2); 300 seeds for the
  retention block (1e); Zipf-resampled cells for thm:tail (Block 3). Stationary quantities are read on the last 5,000 epochs.
* **Verifier**: exact = `score >= theta`, theta in {1.0, 0.5}. Noisy: the exact verdict is flipped with false-accept probability
  beta_fa in {0, 0.05, 0.1, 0.2, 0.4} (plus one config beta_fa = 0.1, beta_fr = 0.2). Noise enters only the writes; measured values are
  always true scores.
* **Cold policy U**: DimensionBest fitted on probing (per-dimension argmax of mean score: algorithm -> qwen3.5-plus, bug_fixing /
  code_completion / code_generation / multi_language -> claude-opus-4-6, code_refactoring -> gpt-5.4, code_understanding /
  data_science -> kimi-k2.5, test_generation -> Qwen3-Max), and always-Opus (`opus`).
* **Write mechanism (the paper's omega_j, psi_j)**: on every epoch, with probability eps = 0.25 the router *explores*: it fields a
  uniformly random model instead of its policy's model, and the verifier's verdict on that real outcome decides the write. Accepted
  -> a record naming the explored model is written to the arriving cell (last write wins, written after the decision, first used at
  epoch n+1). If the explored model truly hit the record is informative (I); if it was falsely accepted it is erroneous (W). Write
  types are therefore independent of the current record and of the past, as eq:record-rate requires, with
  `omega_j = eps (1-beta_fr) mean_m P_jm`, `psi_j = eps beta_fa mean_m (1-P_jm)`, `r_j = omega_j + psi_j`,
  P_jm = P(score_m >= theta | cell j). Exploration is part of both the cold reference and the informed policy ("collects records at the
  same cost"), so v_j^U, c_j^U include it: `v_j^U = (1-eps) V_j[cold] + eps mean_m V_jm`, and analogously for costs.
* **Informed policy I / erroneous W**: on non-exploration epochs field the recorded model. Given an informative record the model is m
  with probability P_jm / sum_m P_jm, given an erroneous one (1-P_jm)/sum_m(1-P_jm); so
  `v_j^I = (1-eps) sum_m q^I_jm V_jm + eps mean_m V_jm`, etc. (`primitives()` in `run.py`).
* **Expiration**: geometric with delta_j = delta in {0, 0.01, 0.05} (one lifetime coin per write: P(survives k expiration coins) =
  (1-delta)^k; a record written after epoch n_w is present at n_w+1 with probability 1 and at n_w+k with probability (1-delta)^(k-1),
  exactly the paper's timing; `alpha_j = (1-delta)(1-nu_j r_j)`).
* **Resources**: dollars = `cost_usd` of the fielded call (ex-ante router: one call per epoch); seconds = its latency. Both are run.
* **eta_U, g_j^s, G_j**: eq:cold, eq:record-surplus, eq:stationary-record-gain evaluated from the primitives above ("in": estimated on
  the id_test pool, i.e. the population of the stream; "oos": estimated on the 7,080 probing tasks with probing nu_j).
* **Bootstrap**: 500 task-cluster resamples of the 2,919 id_test tasks for every quantity that is a ratio or sum of per-epoch outcomes
  (b, c, d, f, overhead; dynamics held fixed, epoch contributions re-weighted by task); 500 resamples of the independent streams
  (seed bootstrap) for coverage / record-probability quantities (a, e, g, h), whose per-epoch indicators depend on the whole history and
  have no task-cluster decomposition.

## 3. Semi-synthetic components (documented mechanisms; no outcome or cost is fabricated)

1. i.i.d. arrival order of real id_test tasks (Zipf frequencies over real tasks in Block 3).
2. Exploration coin (eps = 0.25) and the explored model (uniform over 8).
3. Verifier-noise coins (beta_fa, beta_fr) applied to the exact verdict on the real outcome.
4. Expiration lifetimes (geometric) and TTL / FIFO retention windows.
5. Block 3 (thm:tail): 20,000 Zipf cells, each assigned (uniformly with replacement) one of the 234 id_test tasks on which DimensionBest
   misses at theta = 1 and some model hits; the clean exact-reveal logger records, with probability omega = 0.5 on arrival, the cheapest
   hitting model. This realises the theorem's hypotheses exactly (clean writes, common omega, common Delta = 1) with real outcomes.
   A hypothesis-violating control (2,919 singleton id_test cells, uniform-model logger, heterogeneous omega_j, Delta_j, 46-97% of
   arrivals with zero gap) is run alongside.
6. Block 4 (lem:app-arm-reveal): arms = the 8 models, mu_m = real id_test mean score in the 21-cell, logger reveals mu_m exactly with
   probability e_m per visit (a logging-coin model, as in the lemma).

## 4. Circularity statement

The primitives (P_jm, V_jm, C_jm, omega_j, psi_j) and the measured stream are functions of the same outcome matrix, so the "in-sample"
checks of the identities (a, b, f, overhead identity, e, g-mechanism, h) test whether the *mechanism* as implemented matches the
paper's model — they are identities plus sampling checks, not findings about routers. The hypothesis tests (c, d, e-inexactness,
overhead threshold) ask whether the *sign* predicted from the primitives matches an independently measured paired difference of
efficiencies; in-sample they test the theorem's logic on this data, out-of-sample (probing primitives, id_test measurement) they also
test probing -> id_test shift. The write is decided by the verified outcome of the *explored* model and the value is the true outcome of
the *fielded* model on a *different* later task, so the measured gain is not a restatement of the primitives. What is NOT tested: that
any deployed router logs the way eq:record-rate stipulates (a greedy logger has state-dependent writes; see E7 C10 in the earlier
experiment set), and that informative records help by default (g_j^I < 0 for most cells against DimensionBest here).

## 5. Results (verdict per claim; every number is in `results.json`)

Verdict criteria were fixed before the first full run except where marked "changed after first run" (Sec. 6 reports the first-run values).

### (a) eq:record-probs / eq:app-memory-recurrence — IDENTITY (mechanism check) + sampling check — **PASS**
For all 21 cells, states I and W, all 32 (theta, beta_fa, beta_fr, delta) configs (1,106 cell-state-configs): the fraction of 250-epoch
bins with |z| < 2 is 0.981 (z uses max(seed SE, model SE with lag-k autocorrelation alpha_j^k)); max |z| over all bins 3.66; the late-window
prediction is covered (seed CI or 0.005 tolerance) in 96.1% (strict CI 87.6%; the strict failures are degenerate CIs at coverage 1 or 0);
late-window model-z has mean 0.03, sd 0.94, 98.4% within |2|. Median RMSE of the 20,000-epoch curve 0.018. Example: algorithm|lcb,
theta = 1, beta_fa = 0.2, delta = 0: i* = nu omega/(1-alpha) = 0.0956 x 0.0850/(1-0.98872) = 0.720, w* = 0.280 (late-window measured
0.716 / 0.284) (Fig. 1). [Audit: the numbers previously printed here, 0.64 / 0.27, were stale and do not match `results.json`; at delta = 0
coverage is 1 so i* + w* = 1 necessarily.]

### (b) eq:finite-memory — IDENTITY + sampling check; OUT-OF-SAMPLE prediction — **PASS / PASS**
Prediction `sum_j nu_j G_j [N - (1-alpha_j^N)/(1-alpha_j)]` vs measured `sum_{n<N} E[V_n - eta_U D_n]` at N in {100, ..., 20000},
32 configs x 2 cold x 2 resources (128 configs, 768 points). In-sample: paired task-bootstrap CI covers the prediction in 96.7% of points
(unpaired 99.7%); median normalised error |meas-pred| / (N sum nu_j |G_j|) at N = 20,000 is 0.0074. Example theta = 1, beta_fa = 0,
delta = 0, Opus cold, dollars, in-sample: predicted +3357 (units of score; eta_U = 42.8 score/$), measured +3402 [3070, 3765] unpaired,
+3393 [3108, 3707] paired (Fig. 2 left). The same config out-of-sample (probing primitives, eta_U^probe = 39.6): predicted +3484,
measured +3776 [3449, 4128] unpaired, +3143 [2870, 3444] paired -- the paired CI excludes the OOS prediction. [Audit: the original
text quoted the OOS numbers as the in-sample example.] Out-of-sample (all primitives and nu from probing): sign agrees in 100% of 128 configs; median
normalised error 0.057 (unpaired, CI covers 80%) / 0.086 (paired, CI covers only 25%: the probing -> id_test shift in the record effect is
statistically detectable at this precision, about 8% of N sum nu|G|, always with the right sign). PASS criterion was sign >= 0.80 and
median normalised error < 0.15.

### (c) Stationary sign criterion eta_inf >= eta_U iff sum_j nu_j G_j >= 0, and eq:app-memory-relative-efficiency — IN-SAMPLE and OOS hypothesis tests — **PASS / PASS**
Measured: paired late-window eta^I - eta^U on the same epochs (task bootstrap). Confusion (in-sample) TP 32 / TN 96 / FP 0 / FN 0,
no inconclusive; identical out-of-sample. Both signs occur, but the sign is set entirely by (cold, resource): DimensionBest cold ->
memory hurts in dollars and seconds (sum nu G in [-0.13, -0.01] and [-0.32, -0.01]); Opus cold -> memory helps in dollars
(sum nu G in [0.02, 0.20]; eta rises from 42.9 to 72-83 score/$) but hurts in seconds (a random accepted model is slower than Opus,
whose 7.7 s mean latency is the lowest). Magnitude: predicted eta_inf - eta_U = sum nu G / c_inf covers the measured difference in
99.2% of configs, median relative error 1.5% (Fig. 2 right). The J = 9 variant (16 configs) gives TP 4 / TN 12, accuracy 1.0, and the
finite-memory identity covered in 16/16 (Block 1f).

### (d) eq:noise-threshold psi_j/r_j < g_j^I/(g_j^I - g_j^W) for cells with g_j^I > 0 > g_j^W — per-cell hypothesis test — **PASS (one-sided in-sample) / PASS (OOS, 0.96)**
The hypothesis g^I > 0 > g^W is met in only 144 of 2,604 in-sample cell-configs (3 cells: code_completion|he, code_generation|humaneval,
code_refactoring|bugs2fix at theta = 0.5 with Opus cold; all dollars) and 128 OOS (adds code_refactoring|canitedit, algorithm|lcb).
Measured: paired late-window surplus difference restricted to the cell's epochs. In-sample: 84 conclusive, all predicted and measured
positive (TP 84, accuracy 1.0); the predicted negative side (err = 0.259 > thr = 0.181, code_refactoring|bugs2fix at beta_fa = 0.4) is
inconclusive (CI [-0.018, 0.008]). OOS: TP 62 / TN 14 / FP 3 / FN 0 (accuracy 0.962); the 14 TN are algorithm|lcb (Opus, seconds) with
threshold 0.007 and err >= 0.089; the 3 FP are the same cell at beta_fa = 0 where OOS g^I = +0.0003 is within noise of zero (in-sample
g^I < 0, hypothesis not met). Example plugged in: code_completion|he, theta = 1, DimensionBest, dollars: g^I = 0.262, g^W = -0.090,
threshold 0.745; err ranges 0 to 0.061 across beta_fa, measured gain 0.26 -> 0.23 (all CIs > 0) (Fig. 3). The negative side of the
threshold is therefore exercised only out-of-sample; the in-sample verdict is flagged one-sided.

### (e) app:retention — eq:app-ttl, eq:app-fifo — IDENTITY + sampling check; inexactness of E -> M/beta — IN-SAMPLE hypothesis test — **PASS / PASS / PASS**
Dedicated 300-seed x 20,000-epoch replay of the same real-outcome write mechanism (theta x beta_fa in {0, 0.2}), E, M in {5, 10, 20, 50,
200, 1000}, 462 (cell, E) points; FIFO measured only on epochs where >= M writes have occurred. TTL: prediction covered in 96.8%
(strict CI 84.8%); FIFO: 94.8% (strict 80.7%; most strict failures have predicted coverage > 0.99 where the seed CI of a measured 1.0 is
degenerate; the rest cluster in 4 cells whose empirical mark frequency deviates by ~2 SE from q_j and are shared by all four configs
because they use one replay stream — `code_checks.fifo_mark_law_humaneval`). [Audit: the original run reported TTL 97.2% / FIFO 90.7%;
the 17 cell points and the aggregate at theta = 1, beta_fa = 0, M = 1000 were NaN (FIFO never full in 39 of 300 streams) and were counted
as not covered; after the fix all 462 points are defined and the 18 formerly-NaN points are all covered. TTL point estimates are
unchanged; its coverage moved by 2 borderline points because the bootstrap RNG stream shifted.] Inexactness: the naive substitution E -> M/beta always
UNDER-predicts FIFO coverage (naive - exact < 0 in 100% of points, up to 0.042 per cell, mean 0.007 at M = 5); among the 84 resolvable
points (|naive - exact| > 2 SE, M <= 20) the measurement is closer to the exact formula in 97.6%, the naive formula is rejected in 97.6%
and the exact one in 3.6% (original run: 86 / 97.7% / 96.5% / 5.8%); on the arriving-cell aggregate (sum_j nu_j chi_j) at M <= 20 the naive formula is rejected in 12/12 configs
with mean |z| = 16.6 while the exact formula has mean |z| = 0.66 (e.g. theta = 1, beta_fa = 0, M = 5, beta = 0.0516: measured 0.2782
[0.2770, 0.2793], exact 0.2775, naive 0.2659) (Fig. 4).

### (f) eq:app-regret — IDENTITY + sampling check; OOS prediction — **PASS / PASS**
s = v - eta_U c, s^O_j = best fixed arm of the cell, delta = 0, empty start; 48 configs x 6 horizons. In-sample: CI covers 98.3% of
points; median relative error at N = 20,000 is 0.34%. OOS: 3.1% median relative error, CI covers 91.7% at N = 20,000. The transient
term is signed as the appendix warns: in every one of the 48 configs some cells have s_j^inf < s_j^U (2-20 of the 17-21 writable cells), and
the transient is negative in 26/48 configs (e.g. theta = 1, DimensionBest, dollars: linear 10611, transient -551, measured 9977
[9375, 10551]) (Fig. 5).

### app:memory-overhead — eq:app-arbitrary-memory-surplus (lambda = eta_0) IDENTITY; eq:app-memory-maintenance-threshold IN-SAMPLE — **PASS / PASS**
Instantiation: memory-enabled cold state = cold policy + one shadow call (uniform model, recorded token-price cost estimate, verified and written every
epoch); no-memory reference = the cold policy alone (same value, cost c_0), h = mean shadow cost = $0.00497/epoch (30% of the Opus
cold bill, 53% of the DimensionBest bill). J_U(eta_0) = -eta_0 h holds to 1e-16; the N = 20,000 identity's CI covers the prediction in 8/8
configs (median relative error 0.9%); the threshold sum nu G(eta_0) >= eta_0 h predicts the sign of the measured eta_late - eta_0 in
8/8 (Opus cold: 0.200-0.230 >= 0.187 -> memory beats the no-memory reference despite paying for the shadow call, measured +1.6 to +4.7
score/$; DimensionBest: -0.19 < 0.57 -> loses, measured -72 to -74). [Audit: h/(c_0+h) = 30% / 53% is the shadow call's share of the
memory-enabled bill, i.e. h = 43% of the Opus-alone bill and 111% of the DimensionBest-alone bill.]

### (g) thm:tail — mechanism IDENTITY + sampling check; exponents = asymptotic prediction at finite E, N — **PASS / PASS**
z in {1.5, 2, 3}, 20,000 cells, omega = 0.5, Delta = 1 (verified on every epoch: recorded model hits, cold misses). Seed CI covers the exact
finite-J occupancy sum in 48/48 points. Log-log slopes over E in [200, 2000] / N in [2000, 20000]: TTL deficit -0.349 / -0.504 / -0.672 vs
-(z-1)/z = -0.333 / -0.500 / -0.667; cumulative 0.649 / 0.503 / 0.346 vs 1/z = 0.667 / 0.500 / 0.333 (all within 0.02 of the target and
within 0.004 of the exact finite-J sums; the z = 1.5 residual 0.016-0.018 is finite-J truncation: the infinite sum's slope over the same
range is -0.334 / 0.666). Hypothesis-violating control (singleton id_test cells, heterogeneous omega_j, Delta_j): slopes -0.27 / -0.80 /
-0.59 — the exponents do not hold when the theorem's uniformity hypotheses fail, as the appendix states (Fig. 6).

### (h) lem:app-arm-reveal — probability bound + sampling check — **PASS**
21 cells x 100 visits x 4,000 replicates, e_{m*} in {0.02, 0.05, 0.1, 0.2}, e_other in {0.1, 0.3}: the bound (1-e_{m*})^v is violated
(measured > bound + 2.5 SE) in at most 0.24% of (cell, v) points; the measured probability is 0.57-0.83 of the bound (the bound is not
tight because an unrecorded best arm is often replaced by a recorded equal-valued or near-best arm). With e_{m*} = 0 the recommendation
equals the best value within the logger's support after 200 visits in 100% of replicates in every cell (Fig. 7).

## 6. Test-design changes made after the first full run (first-run values kept for the record)

* (a) first run: bins with |z| < 2 = 0.813, late CI coverage 0.873, max |z| = 8e7, verdict FAIL — caused by seed SEs that are exactly 0
  when all 50 seeds show coverage 1.0 (prediction 0.99999) or 0 (a W-record probability of 0.0016, about one write in the whole run).
  Changed: SE floor from the record chain's autocorrelation, and late coverage = CI or 0.005 tolerance. The mechanism was not changed;
  the sequential re-implementation check (`code_checks.sequential_record_replay_seed0`) reproduces the vectorised states bit for bit.
* (e) first run (50 seeds, same streams as Block 1, FIFO not masked): TTL 0.939 PASS, FIFO 0.827 FAIL, inexactness INCONCLUSIVE (naive
  rejected in 19% of M <= 20 points, exact in 3.7%, naive gap larger in 59%). Diagnosis: degenerate CIs at coverage ~1, the M = 1000 FIFO
  not yet full at epoch 15,000 (beta ~ 0.05-0.13 writes/epoch), and the 0.007 naive-exact gap below the 50-seed resolution. Changed:
  dedicated 300-seed replay, FIFO measured on full-FIFO epochs, resolvable-point and aggregate criteria. Brute-force FIFO/TTL indicators
  on seed 0 equal the vectorised ones (`code_checks`).
* (g) first run with J = 4,000 cells: z = 1.5 slopes -0.366 / 0.616 (deviations 0.033 / 0.051 from target, still within the 0.10 tolerance);
  J raised to 20,000 to reduce truncation (tail mass bound 5e-4 for z = 1.5). Cells were also moved from all 9,999 eligible tasks to the
  234 id_test-eligible tasks so that every measurement uses id_test only. A broadcasting bug in the control block (`.any(1)` on a 3-d
  array) was fixed; it had crashed the first run before Block 4.

## 7. Limitations

* The write mechanism is an explicit exploration logger designed to satisfy eq:record-rate's state-independence; it is not the greedy
  logger of any deployed router, and against DimensionBest a single "model m passed once" record is a worse decision than the cold prior
  in most cells (g_j^I < 0), so the memory hurts there. The theorem's sign rule predicts this correctly, but the experiment does not show
  that memory helps a strong cold policy.
* The noise threshold (d) is instantiable in only 3-5 of 21 cells, and its negative side is conclusive only out-of-sample (17 cell-configs,
  one cell); in-sample it is one-sided.
* Coverage-type CIs are seed-bootstrap CIs; the four retention configs share one replay stream, so their per-cell deviations are
  correlated and the per-cell strict-CI failure rates (15-23%) overstate independent evidence against the formulas; the aggregate tests
  and the 0.005 tolerance are the operative criteria.
* Out-of-sample predictions inherit a probing -> id_test shift of roughly 8% of the record-effect scale in (b) (sign always right).
* thm:tail is tested with Zipf frequencies constructed over real tasks (each of 234 tasks serves ~85 cells); the exponents are asymptotic
  and were read off one decade of E and N.
* Streams are i.i.d. with replacement, so tasks recur; 20,000 epochs do not fill the rarest cells (canitedit: nu r ~ 5e-6 at theta = 1).

## 8. Files

* `run.py` — the whole experiment (`python3 run.py`, ~50 s).
* `results.json` — claims (verdicts, criteria, numbers), tables (per-cell/per-config rows for every block), primitives, code checks.
* `figures/fig1_record_probs.pdf` (a), `fig2_finite_memory_sign.pdf` (b, c), `fig3_noise_threshold.pdf` (d), `fig4_retention.pdf` (e),
  `fig5_regret.pdf` (f), `fig6_heavy_tail.pdf` (g), `fig7_arm_reveal.pdf` (h).

## 9. Audit (skeptical review, 2026-09-11; original script kept as `run_orig.py`, original outputs as `results_orig.json`, `README_orig.md`, `figures_orig/`; independent checks in `audit/`)

**Re-run.** `run_orig.py` re-executed from scratch (56 s): every number in `results_orig.json` reproduced exactly except `runtime_sec`.
The corrected `run.py` (one bookkeeping fix, marked `AUDIT`) leaves every block except 1e bit-identical (`claims`, `tables`,
`code_checks` compared field by field); figures regenerated (vector PDF, Times/NimbusRomNo9L + CM math via usetex, same fonts as
`paper/figures/memory.pdf`).

**Bug fixed (direction: conservative).** `seed_ci` propagated NaN per-seed means. In Block 1e the FIFO of size M = 1000 never fills
within the late window in 39 of the 300 streams at theta = 1, beta_fa = 0 (beta = 0.0516 writes/epoch), so the per-seed FIFO coverage
was NaN there, the seed-bootstrap mean was NaN, and the 17 cell points plus the aggregate row were counted as *not covered*. Dropping
undefined seeds is unbiased (the mark law is independent of the write count). Corrected (e): FIFO covered 94.8% (was 90.7%), strict
80.7% (was 77.5%); all 18 formerly-NaN points are covered (e.g. aggregate M = 1000: measured 0.7550 [0.7492, 0.7609] vs exact 0.7565);
TTL point estimates unchanged, covered 96.8% (was 97.2%, two borderline points flipped because the bootstrap RNG stream shifted);
resolvable points 84 (was 86), measurement closer to exact 97.6%, naive rejected 97.6%, exact rejected 3.6%. Verdicts unchanged.

**README errors corrected in place (marked `[Audit: ...]`).** (i) The (a) example printed i* = 0.64, w* = 0.27 — stale numbers
inconsistent with `results.json` (0.720 / 0.280) and with delta = 0, where i* + w* = 1. (ii) The (b) "in-sample" example was the
out-of-sample row (pred +3484, meas +3776 / +3143); the in-sample row is pred +3357, meas +3402 [3070, 3765] unpaired / +3393
[3108, 3707] paired. (iii) "30% / 53% of the cold bill" is h/(c_0+h), i.e. the share of the memory-enabled bill.

**Independent checks that did not change a verdict (`audit/audit_checks.py`, `audit/hitrate.py`, `audit/marklaw.py`).**
* Write coins recomputed from first principles for seed 0 (three noise configs) equal `Writes.acc`/`wtype` on all 20,000 epochs.
* A from-scratch per-epoch-coin simulation of the paper's record process (decision -> expiration coin -> arrival -> write coin; nu = 0.05,
  omega = 0.3, psi = 0.1, delta = 0.02; 4,000 chains x 600 epochs) matches eq:record-probs (99.2% / 96.3% of epochs with |z| < 2 for I / W,
  max |z| 2.76). The lifetime trick in `Writes` is therefore equivalent to the paper's per-epoch expiration coin.
* The FIFO deviation of code_generation|humaneval flagged in `code_checks.fifo_mark_law_humaneval` (mark share +2.0 SE above q_j in the
  late window of the seed-555 replay) recurred in two fresh 300-seed replays (+2.4, +1.8 SE), but 16 further independent 100-stream
  replays give late-window mean z = -0.14 +/- 0.25 for that cell and a mean chi-square of 17.8 on 16 df over all cells, and the
  per-cell write rate nu_j r_j is unbiased across 5 replays (z = 2.2, 0.8, -1.3, -0.4, -1.1). Pooled, the deviation is ~1.7 SE; no
  mechanism defect was found.
* No probing -> id_test leakage: the cold policy and the "oos" primitives are fitted on the 7,080 probing tasks only; every measured
  V, D is an id_test entry; "in-sample" primitives are the stream's own population parameters and are labelled as identity checks.

**Qualifications the verdict labels understate (not verdict reversals).**
* (b) OOS "PASS" is by the pre-declared sign/normalised-error criterion. The paired 95% CI excludes the OOS prediction in 75% of the
  128 configs at N = 20,000 (median relative error 9.8%; signed shift +0.13 of the record-effect scale for DimensionBest/dollars, -0.11
  for Opus/dollars). The OOS prediction is directionally right and within ~10%, but statistically distinguishable from the measurement.
* (c) "TP 32 / TN 96" counts configurations, but the predicted and measured signs are constant within each of the four (cold, resource)
  groups: the evidence is four sign predictions (plus the same four in the J = 9 variant), not 128 independent ones. Similarly the
  overhead threshold's "8/8" is two sign cases (Opus vs DimensionBest) repeated over theta and beta_fa.
* (d) The negative side of eq:noise-threshold is exercised only out-of-sample and only in algorithm|lcb (Opus, seconds), where
  g^I = +0.0003 is within noise of zero and the threshold is 0.007 — the trivial case in which any erroneous record hurts. The 3 FP are
  the same cell at beta_fa = 0, where in-sample g^I = -0.095 (hypothesis not met); this is a probing -> id_test shift of g^I, not a
  failure of the theorem's logic. Within the hypothesis-met cells the test is effectively one-sided.
* (g) and (h) use real outcomes only as labels: in Block 3 every eligible task has cold miss / recorded hit by construction, so the
  measured deficit is exactly the occupancy indicator (Delta = 1) and the block is a Monte-Carlo check of the occupancy sums, not a data
  result; in Block 4 the process is pure logging coins over fixed cell means. Both are appropriate mechanism checks of pure-math
  statements but should not be described as real-data tests.
* Verification is free in all blocks (the fielded call's cost is charged; the verifier's own computation is not), so c_j^s "including the
  computation used to update the record" is instantiated with zero verification cost.

**Verdict summary after audit:** (a) PASS; (b) identity PASS, OOS PASS (qualified as above); (c) PASS / PASS (four distinct sign cases);
(d) PASS one-sided in-sample / PASS OOS (negative side from one degenerate cell); (e) PASS / PASS / PASS with corrected FIFO numbers;
(f) PASS / PASS; overhead PASS / PASS (two sign cases); (g) PASS / PASS (mechanism); (h) PASS. Runtime of the corrected script 47 s.

