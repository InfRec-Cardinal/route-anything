# RouterBench candidate-selection recheck

Independent replay from the released scores and dollar costs; no model calls.

## Table 1: pooled rerun results

Table 1 reports all ten rerun splits, with equal weight per retained split–context–base-portfolio comparison. The same weights define pooled hit rates, relative regret, and relative gain. These comparisons share tasks and nested contexts; they are not independent tasks or independent confirmation data. The original split is retained below as a control and is excluded from this aggregate. Normalized residual is an additional diagnostic and is not a Table 1 column.

| Threshold | Rule | Pooled hits / comparisons | Pooled rate | Pooled relative regret | Pooled relative gain |
|---|---|---:|---:|---:|---:|
| 0.5 | paired | 3537 / 5539 | 63.856% | 3.558% | 30.969% |
| 0.5 | residual | 3428 / 5539 | 61.888% | 3.711% | 30.770% |
| 0.5 | independence | 1986 / 5539 | 35.855% | 8.456% | 24.070% |
| 0.5 | saving | 3516 / 5539 | 63.477% | 3.540% | 31.010% |
| 0.5 | marginal | 807 / 5539 | 14.569% | 40.030% | -18.005% |
| 0.5 | normalized_residual | 3445 / 5539 | 62.195% | 3.682% | 30.809% |
| 1.0 | paired | 137 / 212 | 64.623% | 0.937% | 12.272% |
| 1.0 | residual | 114 / 212 | 53.774% | 1.359% | 11.794% |
| 1.0 | independence | 77 / 212 | 36.321% | 2.063% | 11.005% |
| 1.0 | saving | 102 / 212 | 48.113% | 1.682% | 11.438% |
| 1.0 | marginal | 0 / 212 | 0.000% | 35.970% | -27.458% |
| 1.0 | normalized_residual | 122 / 212 | 57.547% | 1.193% | 11.988% |

## All ten rerun splits

Every specified seed is included, without selecting favorable runs. Each split uses 25,542 fitting and 10,955 held-out tasks; filtering and ranking use fitting tasks only. The threshold-1 results exclude binary-score contexts.

| Seed | Threshold | Count | Paired | Residual | Independence | Saving | Marginal | Normalized residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20260924 | 0.5 | 564 | 62.411% | 60.284% | 31.383% | 61.525% | 15.603% | 61.170% |
| 20260924 | 1.0 | 21 | 42.857% | 76.190% | 61.905% | 61.905% | 0.000% | 76.190% |
| 20260925 | 0.5 | 557 | 63.196% | 60.682% | 36.984% | 62.118% | 15.619% | 61.041% |
| 20260925 | 1.0 | 23 | 78.261% | 56.522% | 39.130% | 52.174% | 0.000% | 60.870% |
| 20260926 | 0.5 | 552 | 63.768% | 61.594% | 33.696% | 63.587% | 14.674% | 61.232% |
| 20260926 | 1.0 | 21 | 47.619% | 23.810% | 9.524% | 23.810% | 0.000% | 33.333% |
| 20260927 | 0.5 | 551 | 62.795% | 62.976% | 33.938% | 62.613% | 12.523% | 62.795% |
| 20260927 | 1.0 | 21 | 71.429% | 66.667% | 42.857% | 33.333% | 0.000% | 66.667% |
| 20260928 | 0.5 | 556 | 65.468% | 62.050% | 37.230% | 65.108% | 14.388% | 62.410% |
| 20260928 | 1.0 | 21 | 52.381% | 33.333% | 14.286% | 19.048% | 0.000% | 38.095% |
| 20260929 | 0.5 | 560 | 63.571% | 62.679% | 35.179% | 65.357% | 12.500% | 63.036% |
| 20260929 | 1.0 | 21 | 85.714% | 52.381% | 23.810% | 71.429% | 0.000% | 52.381% |
| 20260930 | 0.5 | 558 | 62.545% | 58.423% | 36.201% | 60.036% | 11.111% | 59.319% |
| 20260930 | 1.0 | 21 | 66.667% | 38.095% | 19.048% | 33.333% | 0.000% | 38.095% |
| 20260931 | 0.5 | 549 | 62.295% | 59.563% | 36.248% | 60.109% | 14.936% | 59.745% |
| 20260931 | 1.0 | 21 | 90.476% | 57.143% | 42.857% | 47.619% | 0.000% | 61.905% |
| 20260932 | 0.5 | 537 | 64.060% | 64.432% | 39.665% | 66.294% | 18.436% | 64.246% |
| 20260932 | 1.0 | 21 | 38.095% | 80.952% | 61.905% | 80.952% | 0.000% | 85.714% |
| 20260933 | 0.5 | 555 | 68.468% | 66.306% | 38.198% | 68.108% | 16.036% | 67.027% |
| 20260933 | 1.0 | 21 | 71.429% | 52.381% | 47.619% | 57.143% | 0.000% | 61.905% |

## Equal-split sensitivity summaries

The following means give each split equal weight, unlike the configuration-weighted Table 1 aggregate above.

| Threshold | Rule | Split rate range | Mean split rate | Mean split regret | Mean split gain |
|---|---|---:|---:|---:|---:|
| 0.5 | paired | 62.30%–68.47% | 63.858% | 3.558% | 30.977% |
| 0.5 | residual | 58.42%–66.31% | 61.899% | 3.709% | 30.778% |
| 0.5 | independence | 31.38%–39.66% | 35.872% | 8.453% | 24.077% |
| 0.5 | saving | 60.04%–68.11% | 63.486% | 3.539% | 31.018% |
| 0.5 | marginal | 11.11%–18.44% | 14.583% | 40.040% | -18.016% |
| 0.5 | normalized_residual | 59.32%–67.03% | 62.202% | 3.680% | 30.817% |
| 1.0 | paired | 38.10%–90.48% | 64.493% | 0.938% | 12.268% |
| 1.0 | residual | 23.81%–80.95% | 53.747% | 1.357% | 11.795% |
| 1.0 | independence | 9.52%–61.90% | 36.294% | 2.063% | 11.003% |
| 1.0 | saving | 19.05%–80.95% | 48.075% | 1.681% | 11.437% |
| 1.0 | marginal | 0.00%–0.00% | 0.000% | 35.980% | -27.467% |
| 1.0 | normalized_residual | 33.33%–85.71% | 57.516% | 1.191% | 11.988% |

## Original-split control

This control reproduces the previously reported single-split results; it does not supply the current Table 1 RouterBench rows.

| Threshold | Comparisons | Rule | Hits | Rate | Mean relative regret | Mean relative gain |
|---|---:|---|---:|---:|---:|---:|
| 0.5 | 566 | paired | 378 | 66.7845% | 0.033474 | 0.336581 |
| 0.5 | 566 | residual | 381 | 67.3145% | 0.034688 | 0.334295 |
| 0.5 | 566 | independence | 206 | 36.3958% | 0.084962 | 0.263517 |
| 0.5 | 566 | saving | 382 | 67.4912% | 0.033963 | 0.335320 |
| 0.5 | 566 | marginal | 76 | 13.4276% | 0.435077 | -0.214026 |
| 0.5 | 566 | normalized_residual | 382 | 67.4912% | 0.033972 | 0.335867 |
| 1.0 | 21 | paired | 14 | 66.6667% | 0.009463 | 0.144695 |
| 1.0 | 21 | residual | 13 | 61.9048% | 0.022445 | 0.129187 |
| 1.0 | 21 | independence | 11 | 52.3810% | 0.033630 | 0.116493 |
| 1.0 | 21 | saving | 8 | 38.0952% | 0.036320 | 0.113500 |
| 1.0 | 21 | marginal | 0 | 0.0000% | 0.389622 | -0.294614 |
| 1.0 | 21 | normalized_residual | 14 | 66.6667% | 0.013783 | 0.139592 |

Threshold 0.5: all 566 paired selections maximize fitted efficiency. Maximum identity absolute/relative errors: 1.24e-10 / 7.54e-14. Held-out ties: 0.


Threshold 1.0: all 21 paired selections maximize fitted efficiency. Maximum identity absolute/relative errors: 4.86e-12 / 2.03e-14. Held-out ties: 0.

For threshold 0.5 on the original split, paired and residual select different candidates in 62 comparisons; paired yields higher held-out efficiency in 31 and lower efficiency in 31. Paired is uniquely top-ranked in 25 of these comparisons, while residual is uniquely top-ranked in 28. This explains the three-hit deficit despite lower average regret.


The paired identity orders candidate policies exactly on the fitting sample. It does not order their held-out estimates. Best-candidate frequency also ignores how much efficiency is lost when a non-best candidate is selected; relative regret and mean gain describe that distinction.

