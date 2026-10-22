# X16: supplementary reference, resource, and noise sensitivity

This appendix-only analysis studies how reference policy, resource units, prices, verifier noise, and the score zero affect the retained routing and record results. It is supplementary evidence, not a main-paper experiment.

The historical `run.py`, `results.json`, and generated figures preserve a broader diagnostic sweep than the current paper and may regenerate analyses no longer reported. This README and the paper's supplementary section define the current reported scope; historical outputs are not an additional list of paper claims.

## Design

We use the released CodeRouterBench matrix: 9,999 tasks and eight models, with primitives fitted on 7,080 probing tasks and evaluated on 2,919 `id_test` tasks. The main random seed is 20260917. Task comparisons use 1,000 paired bootstrap resamples; the more expensive record-stream noise analysis uses 500.

The reference is Opus, the released DimensionBest choice, a per-task score oracle with a cheapest-dollar tie-break, or a uniform random model assignment from 20 draws. Changing the reference also changes a cascade's fallback. DimensionBest here uses the released choice file; the record-store experiment's probing-fitted prior is a separate protocol.

Resources are dollars, sequential seconds, and tokens. Exogenous dollar prices are 0.1, 1, 10, and 100; reference-efficiency pricing is also examined. Dollar costs are token-price estimates. The comparison includes seven fixed primary models with two acceptance thresholds, each followed by the chosen reference after rejection.

False-accept and false-reject probabilities each range over {0, 0.05, 0.1, 0.2, 0.4}, giving 25 combinations. Point estimates integrate the verifier coin. In the noise bootstrap each resampled task copy draws a fresh verifier outcome; for records, fresh outcomes replay the entire writing process. A separate value-zero check adds 0.1 or 0.5 to every score.

The record control uses 50 i.i.d. streams of 20,000 test tasks over 21 cells, with exploration probability 0.25, last-write replacement, no expiration, and a final 5,000-round stationary window. The informed and uninformed policies share the exploration cost and task stream. This is the oracle-write control retained alongside X13; the arrival and storage processes remain constructed.

## Reference and resource results

The same record system can improve one comparison and worsen another. At the primary threshold of 1:

| Reference and resource | Record efficiency difference, informed minus uninformed | Paired 95% interval |
| --- | --- | --- |
| Opus, score per dollar | +28.62 | [26.22, 30.90] |
| Opus, score per second | −0.0153 | [−0.0166, −0.0142] |
| DimensionBest, score per dollar | −23.83 | [−27.47, −20.17] |

The probing-fitted record criterion predicts these changes of sign. Thus an efficiency level alone cannot show whether records improve the existing policy.

None of the 14 fixed primary-then-Opus cascades improves seconds-based efficiency in the reported latency-valid comparison. Opus averages 7.8 seconds, compared with 8.2–44.9 for the other models. These are recorded-call comparisons, not a controlled concurrent-system measurement.

The historical sensitivity script retains zero-latency entries and gives a different mean: Opus 7.40 seconds. It also finds no sequential-seconds wins among the 14 cascades. The paper's latency-valid comparison excludes invalid latency entries; its means must not be mixed with the raw-script means. Missing Qwen3-Max runs recorded with zero score and cost remain in the historical dollar analysis.

## Verifier noise

Within the tested region where both error probabilities are at most 0.1, the retained qualitative conclusions do not reverse. This statement is restricted to that evaluated grid.

Redrawing verification errors widens record-stream efficiency intervals by a median factor of 1.214 over the 48 noise-dependent configurations, rounded to 1.21 in the paper; 94% of those intervals widen. A single writing error can determine the stored model for many later rounds, so reweighting tasks with an unchanged record history does not reproduce all of that uncertainty. The per-cell gain interval width ratio is 1.073 over 96 noise-dependent configurations.

The benefit of redrawing is specific to the statistic: per-task comparisons change little because task resampling already resamples the realized per-task errors. Larger intervals need not change a conclusion's sign.

## Value zero

For a common shift c added to every score, the reference-priced net gain changes by c S / E[C_B], where S is saved resource. Quality differences stay unchanged, but efficiency comparisons can change because the reference efficiency changes.

The evaluated shifts change the corresponding ratio-based criteria. For the Opus/dollar aggregate policy comparisons, neither shift reverses the realized efficiency-difference sign; movement of a criterion is not itself evidence of a ranking reversal. At c = 0.5, 16 hypothesis-met record error-threshold decisions change. Fixed exogenous-price comparisons and the cost-only information condition are shift-invariant.

## Interpretation and reproduction

Reference policy, resource unit, price, and value zero are part of a routing claim. Report them alongside quality and efficiency. The record findings concern the specified simulated store, and the noise findings concern the stated error grid.

The design changes one axis at a time around the baseline, plus a reference-by-resource grid; it does not enumerate every interaction. Probing and test are random splits of the same task pools. Full historical numerical outputs and the offline runner remain available for reproduction, while the current paper reports the focused results above.

