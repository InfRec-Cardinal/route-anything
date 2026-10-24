# October 24 manuscript: paper-to-code map

This map follows the supplied 33-page manuscript, whose title page reads **Route Anything: an attempt for AGI through a simple omni-modal router**. The file name of a downloaded PDF may retain an older title. Citation metadata follows the title page. No arXiv identifier is inferred from that file name.

## Current studies

| Paper location | Study | Code and reproduction scope |
| --- | --- | --- |
| §2, Theorem 2; §6.1 | Value minus priced cost; reference-relative quality floor | `experiments_x/X9_optimal_routing/`; `experiments/quality_floor/` adds independent certification. |
| §4, Tables 1–2 and 6 | Fixed arm, type-conditioned headroom, per-input oracle | `experiments/omnimodal/` supplies reusable matched evaluation; exact scores require the private matrix. |
| §5.1, Tables 3 and 7 | Capability-tree router | See the exact training/mapping coverage in `experiments/omnimodal/README.md`; no tree labels, fitted tables, or trained adapters are released. |
| §§5.2–5.3, Figure 3, Tables 3–4 | Robot outcome prediction and winner-label comparison | `experiments/omnimodal/robot/`; separately supplied camera frames, state and outcome labels are required. |
| §§5.3–5.4, Tables 4 and 8 | Outcome/type targets and matched fixed baseline | `experiments/omnimodal/`; the public code must be evaluated on the chosen matched panel. |
| §5.1 and Appendix B.3 | 124 image-routing configurations and twelve winner-label recipes | Not all historical training recipes are packaged; directory-level coverage takes precedence over a claim of complete reproduction. |
| §6.1, Table 5; Appendix C.2 | Fitted and certified code/text price routes | X9 plus `experiments/quality_floor/`; CodeRouterBench and separately obtained RouterBench. |
| Appendix C.2, selected rows of Tables 11–12 | Learned quality–cost routers and baselines | `experiments/learned_price/`; see its prompt preprocessing and implementation-coverage notes. |
| §6.1, Tables 5 and 9 | Image/speech price routes | Private matrix-dependent results; these data and derived price-route tables are not distributed. |
| §6.2 and Appendix C.3 | Portfolio additions on the same tasks | X2, X11, `paired_portfolio_analysis.py`, `router_selection_recheck.py`. |
| §6.2 and Appendix C.4 | Conditional failures and escalation | X10 and `additional_sign_baselines.py`; X7b and `extra_test_retry_analysis.py` retain earlier feedback protocols. |
| Appendix C.4, Table 14 | Retries with disjoint feedback and scoring tests | `experiments/disjoint_retry/` supplies the offline analysis; its newer call records are not in the legacy HF bundle. |
| §6.2 and Appendix C.5 | Verifiers, prices and overheads | X8 and `verifier_price_analysis.py` / `verifier_price_intervals.py`; the X8 archive uses a blended-price proxy, while main-text list-price claims come from the price sensitivity. |
| §6.2 and Appendix C.6 | Released signals and paid forecasts | X12a, X12b, and archived follow-up analyses; not every newer training/forecast recipe is bundled. |
| §6.2 and Appendix C.7 | Memory relative to its uninformed prior | X13 with X4 as an ideal-write control. |

The certification directory includes fitted family/dataset-state RouterBench policies, and the learned-price directory adds selected CPU learned-router rows. The release does not include every predicted-state or nested-resplit experiment in Tables 11–12; see each implementation’s coverage notes. In particular, the archived Table 11 TF-IDF prompt preparation differs from the manuscript’s wrapper-removal description; [reproduction notes](reproducibility.md#learned-router-prompt-provenance) explain the unresolved provenance difference.

X identifiers preserve the original archive names. X3, X5, X6, X7a, X12c, X16, portfolio stopping and R6 remain available as historical or supplementary analyses; their presence does not imply a current main-text claim. Old generated READMEs and archived reproduction notes may refer to earlier theorem and table numbers.

## Evaluation panels (Table 1)

These manuscript-level summaries document how to read the paper; no task-level records are included.

| Panel | Benchmarks | Test inputs | Training seeds | Fixed score | Oracle score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Headroom | 24 | 19,814 | — | 82.60 | 87.98 |
| Rule-based tree | 18 | 5,034 | — | 77.03 | 87.44 |
| One capability-tree router | 16 | 3,444 | 1 | 79.68 | 84.80 |
| Robot routers | 1 | 2,000 | 1 | 70.45 | 88.85 |
| Outcome/type targets | 11 | 1,302 | 4 | 68.30 | 75.84 |
| Winner-label recipes | 16 | 3,211 | 1 | 79.44 | 85.77 |

Scores use a 0–100 scale here. The image-routing suite has a separate 158 GenEval2 / 193 Qwen-Image-Bench panel. Image price routes use 1,515 inputs. Counts and baselines from different panels cannot be substituted for one another.

The headline 374,401 runs, 452 arm versions, 103,307 tasks, 104,050 routing decisions and 19,814 headroom inputs count different units. SciCode's model-dependent continuation inputs explain why it is excluded from one complete-panel comparison but appears in a router panel.

## Release boundary

The omni-modal matrix is proprietary. This release contains implementation code and manuscript-level descriptions only: no raw or derived matrix data, per-input results, media, split assignments, embeddings, fitted lookup tables, or matrix-trained weights. There is no Hugging Face upload for the October refresh. The public replay bundle is the earlier code/text dataset documented in [data.md](data.md).

Released synthetic checks demonstrate software behavior. They are not substitute measurements for Tables 1–9. GPU retraining and the full private-matrix evaluation have not been performed as part of preparing this release.
