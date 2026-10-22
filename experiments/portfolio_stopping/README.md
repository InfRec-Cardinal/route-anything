# Portfolio stopping ablation

This supplement compares forcing a portfolio to two or three members with stopping when its fitted incremental surplus is nonpositive. Candidate choices use probing tasks; quality, cost and paired intervals use the held-out CodeRouterBench tasks. It is an offline replay with oracle acceptance and recorded backend costs.

From the repository root:

```bash
python experiments/portfolio_stopping/portfolio_stopping.py --data data/agent-as-a-router/data/coderouterbench/id_results_long.csv
python experiments/portfolio_stopping/plot_portfolio_stopping.py --data experiments/portfolio_stopping/portfolio_stopping_results/portfolio_aggregate.csv
```

The first command writes task-level selections, aggregate results and 2,000 paired task-bootstrap resamples under `portfolio_stopping_results/`. The plotting script defaults to the archived aggregate CSV when `--data` is omitted. The archived figure reports residual minus independence as 4.30 percentage points [3.67, 4.90] for the stopping condition.
