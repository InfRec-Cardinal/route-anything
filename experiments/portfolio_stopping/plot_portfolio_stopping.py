"""Render the separate stopping ablation from archived aggregate results."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, default=Path(__file__).parent / 'portfolio_aggregate.csv')
p.add_argument('--output', type=Path, default=Path(__file__).parent)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(a.data)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'axes.edgecolor': '#8f969c', 'grid.color': '#e0e4e8', 'grid.linewidth': .6})
fig, ax = plt.subplots(figsize=(6.7, 3.25))
fig.subplots_adjust(left=.13, right=.99, bottom=.17, top=.74)
groups = [('forced_size', 2), ('forced_size', 3), ('positive_stop', 3)]
styles = [('standalone', 'Marginal success', '#7e8790', 'o'),
          ('independence', 'Priced independence', '#be8735', 's'),
          ('residual', 'Residual surplus', '#1b7b83', 'D'),
          ('full_conditional', 'Paired surplus', '#365b95', '^')]
for i, (rule, label, color, marker) in enumerate(styles):
    rows = pd.DataFrame([df[(df['mode'] == mode) & (df['size'] == size) &
                           (df.rule == rule)].iloc[0] for mode, size in groups])
    y = rows.relative_efficiency_gain_pct.to_numpy()
    ax.errorbar(np.arange(3) + (i - 1.5) * .15, y,
                yerr=np.vstack([y - rows.ci_low_pct, rows.ci_high_pct - y]),
                fmt=marker, ms=5.5, capsize=3, color=color, label=label, lw=1.2)
ax.axhline(0, color='#7e8790', ls='--', lw=.8)
ax.set_xticks(range(3), ['Exactly 2 members', 'Exactly 3 members', 'Up to 3; allow stopping'])
ax.set_xlim(-.48, 2.48)
ax.set_ylim(-26, 49)
ax.set_yticks([-20, 0, 20, 40])
ax.set_ylabel('Efficiency gain over\nsingleton + fallback (%)')
ax.grid(axis='y')
ax.set_axisbelow(True)
ax.text(2, 42.5, 'Residual minus independence\n4.30 pp [3.67, 4.90]',
        ha='center', va='center', fontsize=8, color='#176b72')
fig.legend(loc='upper center', bbox_to_anchor=(.55, .995), ncol=2,
           frameon=False, fontsize=9, columnspacing=2.4, handletextpad=.5)
for ext in ['pdf', 'png', 'svg']:
    fig.savefig(a.output / f'portfolio_stopping.{ext}', dpi=220,
                bbox_inches='tight', pad_inches=.05)
plt.close(fig)

