"""Reproduce the paper's analytical figures and finite-example checks.

Run from any directory with Python 3, numpy, scipy, and matplotlib installed.
A LaTeX installation is optional: when ``latex`` is on the PATH the figures are
typeset with Matplotlib's usetex mode and the Times family (this is how the
committed PDFs were produced); otherwise Matplotlib's own text engine is used,
which changes fonts but no numbers. The CSV tables and summary.json are
deterministic and do not depend on the font setting. (The ``epoch`` column of
``memory.csv`` keeps its historical name so that the committed table is
reproduced byte for byte; the paper calls the same index a round.)

Outputs go to ``figures/`` and ``numerical_results/`` in the source root
(the parent of ``scripts/``) unless overridden with ``--outdir DIR`` or the
environment variable ``REPRODUCE_OUTDIR``, in which case they go to
``DIR/figures`` and ``DIR/numerical_results``.

All plotted outcomes are expectations in specified models, not LLM benchmarks.
"""
import argparse
import csv
import json
import os
import shutil
from fractions import Fraction as F
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, zeta

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--outdir', default=os.environ.get('REPRODUCE_OUTDIR'),
                    help='write figures/ and numerical_results/ under this directory '
                         '(default: the source root; also settable via REPRODUCE_OUTDIR)')
args = parser.parse_args()
ROOT = Path(args.outdir).expanduser().resolve() if args.outdir else Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
DATA = ROOT / 'numerical_results'
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

USETEX = shutil.which('latex') is not None
plt.rcParams.update({
    'font.family': 'serif', 'text.usetex': USETEX,
    'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': .65, 'lines.linewidth': 1.6,
})
if USETEX:
    # Times text and (via Matplotlib's TeX preamble) mathptmx math, as in the committed figures.
    plt.rcParams['font.serif'] = ['Times']
# No timestamps in the PDF Info dictionary, so repeated runs give identical bytes.
PDF_METADATA = {'Creator': 'Analytical reproduction script', 'CreationDate': None, 'ModDate': None}
BLUE, RED, GREEN = '#23639b', '#ba5934', '#3a7c5e'

def save_csv(name, fields, rows):
    with (DATA / name).open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(fields)
        writer.writerows(rows)

q = np.array([.30, .20, .40, .10])
p1 = np.array([.45, .45, .05, .05])
p2 = np.array([.40, .40, .10, .10])
rho = np.linspace(1, 0, 201)
steered = np.array([p2 * np.array([r, r, 1, 1]) / (.8*r+.2) for r in rho])
fail = (1-p1) * (1-steered)
hit = 1-fail @ q
cost = 2+10*(1-hit)
eta_uniform = (hit+.8*(1-hit))/cost
eta_typed = (hit+fail @ (q*np.array([0, 0, 1, 1])))/cost
assert np.allclose(steered @ q, .25)
assert np.all(np.diff(hit) > 0)
assert eta_uniform[-1] > eta_uniform[0]
assert eta_typed[-1] < eta_typed[0]

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.13), layout='constrained')
ax = axes[0]
ax.plot(1-rho, eta_uniform/eta_uniform[0], color=BLUE, label='Type-independent fallback')
ax.plot(1-rho, eta_typed/eta_typed[0], color=RED, label='Type-dependent fallback')
ax.axhline(1, color='.55', lw=.65, ls=':')
ax.set(xlabel=r'Steering strength $1-\rho$', ylabel='Relative efficiency', xlim=(0, 1))
ax.legend(loc='upper left', frameon=False)
ax.text(.98, .03, r'Acceptance: $0.4075\to0.4875$', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=8)

s = np.linspace(0, 1, 1001)
primary, backup, router = .1, 1., .02
cascade = primary+(1-s)*backup
direct = np.full_like(s, backup)
informed = router+s*primary+(1-s)*backup
low, high = router/(backup-primary), 1-router/primary
ax = axes[1]
ax.axvspan(low, high, color=GREEN, alpha=.10)
ax.plot(s, cascade, color=BLUE, label='Cheap-first')
ax.plot(s, direct, color='.4', ls='--', label='Direct fallback')
ax.plot(s, informed, color=GREEN, label='Exact routing signal')
ax.axvline(low, color=GREEN, lw=.75, ls=':')
ax.axvline(high, color=GREEN, lw=.75, ls=':')
ax.set(xlabel=r'Sufficiency probability $s$', ylabel='Expected resource', xlim=(0, 1), ylim=(0, 1.16))
ax.legend(loc='lower left', frameon=False)
fig.savefig(FIG/'allocation.pdf', metadata=PDF_METADATA)
plt.close(fig)
save_csv('steering.csv', ['rho','acceptance','standalone_quality','expected_cost','eta_uniform','eta_typed'],
         zip(rho, hit, steered@q, cost, eta_uniform, eta_typed))
save_csv('routing_band.csv', ['s','cascade_cost','direct_cost','informed_cost'],
         zip(s, cascade, direct, informed))

n = np.arange(121)
nu, write_rate, expiration = .1, .5, .01
retention = (1-expiration)*(1-nu*write_rate)
coverage = nu*write_rate/(1-retention)*(1-retention**n)
errors = [0.1, .375, .65]
curves = [.6+coverage*(.3-.8*err) for err in errors]
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.12), layout='constrained')
for error, y, color in zip(errors, curves, [BLUE, '.45', RED]):
    axes[0].plot(n, y, color=color, label=fr'Erroneous write fraction $e={error:g}$')
axes[0].set(xlabel='Round', ylabel='Expected value / unit cost', ylim=(.39, .81))
for e, y, color in [(0.1, .745, BLUE), (.375, .63, '.4'), (.65, .46, RED)]:
    axes[0].text(75, y, fr'$e={e:g}$', color=color, fontsize=8)

# Infinite Zipf(z=2) sums: add the unseen tail's first-order term and
# record a rigorous bound for the omitted higher-order correction.
J = 200_000
z = 2.
normalizer = 1/zeta(z, 1)
freq = normalizer/np.arange(1, J+1, dtype=float)**z
tail1 = normalizer*zeta(z, J+1)
tail2 = normalizer**2*zeta(2*z, J+1)
N = np.unique(np.geomspace(10, 1_000_000, 81).astype(int))
missing, cumulative = [], []
for horizon in N:
    exponent = horizon*np.log1p(-freq)
    missing.append(np.sum(freq*np.exp(exponent))+tail1)
    cumulative.append(np.sum(-np.expm1(exponent))+horizon*tail1)
missing, cumulative = np.array(missing), np.array(cumulative)
constant = normalizer**(1/z)*gamma(1-1/z)
# Per-round uncovered deficit L_N (= stationary deficit under a TTL of N rounds) and its
# cumulative sum R_N, in hits (Delta = 1); dashed: leading asymptotic equivalents.
axes[1].loglog(N, missing, color=BLUE, label='Occupancy sum')
axes[1].loglog(N, constant/z*N**(1/z-1), color='.5', ls='--', label=r'$\sqrt{c\pi}\,N^{-1/2}/2$')
axes[1].set(xlabel='Round / TTL horizon $N$', ylabel='Uncovered deficit $L_N$')
axes[1].legend(loc='lower left', fontsize=7.5, frameon=False)
axes[2].loglog(N, cumulative, color=GREEN, label='Occupancy sum')
axes[2].loglog(N, constant*N**(1/z), color='.5', ls='--', label=r'$\sqrt{c\pi N}$')
axes[2].set(xlabel='Number of rounds $N$', ylabel='Cumulative deficit $R_N$')
axes[2].legend(loc='upper left', fontsize=7.5, frameon=False)
fig.savefig(FIG/'memory.pdf', metadata=PDF_METADATA)
plt.close(fig)
# Column name 'epoch' (= round) kept so the committed CSV is reproduced byte for byte.
save_csv('memory.csv', ['epoch']+[f'value_error_{e}' for e in errors], zip(n, *curves))
save_csv('zipf.csv', ['horizon','missing_coverage','cumulative_missing','missing_tail_error_bound','cumulative_tail_error_bound'],
         zip(N, missing, cumulative, N*tail2, N*(N-1)/2*tail2))

# The two dashed curves are the leading asymptotic equivalents sqrt(c pi) N^{-1/2}/2 and
# sqrt(c pi) N^{1/2} (c = 1/zeta(2)); their product is c pi/2 = 3/pi, and L_N R_N converges
# to it with a gap of order N^{-1/2}: sqrt(N) times the gap stays between 0.34 and 0.36 for
# N >= 1e3 (0.353 at N = 1e3, 0.346 at N = 1e6; the gap itself is about 3.5e-4 at N = 1e6).
product_limit = constant**2/z
assert np.isclose(product_limit, 3/np.pi) and np.isclose(product_limit, normalizer*np.pi/2)
product_gap = product_limit-missing*cumulative
assert np.all(product_gap > 0) and abs(product_gap[-1]) < 1e-3
scaled_gap = product_gap*np.sqrt(N)
assert np.all((scaled_gap[N >= 1000] > .34) & (scaled_gap[N >= 1000] < .36))

# Exact rational checks of the source construction and lower-bound example
# (Appendix "An exact four-type construction"). The proposal laws, costs (one per
# proposer, ten for the fallback) and v_h = 1 are shared by both fallback variants.
qf = [F(3,10),F(1,5),F(2,5),F(1,10)]
pf = [F(9,20),F(9,20),F(1,20),F(1,20)]
sf = [F(2,5),F(2,5),F(1,10),F(1,10)]
expected = [F(163,400),F(7,16),F(39,80)]
expected_eta_uniform = [F(1763,15850),F(71,610),F(359,2850)]
expected_eta_typed = [F(167,1585),F(127,1220),F(29,285)]
exact, typed = [], []
for r, target, target_uniform, target_typed in zip([F(1),F(1,4),F(0)], expected,
                                                   expected_eta_uniform, expected_eta_typed):
    weights = [r,r,F(1),F(1)]
    normal = sum(x*w for x,w in zip(sf,weights))          # Z(rho) = 0.8 rho + 0.2
    law = [x*w/normal for x,w in zip(sf,weights)]         # steered law p_2^rho
    residual = [qq*(1-pp)*(1-ss) for qq,pp,ss in zip(qf,pf,law)]   # joint-failure mass by type
    p = 1-sum(residual)
    quality = sum(qq*ss for qq,ss in zip(qf,law))
    cost_exact = 2+10*(1-p)
    eta = (p+F(4,5)*(1-p))/cost_exact                     # type-independent fallback, value 4/5
    value_typed = p+residual[2]+residual[3]                # fallback succeeds exactly on types 3, 4
    assert value_typed == F(725,1000)+F(11,100)*r/normal  # v(rho) = 0.725 + 0.11 rho / Z(rho)
    eta_typed_exact = value_typed/cost_exact
    assert p == target and quality == F(1,4)
    assert eta == target_uniform and eta_typed_exact == target_typed
    exact.append({'rho':str(r),'hit':str(p),'quality':str(quality),'efficiency':str(eta)})
    typed.append({'rho':str(r),'value':str(value_typed),'efficiency':str(eta_typed_exact)})
assert expected_eta_typed[0] > expected_eta_typed[1] > expected_eta_typed[2]   # decreases with steering
# The plotted floating-point curves agree with the exact fractions at rho = 1, 1/4, 0
# (grid indices 0, 150, 200 of the 201-point rho grid).
for index, r_exact, target_uniform, target_typed in zip([0, 150, 200], [F(1),F(1,4),F(0)],
                                                         expected_eta_uniform, expected_eta_typed):
    assert np.isclose(rho[index], float(r_exact))
    assert np.isclose(eta_uniform[index], float(target_uniform)) and np.isclose(eta_typed[index], float(target_typed))

# Selection score Sel_alpha(P) = (1-alpha) sum_{i in P} a_i + alpha p(P): the fully steered pair
# against {1,3} with p_3 = (0.6, 0.4, 0, 0). The difference is -1/100 + (209/2000) alpha,
# positive exactly when alpha > 20/209.
p3 = [F(3,5),F(2,5),F(0),F(0)]
a1 = sum(qq*pp for qq,pp in zip(qf,pf))
a3 = sum(qq*pp for qq,pp in zip(qf,p3))
law_steered = [x*w/sum(y*v for y,v in zip(sf,[F(0),F(0),F(1),F(1)])) for x,w in zip(sf,[F(0),F(0),F(1),F(1)])]
p_steered = 1-sum(qq*(1-pp)*(1-ss) for qq,pp,ss in zip(qf,pf,law_steered))
p13 = 1-sum(qq*(1-pp)*(1-ss) for qq,pp,ss in zip(qf,pf,p3))
r3 = sum(qq*(1-pp)*ss for qq,pp,ss in zip(qf,pf,p3))
assert a1 == F(1,4) and a3 == F(13,50) and p_steered == F(39,80) and p13 == F(393,1000)
assert r3 == F(143,1000) and (p13+F(4,5)*(1-p13))/(2+10*(1-p13)) == F(4393,40350)
sel_intercept = F(1,4)-a3                     # a_2 - a_3 = -1/100
sel_slope = (p_steered-p13)-sel_intercept     # 209/2000 = 0.1045
assert sel_intercept == F(-1,100) and sel_slope == F(209,2000)
selection_crossing = -sel_intercept/sel_slope
assert selection_crossing == F(20,209)

# Value-only noise threshold of the memory figure and the routing-information band.
noise_threshold_exact = (F(9,10)-F(3,5))/((F(9,10)-F(3,5))-(F(1,10)-F(3,5)))   # g^I/(g^I-g^W)
assert noise_threshold_exact == F(3,8) and np.isclose(float(noise_threshold_exact), .3/.8)
band_exact = [F(1,50)/(F(1)-F(1,10)), 1-F(1,50)/F(1,10)]                        # k/(b-a), 1-k/a
crossover_exact = F(1,10)                                                        # a/b
assert band_exact == [F(1,45),F(4,5)] and band_exact[0] < crossover_exact < band_exact[1]
assert np.isclose(float(band_exact[0]), low) and np.isclose(float(band_exact[1]), high)

# TTL and ordinary event-FIFO differ even when mean time spans coincide.
write_probability, total_write_probability, capacity = F(1,10), F(1,2), 2
fifo = 1-(1-write_probability/total_write_probability)**capacity
ttl_mean_span = 1-(1-write_probability)**4
assert fifo == F(9,25) and ttl_mean_span == F(3439,10000)

# Finite-horizon memory formula against an explicit iteration of the record recurrence
# i_{n+1} = nu*omega + alpha*i_n, w_{n+1} = nu*psi + alpha*w_n (omega = r(1-e), psi = r e),
# for the ten identical cells of the memory figure, both gain signs and the boundary case.
gain_I, gain_W = .3, -.5
for error, curve in zip(errors, curves):
    omega, psi = write_rate*(1-error), write_rate*error
    i = w = 0.
    explicit, iterated_values = 0., []
    for _ in n:                                   # records present before rounds 0..120
        iterated_values.append(.6+i*gain_I+w*gain_W)
        explicit += 10*nu*(i*gain_I+w*gain_W)     # per-round gain summed over the cells
        i, w = nu*omega+retention*i, nu*psi+retention*w
    assert np.allclose(iterated_values, curve, atol=1e-12)
    g = nu*write_rate*(.3-.8*error)/(1-retention)
    closed = g*(len(n)-(1-retention**len(n))/(1-retention))
    assert np.isclose(explicit,closed,atol=1e-12)

summary = {
    'construction': exact,
    'routing_information_band': [low,high],
    'noise_threshold': .3/.8,
    'fifo_example': {'ordinary_fifo':str(fifo),'mean_span_ttl':str(ttl_mean_span)},
    'zipf_truncation_cells': J,
    'max_missing_tail_error': float(np.max(N*tail2)),
    'max_cumulative_tail_error': float(np.max(N*(N-1)/2*tail2)),
    'routing_information_band_exact': [str(x) for x in band_exact],
    'routing_crossover_exact': str(crossover_exact),
    'noise_threshold_exact': str(noise_threshold_exact),
    'typed_fallback': typed,
    'selection_score_crossing': str(selection_crossing),
    'zipf_product_check': {'L_N_R_N_at_max_horizon': float(missing[-1]*cumulative[-1]),
                           'limit_c_pi_over_2': float(product_limit),
                           'sqrt_N_times_gap_at_max_horizon': float(scaled_gap[-1])},
    'exact_checks': 'Uniform-fallback efficiencies 1763/15850, 71/610, 359/2850; type-dependent '
                    'fallback efficiencies 167/1585, 127/1220, 29/285; selection-score crossing 20/209; '
                    'noise threshold 3/8; band 1/45 to 4/5 with crossover 1/10; explicit recurrence '
                    'iteration equals the finite-horizon formula.',
    'checks': 'Exact construction, steering signs, FIFO distinction, and finite-horizon recurrence passed.',
}
(DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary['checks'])
print(summary['exact_checks'])
print(f"usetex: {'on' if USETEX else 'off (latex not found; Matplotlib text engine used)'}")
print(f'Created {FIG/"allocation.pdf"} and {FIG/"memory.pdf"}; CSV tables and summary.json in {DATA}/.')

