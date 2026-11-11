import numpy as np, csv, json, sys
from collections import defaultdict
from scipy.stats import binomtest, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open('episodes.csv')))
for r in rows:
    for k in ['eps', 'ptrap', 'kappa', 'tau0']:
        r[k] = float(r[k])
    for k in ['board', 'solvable', 'steps', 'defers', 'neutral']:
        r[k] = int(r[k])
    r['final_doubt'] = float(r['final_doubt']) if r['final_doubt'] not in ('', 'nan') else np.nan

AG = ['B0', 'SP', 'DA', 'TW', 'BF', 'WO']
NAMES = {'B0': 'Reactive (original)', 'SP': 'Single-pathway', 'DA': 'Doubt-only',
         'TW': 'Three-way-only', 'BF': 'Bifrontal (full)', 'WO': 'Withdrawal-only'}
EPS = [0.0, 0.1, 0.2, 0.3, 0.4]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def util(o, r):
    return 1.0 if o == 'success' else (-r if o == 'trap' else 0.0)


main = [r for r in rows if r['exp'] == 'main']
by = defaultdict(dict)
for r in main:
    by[(r['agent'], r['eps'])][r['board']] = r

out = {'main': {}, 'paired': {}, 'density': {}, 'sens': {}, 'doubt': {}, 'solv': {}}
for a in AG:
    for e in EPS:
        d = by[(a, e)]
        n = len(d)
        oc = [x['outcome'] for x in d.values()]
        res = {}
        for o in ['success', 'trap', 'abstain', 'timeout']:
            res[o] = wilson(oc.count(o), n)
        for rr in [1, 3]:
            u = np.array([util(o, rr) for o in oc])
            res[f'U{rr}'] = (u.mean(), u.mean() - 1.96 * u.std(ddof=1) / np.sqrt(n), u.mean() + 1.96 * u.std(ddof=1) / np.sqrt(n))
        succ_steps = [x['steps'] for x in d.values() if x['outcome'] == 'success']
        res['steps_success'] = (float(np.mean(succ_steps)) if succ_steps else np.nan,
                                float(np.std(succ_steps)) if succ_steps else np.nan)
        res['defers'] = float(np.mean([x['defers'] for x in d.values()]))
        res['n'] = n
        # solvable / unsolvable split
        sol = [x for x in d.values() if x['solvable'] == 1]
        uns = [x for x in d.values() if x['solvable'] == 0]
        res['succ_on_solvable'] = wilson(sum(x['outcome'] == 'success' for x in sol), len(sol))
        res['trap_on_solvable'] = wilson(sum(x['outcome'] == 'trap' for x in sol), len(sol))
        res['trap_on_unsolvable'] = wilson(sum(x['outcome'] == 'trap' for x in uns), len(uns))
        res['n_solvable'] = len(sol)
        res['doubt_mean'] = float(np.nanmean([x['final_doubt'] for x in d.values()])) if a in ('DA', 'BF') else np.nan
        out['main'][f'{a}|{e}'] = res

# paired comparisons BF vs SP, BF vs TW, TW vs SP, DA vs SP
rng = np.random.default_rng(7)
pvals = []
for (a1, a2) in [('BF', 'SP'), ('TW', 'SP'), ('DA', 'SP'), ('BF', 'TW')]:
    for e in EPS:
        boards = sorted(set(by[(a1, e)]) & set(by[(a2, e)]))
        o1 = [by[(a1, e)][b]['outcome'] for b in boards]
        o2 = [by[(a2, e)][b]['outcome'] for b in boards]
        res = {}
        for rr in [1, 3]:
            d = np.array([util(x, rr) - util(y, rr) for x, y in zip(o1, o2)])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            try:
                p = wilcoxon(d[d != 0]).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            res[f'dU{rr}'] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(p))
        # exact McNemar on trap
        b10 = sum((x == 'trap') and (y != 'trap') for x, y in zip(o1, o2))
        b01 = sum((x != 'trap') and (y == 'trap') for x, y in zip(o1, o2))
        p = binomtest(b10, b10 + b01, 0.5).pvalue if (b10 + b01) > 0 else 1.0
        res['trap_mcnemar'] = (b10, b01, float(p))
        b10s = sum((x == 'success') and (y != 'success') for x, y in zip(o1, o2))
        b01s = sum((x != 'success') and (y == 'success') for x, y in zip(o1, o2))
        ps = binomtest(b10s, b10s + b01s, 0.5).pvalue if (b10s + b01s) > 0 else 1.0
        res['succ_mcnemar'] = (b10s, b01s, float(ps))
        out['paired'][f'{a1}-{a2}|{e}'] = res
        pvals.append((f'{a1}-{a2}|{e}', 'dU1', res['dU1'][3]))
        pvals.append((f'{a1}-{a2}|{e}', 'dU3', res['dU3'][3]))
        pvals.append((f'{a1}-{a2}|{e}', 'trap', res['trap_mcnemar'][2]))

# Holm correction across all paired tests
pvals_sorted = sorted(pvals, key=lambda x: x[2])
m = len(pvals_sorted)
holm = {}
running = 0
for i, (k, t, p) in enumerate(pvals_sorted):
    adj = min(1.0, (m - i) * p)
    running = max(running, adj)
    holm[f'{k}|{t}'] = running
out['holm'] = holm
out['n_tests'] = m

# density
dens = [r for r in rows if r['exp'] == 'density'] + [r for r in main if r['eps'] == 0.2 and r['agent'] in ('SP', 'TW', 'BF')]
for pt in [0.15, 0.25, 0.35]:
    for a in ['SP', 'TW', 'BF']:
        oc = [r['outcome'] for r in dens if r['ptrap'] == pt and r['agent'] == a]
        n = len(oc)
        out['density'][f'{a}|{pt}'] = dict(n=n, success=wilson(oc.count('success'), n), trap=wilson(oc.count('trap'), n),
                                          abstain=wilson(oc.count('abstain'), n), timeout=wilson(oc.count('timeout'), n),
                                          U1=float(np.mean([util(o, 1) for o in oc])), U3=float(np.mean([util(o, 3) for o in oc])))

# sensitivity
sens = [r for r in rows if r['exp'] == 'sens']
K = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
TAU = [0.1, 0.2, 0.3, 0.4, 0.5]
U1 = np.zeros((len(K), len(TAU))); U3 = np.zeros_like(U1); TR = np.zeros_like(U1); SU = np.zeros_like(U1)
for i, k in enumerate(K):
    for j, t in enumerate(TAU):
        oc = [r['outcome'] for r in sens if r['kappa'] == k and abs(r['tau0'] - t) < 1e-9]
        U1[i, j] = np.mean([util(o, 1) for o in oc]); U3[i, j] = np.mean([util(o, 3) for o in oc])
        TR[i, j] = oc.count('trap') / len(oc); SU[i, j] = oc.count('success') / len(oc)
out['sens'] = dict(K=K, TAU=TAU, U1=U1.tolist(), U3=U3.tolist(), trap=TR.tolist(), succ=SU.tolist())
json.dump(out, open('results.json', 'w'), indent=1, default=float)

# ---------------- figures ----------------
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'B0': '#9a9a9a', 'SP': '#2b6cb0', 'DA': '#6b9bd1', 'TW': '#d69e2e', 'BF': '#c53030', 'WO': '#2f855a'}
MK = {'B0': 'x', 'SP': 'o', 'DA': 's', 'TW': '^', 'BF': 'D', 'WO': 'v'}

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
for ax, (o, ttl) in zip(axes.flat, [('success', 'A  Goal reached'), ('trap', 'B  Hazard entered (failure)'),
                                     ('abstain', 'C  Explicit abstention'), ('timeout', 'D  Time-out (no commitment)')]):
    for a in AG:
        p = [out['main'][f'{a}|{e}'][o] for e in EPS]
        y = [x[0] for x in p]; lo = [x[0] - x[1] for x in p]; hi = [x[2] - x[0] for x in p]
        ax.errorbar(EPS, y, yerr=[lo, hi], color=COL[a], marker=MK[a], ms=4, lw=1.3, capsize=2, label=NAMES[a])
    ax.set_title(ttl, loc='left', fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Sensor error rate ε'); ax.set_ylabel('Proportion of episodes')
    ax.set_xticks(EPS); ax.set_ylim(-0.02, 1.02)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig('fig3_outcomes.png', dpi=300)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, rr in zip(axes, [1, 3]):
    for a in AG:
        p = [out['main'][f'{a}|{e}'][f'U{rr}'] for e in EPS]
        y = [x[0] for x in p]; lo = [x[0] - x[1] for x in p]; hi = [x[2] - x[0] for x in p]
        ax.errorbar(EPS, y, yerr=[lo, hi], color=COL[a], marker=MK[a], ms=4, lw=1.3, capsize=2, label=NAMES[a])
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title(f"{'A' if rr == 1 else 'B'}  Expected utility, failure cost r = {rr}", loc='left', fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Sensor error rate ε'); ax.set_ylabel('Mean utility per episode'); ax.set_xticks(EPS)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig('fig4_utility.png', dpi=300)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (M, ttl) in zip(axes, [(U1, 'A  Utility (r = 1)'), (TR, 'B  Hazard rate')]):
    im = ax.imshow(M, cmap='viridis' if M is U1 else 'magma_r', aspect='auto')
    ax.set_xticks(range(len(TAU))); ax.set_xticklabels(TAU); ax.set_yticks(range(len(K))); ax.set_yticklabels(K)
    ax.set_xlabel('Acceptance threshold τ₀'); ax.set_ylabel('Doubt gain κ')
    for i in range(len(K)):
        for j in range(len(TAU)):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if (M[i, j] < M.mean()) ^ (M is TR) else 'black')
    ax.set_title(ttl, loc='left', fontsize=9.5, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig('fig5_sensitivity.png', dpi=300)
plt.close(fig)

# doubt calibration
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for a in ['DA', 'BF']:
    vals = [[x['final_doubt'] for x in by[(a, e)].values() if not np.isnan(x['final_doubt'])] for e in EPS]
    ax.errorbar(np.array(EPS) + (0.005 if a == 'BF' else -0.005), [np.mean(v) for v in vals],
                yerr=[np.std(v) for v in vals], color=COL[a], marker=MK[a], ms=4, capsize=2, label=NAMES[a])
ax.plot([0, 0.4], [0, 0.4], 'k--', lw=0.8, label='Identity')
ax.axhline(0.1, color='gray', lw=0.6, ls=':', xmax=0.95)
ax.text(0.20, 0.075, 'fixed prior ε̂₀ = 0.1', fontsize=7, color='gray')
ax.set_xlabel('True sensor error rate ε'); ax.set_ylabel('Terminal doubt D (mean ± SD)')
ax.legend(frameon=False, fontsize=7.5); ax.set_xticks(EPS)
fig.tight_layout(); fig.savefig('fig6_doubt.png', dpi=300); plt.close(fig)
print('analysis done')
