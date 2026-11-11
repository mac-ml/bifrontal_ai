import numpy as np, sys, json
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from bifrontal_sim_path import *

tun = json.load(open('tuning.json'))
cfg = Config(eps=0.2, lam0=tun['best_lam'], tau0=tun['best_tau'])
# find the first test board (seed order) where SP enters a hazard and BF reaches the goal
cands = []
for i in range(400):
    b = make_board(np.random.default_rng(i), 8)
    r1 = run_episode(b, cfg, 'SP', seed=100000 + i)
    r2 = run_episode(b, cfg, 'BF', seed=100000 + i)
    if r1['outcome'] == 'trap' and r2['outcome'] == 'success':
        cands.append((len(r2['path']), i, b, r1, r2))
print('n candidates', len(cands), sorted(c[0] for c in cands))
_, i, b, r1, r2 = min(cands, key=lambda c: c[0])
print('board', i, len(r1['path']), len(r2['path']))
cmap = ListedColormap(['#f7fafc', '#e2e8f0', '#feb2b2'])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
for ax, r, ttl in zip(axes, [r1, r2], ['A  Single-pathway agent', 'B  Bifrontal agent']):
    ax.imshow(b, cmap=cmap, vmin=0, vmax=2)
    for x in range(8):
        for y in range(8):
            ax.text(y, x, 'SNT'[b[x, y]], ha='center', va='center', fontsize=7, color='#4a5568')
    p = np.array(r['path'])
    ax.plot(p[:, 1], p[:, 0], '-', color='#2b6cb0' if r is r1 else '#c53030', lw=2, alpha=0.85)
    ax.plot(p[0, 1], p[0, 0], 'o', color='k', ms=5)
    ax.plot(p[-1, 1], p[-1, 0], 'X' if r['outcome'] == 'trap' else '*', color='k', ms=9)
    if 'vetoed' in r:
        vx, vy = np.where(r['vetoed'])
        ax.plot(vy, vx, 's', mfc='none', mec='#c53030', ms=13, mew=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{ttl}: {r['outcome']} ({len(p) - 1} moves)", loc='left', fontsize=9, fontweight='bold')
fig.tight_layout()

fig.savefig('fig2_example.png', dpi=300)
json.dump(dict(board=i, sp=r1['outcome'], bf=r2['outcome'], sp_moves=len(r1['path']) - 1, bf_moves=len(r2['path']) - 1,
               bf_defers=r2['defers'], bf_vetoes=int(r2['vetoed'].sum())), open('fig2_meta.json', 'w'))
