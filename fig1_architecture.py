import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_xlim(0, 100); ax.set_ylim(2, 71); ax.axis('off')


def box(x, y, w, h, title, body, fc, ec):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.6', fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='top', fontweight='bold', fontsize=8.8)
    ax.text(x + w / 2, y + h - 7.2, body, ha='center', va='top', fontsize=7.4, linespacing=1.35)


def arrow(x1, y1, x2, y2, txt=None, style='-|>', color='#333', ls='-'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color, lw=1.1, linestyle=ls)
    ax.add_patch(a)
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, txt, ha='center', fontsize=7, color=color, style='italic')


box(1, 24, 18, 20, 'Shared perception', 'noisy observations o\nBayesian belief b(x)\nover {safe, neutral,\nhazard}', '#edf2f7', '#4a5568')
box(28, 47, 26, 15, 'Approach channel A(a)', '"left-frontal" analogue: goal\nprogress and novelty\nA = −β·d̂(a) − η·v(a)', '#ebf4ff', '#2b6cb0')
box(28, 26.5, 26, 14.5, 'Doubt D (gain control)', 'learned sensor distrust\nD = (a₀+m)/(a₀+b₀+o)\nλ(D)=λ₀(1+κD); τ(D)=τ₀/(1+κD)', '#fffaf0', '#b7791f')
box(28, 5, 26, 15, 'Withdrawal channel W(a)', '"right-frontal" analogue:\nthreat salience\nW = P(hazard | evidence)', '#fff5f5', '#c53030')
box(62, 24, 17, 20, 'Arbitration', 'a* = argmax\nA(a) − λ(D)·W(a)\n(integration across\nchannels)', '#f0fff4', '#2f855a')
box(85, 44, 14, 12, 'ACT', 'W(a*) ≤ τ(D):\nmove', '#f0fff4', '#2f855a')
box(85, 26, 14, 12, 'DEFER', 're-sense\n(budget k)', '#fffaf0', '#b7791f')
box(85, 8, 14, 12, 'VETO', 'reject a*; no\noption → abstain', '#fff5f5', '#c53030')

arrow(19.6, 38, 27.4, 53)
arrow(19.6, 30, 27.4, 12)
arrow(41, 41.6, 41, 46.4, color='#b7791f')
arrow(41, 25.9, 41, 20.6, color='#b7791f')
arrow(54.6, 54, 61.4, 41)
arrow(54.6, 12, 61.4, 27)
arrow(54.6, 34, 61.4, 34, color='#b7791f')
arrow(79.6, 40, 84.4, 49)
arrow(79.6, 34, 84.4, 32)
arrow(79.6, 28, 84.4, 15)
# feedback loop along the top
ax.plot([92, 92, 10, 10], [56.6, 67, 67, 45], color='#777', lw=0.9, ls='--')
arrow(10, 46.5, 10, 44.6, color='#777')
ax.text(51, 68.2, 'outcome revealed → prediction error m updates doubt D; vetoes persist in memory', ha='center', fontsize=7.2, color='#555', style='italic')
fig.tight_layout()
fig.savefig('fig1_architecture.png', dpi=300)
print('ok')
