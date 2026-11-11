import numpy as np, sys, json, csv, time
from multiprocessing import Pool
sys.path.insert(0, '.')
from bifrontal_sim import *

tun = json.load(open('tuning.json'))
LAM0, TAU0 = tun['best_lam'], tun['best_tau']
AGENTS = ['B0', 'SP', 'DA', 'TW', 'BF', 'WO']
NB = 2000


def ep(args):
    exp, i, agent, eps, ptrap, kw = args
    p = ((1 - ptrap) * 0.40 / 0.75, (1 - ptrap) * 0.35 / 0.75, ptrap)
    board = make_board(np.random.default_rng(i), 8, p)
    base = dict(lam0=LAM0, tau0=TAU0); base.update(kw)
    cfg = Config(eps=eps, prior=p, **base)
    r = run_episode(board, cfg, agent, seed=100000 + i)
    d = r['doubt']
    return dict(exp=exp, board=i, agent=agent, eps=eps, ptrap=ptrap,
                kappa=kw.get('kappa', 4.0), tau0=kw.get('tau0', TAU0),
                solvable=int(solvable(board)), outcome=r['outcome'], steps=r['steps'],
                defers=r['defers'], neutral=r['neutral'],
                final_doubt=(d[-1] if d else np.nan))


if __name__ == '__main__':
    jobs = []
    # Experiment 1: sensor-noise sweep (main)
    for eps in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for a in AGENTS:
            for i in range(NB):
                jobs.append(('main', i, a, eps, 0.25, {}))
    # Experiment 2: hazard-density sweep at eps = 0.2
    for pt in [0.15, 0.35]:
        for a in ['SP', 'TW', 'BF']:
            for i in range(NB):
                jobs.append(('density', i, a, 0.2, pt, {}))
    # Experiment 3: sensitivity of BF to kappa and tau0 at eps = 0.3
    for kap in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]:
        for tau in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for i in range(1000):
                kw = {'kappa': kap}
                if tau != TAU0:
                    kw['tau0'] = tau
                jobs.append(('sens', i, 'BF', 0.3, 0.25, kw))
    t0 = time.time()
    with Pool(2) as p:
        rows = p.map(ep, jobs, chunksize=200)
    keys = list(rows[0].keys())
    with open('episodes.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('done', len(rows), round(time.time() - t0, 1), 's')
