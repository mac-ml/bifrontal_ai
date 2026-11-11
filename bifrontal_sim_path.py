"""
Computational proof-of-concept for the Bifrontal (approach/withdrawal) decision architecture.

Environment: N x N grid world with hidden cell types {S: safe, N: neutral, T: trap}.
The agent starts at (0,0) and must reach (N-1,N-1). It only perceives the 4-neighbourhood,
and each perceived label is correct with probability 1-eps; otherwise it is replaced by one of the
two other labels uniformly at random (a "true value may actually be false" sensor).
Entering a trap ends the episode (failure). Entering any other cell reveals its true type.

Agents (all share the same Bayesian perception module, so differences are due to the decision layer):
  B0  Reactive (original prototype): no goal knowledge, no memory, takes perceived safe > neutral at random.
  SP  Single-pathway: one scalar score S = A - lam0*W, forced choice (argmax), fixed sensor trust.
  DA  + Doubt adaptation only (adaptive sensor trust and risk gain), forced choice.
  TW  + Three-way output only (act / defer-and-resense / abstain), fixed trust and gain.
  BF  Bifrontal (full): adaptive doubt-modulated withdrawal gain + three-way output.
  WO  Withdrawal-only ("pure coward"): avoids threat, no approach drive (progress weight = 0).

A(c): approach pathway (goal progress, novelty, neutral-cell cost).
W(c): withdrawal pathway (posterior trap probability).
D   : doubt = online Beta-posterior estimate of sensor error rate learned from prediction errors
      when cells are revealed; it sets (i) sensor trust and (ii) the gain of the withdrawal channel.
"""
import numpy as np
from collections import deque

S, NEU, T = 0, 1, 2
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def make_board(rng, n, p=(0.40, 0.35, 0.25)):
    b = rng.choice(3, size=(n, n), p=p)
    b[0, 0] = S
    b[n - 1, n - 1] = S
    return b


def solvable(board):
    n = board.shape[0]
    seen = np.zeros_like(board, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        x, y = q.popleft()
        if (x, y) == (n - 1, n - 1):
            return True
        for dx, dy in DIRS:
            a, b_ = x + dx, y + dy
            if 0 <= a < n and 0 <= b_ < n and not seen[a, b_] and board[a, b_] != T:
                seen[a, b_] = True
                q.append((a, b_))
    return False


def neighbours(x, y, n):
    out = []
    for dx, dy in DIRS:
        a, b = x + dx, y + dy
        if 0 <= a < n and 0 <= b < n:
            out.append((a, b))
    return out


def observe(board, cell, eps, rng):
    true = board[cell]
    if rng.random() < eps:
        others = [k for k in (S, NEU, T) if k != true]
        return others[rng.integers(2)]
    return true


class Config:
    def __init__(self, n=8, eps=0.2, prior=(0.40, 0.35, 0.25), eps_hat0=0.10,
                 beta=1.0, eta=0.30, c_neu=0.05, mu=3.0, lam0=8.0, kappa=4.0, tau0=0.25,
                 defer_budget=2, t_max=150, a0=1.0, b0=9.0):
        self.__dict__.update(locals())
        del self.__dict__['self']


def run_episode(board, cfg, agent, seed):
    """Returns dict with outcome in {'success','trap','abstain','timeout'} and diagnostics."""
    rng = np.random.default_rng(seed)
    n = board.shape[0]
    goal = (n - 1, n - 1)
    prior = np.array(cfg.prior, dtype=float)
    counts = np.zeros((n, n, 3), dtype=int)          # perceived-label counts per cell
    known = -np.ones((n, n), dtype=int)               # revealed true type (-1 unknown)
    known[0, 0] = S
    visits = np.zeros((n, n), dtype=int)
    visits[0, 0] = 1
    pos = (0, 0)
    mism, nobs_rev = 0, 0                             # prediction errors for doubt
    defers_here = 0
    n_defer = 0
    n_neutral = 0
    doubt_trace = []

    adaptive = agent in ('DA', 'BF')
    three_way = agent in ('TW', 'BF')

    def eps_hat():
        if adaptive:
            return (cfg.a0 + mism) / (cfg.a0 + cfg.b0 + nobs_rev)
        return cfg.eps_hat0

    def posterior(cell, e):
        if known[cell] >= 0:
            p = np.zeros(3); p[known[cell]] = 1.0
            return p
        c = counts[cell]
        tot = c.sum()
        e = min(max(e, 1e-3), 0.66)
        loglik = np.array([c[k] * np.log(1 - e) + (tot - c[k]) * np.log(e / 2) for k in range(3)])
        lp = np.log(prior) + loglik
        lp -= lp.max()
        p = np.exp(lp)
        return p / p.sum()

    def sense(cell_list):
        for c in cell_list:
            if known[c] < 0:
                counts[c][observe(board, c, cfg.eps, rng)] += 1

    # ---------------- B0: original reactive prototype ----------------
    if agent == 'B0':
        for t in range(cfg.t_max):
            nb = neighbours(*pos, n)
            obs = {c: observe(board, c, cfg.eps, rng) for c in nb}
            safe = [c for c in nb if obs[c] == S]
            neu = [c for c in nb if obs[c] == NEU]
            if safe:
                nxt = safe[rng.integers(len(safe))]
            elif neu:
                nxt = neu[rng.integers(len(neu))]
            else:
                return dict(outcome='abstain', steps=t, defers=0, neutral=n_neutral, doubt=[])
            pos = nxt
            if board[pos] == T:
                return dict(outcome='trap', steps=t + 1, defers=0, neutral=n_neutral, doubt=[])
            if board[pos] == NEU:
                n_neutral += 1
            if pos == goal:
                return dict(outcome='success', steps=t + 1, defers=0, neutral=n_neutral, doubt=[])
        return dict(outcome='timeout', steps=cfg.t_max, defers=0, neutral=n_neutral, doubt=[])

    # ---------------- model-based agents ----------------
    import heapq
    beta = 0.0 if agent == 'WO' else cfg.beta
    vetoed = np.zeros((n, n), dtype=bool)          # cells rejected by the withdrawal channel
    path = [pos]
    t = 0
    sense(neighbours(*pos, n))

    def plan(e):
        """Shared approach planner: expected path cost to goal over believed-passable cells."""
        e_ = min(max(e, 1e-3), 0.66)
        tot = counts.sum(axis=2, keepdims=True)
        loglik = counts * np.log(1 - e_) + (tot - counts) * np.log(e_ / 2)
        lp = np.log(prior)[None, None, :] + loglik
        lp -= lp.max(axis=2, keepdims=True)
        P = np.exp(lp)
        P /= P.sum(axis=2, keepdims=True)
        kn = known >= 0
        P[kn] = np.eye(3)[known[kn]]
        cost = 1.0 + cfg.c_neu * P[:, :, NEU] + cfg.mu * P[:, :, T]
        dist = np.full((n, n), np.inf)
        dist[goal] = 0.0
        h = [(0.0, goal)]
        while h:
            d, (x, y) = heapq.heappop(h)
            if d > dist[x, y]:
                continue
            for a, b in neighbours(x, y, n):
                if vetoed[x, y] and (x, y) != goal:
                    continue
                nd = d + cost[x, y]
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(h, (nd, (a, b)))
        return P, dist

    while t < cfg.t_max:
        e = eps_hat()
        D = e  # doubt = current estimate of perceptual unreliability
        doubt_trace.append(D)
        lam = cfg.lam0 * (1 + cfg.kappa * D) if adaptive else cfg.lam0
        tau = cfg.tau0 / (1 + cfg.kappa * D) if adaptive else cfg.tau0
        nb = neighbours(*pos, n)
        decided = False
        while not decided:
            P, dist = plan(e)
            best, best_s, best_w = None, -np.inf, None
            for c in nb:
                if vetoed[c]:
                    continue
                W = P[c][T]                                   # withdrawal pathway
                if np.isinf(dist[c]):
                    continue
                A = -beta * dist[c] - cfg.eta * visits[c]     # approach pathway
                s = A - lam * W + 1e-9 * rng.random()
                if s > best_s:
                    best, best_s, best_w = c, s, W
            if best is None:
                return dict(outcome='abstain', steps=t, defers=n_defer, neutral=n_neutral, doubt=doubt_trace, path=path, vetoed=vetoed.copy())
            if three_way and best_w > tau:
                if defers_here < cfg.defer_budget:
                    sense(nb)                                 # defer: look again (one time step)
                    defers_here += 1
                    n_defer += 1
                    t += 1
                    break
                vetoed[best] = True                           # reject: withdrawal veto persists
                continue
            decided = True
        if not decided:
            continue
        # act
        t += 1
        defers_here = 0
        pos = best
        path.append(pos)
        visits[pos] += 1
        if board[pos] == T:
            return dict(outcome='trap', steps=t, defers=n_defer, neutral=n_neutral, doubt=doubt_trace, path=path, vetoed=vetoed.copy())
        if known[pos] < 0:
            c = counts[pos]
            mism += int(c.sum() - c[board[pos]])
            nobs_rev += int(c.sum())
            known[pos] = board[pos]
            if board[pos] == NEU:
                n_neutral += 1
        if pos == goal:
            return dict(outcome='success', steps=t, defers=n_defer, neutral=n_neutral, doubt=doubt_trace, path=path, vetoed=vetoed.copy())
        sense(neighbours(*pos, n))
    return dict(outcome='timeout', steps=t, defers=n_defer, neutral=n_neutral, doubt=doubt_trace, path=path, vetoed=vetoed.copy())
