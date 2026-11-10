"""
E-NSGA-II and the benchmark algorithms (Section 4).

All algorithms share:
  * the two-part encoding (assignment genes a, intensity genes rho)
  * the decoder with proportional capacity repair (model.decode)
  * the same function-evaluation budget
  * the same in-sample scenario set (common random numbers)

E-NSGA-II adds, relative to plain NSGA-II:
  Init   : capacity-aware greedy seeding of a fraction eta of the population
  Adapt  : mutation probability and creep width driven by front spread
           and hypervolume stagnation
(The repair operator lives in the decoder and is therefore present in every
variant, as stated in the manuscript.)
"""

import numpy as np
from .model import decode, decode_batch


# ----------------------------------------------------------------------
# Non-dominated sorting and crowding distance
# ----------------------------------------------------------------------

def fast_nondominated_sort(F):
    """F: (N,2) minimisation objectives. Returns list of fronts (index arrays)."""
    N = F.shape[0]
    dominated_count = np.zeros(N, dtype=int)
    dominates = [[] for _ in range(N)]
    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]
    dom = le.all(axis=2) & lt.any(axis=2)      # dom[p,q] : p dominates q
    for p in range(N):
        q_idx = np.nonzero(dom[p])[0]
        dominates[p] = q_idx
    dominated_count = dom.sum(axis=0)
    fronts, current = [], np.nonzero(dominated_count == 0)[0]
    counts = dominated_count.copy()
    while current.size:
        fronts.append(current)
        nxt = []
        for p in current:
            for q in dominates[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    nxt.append(q)
        current = np.array(sorted(set(nxt)), dtype=int)
    return fronts


def front_extent(F):
    """
    Extent of a front in objective space: the geometric mean of the per-objective
    ranges. Unlike mean crowding distance this does NOT fall as the front fills
    in -- it measures how much of the trade-off surface has been reached, which
    is what the adaptive rule in Section 4 is supposed to respond to.
    """
    if F.shape[0] < 2:
        return 0.0
    r = F.max(axis=0) - F.min(axis=0)
    return float(np.sqrt(max(r[0], 0.0) * max(r[1], 0.0)))


def crowding_distance(F):
    """Standard NSGA-II crowding distance for one front."""
    n = F.shape[0]
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for m in range(F.shape[1]):
        order = np.argsort(F[:, m])
        fmin, fmax = F[order[0], m], F[order[-1], m]
        d[order[0]] = d[order[-1]] = np.inf
        if fmax - fmin < 1e-12:
            continue
        d[order[1:-1]] += (F[order[2:], m] - F[order[:-2], m]) / (fmax - fmin)
    return d


# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------

def hypervolume_2d(F, ref):
    """Dominated hypervolume for 2-D minimisation, reference point ref."""
    P = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    if P.size == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    keep, best = [], np.inf
    for p in P:                      # keep the non-dominated staircase
        if p[1] < best:
            keep.append(p)
            best = p[1]
    P = np.array(keep)
    hv, prev_y = 0.0, ref[1]
    for x, y in P:
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(hv)


def igd_plus(F, ref_front):
    """IGD+ of front F w.r.t. a reference front (both minimisation)."""
    d = np.maximum(F[None, :, :] - ref_front[:, None, :], 0.0)
    return float(np.sqrt((d ** 2).sum(axis=2)).min(axis=1).mean())


def spread(F):
    """Delta spread indicator (lower is better), 2-D."""
    if F.shape[0] < 3:
        return 1.0
    P = F[np.argsort(F[:, 0])]
    d = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(axis=1))
    dbar = d.mean()
    if dbar < 1e-12:
        return 0.0
    df = np.sqrt(((P[0] - P[0]) ** 2).sum())     # extremes already in the set
    return float((np.abs(d - dbar).sum()) / ((len(d)) * dbar))


def epsilon_plus(F, ref_front):
    """Additive epsilon-indicator w.r.t. a reference front."""
    return float(np.max(np.min(np.max(F[None, :, :] - ref_front[:, None, :],
                                      axis=2), axis=1)))


# ----------------------------------------------------------------------
# Population helpers
# ----------------------------------------------------------------------

def evaluate_pop(inst, A, Rho):
    return decode_batch(inst, A, Rho)


def heuristic_init(inst, n, rng):
    """Capacity-aware greedy assignments + economically graded intensities."""
    I, J = inst.nI, inst.nJ
    A = np.zeros((n, I), dtype=int)
    Rho = np.zeros((n, I))
    Rbar, Cbar, Wbar, Cabar = inst.R.mean(0), inst.C.mean(0), inst.W_.mean(0), inst.Ca.mean(0)
    unit_cost = inst.T_ij + (Cbar + Wbar)[None, :]          # (I,J)
    for s in range(n):
        residual = Cabar.copy()
        order = np.argsort(-Rbar)
        if s > 0:                                            # randomised tie-breaking
            noise = rng.normal(0, 0.05 * Rbar.std() + 1e-9, I)
            order = np.argsort(-(Rbar + noise))
        for i in order:
            cand = np.argsort(unit_cost[i])
            placed = False
            for j in cand:
                if residual[j] >= Rbar[i]:
                    A[s, i] = j
                    residual[j] -= Rbar[i]
                    placed = True
                    break
            if not placed:
                A[s, i] = cand[0]
        # intensity graded by penalty-to-marginal-cost ratio
        marg = unit_cost[np.arange(I), A[s]]
        ratio = inst.phi1 / np.maximum(marg, 1e-9)
        ratio = (ratio - ratio.min()) / max(np.ptp(ratio), 1e-9)
        lo = s / max(n - 1, 1)                               # spread seeds along the front
        Rho[s] = np.clip(lo + (1 - lo) * ratio, 0, 1)
    return A, Rho


def random_init(inst, n, rng):
    return (rng.integers(0, inst.nJ, (n, inst.nI)),
            rng.random((n, inst.nI)))


def sbx(p1, p2, eta, rng):
    """Bounded simulated binary crossover on [0,1]^n. Returns one child."""
    n = p1.size
    u = rng.random(n)
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta + 1)),
                    (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1)))
    swap = rng.random(n) < 0.5
    c = np.where(swap, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2),
                 0.5 * ((1 - beta) * p1 + (1 + beta) * p2))
    return np.clip(c, 0.0, 1.0)


def poly_mut(x, pm, eta, rng):
    """Polynomial mutation on [0,1]^n."""
    y = x.copy()
    m = rng.random(x.size) < pm
    if not m.any():
        return y
    u = rng.random(m.sum())
    d = np.where(u < 0.5,
                 (2 * u) ** (1.0 / (eta + 1)) - 1.0,
                 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1)))
    y[m] = np.clip(y[m] + d, 0.0, 1.0)
    return y


def int_cx(a1, a2, rng):
    """Uniform crossover on the integer assignment genes."""
    take = rng.random(a1.size) < 0.5
    return np.where(take, a1, a2)


def int_mut(a, pm, nJ, rng):
    """Random-reset mutation on the integer assignment genes."""
    b = a.copy()
    m = rng.random(a.size) < pm
    if m.any():
        b[m] = rng.integers(0, nJ, m.sum())
    return b


def variation(A, Rho, parents, pc, pm, eta_c, eta_m, nJ, rng):
    """
    Offspring generation shared by E-NSGA-II and the benchmarks: SBX plus
    polynomial mutation on the continuous intensity genes, uniform crossover
    plus random-reset mutation on the integer assignment genes. Using the same
    operators everywhere isolates the algorithmic differences (selection,
    seeding, adaptation) from operator quality.
    """
    N, I = A.shape
    Ac = np.empty_like(A)
    Rc = np.empty_like(Rho)
    mates = parents[rng.integers(0, N, N)]
    for s in range(N):
        p, q = parents[s], mates[s]
        if rng.random() < pc:
            Rc[s] = sbx(Rho[p], Rho[q], eta_c, rng)
            Ac[s] = int_cx(A[p], A[q], rng)
        else:
            Rc[s], Ac[s] = Rho[p].copy(), A[p].copy()
        Rc[s] = poly_mut(Rc[s], pm, eta_m, rng)
        Ac[s] = int_mut(Ac[s], pm, nJ, rng)
    return Ac, Rc


def tournament(F, rank, cd, rng, k=2):
    N = F.shape[0]
    cand = rng.integers(0, N, (N, k))
    win = cand[:, 0].copy()
    for c in range(1, k):
        chall = cand[:, c]
        better = (rank[chall] < rank[win]) | (
            (rank[chall] == rank[win]) & (cd[chall] > cd[win]))
        win[better] = chall[better]
    return win


def crossover(A, Rho, parents, pc, rng):
    N, I = A.shape
    Ac, Rc = A[parents].copy(), Rho[parents].copy()
    for s in range(0, N - 1, 2):
        if rng.random() < pc:
            p, q = sorted(rng.integers(0, I, 2))
            Ac[s, p:q], Ac[s + 1, p:q] = Ac[s + 1, p:q].copy(), Ac[s, p:q].copy()
            Rc[s, p:q], Rc[s + 1, p:q] = Rc[s + 1, p:q].copy(), Rc[s, p:q].copy()
    return Ac, Rc


def mutate(A, Rho, pm, sigma, nJ, rng):
    N, I = A.shape
    mask = rng.random((N, I)) < pm
    A = A.copy()
    A[mask] = rng.integers(0, nJ, mask.sum())
    Rho = Rho + mask * rng.uniform(-sigma, sigma, (N, I))
    return A, np.clip(Rho, 0.0, 1.0)


def environmental_selection(F, N):
    fronts = fast_nondominated_sort(F)
    chosen, rank = [], np.zeros(F.shape[0], dtype=int)
    for r, fr in enumerate(fronts):
        rank[fr] = r
        if len(chosen) + len(fr) <= N:
            chosen.extend(fr)
        else:
            cd = crowding_distance(F[fr])
            order = fr[np.argsort(-cd)]
            chosen.extend(order[:N - len(chosen)])
            break
    return np.array(chosen, dtype=int)


# ----------------------------------------------------------------------
# The algorithm
# ----------------------------------------------------------------------

def run_ensga2(inst, budget, N, rng, ref_point,
               use_init=True, use_adapt=True,
               eta=0.3, pc=0.9, kmin=1.0, kmax=6.0, k_fixed=1.0,
               eta_c=15, eta_m=20, gs=10):
    """
    One run. Returns (final_front_objectives, evaluations_used).

    use_init  : heuristic seeding on/off
    use_adapt : adaptive mutation on/off (canonical 1/I rate and decaying sigma
                if off, i.e. textbook NSGA-II)
    Plain NSGA-II = use_init=False, use_adapt=False.

    Mutation rate is parameterised as a MULTIPLE k of the canonical per-gene
    rate 1/I, so the expected number of perturbed assignment genes is k
    regardless of instance size. A fixed per-gene probability (the earlier
    formulation) mutated ~0.3*I genes on large instances, which destroyed the
    search as I grew.
    """
    I = inst.nI
    n_seed = int(round(eta * N)) if use_init else 0
    if n_seed > 0:
        As, Rs = heuristic_init(inst, n_seed, rng)
        Ar, Rr = random_init(inst, N - n_seed, rng)
        A, Rho = np.vstack([As, Ar]), np.vstack([Rs, Rr])
    else:
        A, Rho = random_init(inst, N, rng)

    F = evaluate_pop(inst, A, Rho)
    evals = N
    T = max(1, (budget - N) // N)          # generations affordable

    hv_best, stall, ext_max = -np.inf, 0, 0.0
    for t in range(T):
        fronts = fast_nondominated_sort(F)
        rank = np.zeros(N, dtype=int)
        cd = np.zeros(N)
        for r, fr in enumerate(fronts):
            rank[fr] = r
            cd[fr] = crowding_distance(F[fr])

        if use_adapt:
            f1 = F[fronts[0]]
            raw = front_extent(f1)
            ext_max = max(ext_max, raw)
            delta = raw / ext_max if ext_max > 1e-12 else 0.0
            hv_now = hypervolume_2d(f1, ref_point)
            if hv_now > hv_best:
                hv_best, stall = hv_now, 0
            else:
                stall += 1
            k = kmin + (kmax - kmin) * (1.0 - delta)
            if stall >= gs:                      # stagnation: temporary boost
                k = min(2.0 * k, 2.0 * kmax)
            pm = k / I
        else:
            pm = k_fixed / I

        parents = tournament(F, rank, cd, rng)
        Ac, Rc = variation(A, Rho, parents, pc, pm, eta_c, eta_m, inst.nJ, rng)
        Fc = evaluate_pop(inst, Ac, Rc)
        evals += N

        Aall = np.vstack([A, Ac]); Rall = np.vstack([Rho, Rc])
        Fall = np.vstack([F, Fc])
        keep = environmental_selection(Fall, N)
        A, Rho, F = Aall[keep], Rall[keep], Fall[keep]

    fronts = fast_nondominated_sort(F)
    return F[fronts[0]], evals
