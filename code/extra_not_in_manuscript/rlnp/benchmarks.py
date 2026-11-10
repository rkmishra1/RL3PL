"""
Benchmark multi-objective algorithms for Section 5.3.

All three share with E-NSGA-II:
  * the two-part encoding (integer assignment genes a, continuous intensity rho)
  * the decoder with proportional capacity repair (model.decode_batch), so every
    solution examined by every algorithm is feasible by construction
  * the same function-evaluation budget
  * the same in-sample scenario set (common random numbers)

Parameter settings are those recommended in each algorithm's original
publication; no problem-specific tuning is applied to any method.

  MOEA/D-DE  Li and Zhang (2009).  T = 20, delta = 0.9, n_r = 2, F = 0.5,
             CR = 1.0, polynomial mutation eta_m = 20 at rate 1/n.
             Normalised Tchebycheff scalarisation (the two objectives differ by
             two orders of magnitude, so the unnormalised form would collapse
             onto Z2).
  SMS-EMOA   Beume, Naujoks and Emmerich (2007).  Steady state (mu + 1); the
             member of the worst front with least exclusive hypervolume
             contribution is discarded. SBX eta_c = 15, polynomial mutation
             eta_m = 20 at rate 1/n.
  SPEA2      Zitzler, Laumanns and Thiele (2001).  Population = archive = N,
             raw fitness plus k-th nearest neighbour density with
             k = sqrt(N + Nbar), truncation-based environmental selection,
             binary tournament on the archive. SBX eta_c = 15, polynomial
             mutation eta_m = 20 at rate 1/n.

The published variation operators act on real-valued genes. The integer
assignment part of the chromosome is handled identically in every algorithm
(uniform crossover plus random-reset mutation at rate 1/I), so no method is
advantaged or penalised by the encoding.
"""

import numpy as np

from .model import decode, decode_batch
from .algorithms import (fast_nondominated_sort, sbx, poly_mut, int_cx,
                         int_mut)


# ----------------------------------------------------------------------
# Shared machinery
# ----------------------------------------------------------------------

def evaluate(inst, A, Rho):
    return decode_batch(inst, A, Rho)


def evaluate_one(inst, a, rho):
    """Single-solution evaluation. Identical arithmetic to decode_batch, but
    without the batch-axis overhead, which matters for the two steady-state /
    sequential benchmarks that evaluate one offspring at a time."""
    return np.asarray(decode(inst, a, rho), dtype=float)


def random_pop(inst, n, rng):
    return (rng.integers(0, inst.nJ, (n, inst.nI)), rng.random((n, inst.nI)))


def nds_2d(F):
    """Front index of every row. 2-D minimisation, O(N log N)."""
    n = F.shape[0]
    order = np.lexsort((F[:, 1], F[:, 0]))
    rank = np.empty(n, dtype=int)
    tails = []                      # tails[k] = min f2 so far on front k, increasing in k
    for i in order:
        f2 = F[i, 1]
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] > f2:
                hi = mid
            else:
                lo = mid + 1
        if lo == len(tails):
            tails.append(f2)
        else:
            tails[lo] = f2
        rank[i] = lo
    return rank


def hv_contributions_2d(F, ref):
    """Exclusive hypervolume contribution of each point of a 2-D front."""
    n = F.shape[0]
    order = np.argsort(F[:, 0])
    S = F[order]
    c = np.empty(n)
    for j in range(n):
        top = ref[1] if j == 0 else S[j - 1, 1]
        right = ref[0] if j == n - 1 else S[j + 1, 0]
        c[j] = max(right - S[j, 0], 0.0) * max(top - S[j, 1], 0.0)
    out = np.empty(n)
    out[order] = c
    return out


def final_front(F):
    return F[fast_nondominated_sort(F)[0]]


# ----------------------------------------------------------------------
# MOEA/D-DE  (Li and Zhang, 2009)
# ----------------------------------------------------------------------

def run_moead_de(inst, budget, N, rng,
                 T=20, delta=0.9, nr=2, CR=1.0, Fw=0.5, eta_m=20):
    I, nJ = inst.nI, inst.nJ
    pm = 1.0 / I

    lam = np.column_stack([np.linspace(0, 1, N), 1 - np.linspace(0, 1, N)])
    d = np.abs(lam[:, None, 0] - lam[None, :, 0])
    B = np.argsort(d, axis=1)[:, :T]

    A, Rho = random_pop(inst, N, rng)
    F = evaluate(inst, A, Rho)
    evals = N
    z = F.min(axis=0).copy()                     # ideal point
    znad = F.max(axis=0).copy()                  # running nadir, for normalisation

    def g(fv, w):
        s = np.maximum(znad - z, 1e-12)
        return np.max(w * (fv - z) / s, axis=-1)

    while evals < budget:
        for i in range(N):
            if evals >= budget:
                break
            pool = B[i] if rng.random() < delta else np.arange(N)
            r = rng.choice(pool, 3, replace=False)
            r1, r2, r3 = r[0], r[1], r[2]

            # DE/rand/1 on the continuous genes
            v = Rho[r1] + Fw * (Rho[r2] - Rho[r3])
            mask = rng.random(I) < CR
            child_rho = np.clip(np.where(mask, v, Rho[i]), 0.0, 1.0)
            child_rho = poly_mut(child_rho, pm, eta_m, rng)
            # matching operators on the integer genes
            child_a = int_mut(int_cx(A[r1], A[i], rng), pm, nJ, rng)

            fc = evaluate_one(inst, child_a, child_rho)
            evals += 1
            z = np.minimum(z, fc)
            znad = np.maximum(znad, fc)

            # replace at most nr neighbours
            cnt = 0
            for j in rng.permutation(pool):
                if cnt >= nr:
                    break
                if g(fc, lam[j]) <= g(F[j], lam[j]):
                    A[j], Rho[j], F[j] = child_a, child_rho, fc
                    cnt += 1

    return final_front(F), evals


# ----------------------------------------------------------------------
# SMS-EMOA  (Beume, Naujoks and Emmerich, 2007)
# ----------------------------------------------------------------------

def run_sms_emoa(inst, budget, N, rng, eta_c=15, eta_m=20):
    I, nJ = inst.nI, inst.nJ
    pm = 1.0 / I

    A, Rho = random_pop(inst, N, rng)
    F = evaluate(inst, A, Rho)
    evals = N

    while evals < budget:
        p, q = rng.integers(0, N, 2)
        child_rho = poly_mut(sbx(Rho[p], Rho[q], eta_c, rng), pm, eta_m, rng)
        child_a = int_mut(int_cx(A[p], A[q], rng), pm, nJ, rng)
        fc = evaluate_one(inst, child_a, child_rho)
        evals += 1

        Aa = np.vstack([A, child_a[None, :]])
        Rr = np.vstack([Rho, child_rho[None, :]])
        Ff = np.vstack([F, fc[None, :]])

        rank = nds_2d(Ff)
        worst = np.nonzero(rank == rank.max())[0]
        if worst.size == 1:
            drop = worst[0]
        else:
            W = Ff[worst]
            ref = W.max(axis=0) + 0.01 * (W.max(axis=0) - W.min(axis=0) + 1.0)
            drop = worst[int(np.argmin(hv_contributions_2d(W, ref)))]

        keep = np.ones(N + 1, dtype=bool)
        keep[drop] = False
        A, Rho, F = Aa[keep], Rr[keep], Ff[keep]

    return final_front(F), evals


# ----------------------------------------------------------------------
# SPEA2  (Zitzler, Laumanns and Thiele, 2001)
# ----------------------------------------------------------------------

def _spea2_fitness(F):
    n = F.shape[0]
    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]
    dom = le.all(axis=2) & lt.any(axis=2)              # dom[p,q]: p dominates q
    strength = dom.sum(axis=1).astype(float)
    raw = (dom * strength[:, None]).sum(axis=0)        # sum of strengths of dominators
    # normalised objective space for the density term
    rngs = np.maximum(F.max(axis=0) - F.min(axis=0), 1e-12)
    Z = (F - F.min(axis=0)) / rngs
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    k = int(np.sqrt(n))
    Ds = np.sort(D, axis=1)
    sigma = Ds[:, min(k, n - 1)]
    return raw + 1.0 / (sigma + 2.0), D


def _spea2_environmental(F, Nbar):
    fit, D = _spea2_fitness(F)
    sel = list(np.nonzero(fit < 1.0)[0])               # non-dominated members
    if len(sel) < Nbar:
        rest = np.argsort(fit)
        for i in rest:
            if i not in sel:
                sel.append(int(i))
            if len(sel) == Nbar:
                break
    else:
        sel = list(sel)
        while len(sel) > Nbar:
            sub = D[np.ix_(sel, sel)].copy()
            np.fill_diagonal(sub, np.inf)
            order = np.sort(sub, axis=1)
            # remove the individual with the smallest distance to its neighbours,
            # ties broken on successive nearest neighbours (lexicographic)
            drop = int(np.lexsort(order.T[::-1])[0])
            sel.pop(drop)
    return np.array(sel[:Nbar], dtype=int)


def run_spea2(inst, budget, N, rng, Nbar=None, eta_c=15, eta_m=20, pc=0.9):
    I, nJ = inst.nI, inst.nJ
    Nbar = Nbar or N
    pm = 1.0 / I

    A, Rho = random_pop(inst, N, rng)
    F = evaluate(inst, A, Rho)
    evals = N
    aA, aR, aF = A.copy(), Rho.copy(), F.copy()

    while evals < budget:
        UA = np.vstack([A, aA]); UR = np.vstack([Rho, aR]); UF = np.vstack([F, aF])
        keep = _spea2_environmental(UF, Nbar)
        aA, aR, aF = UA[keep], UR[keep], UF[keep]

        fit, _ = _spea2_fitness(aF)
        m = aF.shape[0]
        cand = rng.integers(0, m, (N, 2))
        win = np.where(fit[cand[:, 0]] <= fit[cand[:, 1]], cand[:, 0], cand[:, 1])

        A = np.empty((N, I), dtype=int)
        Rho = np.empty((N, I))
        for s in range(N):
            p, q = win[s], win[rng.integers(0, N)]
            if rng.random() < pc:
                Rho[s] = sbx(aR[p], aR[q], eta_c, rng)
                A[s] = int_cx(aA[p], aA[q], rng)
            else:
                Rho[s], A[s] = aR[p].copy(), aA[p].copy()
            Rho[s] = poly_mut(Rho[s], pm, eta_m, rng)
            A[s] = int_mut(A[s], pm, nJ, rng)

        F = evaluate(inst, A, Rho)
        evals += N

    # SPEA2 reports the archive, so run environmental selection once more and
    # return its non-dominated members (at most Nbar, matching the other methods)
    UF = np.vstack([F, aF])
    aF = UF[_spea2_environmental(UF, Nbar)]
    return final_front(aF), evals


ALGORITHMS = {
    "moead_de": run_moead_de,
    "sms_emoa": run_sms_emoa,
    "spea2": run_spea2,
}
