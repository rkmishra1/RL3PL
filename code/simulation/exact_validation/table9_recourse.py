"""
Table 9 re-run with the recourse solved inside the decoder.

Chromosome: (a, lam).  a_i = collection centre of zone i (first stage).
lam in [0,1] = weight on Z1 against Z2 used to solve the recourse.
For each scenario and centre the weighted recourse is a continuous knapsack:
collect units in increasing order of  w = lam*c + (1-lam)*e  while w < 0,
where c is the unit cost (Z1) and e = phi2*beta_S*delta - phi1 the unit
change in Z2.  Service-centre choice is the cost-based one used by the
exact MILP of study.py, so the two are compared on the same model.

Single process, no Pool.  Usage:
  python3 table9_recourse.py <code_dir> <saved_results_dir> <out_dir>
"""
import sys, time, pickle, json
import numpy as np

code_dir, res_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, code_dir)
import study as st


class RecourseDecoder:
    def __init__(self, inst):
        self.inst = inst
        self.dec = st.Decoder(inst)
        self.R, self.Ca = inst['R'], inst['Ca']
        self.C_unit, self.delta = self.dec.precompute_scenario_costs(
            self.R, self.Ca, inst['C'], inst['W'], inst['t'])
        self.I, self.J = inst['I'], inst['J']
        self.W = self.R.shape[0]
        d = self.dec
        self.E_unit = d.phi_2 * d.beta_S * self.delta - d.phi_1   # (W,I,J)
        self.z2_const = d.phi_1 * self.R.sum(axis=1).mean()

    def decode(self, a, lam):
        """a: (N,I) int, lam: (N,) float -> Z1, Z2 arrays of length N."""
        N, I, W = a.shape[0], self.I, self.W
        ii = np.arange(I)[None, :, None]
        ww = np.arange(W)[None, None, :]
        c = self.C_unit.transpose(1, 2, 0)[ii, a[:, :, None], ww]   # (N,I,W)
        e = self.E_unit.transpose(1, 2, 0)[ii, a[:, :, None], ww]
        w = lam[:, None, None] * c + (1.0 - lam[:, None, None]) * e
        order = np.argsort(w, axis=1, kind='stable')                 # (N,I,W)
        remaining = np.broadcast_to(self.Ca.T[None, :, :], (N, self.J, W)).copy()
        Rw = self.R.T                                                # (I,W)
        Q = np.zeros((N, I, W))
        nn = np.arange(N)[:, None]
        wv = np.arange(W)[None, :]
        for k in range(I):
            z = order[:, k, :]                                       # (N,W)
            j = a[nn, z]                                             # (N,W)
            wz = w[nn, z, wv]
            rem = remaining[nn, j, wv]
            take = np.where(wz < 0.0, np.minimum(Rw[z, wv], rem), 0.0)
            Q[nn, z, wv] = take
            remaining[nn, j, wv] = rem - take
        Z1 = (c * Q).sum(axis=1).mean(axis=1)
        Z2 = self.z2_const + (e * Q).sum(axis=1).mean(axis=1)
        return Z1, Z2


def run_ensga2_recourse(inst, seed, return_pop=False):
    """E-NSGA-II of study.run_nsga2_base with the (a, lam) chromosome."""
    N, T_eval, pc = 100, 20000, 0.9
    I, J = inst['I'], inst['J']
    rd = RecourseDecoder(inst)
    rng = np.random.default_rng(seed)

    N_seed = int(np.ceil(0.3 * N))
    seed_a, _ = st.heuristic_seeding(inst, N_seed, rng)
    pop_a = np.vstack([seed_a, rng.integers(0, J, (N - N_seed, I))])
    pop_l = np.concatenate([np.linspace(0.0, 0.9, N_seed),
                            rng.uniform(0.0, 1.0, N - N_seed)])
    Z1, Z2 = rd.decode(pop_a, pop_l)
    evals = N
    ranks, fronts = st.fast_non_dominated_sort(Z1, Z2)
    cd = st.calculate_crowding_distance(Z1, Z2, fronts)
    ref_pt = np.array([1.1 * Z1.max(), 1.1 * Z2.max()])
    p_min, p_max, sigma_0, g_s = 0.05, 0.30, 0.2, 10
    t, T_max, spread_max, h_best, s_stall = 0, T_eval // N - 1, 0.0, -1.0, 0

    def pick():
        i1, i2 = rng.choice(N, 2, replace=False)
        if ranks[i1] != ranks[i2]:
            return i1 if ranks[i1] < ranks[i2] else i2
        return i1 if cd[i1] > cd[i2] else i2

    while evals < T_eval:
        f1 = fronts[0]
        fin = cd[f1][cd[f1] < np.inf]
        cd_avg = fin.mean() if len(fin) else 0.0
        spread_max = max(spread_max, cd_avg)
        d_t = cd_avg / spread_max if spread_max > 0 else 0.5
        h = st.compute_hypervolume(np.column_stack([Z1[f1], Z2[f1]]), ref_pt)
        if h > h_best:
            h_best, s_stall = h, 0
        else:
            s_stall += 1
        pm = p_min + (p_max - p_min) * (1.0 - d_t)
        sigma = sigma_0 * (1.0 - t / T_max) * (2.0 if s_stall >= g_s else 1.0)

        off_a, off_l = [], []
        for _ in range(N // 2):
            i1, i2 = pick(), pick()
            dummy = np.zeros(I)
            c1, _, c2, _ = st.crossover_two_point(pop_a[i1], dummy, pop_a[i2], dummy, pc, rng)
            u = rng.random()
            l1 = u * pop_l[i1] + (1 - u) * pop_l[i2]
            l2 = (1 - u) * pop_l[i1] + u * pop_l[i2]
            for ca, cl in ((c1, l1), (c2, l2)):
                m = rng.random(I) < pm
                ca = ca.copy()
                if m.any():
                    ca[m] = rng.integers(0, J, size=m.sum())
                if rng.random() < pm:
                    cl = float(np.clip(cl + rng.uniform(-sigma, sigma), 0.0, 1.0))
                off_a.append(ca); off_l.append(cl)
        off_a, off_l = np.array(off_a), np.array(off_l)
        oZ1, oZ2 = rd.decode(off_a, off_l)
        evals += N

        ca_, cl_ = np.vstack([pop_a, off_a]), np.concatenate([pop_l, off_l])
        cZ1, cZ2 = np.concatenate([Z1, oZ1]), np.concatenate([Z2, oZ2])
        ranks, fronts = st.fast_non_dominated_sort(cZ1, cZ2)
        cd = st.calculate_crowding_distance(cZ1, cZ2, fronts)
        keep = []
        for fr in fronts:
            if len(keep) + len(fr) <= N:
                keep.extend(fr)
            else:
                keep.extend(fr[np.argsort(-cd[fr])][:N - len(keep)])
                break
        pop_a, pop_l, Z1, Z2 = ca_[keep], cl_[keep], cZ1[keep], cZ2[keep]
        ranks, fronts = st.fast_non_dominated_sort(Z1, Z2)
        cd = st.calculate_crowding_distance(Z1, Z2, fronts)
        t += 1

    f1 = fronts[0]
    front = st.filter_non_dominated(np.unique(np.column_stack([Z1[f1], Z2[f1]]), axis=0))
    if return_pop:
        return front, pop_a[f1], pop_l[f1], Z1[f1], Z2[f1]
    return front


def brute_force_front(inst, n_lam=2001):
    """All assignments x a fine lam grid. Only for tiny instances."""
    import itertools
    I, J = inst['I'], inst['J']
    rd = RecourseDecoder(inst)
    A = np.array(list(itertools.product(range(J), repeat=I)))
    lam = np.linspace(0.0, 1.0, n_lam)
    pts = []
    for l_chunk in np.array_split(lam, max(1, n_lam // 50)):
        a = np.repeat(A, len(l_chunk), axis=0)
        l = np.tile(l_chunk, len(A))
        z1, z2 = rd.decode(a, l)
        pts.append(st.filter_non_dominated(np.unique(np.column_stack([z1, z2]), axis=0)))
    return st.filter_non_dominated(np.unique(np.vstack(pts), axis=0))


def main():
    exact = pickle.load(open(f"{res_dir}/exact_results.pkl", "rb"))
    comp = pickle.load(open(f"{res_dir}/comparison_results.pkl", "rb"))
    seeds = np.random.default_rng(9999).integers(10000, 99999, (12, 4, 30))
    out = {}
    for inst_id in (1, 2, 3):
        inst = st.generate_instance(inst_id)
        ex = exact[inst_id]['front']
        ref = 1.1 * ex.max(axis=0)
        ex_hv = st.compute_hypervolume(ex, ref)

        old = [st.compute_hypervolume(comp[inst_id]['E-NSGA-II'][r]['front'], ref) / ex_hv
               for r in range(30)]
        old_igd = [st.compute_igd_plus(comp[inst_id]['E-NSGA-II'][r]['front'], ex)
                   for r in range(30)]

        new, new_igd, cpu, sizes = [], [], [], []
        for r in range(30):
            t0 = time.time()
            fr = run_ensga2_recourse(inst, int(seeds[inst_id - 1, 0, r]))
            cpu.append(time.time() - t0)
            new.append(st.compute_hypervolume(fr, ref) / ex_hv)
            new_igd.append(st.compute_igd_plus(fr, ex))
            sizes.append(len(fr))
        row = dict(exact_size=len(ex),
                   rule_hv=(np.mean(old), np.std(old)),
                   rule_igd=(np.mean(old_igd), np.std(old_igd)),
                   rec_hv=(np.mean(new), np.std(new)), rec_hv_min=min(new),
                   rec_igd=(np.mean(new_igd), np.std(new_igd)),
                   rec_cpu=np.mean(cpu), rec_front=np.mean(sizes))
        if inst_id == 1:
            bf = brute_force_front(inst)
            row['bf_hv'] = st.compute_hypervolume(bf, ref) / ex_hv
            row['bf_igd'] = st.compute_igd_plus(bf, ex)
            row['bf_size'] = len(bf)
        out[inst_id] = row
        print(inst_id, json.dumps(row, default=float), flush=True)
    pickle.dump(out, open(f"{out_dir}/table9_recourse.pkl", "wb"))


if __name__ == "__main__":
    main()
