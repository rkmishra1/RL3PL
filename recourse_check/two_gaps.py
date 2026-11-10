"""Two different gaps for the knee-point plan on instances 1-3, reported side by side.

(a) out-of-sample gap: 50 in-sample scenarios -> 500 fresh scenarios (same plan).
(b) gap to the exact front (in-sample): extra cost of the plan against the exact front
    at the same or better service level, and extra penalty at the same or lower cost.
Both versions of the algorithm: collection rule (a, rho) and recourse in decoder (a, lam).
Knee plan taken from the median-hypervolume replication, as in study.py step 4.
"""
import sys, pickle, importlib.util
import numpy as np
code_dir, res_dir = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], code_dir, res_dir, "."]
spec = importlib.util.spec_from_file_location("t9", "table9_recourse.py")
t9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t9)
st = t9.st
dense = pickle.load(open("exact_dense.pkl", "rb"))
comp = pickle.load(open(f"{res_dir}/comparison_results.pkl", "rb"))
recf = pickle.load(open("recourse_fronts.pkl", "rb"))
seeds = np.random.default_rng(9999).integers(10000, 99999, (12, 4, 30))

def exact_gaps(z1, z2, ex):
    """Cost excess at equal-or-better service, and penalty excess at equal-or-lower cost."""
    m = ex[:, 1] <= z2
    g1 = (z1 - ex[m, 0].min()) / ex[m, 0].min() * 100 if m.any() else np.nan
    m = ex[:, 0] <= z1
    g2 = (z2 - ex[m, 1].min()) / ex[m, 1].min() * 100 if m.any() else np.nan
    return g1, g2

def knee(Z1, Z2):
    fr = st.filter_non_dominated(np.column_stack([Z1, Z2]))
    kv, _ = st.select_knee_point(fr)
    return int(np.argmin(np.abs(Z1 - kv[0]) + np.abs(Z2 - kv[1])))

out = {}
for inst_id in (1, 2, 3):
    inst = st.generate_instance(inst_id)
    ex = dense[inst_id]['front']
    ref = 1.1 * ex.max(axis=0)
    oos = dict(inst)
    oos.update(R=inst['R_oos'], Ca=inst['Ca_oos'], C=inst['C_oos'], W=inst['W_oos'], t=inst['t_oos'])

    # collection rule
    hv = sorted((st.compute_hypervolume(comp[inst_id]['E-NSGA-II'][r]['front'], ref), r) for r in range(30))
    med = hv[15][1]
    _, pa, pr = st.run_nsga2_base(inst, is_enhanced=True, seed=int(seeds[inst_id - 1, 0, med]))
    dec = st.Decoder(inst)
    Cu, de = dec.precompute_scenario_costs(inst['R'], inst['Ca'], inst['C'], inst['W'], inst['t'])
    Z1, Z2 = dec.decode(pa, pr, inst['R'], inst['Ca'], Cu, de)
    k = knee(Z1, Z2)
    Cuo, deo = dec.precompute_scenario_costs(oos['R'], oos['Ca'], oos['C'], oos['W'], oos['t'])
    o1, o2 = dec.decode(pa[k:k+1], pr[k:k+1], oos['R'], oos['Ca'], Cuo, deo)
    rule = dict(z1=Z1[k], z2=Z2[k], oos1=(o1[0]-Z1[k])/Z1[k]*100, oos2=(o2[0]-Z2[k])/Z2[k]*100,
                ex=exact_gaps(Z1[k], Z2[k], ex))

    # recourse in decoder
    hv = sorted((st.compute_hypervolume(recf[inst_id]['rec'][r], ref), r) for r in range(30))
    med = hv[15][1]
    _, pa, pl, Z1, Z2 = t9.run_ensga2_recourse(inst, int(seeds[inst_id - 1, 0, med]), return_pop=True)
    k = knee(Z1, Z2)
    rdo = t9.RecourseDecoder(oos)
    o1, o2 = rdo.decode(pa[k:k+1], pl[k:k+1])
    rec = dict(z1=Z1[k], z2=Z2[k], oos1=(o1[0]-Z1[k])/Z1[k]*100, oos2=(o2[0]-Z2[k])/Z2[k]*100,
               ex=exact_gaps(Z1[k], Z2[k], ex))
    out[inst_id] = dict(rule=rule, rec=rec)
    for nm, d in (("RULE", rule), ("REC ", rec)):
        print(f"inst {inst_id} {nm}: knee Z1={d['z1']:.0f} Z2={d['z2']:.0f} | OOS gap Z1 {d['oos1']:+.2f}% Z2 {d['oos2']:+.2f}% | "
              f"vs exact: cost excess {d['ex'][0]:+.2f}% penalty excess {d['ex'][1]:+.2f}%", flush=True)
pickle.dump(out, open("two_gaps.pkl", "wb"))
