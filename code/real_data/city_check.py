"""Recompute what the saved case-study plans actually do (no new optimisation of the paper's results).
Rebuilds the two city instances from the original script (same seeds), then for the saved knee plan
and for the nearest-assignment plan reports cost, penalty and collection rate."""
import sys, pickle, re
import numpy as np
code_dir, script, pkl = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, code_dir)
import study as st
src = open(script).read()
src = src.split("# Save results dictionary")[0]
src = src.replace("import matplotlib.pyplot as plt", "")
src = re.sub(r"os\.makedirs\(.*\)", "pass", src)
ns = {}
exec(compile(src, "city_instances", "exec"), ns)
saved = pickle.load(open(pkl, "rb"))

def rate(inst, a, rho):
    dec = st.Decoder(inst)
    R, Ca = inst['R'], inst['Ca']
    Cu, de = dec.precompute_scenario_costs(R, Ca, inst['C'], inst['W'], inst['t'])
    z1, z2 = dec.decode(a[None, :], rho[None, :], R, Ca, Cu, de)
    I, J = inst['I'], inst['J']
    mask = (a[:, None] == np.arange(J)[None, :])
    V = (mask[:, :, None] * (rho[:, None, None] * R.T[:, None, :])).sum(axis=0)      # (J,W)
    sc = np.where(V > 0, np.minimum(1.0, Ca.T / np.maximum(V, 1e-9)), 1.0)
    Q = rho[:, None] * R.T * sc[a, :]                                                # (I,W)
    late = (dec.phi_2 * dec.beta_S * (de.transpose(1, 2, 0)[np.arange(I), a, :] * Q).sum(axis=0)).mean()
    return z1[0], z2[0], Q.sum() / R.sum() * 100, Q.mean(axis=1), late, (Q.sum(axis=0) / R.sum(axis=1) * 100)

for city, inst in (("delhi", ns['inst_delhi']), ("london", ns['inst_london'])):
    s = saved[city]
    front = s['front']
    k = rate(inst, np.asarray(s['a']), np.asarray(s['rho']))
    b = rate(inst, np.argmin(inst['T_ij'], axis=1), np.ones(inst['I']))
    print(f"\n{city}: total mean returns {inst['R'].sum(axis=1).mean():.1f}, total mean capacity {inst['Ca'].sum(axis=1).mean():.1f}")
    print(f"  saved knee  Z1={s['knee'][0]:.2f} Z2={s['knee'][1]:.2f}  recomputed Z1={k[0]:.2f} Z2={k[1]:.2f}")
    print(f"  knee   collection rate {k[2]:.1f}% (scenario range {k[5].min():.1f}-{k[5].max():.1f}), lateness part of Z2 {k[4]:.1f}")
    print(f"  knee   a={list(s['a'])} mean collected per zone {np.round(k[3],2)}")
    print(f"  nearest Z1={b[0]:.2f} Z2={b[1]:.2f} collection rate {b[2]:.1f}%  (saved {s['baseline']})")
    print(f"  front: n={len(front)} Z1 {front[:,0].min():.1f}..{front[:,0].max():.1f}  Z2 {front[:,1].min():.1f}..{front[:,1].max():.1f}")
    m = front[:, 1] <= b[1]
    print("  front points with penalty <= nearest plan:", int(m.sum()),
          (f"cheapest Z1={front[m,0].min():.2f}" if m.any() else ""))
    m = front[:, 0] <= b[0]
    print(f"  front point with cost <= nearest plan and lowest penalty: Z2={front[m,1].min():.2f}")
    d = (front[:, 0] >= b[0]) & (front[:, 1] >= b[1])
    print("  nearest plan dominated by a front point:", bool(((front[:,0] <= b[0]) & (front[:,1] <= b[1])).any()))

print("\n--- true knee plans (matched by objective value in the final population) ---")
out = {}
for city, inst, A, RHO in (("delhi", ns['inst_delhi'], ns['a_delhi'], ns['rho_delhi']),
                           ("london", ns['inst_london'], ns['a_london'], ns['rho_london'])):
    s = saved[city]
    dec = st.Decoder(inst)
    Cu, de = dec.precompute_scenario_costs(inst['R'], inst['Ca'], inst['C'], inst['W'], inst['t'])
    Z1, Z2 = dec.decode(A, RHO, inst['R'], inst['Ca'], Cu, de)
    j = int(np.argmin(np.abs(Z1 - s['knee'][0]) + np.abs(Z2 - s['knee'][1])))
    k = rate(inst, A[j], RHO[j])
    b = rate(inst, np.argmin(inst['T_ij'], axis=1), np.ones(inst['I']))
    util = []
    for c in range(inst['J']):
        util.append(k[3][A[j] == c].sum() / inst['Ca'][:, c].mean() * 100)
    print(f"{city}: knee Z1={k[0]:.2f} Z2={k[1]:.2f} (saved {s['knee'][0]:.2f}, {s['knee'][1]:.2f}) collection {k[2]:.1f}% "
          f"[{k[5].min():.1f}-{k[5].max():.1f}] lateness part {k[4]:.1f}")
    print("   a  =", [int(x) for x in A[j]], " nearest =", [int(x) for x in np.argmin(inst['T_ij'], axis=1)])
    print("   rho=", np.round(RHO[j], 2))
    print("   mean collected per zone =", np.round(k[3], 2), " mean returns =", np.round(inst['R'].mean(axis=0), 1))
    print("   centre utilisation % =", np.round(util, 1), " nearest-plan collection", round(b[2], 1))
    out[city] = dict(a=A[j], rho=RHO[j], Z=(k[0], k[1]), rate=k[2], coll=k[3], util=util, late=k[4],
                     near=(b[0], b[1], b[2]), front=s['front'])
pickle.dump(out, open("city_check.pkl", "wb"))
