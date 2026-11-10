"""Summarise the budgeted exact runs and compare both E-NSGA-II versions against the incumbent fronts."""
import sys, pickle, os
import numpy as np
code_dir, res_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, code_dir)
import study as st
comp = pickle.load(open(f"{res_dir}/comparison_results.pkl", "rb"))
for i in [int(x) for x in sys.argv[3].split(",")]:
    f = f"exact_budget_i{i}.pkl"
    if not os.path.exists(f):
        continue
    e = pickle.load(open(f, "rb"))
    pts = e['pts']
    npt = len(pts)
    opt_n = sum(p['status'] == 0 for p in pts)
    to = [p for p in pts if p['status'] != 0]
    gaps = [100 * p['gap'] for p in to if p.get('gap') is not None]
    inc = np.array([[p['z1'], p['z2']] for p in pts if 'z1' in p])
    inc = st.filter_non_dominated(inc)
    ref = 1.1 * inc.max(axis=0)
    hv = st.compute_hypervolume(inc, ref)
    # optimistic front from the bounds: for Z2 <= eps_k, Z1 >= bound_k
    bnd = np.array([[p['bound'], p['eps']] for p in pts])
    line = (f"inst {i} I={e['I']} J={e['J']} vars={e['nvars']} pts={npt}/{e['grid']} optimal={opt_n} timeouts={len(to)} "
            f"maxgap={max(gaps) if gaps else 0:.2f}% meangap={np.mean(gaps) if gaps else 0:.2f}% "
            f"minZ2 gap={100*e['minz2']['gap']:.2f}% total={e.get('total_time',0):.0f}s")
    rule = [comp[i]['E-NSGA-II'][r]['front'] for r in range(30)]
    h = [st.compute_hypervolume(x, ref) / hv for x in rule]
    g = [st.compute_igd_plus(x, inc) for x in rule]
    line += f"\n   RULE hv {np.mean(h):.4f}+-{np.std(h):.4f} igd {np.mean(g):.0f}+-{np.std(g):.0f}"
    if os.path.exists(f"rec_fronts_i{i}.pkl"):
        rec = pickle.load(open(f"rec_fronts_i{i}.pkl", "rb"))
        h = [st.compute_hypervolume(x, ref) / hv for x in rec['rec']]
        g = [st.compute_igd_plus(x, inc) for x in rec['rec']]
        line += f" | REC hv {np.mean(h):.4f}+-{np.std(h):.4f} igd {np.mean(g):.0f}+-{np.std(g):.0f} cpu {np.mean(rec['cpu']):.2f}s"
    # how far can the incumbent front be from exact: largest relative bound gap in Z1 over grid points
    rel = [(p['z1'] - p['bound']) / max(p['z1'], 1e-9) * 100 for p in pts if 'z1' in p and p['z1'] > 0]
    line += f"\n   largest (incumbent-bound)/incumbent over grid: {max(rel):.2f}%"
    print(line, flush=True)
