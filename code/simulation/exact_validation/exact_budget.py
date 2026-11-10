"""Epsilon-constraint runs with a stated budget: 21 grid points, a time limit per subproblem,
no overall cutoff.  For every subproblem the incumbent, the best bound and the gap are stored.
Usage: python3 exact_budget.py <code_dir> <instance ids, comma separated> <time limit s> <grid>
"""
import sys, time, pickle
import numpy as np
code_dir = sys.argv[1]
sys.path.insert(0, code_dir)
import study as st
import exact_sparse as es

ids = [int(x) for x in sys.argv[2].split(",")]
tl = float(sys.argv[3])
grid = int(sys.argv[4])
for i in ids:
    inst = st.generate_instance(i)
    t_all = time.time()
    m = es.build(st, inst)
    first = es.solve(m, m['c2'], time_limit=tl)
    zmin2, zmax2 = first['z2'], m['const2']
    # bound on min Z2 (objective was c2 without the constant)
    lb2 = (first['bound'] + m['const2']) if first['bound'] is not None else None
    rec = dict(inst=i, I=inst['I'], J=inst['J'], nvars=m['n'], tl=tl, grid=grid,
               minz2=dict(z2=zmin2, lb=lb2, gap=first['gap'], status=first['status'], time=first['time']),
               pts=[])
    print(f"inst {i}: minZ2 st={first['status']} t={first['time']:.1f}s gap={first['gap']:.2e}", flush=True)
    for k, eps in enumerate(np.linspace(zmin2, zmax2, grid)):
        r = es.solve(m, m['c1'], eps=eps * (1 + 1e-9), time_limit=tl)
        r.pop('x', None)
        r['eps'] = float(eps)
        rec['pts'].append(r)
        print(f"inst {i} pt {k+1}/{grid}: st={r['status']} t={r['time']:.1f}s gap={r['gap']} z1={r.get('z1')} bound={r['bound']}", flush=True)
        rec['total_time'] = time.time() - t_all
        pickle.dump(rec, open(f"exact_budget_i{i}.pkl", "wb"))
    print(f"inst {i} done total {rec['total_time']:.1f}s", flush=True)
