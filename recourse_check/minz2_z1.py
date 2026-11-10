"""Re-solve the min-Z2 problem (deterministic) to store the cost of its incumbent, which is the
feasible point at the high-service end of the grid."""
import sys, pickle
sys.path.insert(0, sys.argv[1])
import study as st
import exact_sparse as es
i = int(sys.argv[2])
m = es.build(st, st.generate_instance(i))
r = es.solve(m, m['c2'], time_limit=float(sys.argv[3]))
r.pop('x', None)
pickle.dump(r, open(f"minz2_i{i}.pkl", "wb"))
print(i, r['status'], r['z1'], r['z2'], r['bound'] + m['const2'], r['gap'], flush=True)
