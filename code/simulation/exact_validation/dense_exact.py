"""Denser epsilon-constraint reference fronts for instances 1-3 (201 grid points)."""
import sys, time, pickle
import numpy as np
sys.path.insert(0, sys.argv[1])
import study as st
out = {}
for inst_id in (1, 2, 3):
    inst = st.generate_instance(inst_id)
    t0 = time.time()
    fr = st.solve_exact(inst, grid_size=201)
    out[inst_id] = dict(front=fr, runtime=time.time() - t0)
    print(inst_id, len(fr), round(time.time() - t0, 1), flush=True)
    pickle.dump(out, open(sys.argv[2] + "/exact_dense.pkl", "wb"))
