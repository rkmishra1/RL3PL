"""Recourse-in-decoder E-NSGA-II on further instances (30 seeds, same seeds as the stored runs)."""
import sys, time, pickle, importlib.util
import numpy as np
code_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2].split(",")]
sys.argv = [sys.argv[0], code_dir, ".", "."]
spec = importlib.util.spec_from_file_location("t9", "table9_recourse.py")
t9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t9)
st = t9.st
seeds = np.random.default_rng(9999).integers(10000, 99999, (12, 4, 30))
for i in ids:
    inst = st.generate_instance(i)
    fr, cpu = [], []
    for r in range(30):
        t0 = time.time()
        fr.append(t9.run_ensga2_recourse(inst, int(seeds[i - 1, 0, r])))
        cpu.append(time.time() - t0)
    pickle.dump(dict(rec=fr, cpu=cpu), open(f"rec_fronts_i{i}.pkl", "wb"))
    print(i, round(float(np.mean(cpu)), 2), flush=True)
