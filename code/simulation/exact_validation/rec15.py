"""Recourse-in-decoder E-NSGA-II on instance 15 (10 seeds) and comparison with the budgeted exact front."""
import sys, time, pickle, importlib.util
import numpy as np
code_dir, res_dir = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], code_dir, ".", "."]
spec = importlib.util.spec_from_file_location("t9", "table9_recourse.py")
t9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t9)
st = t9.st
i = 15
seeds = np.random.default_rng(9999 + i).integers(10000, 99999, 10)
inst = st.generate_instance(i)
fr, cpu = [], []
for r in range(10):
    t0 = time.time()
    fr.append(t9.run_ensga2_recourse(inst, int(seeds[r])))
    cpu.append(time.time() - t0)
    print("rep", r, round(cpu[-1], 1), flush=True)
    pickle.dump(dict(rec=fr, cpu=cpu), open(f"rec_fronts_i{i}.pkl", "wb"))
e = pickle.load(open(f"exact_budget_i{i}.pkl", "rb"))
inc = st.filter_non_dominated(np.array([[p['z1'], p['z2']] for p in e['pts'] if 'z1' in p]))
ref = 1.1 * inc.max(axis=0)
hv = st.compute_hypervolume(inc, ref)
sc = pickle.load(open(f"{res_dir}/scalability_results.pkl", "rb"))[i]['E-NSGA-II']
rule = [sc[k]['front'] for k in sorted(sc)]
print("n rule reps", len(rule), "keys", list(sc[0].keys()))
for name, fs in (("RULE", rule), ("REC", fr)):
    h = [st.compute_hypervolume(x, ref) / hv for x in fs]
    g = [st.compute_igd_plus(x, inc) for x in fs]
    print(name, "hv %.4f+-%.4f igd %.0f+-%.0f" % (np.mean(h), np.std(h), np.mean(g), np.std(g)))
print("rec cpu mean", np.mean(cpu), "done15")
