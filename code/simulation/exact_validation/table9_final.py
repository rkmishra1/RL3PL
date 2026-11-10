"""Table 9 numbers against the 21-point and the 201-point exact references.
Re-runs the recourse variant (same seeds as the published E-NSGA-II runs) and saves the fronts."""
import sys, time, pickle, importlib.util
import numpy as np
code_dir, res_dir = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], code_dir, res_dir, "."]
spec = importlib.util.spec_from_file_location("t9", "table9_recourse.py")
t9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t9)
st = t9.st

exact21 = pickle.load(open(f"{res_dir}/exact_results.pkl", "rb"))
dense = pickle.load(open("exact_dense.pkl", "rb"))
comp = pickle.load(open(f"{res_dir}/comparison_results.pkl", "rb"))
seeds = np.random.default_rng(9999).integers(10000, 99999, (12, 4, 30))
saved = {}
for inst_id in (1, 2, 3):
    inst = st.generate_instance(inst_id)
    rec, cpu = [], []
    for r in range(30):
        t0 = time.time()
        rec.append(t9.run_ensga2_recourse(inst, int(seeds[inst_id - 1, 0, r])))
        cpu.append(time.time() - t0)
    rule = [comp[inst_id]['E-NSGA-II'][r]['front'] for r in range(30)]
    rule_cpu = np.mean([comp[inst_id]['E-NSGA-II'][r]['runtime'] for r in range(30)])
    saved[inst_id] = dict(rec=rec, cpu=cpu)
    for name, ex in (("21pt", exact21[inst_id]['front']), ("201pt", dense[inst_id]['front'])):
        ref = 1.1 * ex.max(axis=0)
        exhv = st.compute_hypervolume(ex, ref)
        def stats(fronts):
            hv = [st.compute_hypervolume(f, ref) / exhv for f in fronts]
            ig = [st.compute_igd_plus(f, ex) for f in fronts]
            return np.mean(hv), np.std(hv), np.min(hv), np.max(hv), np.mean(ig), np.std(ig)
        a, b = stats(rule), stats(rec)
        print(f"inst {inst_id} ref {name} (n={len(ex)}): "
              f"RULE hv {a[0]:.4f}+-{a[1]:.4f} [{a[2]:.4f},{a[3]:.4f}] igd {a[4]:.1f}+-{a[5]:.1f} | "
              f"REC hv {b[0]:.4f}+-{b[1]:.4f} [{b[2]:.4f},{b[3]:.4f}] igd {b[4]:.1f}+-{b[5]:.1f}", flush=True)
    print(f"inst {inst_id} cpu: rule(saved) {rule_cpu:.2f}s rec {np.mean(cpu):.2f}s; "
          f"exact21 {exact21[inst_id]['runtime']:.2f}s dense {dense[inst_id]['runtime']:.1f}s", flush=True)
pickle.dump(saved, open("recourse_fronts.pkl", "wb"))
