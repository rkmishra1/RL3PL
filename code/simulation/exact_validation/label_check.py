"""Re-run E-NSGA-II and SMS-EMOA once on instance 12 with the stored seeds and compare with the stored fronts."""
import sys, time, pickle
import numpy as np
sys.path.insert(0, sys.argv[1])
import study as st
c = pickle.load(open(sys.argv[2] + "/comparison_results.pkl", "rb"))
seeds = np.random.default_rng(9999).integers(10000, 99999, (12, 4, 30))
i = 12
inst = st.generate_instance(i)
allf = np.vstack([c[i][a][r]['front'] for a in c[i] for r in range(30)])
rp = 1.1 * allf.max(axis=0)
t0 = time.time()
fe = st.run_nsga2_base(inst, is_enhanced=True, seed=int(seeds[i - 1, 0, 0]))[0]
te = time.time() - t0
t0 = time.time()
fs = st.run_sms_emoa(inst, seed=int(seeds[i - 1, 2, 0]))[0]
ts = time.time() - t0
for name, f, t, key in (("fresh E-NSGA-II", fe, te, 'E-NSGA-II'), ("fresh SMS-EMOA", fs, ts, 'SMS-EMOA')):
    s = c[i][key][0]
    print(name, "time %.1f hv %.3e | stored under" % (t, st.compute_hypervolume(f, rp)), key,
          "time %.1f hv %.3e" % (s['runtime'], st.compute_hypervolume(s['front'], rp)),
          "| same front:", f.shape == s['front'].shape and bool(np.allclose(np.sort(f, 0), np.sort(s['front'], 0))))
