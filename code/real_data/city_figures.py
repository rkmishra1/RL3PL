"""Redraw the four case-study figures from the original script with three corrections:
 1. the plan drawn in the flow diagrams is the knee plan itself (the original indexed the
    unsorted population with an index into the sorted front),
 2. the comparator is labelled as what it is, a nearest-hub assignment with all returns tendered,
 3. titles no longer call the front or the plan optimal.
Same seeds as the original, so the fronts and knee values are unchanged."""
import sys, re, os
code_dir, script, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, code_dir)
import matplotlib
matplotlib.use("Agg")
src = open(script).read()
head, tail = src.split("# Save results dictionary")
tail = tail[tail.index("# 3. PLOT FIGURES"):]
src = head + tail
src = re.sub(r"os\.makedirs\(.*\)", "pass", src)
src = src.replace("/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/", out.rstrip("/") + "/")
fix = """
def _match(A, RHO, inst, knee):
    d = Decoder(inst)
    cu, de = d.precompute_scenario_costs(inst['R'], inst['Ca'], inst['C'], inst['W'], inst['t'])
    z1, z2 = d.decode(A, RHO, inst['R'], inst['Ca'], cu, de)
    j = int(np.argmin(np.abs(z1 - knee[0]) + np.abs(z2 - knee[1])))
    assert abs(z1[j] - knee[0]) < 1e-6 and abs(z2[j] - knee[1]) < 1e-6
    return A[j], RHO[j]
"""
for c, s in (("delhi", "d"), ("london", "l")):
    old = f"knee_a_{s} = a_{c}[knee_idx_{s}]\nknee_rho_{s} = rho_{c}[knee_idx_{s}]"
    assert old in src
    src = src.replace(old, f"knee_a_{s}, knee_rho_{s} = _match(a_{c}, rho_{c}, inst_{c}, knee_{c})")
src = src.replace("from study import", fix + "\nfrom study import", 1) if False else src.replace(
    "out_dir =", fix + "\nout_dir =", 1)
for a, b in [
    ("'Status-Quo Heuristic Baseline'", "'Nearest-hub assignment, all returns tendered'"),
    ("'Status-Quo Baseline\\n(Nearest District Assignment)'", "'Nearest-hub assignment\\n(all returns tendered)'"),
    ("'Status-Quo Baseline\\n(Nearest Borough Assignment)'", "'Nearest-hub assignment\\n(all returns tendered)'"),
    ("'E-NSGA-II Pareto Front'", "'E-NSGA-II non-dominated front'"),
    ("Empirical Pareto Front", "Non-dominated Front"),
    ("Optimal Allocation Topology (Knee Point Plan)", "Allocation at the Knee-Point Plan"),
]:
    assert a in src, a
    src = src.replace(a, b)
exec(compile(src, "city_figures_src", "exec"), {})
