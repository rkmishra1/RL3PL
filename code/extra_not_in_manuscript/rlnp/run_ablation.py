"""
Ablation driver, two phases.

  phase 1 (per variant, parallel over cores):
      python3 -m rlnp.run_ablation worker --variant full --instance 8 --reps 30
      -> saves fronts to runs/ablation_i8_full.npz

  phase 2 (once, after all variants):
      python3 -m rlnp.run_ablation collect --instance 8 --out ablation_raw.csv

Correctness guarantees:
  * replication r uses the SAME instance, scenario set and algorithm seed for
    every variant  -> paired comparison
  * the hypervolume reference point and the IGD+/epsilon reference front are
    computed ONCE in phase 2 from the union of all runs, then applied
    identically to every variant and replication
"""

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

from .model import make_instance
from .algorithms import (run_ensga2, fast_nondominated_sort, hypervolume_2d,
                         igd_plus, spread, epsilon_plus)

INSTANCES = {
    1: (6, 2, 3, 2, 1, 1),       2: (8, 3, 5, 2, 2, 1),
    3: (10, 4, 7, 3, 2, 1),      4: (12, 5, 8, 3, 2, 2),
    5: (15, 6, 9, 3, 3, 2),      6: (18, 6, 9, 4, 3, 2),
    7: (21, 8, 10, 4, 4, 3),     8: (24, 9, 12, 5, 4, 3),
    9: (27, 9, 14, 5, 5, 4),    10: (30, 10, 16, 6, 6, 4),
    11: (33, 10, 18, 6, 6, 5),  12: (36, 12, 18, 8, 8, 5),
    13: (50, 16, 24, 10, 10, 6), 14: (75, 24, 34, 14, 14, 8),
    15: (100, 32, 45, 18, 18, 10),
}

VARIANTS = {"full": (True, True), "no_init": (False, True),
            "no_adapt": (True, False), "plain": (False, False)}

RUNDIR = "runs"
INSTANCE_SEED = 20260813


def build_instance(instance, scenarios, dist="uniform"):
    return make_instance(*INSTANCES[instance], nW=scenarios,
                         rng=np.random.default_rng(INSTANCE_SEED + instance),
                         dist=dist)


_G = {}


def _init_worker(instance, scenarios, dist):
    _G["inst"] = build_instance(instance, scenarios, dist)


def _one_run(task):
    rep, ui, ua, budget, pop = task
    F, ev = run_ensga2(_G["inst"], budget=budget, N=pop,
                       rng=np.random.default_rng(10_000 + rep),
                       ref_point=np.array([np.inf, np.inf]),
                       use_init=ui, use_adapt=ua)
    return rep, F


def cmd_worker(args):
    ui, ua = VARIANTS[args.variant]
    lo, hi = args.rep_start, args.rep_start + args.reps
    tasks = [(r, ui, ua, args.budget, args.pop) for r in range(lo, hi)]
    os.makedirs(RUNDIR, exist_ok=True)
    with Pool(args.procs, initializer=_init_worker,
              initargs=(args.instance, args.scenarios, args.dist)) as pool:
        out = pool.map(_one_run, tasks)
    fronts = {f"rep{r}": F for r, F in out}
    path = (f"{RUNDIR}/ablation_i{args.instance}_{args.variant}"
            f"_r{lo}-{hi}.npz")
    np.savez_compressed(path, **fronts)
    sizes = [F.shape[0] for _, F in out]
    print(f"{args.variant}: {len(out)} runs, front sizes {min(sizes)}-{max(sizes)}"
          f" -> {path}")


def cmd_collect(args):
    import glob
    fronts = {}
    for v in VARIANTS:
        files = sorted(glob.glob(f"{RUNDIR}/ablation_i{args.instance}_{v}_r*.npz"))
        assert files, f"no run files for variant {v}"
        for path in files:
            z = np.load(path)
            for k in z.files:
                fronts[(v, int(k[3:]))] = z[k]
    allF = np.vstack(list(fronts.values()))
    ref_point = allF.max(axis=0) * 1.1
    ref_front = allF[fast_nondominated_sort(allF)[0]]
    print(f"reference point {ref_point}")
    print(f"reference front size {len(ref_front)}, runs collected {len(fronts)}")

    rows = []
    for (v, r), F in sorted(fronts.items(), key=lambda kv: (list(VARIANTS).index(kv[0][0]), kv[0][1])):
        rows.append(dict(variant=v, rep=r + 1,
                         hv=hypervolume_2d(F, ref_point),
                         igdplus=igd_plus(F, ref_front),
                         spread=spread(F),
                         epsplus=epsilon_plus(F, ref_front),
                         front_size=len(F)))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}\n")
    for v in VARIANTS:
        sel = [d for d in rows if d["variant"] == v]
        hv = np.array([d["hv"] for d in sel]); ig = np.array([d["igdplus"] for d in sel])
        sp = np.array([d["spread"] for d in sel])
        print(f"{v:9s} HV {hv.mean():.4g}+-{hv.std(ddof=1):.3g}   "
              f"IGD+ {ig.mean():.4g}+-{ig.std(ddof=1):.3g}   "
              f"spread {sp.mean():.4f}+-{sp.std(ddof=1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker")
    w.add_argument("--variant", required=True, choices=list(VARIANTS))
    w.add_argument("--instance", type=int, default=8)
    w.add_argument("--reps", type=int, default=30)
    w.add_argument("--rep-start", type=int, default=0)
    w.add_argument("--budget", type=int, default=20000)
    w.add_argument("--pop", type=int, default=100)
    w.add_argument("--scenarios", type=int, default=50)
    w.add_argument("--procs", type=int, default=4)
    w.add_argument("--dist", default="uniform")
    w.set_defaults(func=cmd_worker)

    c = sub.add_parser("collect")
    c.add_argument("--instance", type=int, default=8)
    c.add_argument("--out", default="ablation_raw.csv")
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
