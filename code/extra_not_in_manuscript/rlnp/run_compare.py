"""
Four-algorithm comparison driver (Sections 5.3, 5.6 and 5.7).

  phase 1 (per algorithm, parallel over cores):
      python3 -m rlnp.run_compare worker --algo ensga2 --instance 8 --reps 30
      -> runs/cmp_i8_uniform_ensga2_r0-30.npz   (fronts + wall-clock times)

  phase 2 (once, after all algorithms for that instance/distribution):
      python3 -m rlnp.run_compare collect --instance 8 --out compare_raw.csv

Correctness guarantees, identical to the ablation driver:
  * replication r uses the SAME instance, scenario set and algorithm seed for
    every algorithm  -> paired comparison
  * the hypervolume reference point and the IGD+/epsilon reference front are
    computed ONCE in phase 2 from the union of all runs for that instance,
    across every algorithm AND every distribution, then applied identically.
    That makes indicator values comparable within an instance block.
"""

import argparse
import csv
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

from .run_ablation import INSTANCES, build_instance
from .algorithms import (run_ensga2, fast_nondominated_sort, hypervolume_2d,
                         igd_plus, spread, epsilon_plus)
from .benchmarks import run_moead_de, run_sms_emoa, run_spea2

ALGOS = ["ensga2", "moead_de", "sms_emoa", "spea2"]
LABELS = {"ensga2": "E-NSGA-II", "moead_de": "MOEA/D-DE",
          "sms_emoa": "SMS-EMOA", "spea2": "SPEA2"}
DISTS = ["uniform", "normal", "lognormal"]
RUNDIR = "runs"

_G = {}


def _init_worker(instance, scenarios, dist):
    _G["inst"] = build_instance(instance, scenarios, dist)


def _one_run(task):
    rep, algo, budget, pop = task
    inst = _G["inst"]
    seed = np.random.default_rng(10_000 + rep)
    t0 = time.perf_counter()
    if algo == "ensga2":
        F, ev = run_ensga2(inst, budget, pop, seed,
                           ref_point=np.array([np.inf, np.inf]))
    elif algo == "moead_de":
        F, ev = run_moead_de(inst, budget, pop, seed)
    elif algo == "sms_emoa":
        F, ev = run_sms_emoa(inst, budget, pop, seed)
    elif algo == "spea2":
        F, ev = run_spea2(inst, budget, pop, seed)
    else:
        raise ValueError(algo)
    return rep, F, time.perf_counter() - t0


def cmd_worker(args):
    lo, hi = args.rep_start, args.rep_start + args.reps
    tasks = [(r, args.algo, args.budget, args.pop) for r in range(lo, hi)]
    os.makedirs(RUNDIR, exist_ok=True)
    with Pool(args.procs, initializer=_init_worker,
              initargs=(args.instance, args.scenarios, args.dist)) as pool:
        out = pool.map(_one_run, tasks)
    payload = {f"rep{r}": F for r, F, _ in out}
    payload["cpu"] = np.array([c for _, _, c in out])
    payload["reps"] = np.array([r for r, _, _ in out])
    path = (f"{RUNDIR}/cmp_i{args.instance}_{args.dist}_{args.algo}"
            f"_r{lo}-{hi}.npz")
    np.savez_compressed(path, **payload)
    print(f"{args.algo:9s} i{args.instance} {args.dist:9s} {len(out)} runs, "
          f"cpu {payload['cpu'].mean():.2f}s -> {path}")


def _load(instance):
    fronts, cpu = {}, {}
    for path in sorted(glob.glob(f"{RUNDIR}/cmp_i{instance}_*.npz")):
        stem = os.path.basename(path)[:-4].split("_")
        dist, algo = stem[2], "_".join(stem[3:-1])
        z = np.load(path)
        reps = z["reps"]
        for j, r in enumerate(reps):
            fronts[(dist, algo, int(r))] = z[f"rep{int(r)}"]
            cpu[(dist, algo, int(r))] = float(z["cpu"][j])
    return fronts, cpu


def cmd_collect(args):
    fronts, cpu = _load(args.instance)
    assert fronts, f"no run files for instance {args.instance}"
    allF = np.vstack(list(fronts.values()))
    ref_point = allF.max(axis=0) * 1.1
    ref_front = allF[fast_nondominated_sort(allF)[0]]
    print(f"instance {args.instance}: {len(fronts)} runs, "
          f"ref point {ref_point}, ref front {len(ref_front)} points")

    rows = []
    for (dist, algo, r), F in sorted(fronts.items()):
        rows.append(dict(instance=args.instance, dist=dist, algo=algo, rep=r + 1,
                         hv=hypervolume_2d(F, ref_point),
                         igdplus=igd_plus(F, ref_front),
                         spread=spread(F),
                         epsplus=epsilon_plus(F, ref_front),
                         cpu=cpu[(dist, algo, r)],
                         front_size=len(F)))
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {args.out}\n")

    for dist in DISTS:
        sel_d = [d for d in rows if d["dist"] == dist]
        if not sel_d:
            continue
        print(f"  {dist}")
        for a in ALGOS:
            s = [d for d in sel_d if d["algo"] == a]
            if not s:
                continue
            hv = np.array([d["hv"] for d in s])
            ig = np.array([d["igdplus"] for d in s])
            cp = np.array([d["cpu"] for d in s])
            print(f"    {LABELS[a]:10s} HV {hv.mean():.4e}+-{hv.std(ddof=1):.2e}"
                  f"   IGD+ {ig.mean():.4e}+-{ig.std(ddof=1):.2e}"
                  f"   CPU {cp.mean():6.2f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker")
    w.add_argument("--algo", required=True, choices=ALGOS)
    w.add_argument("--instance", type=int, required=True)
    w.add_argument("--dist", default="uniform", choices=DISTS)
    w.add_argument("--reps", type=int, default=30)
    w.add_argument("--rep-start", type=int, default=0)
    w.add_argument("--budget", type=int, default=20000)
    w.add_argument("--pop", type=int, default=100)
    w.add_argument("--scenarios", type=int, default=50)
    w.add_argument("--procs", type=int, default=4)
    w.set_defaults(func=cmd_worker)

    c = sub.add_parser("collect")
    c.add_argument("--instance", type=int, required=True)
    c.add_argument("--out", default="compare_raw.csv")
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
