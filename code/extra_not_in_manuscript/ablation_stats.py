#!/usr/bin/env python3
"""
Turn raw per-replication ablation output into the LaTeX table for Section 5.4.

INPUT  : a CSV with one row per (variant, replication) and columns
             variant,rep,hv,igdplus,spread
         variant must use exactly these labels:
             full, no_init, no_adapt, plain
         (add more variants freely; 'full' is the reference for the tests)

OUTPUT : means, standard deviations, Wilcoxon signed-rank tests against the
         full algorithm with Holm correction, and a ready-to-paste LaTeX table.

USAGE  : python3 ablation_stats.py ablation_raw.csv
         python3 ablation_stats.py ablation_raw.csv --instance 8 --out table.tex

NOTE   : the tests are paired, so replication r of every variant must use the
         same seed and the same scenario set as replication r of 'full'.
         If your harness does not pair seeds across variants, pass --unpaired
         to switch to the Mann-Whitney U test instead.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict

try:
    from scipy.stats import wilcoxon, mannwhitneyu
except ImportError:
    sys.exit("scipy is required:  pip install scipy")

LABELS = {
    "full": "E-NSGA-II",
    "no_init": "w/o Init",
    "no_adapt": "w/o Adapt",
    "plain": "Plain NSGA-II",
}
# indicator name -> (column, higher_is_better)
INDICATORS = [
    ("HV", "hv", True),
    ("IGD+", "igdplus", False),
    ("Spread", "spread", False),
]


def read(path):
    data = defaultdict(lambda: defaultdict(list))  # variant -> col -> [values]
    reps = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            v = row["variant"].strip()
            reps[v].append(int(row["rep"]))
            for _, col, _ in INDICATORS:
                data[v][col].append(float(row[col]))
    for v, r in reps.items():
        if len(set(r)) != len(r):
            sys.exit(f"duplicate replication ids for variant '{v}'")
    return data


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def holm(pvals):
    """Holm-Bonferroni. Returns adjusted p-values in the original order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (len(pvals) - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def fmt_sci(m, sd, digits=4):
    """(mean +/- sd) x 10^e with a single common factor taken from the mean."""
    if m == 0:
        return f"$0 \\pm {sd:.{digits}f}$"
    e = int(math.floor(math.log10(abs(m))))
    return (f"$({m / 10**e:.{digits}f} \\pm {sd / 10**e:.{digits}f})"
            f"\\times 10^{{{e}}}$")


def fmt_plain(m, sd, digits=4):
    return f"${m:.{digits}f} \\pm {sd:.{digits}f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--instance", default="8")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--unpaired", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = read(args.csv)
    if "full" not in data:
        sys.exit("no rows with variant='full'")
    order = ["full"] + [v for v in ("no_init", "no_adapt", "plain") if v in data]
    others = order[1:]

    # ---- statistics -------------------------------------------------------
    marks = {v: {} for v in others}
    print(f"\nAblation, instance {args.instance}, "
          f"n = {len(data['full']['hv'])} replications\n")
    for name, col, higher_better in INDICATORS:
        ref = data["full"][col]
        raw_p, stats = [], []
        for v in others:
            x = data[v][col]
            if args.unpaired:
                _, p = mannwhitneyu(ref, x, alternative="two-sided")
            else:
                if len(x) != len(ref):
                    sys.exit(f"paired test needs equal n: full={len(ref)}, {v}={len(x)}")
                _, p = wilcoxon(ref, x)
            raw_p.append(p)
            stats.append(v)
        adj = holm(raw_p)
        print(f"  {name}")
        for v, p, pa in zip(stats, raw_p, adj):
            m_ref, _ = mean_sd(ref)
            m_v, _ = mean_sd(data[v][col])
            better_full = (m_ref > m_v) if higher_better else (m_ref < m_v)
            if pa >= args.alpha:
                mark = "="
            else:
                mark = "+" if better_full else "-"
            marks[v][name] = mark
            print(f"    full vs {v:<9} p={p:.4g}  Holm={pa:.4g}  -> {mark}")
        print()

    # ---- LaTeX ------------------------------------------------------------
    rows = []
    best = {}
    for name, col, higher_better in INDICATORS:
        vals = {v: mean_sd(data[v][col])[0] for v in order}
        best[name] = (max if higher_better else min)(vals, key=vals.get)

    for v in order:
        cells = []
        for name, col, _ in INDICATORS:
            m, sd = mean_sd(data[v][col])
            s = fmt_plain(m, sd) if name == "Spread" else fmt_sci(m, sd)
            if best[name] == v:
                s = "$\\mathbf{" + s.strip("$") + "}$"
            if v != "full":
                s = s[:-1] + f"{{}}^{{{marks[v][name]}}}$"
            cells.append(s)
        rows.append(f"{LABELS.get(v, v)} & " + " & ".join(cells) + " \\\\")

    tex = f"""\\begin{{table}}[!htb]
\\centering
\\caption{{Ablation of the E-NSGA-II components on instance {args.instance}
({len(data['full']['hv'])} replications, identical evaluation budget; the
feasibility repair operator is retained in all variants). Best value per
column in bold. $^{{+}}$/$^{{-}}$/$^{{=}}$ indicate that the full algorithm is
significantly better/worse/indistinguishable relative to the variant
(Wilcoxon signed-rank, Holm-corrected, $\\alpha={args.alpha}$).}}
\\label{{tab:Ablation}}
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Variant & HV (mean $\\pm$ sd) & IGD+ (mean $\\pm$ sd) & Spread (mean $\\pm$ sd) \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(tex)
        print(f"LaTeX written to {args.out}\n")
    else:
        print(tex)


if __name__ == "__main__":
    main()
