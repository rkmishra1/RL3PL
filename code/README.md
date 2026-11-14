# Code for the reverse logistics network paper (Comput. Mgmt. Science)

Collected on 2026-11-10 from three places: `Downloads/Reverse_Logistix_2` (the main module and the table scripts),
the scratch folders of the July and August assistant sessions (stored results, city script), and
`recourse_check/` in this project (the November revision runs).

Checked with Python 3.9.6, numpy 2.0.2, scipy 1.13.1, matplotlib 3.9.4 (`/usr/bin/python3` on this Mac).
The exact solver is HiGHS, called through `scipy.optimize.milp`.

Read the section "Known problems" before sharing this folder with anyone.

## Layout

| Folder | Content |
|---|---|
| `core/study.py` | Instance generator, scenario sampler, decoder, epsilon-constraint solver, E-NSGA-II, MOEA/D-DE, SMS-EMOA, SPEA2, indicators, knee and TOPSIS selection, and the driver `main()` for the whole computational study. |
| `results/` | The eight result files written by `core/study.py` in July 2026. All scripts below read these. |
| `simulation/main_study/` | `fill_cms.py` and `fill_manuscript.py`: build the tables (means, sd, Friedman, Wilcoxon, Holm) from `results/` and write them into a tex file. `fill_manuscript.py` also draws `Fronts.png` and `Knee.png` from the stored runs. |
| `simulation/exact_validation/` | November revision: exact fronts with a proper time budget, E-NSGA-II with the recourse solved in the decoder, the two-gaps table. Scripts, their result files and logs are kept together because the scripts read and write in the current directory. |
| `real_data/` | Delhi NCR and Greater London case studies. |
| `figure_scripts_recovered/` | Four figure scripts that were never saved as files. Recovered from the August session log. Three of them do not use any algorithm output (see Known problems). |
| `extra_not_in_manuscript/` | Second implementation `rlnp/` with its partial runs, `ablation_stats.py`, and the Karnataka case study. None of this feeds the current manuscript. |

## Simulation study

Full study (instances 1 to 15, 30 replications, four algorithms). It uses a multiprocessing Pool and takes hours.
Run it from a file, in a terminal, and leave the machine idle.

```bash
cd code/core && python3 study.py ../results_rerun
```

The output folder is the first argument (default `results_rerun`), so the stored `results/` are not overwritten.

| Manuscript item | Produced by | Stored result |
|---|---|---|
| HV, IGD+, Spread, epsilon, CPU tables (instances 1 to 12) | `study.py`, then `fill_cms.py` | `comparison_results.pkl` |
| Ablation table | `study.py`, then `fill_cms.py` | `ablation_results.pkl` |
| Out-of-sample table (instances 4, 8, 12) | `study.py` | `oos_results.pkl` |
| Distributional robustness table | `study.py`, then `fill_cms.py` | `robustness_results.pkl` |
| Scalability table (instances 13 to 15) | `study.py`, then `fill_cms.py` | `scalability_results.pkl` |
| Knee and TOPSIS table | `study.py` | `closure_results.pkl` |
| Sensitivity table | `study.py` | `sensitivity_results.pkl` |

Tables from stored results. Both scripts rewrite the tex file given as the second argument, and they were written
for the July versions of the manuscript. Use a copy, never the current manuscript.

```bash
cd code/simulation/main_study && PYTHONPATH=../../core python3 fill_cms.py ../../results /path/to/copy.tex
```

Exact validation and recourse runs (table `tab:Exact`, table `tab:TwoGaps`). Run from inside the folder.
`CORE=../../core`, `RES=../../results`.

| Script | What it does | Command |
|---|---|---|
| `exact_sparse.py` | Sparse scenario-expanded MILP and HiGHS call. Imported by the others. | |
| `dense_exact.py` | 201-point exact fronts, instances 1 to 3. Writes `exact_dense.pkl`. | `python3 dense_exact.py $CORE .` |
| `table9_recourse.py` | E-NSGA-II with the recourse LP solved in the decoder. Also a module for the scripts below. | `python3 table9_recourse.py $CORE $RES .` |
| `table9_final.py` | Panel (a): rule and recourse variants against the 21 and 201 point references. Writes `recourse_fronts.pkl`. | `python3 table9_final.py $CORE $RES` |
| `exact_budget.py` | Panel (b): epsilon-constraint with a time limit per grid point, with gaps. Writes `exact_budget_i*.pkl`. | `python3 exact_budget.py $CORE 4,5 300 21` |
| `minz2_z1.py` | Minimum-penalty problem for one instance. | `python3 minz2_z1.py $CORE 6 300` |
| `rec_runs.py`, `rec15.py` | Recourse variant with 30 seeds on further instances, and with 10 seeds on instance 15. | `python3 rec_runs.py $CORE 4,5,6` |
| `exact_table.py` | Prints the rows of panel (b) from the stored files. Fast. | `python3 exact_table.py $CORE $RES 4,8` |
| `two_gaps.py` | Out-of-sample gap and excess over the exact front, instances 1 to 3. | `python3 two_gaps.py $CORE $RES` |
| `label_check.py` | Re-runs E-NSGA-II and SMS-EMOA once on instance 12 and compares with the stored fronts. | `python3 label_check.py $CORE $RES` |

Time limits and instance lists used for the stored files are in the `.log` files.

## Real data analysis

The case-study inputs (mean returns, hub capacities, distance matrices, sorting fractions) are typed into
`run_city_studies.py` as arrays. There is no raw data file. Scenarios are drawn around these means with fixed seeds
(101 for Delhi, 202 for London).

| Script | What it does |
|---|---|
| `run_city_studies.py` | Original August script, unchanged. Builds both instances, runs E-NSGA-II, draws four figures. Its output paths point to the old project folder, so do not run it directly. The two scripts below read it as text and need it unchanged. |
| `city_figures.py` | Redraws the four figures with the corrected knee plan and labels. These are the figures in the manuscript. |
| `city_check.py` | Recomputes cost, penalty, collection rate and hub utilisation of the knee plan and of the nearest-hub plan (cross-city table). Writes `city_check.pkl` in the current directory. |

```bash
cd code/real_data && python3 city_figures.py ../core run_city_studies.py /path/to/output_folder
```

```bash
cd code/real_data && python3 city_check.py ../core run_city_studies.py case_studies_results.pkl
```

## Changes made to the copies

Only hard-coded paths were changed, in three files. Everything else is a byte copy.

- `core/study.py`: `import sys` was added, and the output folder in `main()` is now the first argument.
- `simulation/main_study/fill_cms.py` and `fill_manuscript.py`: `import sys` was added, the results folder and the tex file are now arguments, and figures are saved next to the tex file.

## What was checked on 2026-11-10

- All 30 scripts compile. `core/study.py` imports.
- `exact_table.py` runs on the stored files for instances 4 and 8.
- `city_figures.py` reproduces the four case-study PNG files in the manuscript folder byte for byte.
- `city_check.py` reproduces the stored knee values: Delhi (1970.89, 9167.98) and London (998.49, 9319.76).
- Recomputed from `results/` and compared with the manuscript: CPU table (instances 1 to 3, 8), out-of-sample table, sensitivity table agree.
- Nothing long was re-run. The full study, the exact runs and the recourse runs were not repeated.

