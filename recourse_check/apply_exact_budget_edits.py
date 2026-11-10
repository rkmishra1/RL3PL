"""Apply the manuscript edits for the comment on the exact-solver budget."""
import sys
path = sys.argv[1]
s = open(path, encoding="utf-8").read()
n_done = 0


def span(start, end, new):
    """Replace the text from the start anchor through the end anchor."""
    global s, n_done
    a = s.index(start)
    assert s.count(start) == 1, start
    b = s.index(end, a) + len(end)
    s = s[:a] + new + s[b:]
    n_done += 1


def rep(old, new, count=1):
    global s, n_done
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
    n_done += 1


# Abstract
rep("as an approximate solution method, because the problem is NP-hard.",
    "as an approximate solution method that returns a front in seconds.")
span("and on three small instances that can be solved exactly they recover",
     "when the recourse is solved inside the decoder.",
     "and measured against $\\epsilon$-constraint reference fronts they recover between 28\\% and 89\\% of the "
     "reference hypervolume, with the loss growing with instance size, while the same algorithm with the recourse "
     "solved inside the decoder stays at the level of the reference on every instance tested.")

# Contribution 3
rep("measures its effect on the instances that can be solved exactly.",
    "measures its effect against $\\epsilon$-constraint reference fronts.")

# Section 3.3
span("Taken together, these properties justify the use of multi-objective metaheuristics",
     "exact solutions of the scalarized single-scenario model.",
     "These properties make the worst case hard. They do not show that the instances studied here are beyond a "
     "general-purpose solver, and Section~\\ref{validation-against-exact-solutions} reports how far the "
     "$\\epsilon$-constraint method gets on them under a stated time budget. The approximate method of Section~4 "
     "is motivated by run time, not by a claim that exact solution is out of reach at benchmark size.")

# Section 4 opening
span("Consequently, exact bi-objective optimization approaches",
     "on the smallest test instances.",
     "Worst-case complexity does not settle whether a given instance can be solved in practice. With an open-source "
     "solver and 300 seconds per subproblem, the $\\epsilon$-constraint method applied to the scenario-expanded MILP "
     "proves optimality at every grid point up to instance~5 and at all but one to three of the 21 points up to "
     "instance~12. On the largest instance it still returns incumbents with certified gaps of about 4\\% at most in a "
     "little under two hours (Section~\\ref{validation-against-exact-solutions}). Exact solution is therefore "
     "available as a reference on this benchmark, and we use it as one. What an evolutionary method offers here is "
     "run time: seconds per run where the exact grid takes from under a minute to about two hours.")
rep("To efficiently solve larger and practically relevant problem instances, we propose",
    "To obtain approximate fronts quickly, we propose")

# Section 4.1
rep("measures the loss on the instances for which the exact front can be computed.",
    "measures the loss against $\\epsilon$-constraint reference fronts.")
rep("it recovers more than 99\\% of the exact hypervolume where the collection rule recovers between 63\\% and 89\\%.",
    "it stays at the level of the reference on every instance tested, where the collection rule recovers between "
    "28\\% and 89\\% of the reference hypervolume.")

# Section 5 intro
rep("(i) validate the proposed model and the E-NSGA-II solution approach against exact solutions on small instances",
    "(i) compare E-NSGA-II with the $\\epsilon$-constraint method run under a stated time budget")

# Section 5.2 opening paragraph
span("For instances 1--3, the scenario-expanded model is solved by the $\\epsilon$-constraint method",
     "the effect of that routing choice is not measured here.",
     "The scenario-expanded model is solved by the $\\epsilon$-constraint method with the HiGHS MILP solver called "
     "through SciPy. We had no access to a commercial solver such as Gurobi or CPLEX, so everything reported here "
     "is what an open-source solver achieves on one thread of a laptop. The budget is as follows. The "
     "minimum-penalty problem is solved first to fix the range of $Z_2$. Cost is then minimized at each of 21 "
     "equally spaced bounds on $Z_2$. Every subproblem has a limit of 300 seconds and a relative gap tolerance of "
     "$10^{-4}$. There is no overall cutoff, so the grid is always completed, and where a subproblem stops at its "
     "limit we keep the incumbent and the best bound. In the exact model the collected quantities "
     "$Q_{ij}^{\\omega}$ are free in every scenario. The service center of each zone is fixed by round-trip cost, as "
     "in the decoder, so the reference is exact for the model with cost-based repair routing, and the effect of "
     "that routing choice is not measured here.\n\n"
     "Two statements in the earlier version were wrong and are withdrawn. The text gave a limit of 180 seconds per "
     "subproblem inside an overall limit of 3{,}600 seconds. A grid of 21 points cannot finish under those two "
     "numbers even in principle. It also reported that instances 4 and 5 exceeded the limit. The exact method had "
     "in fact been run on instances 1--3 only, and the entries for instances 4 and 5 were not the outcome of a run. "
     "Instance~4 takes 40 seconds.\n\n"
     "For instances 1--3 the 21-point grid needs 0.27, 7.77 and 25.43 seconds, with every point proved optimal. "
     "A staircase through 21 points understates the hypervolume of the exact front by between 1\\% and 2\\% on "
     "these instances, enough for a good approximate front to score above one, so for them the grid was refined to "
     "201 points, again all optimal. Panel (a) of Table~\\ref{tab:Exact} uses this reference. Panel (b) covers "
     "instances 4--8, 10, 12 and 15 with the 21-point grid under the budget above. Instances 9, 11, 13 and 14 were "
     "not run.")

rep("Table~\\ref{tab:Exact} reports two versions of E-NSGA-II against this reference, with the same seeds and the same budget of 20{,}000 evaluations.",
    "Table~\\ref{tab:Exact} reports two versions of E-NSGA-II against these references, with the same seeds and the same budget of 20{,}000 evaluations.")

# Table 9: caption and second panel
span("\\caption{E-NSGA-II against the epsilon-constraint reference (201 grid points)",
     "Mean $\\pm$ sd over 30 replications.}",
     "\\caption{E-NSGA-II against the epsilon-constraint method, with the collection rule and with the recourse "
     "solved in the decoder. Panel (a): instances 1--3, reference of 201 grid points, all proved optimal. Panel (b): "
     "larger instances, 21 grid points with 300 seconds per subproblem. Mean $\\pm$ sd over 30 replications (10 for "
     "the recourse variant on instance 15).}")
rep("\\begin{tabular}{@{}llllll@{}}\n\\toprule\n & & \\multicolumn{2}{l}{Collection rule $(a,\\rho)$}",
    "\\textit{(a) Instances 1--3}\\\\[2pt]\n\\begin{tabular}{@{}llllll@{}}\n\\toprule\n & & \\multicolumn{2}{l}{Collection rule $(a,\\rho)$}")
panel_b = r"""3 & 10, 4 & 0.6256 $\pm$ 0.0491 & 20.03 $\pm$ 0.79 & 0.9918 $\pm$ 0.0008 & 0.284 $\pm$ 0.015 \\
\bottomrule
\end{tabular}

\vspace{8pt}
\textit{(b) Instances 4--15}\\[2pt]
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llrcccrllrr@{}}
\toprule
 & & & \multicolumn{4}{l}{Epsilon-constraint method (HiGHS)} & \multicolumn{2}{l}{HV ratio} & \multicolumn{2}{l}{Time per run (s)} \\
\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
Inst. & $|I|,|J|$ & Variables & Optimal & Max gap & Gap, min $Z_2$ & Time (s) & Rule & Recourse & Rule & Recourse \\
\midrule
4  & 12, 5   & 3{,}060   & 21/21 & --     & optimal & 40      & 0.847 $\pm$ 0.025 & 1.013 $\pm$ 0.001 & 3.9  & 1.9 \\
5  & 15, 6   & 4{,}590   & 21/21 & --     & optimal & 495     & 0.797 $\pm$ 0.012 & 1.010 $\pm$ 0.001 & 4.1  & 2.3 \\
6  & 18, 6   & 5{,}508   & 20/21 & n.f.   & 1.39\%  & 705     & 0.806 $\pm$ 0.006 & 1.012 $\pm$ 0.001 & 4.3  & 2.5 \\
7  & 21, 8   & 8{,}568   & 20/21 & n.f.   & 2.93\%  & 888     & 0.807 $\pm$ 0.036 & 1.010 $\pm$ 0.001 & 4.7  & 2.9 \\
8  & 24, 9   & 11{,}016  & 20/21 & n.f.   & 3.19\%  & 892     & 0.762 $\pm$ 0.050 & 1.009 $\pm$ 0.001 & 5.2  & 3.4 \\
10 & 30, 10  & 15{,}300  & 19/21 & 0.28\%, n.f. & 2.13\%  & 1{,}363 & 0.663 $\pm$ 0.044 & 1.008 $\pm$ 0.001 & 5.9  & 6.2 \\
12 & 36, 12  & 22{,}032  & 18/21 & 0.26\%, n.f. & 1.47\%  & 2{,}164 & 0.557 $\pm$ 0.050 & 1.008 $\pm$ 0.001 & 6.5  & 6.9 \\
15 & 100, 32 & 163{,}200 & 7/21  & 4.03\% & no bound & 6{,}691 & 0.284 $\pm$ 0.010 & 1.003 $\pm$ 0.001 & 30.0 & 8.2 \\
\bottomrule
\end{tabular}

\vspace{4pt}
\parbox{\linewidth}{\footnotesize \emph{Optimal}: grid points proved optimal within 300 seconds. \emph{Max gap}: largest relative gap between incumbent and bound among the remaining points. n.f.: the tightest grid point, whose bound on $Z_2$ equals the incumbent of the minimum-penalty problem, returned no feasible solution within the limit and is left out of the reference. \emph{Gap, min $Z_2$}: gap of the minimum-penalty problem at its limit. On instance 15 that problem returned an incumbent but no useful bound. \emph{Time}: total for the minimum-penalty problem and the 21 grid points. HV ratios are relative to the front of the incumbents. The rule columns for instance 15 use the stored runs of Section~\ref{scalability-analysis}.}
\end{table}"""
span("3 & 10, 4 & 0.6256 $\\pm$ 0.0491", "\\end{table}", panel_b)

# New discussion after the two existing paragraphs of Section 5.2
new_par = (
    "\n\n\\paragraph{Beyond instance 3.} Panel (b) of Table~\\ref{tab:Exact} continues the comparison. Three things "
    "follow from it.\n\n"
    "First, exact solution is not out of reach on this benchmark, and the earlier text was wrong to suggest "
    "it. HiGHS proves optimality at every grid point on instances 4 and 5, in 40 and 495 seconds, and at all but "
    "one to three points up to instance~12, in at most 37 minutes. Most subproblems are easy. The median time per "
    "grid point is below half a second up to instance~10 and about 15 seconds on instance~12. The effort is "
    "concentrated at the high-service end of the front, where the minimum-penalty problem stops at its limit with "
    "a gap between 1.4\\% and 3.2\\% on instances 6--12 and the tightest grid point finds no feasible solution. On "
    "instance~15, with 163{,}200 variables, 7 points are proved optimal and the other 14 stop with gaps between "
    "0.06\\% and 4.0\\%, for a total of 6{,}691 seconds. HiGHS checks its limit between node solves, so on this "
    "instance single subproblems ran past 300 seconds, in one case to 1{,}544 seconds. Even at this size the "
    "solver returns a usable front together with a bound on its error. The case for an evolutionary method on "
    "these instances therefore cannot be that exact solution is intractable. It is run time. One run of E-NSGA-II "
    "takes between 2 and 30 seconds, against 40 seconds to almost two hours for the exact grid, and a commercial "
    "solver would be expected to narrow that difference further.\n\n"
    "Second, the shortfall of the collection rule does not stay at the level seen on instances 1--3. It grows with "
    "size. The rule recovers 85\\% of the reference hypervolume on instance~4, 76\\% on instance~8, 56\\% on "
    "instance~12 and 28\\% on instance~15. The fronts reported for the larger instances in the remainder of this "
    "section are therefore further from the two-stage front than those for the small ones.\n\n"
    "Third, with the recourse solved in the decoder the same algorithm stays at the level of the reference on "
    "every instance, in 2 to 8 seconds per run. The ratios in panel (b) are slightly above one. This does not "
    "mean that the algorithm improves on the exact front. The reference is a staircase through at most 21 "
    "incumbents, which understates the exact hypervolume by 1\\% to 2\\% as noted above, it omits the tightest "
    "point on instances 6--12, and on instances 10--15 some of its points carry the gaps shown. What the ratios "
    "support is that the recourse variant is not measurably worse than what the exact method delivers under this "
    "budget. On instance~15 it reaches that level in about 8 seconds, where the exact grid needs close to two "
    "hours.")
anchor = "where the rule recovers less than two thirds of the exact hypervolume and the result also varies across seeds."
rep(anchor, anchor + new_par)

# Section 5.5
rep("and it can be answered only on instances 1--3, where the exact front is available.",
    "and we answer it at the level of single plans on instances 1--3, where a dense exact front is available.")
span("For instances 4, 8 and 12 in Table~\\ref{tab:oos-performance}, and for all larger instances",
     "it may be large.",
     "For instances 4, 8 and 12 in Table~\\ref{tab:oos-performance} the excess of the single plans was not "
     "computed. The reference fronts of Table~\\ref{tab:Exact}, panel (b), are available for them, and the "
     "collection rule recovers 85\\%, 76\\% and 56\\% of their hypervolume. The plans on those instances are "
     "therefore also far from the reference, whatever their out-of-sample gap.")

# Conclusion
rep("Because the model is NP-hard, an", "An")
rep("On the three instances that could be solved exactly, E-NSGA-II with this rule recovered",
    "On instances 1--3, where a dense exact reference was computed, E-NSGA-II with this rule recovered")
rep("This variant was tested on those three instances only.",
    "On instances 4--8, 10, 12 and 15, against reference fronts computed with an open-source solver under a budget "
    "of 300 seconds per subproblem, the rule recovered between 85\\% and 28\\% of the reference hypervolume, "
    "falling as the instance grows, and the recourse variant stayed at the level of the reference. The exact "
    "method proved optimality at every grid point up to instance~5 and returned fronts with gaps of about 4\\% at most "
    "on the largest instance in under two hours, so the argument for the evolutionary method on this benchmark is "
    "its run time of seconds and not the intractability of exact solution. No commercial solver was available to "
    "us. The recourse variant has been run for this comparison only.")
rep("and no measurement of the loss is available for them.",
    "and their fronts carry the loss measured above.")

open(path, "w", encoding="utf-8").write(s)
print("edits applied:", n_done)
