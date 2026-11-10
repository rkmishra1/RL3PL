# Draft replies to the reviewer comments on recourse, Table 9, the two gaps, the case studies and the exact-solver budget

## Comment 1. Recourse in the model and in the algorithm

**Comment (summary).** Table 5 declares the flows and the uncollected volume as second-stage variables, while Section 4.1 encodes a collection intensity that does not depend on the scenario. The algorithm therefore searches a restricted policy class and not the recourse space of the model.

**Response.** The reviewer is right. The model is meant as stated in Table 5: the collected quantities and the uncollected volume are wait-and-see decisions. The algorithm in the submitted version fixes the fraction of each zone's returns that the provider attempts to collect before the scenario is known. Our earlier wording, including our reply to Reviewer 1's second comment, described the model correctly but did not say that the algorithm works with a restriction of it.

**Changes made.**

1. *Abstract, contribution 3 and the end of Section 3.* The method is described as an approximate method that searches over the assignments and a scenario-independent collection rule.
2. *Section 4.1.* A new paragraph, "Relation to the two-stage model", gives the rule in closed form (equation 4.1.1) and lists the three ways in which it restricts the recourse of Table 5. It states that every decoded solution is feasible, so the reported fronts approximate the two-stage front from the inside, and that the four algorithms share the decoder.
3. *The reviewer's suggestion.* For a given assignment, scenario and weight on the two objectives, the recourse is a linear programme that separates into a continuous knapsack problem at each collection centre. We have implemented this inside the decoder, with the weight carried as a gene in place of the intensity genes, and tested it on instances 1 to 3 (see the reply to Comment 2).
4. *Section 5.5 and the Conclusion.* The out-of-sample evaluation is described as applying the rule to new scenarios, and the restriction is now the first methodological limitation.

## Comment 2. The hypervolume ratios in Table 9

**Comment (summary).** Ratios of 0.86 to 0.90 were described as closely reproducing the exact fronts. They do not. On instance 1, with 64 first-stage assignments, the algorithm is 14 per cent short with a standard deviation of 0.0003, which means it cannot represent what it is missing. The reviewer asks for the table to be re-run once the recourse is handled properly.

**Response.** We agree on every point, and the re-run confirms the reviewer's reading.

1. *The wording was wrong.* The phrases "closely reproduces" and "near-exact front quality" have been removed.
2. *Correction to the instance 3 row.* When we recomputed the table from the stored runs, the instance 3 row did not match what we had printed. The stored runs give a hypervolume ratio of 0.6372 with standard deviation 0.0500, and an IGD+ of 2.046 x 10^5, against the 21-point reference. The earlier table showed 0.8772 and 4.646 x 10^4. The rows for instances 1 and 2 were correct. We apologise for this error. The corrected value makes the reviewer's point more strongly.
3. *Denser reference.* The exact reference now uses 201 grid points instead of 21. With 21 points the staircase understates the exact hypervolume by 1 to 2 per cent, and a good front can then score above one.
4. *Re-run with the recourse handled in the decoder.* Table 9 now reports both versions, with the same seeds and the same budget of 20,000 evaluations.

| Instance | Collection rule, HV ratio | Recourse in decoder, HV ratio |
|---|---|---|
| 1 | 0.8459 ± 0.0003 | 0.9958 ± 0.0002 |
| 2 | 0.8850 ± 0.0031 | 0.9944 ± 0.0005 |
| 3 | 0.6256 ± 0.0491 | 0.9918 ± 0.0008 |

The IGD+ falls by a factor of between 20 and 70. For instance 1 we also enumerated all 64 assignments over 2001 weights. Against the 21-point reference the enumeration scores 1.018 and the evolutionary search with the recourse decoder scores 1.014. What remains after the recourse is handled is well under 1 per cent.

5. *Why the rule fails.* Collection capacity is far below the returned volume in every scenario, and the lateness penalty of a unit exceeds the uncollected penalty whenever the lateness is above 2.67 periods. Which zones are worth collecting from therefore changes with the realised turnaround times, and a fraction that is fixed across scenarios cannot follow it. Section 5.2 now says this.
6. *What has not been done.* Only Table 9 was re-run. The algorithm comparison, the ablation, the out-of-sample and robustness analyses, the scalability study and the two case studies still use the collection rule. Section 5.2 and the Conclusion say so, and say that those fronts should be read as fronts of the restricted policy class. [To be replaced if the full study is re-run with the recourse decoder.]
7. *One caveat on the reference.* The exact model used for validation fixes the service centre of each zone by round-trip cost, as the decoder does. The reference is therefore exact for the model with cost-based repair routing. Section 5.2 states this.

## Other change

The bibliography now uses natbib with the plainnat style. Two references on decision rules were added (Ben-Tal et al., 2004 and Kuhn et al., 2011).

## Comment 3. Out-of-sample gap and distance to the best attainable plan

**Comment (summary).** Sections 5.2 and 5.5 answer different questions, and the paper lets the out-of-sample gap of 3.1 per cent stand in for the distance to the best attainable plan. The reviewer asks that both be reported and kept distinct.

**Response.** We agree. The out-of-sample gap compares one plan with itself on two scenario samples. It says whether fifty scenarios are enough. It says nothing about how far the plan is from the exact front. Section 5.5 had also called this quantity a "relative optimality gap", which was the wrong name.

**Changes made.**

1. *Section 5.5.* The quantity is now called the out-of-sample gap, and the text states that it is not an optimality gap. The conclusion drawn from it is limited to the sufficiency of the scenario sample.
2. *New Table 17.* For the knee-point plans of instances 1 to 3, where the exact front is available, the table reports both gaps side by side for the same plans.

| Instance | Decoder | Out-of-sample gap, Z1 | Out-of-sample gap, Z2 | Cost excess over exact front | Penalty excess over exact front |
|---|---|---|---|---|---|
| 1 | Collection rule | -0.23% | -0.65% | 41.2% | 2.95% |
| 1 | Recourse in decoder | -0.50% | -0.31% | -0.17% | -0.07% |
| 2 | Collection rule | 0.18% | 0.83% | 28.1% | 3.24% |
| 2 | Recourse in decoder | -1.42% | 0.05% | -0.06% | -0.10% |
| 3 | Collection rule | -0.49% | 1.09% | 135.1% | 9.64% |
| 3 | Recourse in decoder | -7.72% | 0.75% | -0.04% | -0.09% |

The cost excess is the relative amount by which the cost of the plan exceeds the lowest cost on the exact front at the same or a better service level. The plans of the collection rule generalise within about 1 per cent and are nevertheless 28 to 135 per cent more costly than exact plans with the same service. This is the reviewer's point in numbers. With the recourse solved in the decoder the gap to the exact front is zero up to the resolution of the reference grid.

3. *Instances 4, 8 and 12.* The text now says that only the out-of-sample gap is available for these plans and that their distance to the exact front is unknown.
4. *Managerial Implications and Conclusion.* The 3.1 per cent figure is now described as stability across scenario samples, with a sentence stating that it does not measure closeness to the best attainable plan and a pointer to Table 17. The heading "Knee-point optimality" in Section 5.10 is now "Knee-point behaviour".

## Comment 4. The comparator in the case studies (Section 5.10)

**Comment (summary).** The savings of 38.0 and 67.5 per cent rest on a nearest-facility heuristic described as current industry practice. The reviewer asks for a source or for the percentages to be dropped and the knee-point plans presented on their own terms.

**Response.** We have no source establishing that providers in Delhi or London operate in that way, so we have withdrawn the percentages and the description of the comparator as current practice. In checking the numbers we also found that the comparison was unsound for a second reason, and that two other statements in the section were wrong. We set these out in full.

1. *The comparison was not made at equal service.* The comparator assigns each zone to its nearest hub and tenders all returns. It collects 83.3 per cent of the returns in Delhi and 87.2 per cent in London. The knee-point plans collect 51.8 and 28.5 per cent. The lower cost of the knee-point plans was the cost of the volume they do not collect, paid for through a higher penalty. At the same service level the difference disappears. In Delhi the cheapest front point with no higher penalty costs 3151.61 against 3180.93. In London no front point dominates the comparator.
2. *The collection rate was misreported.* The earlier text said the knee-point plans kept the collection rate in the 81 to 86 per cent band. The recomputed rates are 51.8 per cent (Delhi) and 28.5 per cent (London). The earlier figure is close to the rate of the comparator, not of the knee-point plans.
3. *The flow diagrams showed the wrong plan.* The plotting script selected the plan with an index into the sorted front applied to the unsorted population. Figures 5 and 6 (right panels) have been redrawn with the knee-point plan itself. The fronts and the knee-point objective values are unchanged.

**Changes made.**

1. Section 5.10 now describes each knee-point plan by its cost, penalty, collection rate, the districts it serves and the utilisation of each hub. It says plainly that both are low-collection plans and that the London plan would not be acceptable under collection targets.
2. Table 22 now reports mean returns, mean capacity, knee-point cost and penalty and the collection rate. The baseline column has been removed.
3. The cross-city findings state that the earlier savings figures are withdrawn and why, and that the case studies show the framework runs on these networks, not that it improves on current practice.
4. The comparator is still marked on the two front plots, labelled as what it is (nearest-hub assignment, all returns tendered), because it shows where a full-collection plan sits on the trade-off.
5. Managerial Implications: the first message ("Proximity routing is the costliest habit") has been replaced by one the results support, namely that the collection level drives cost and that plans should be compared on cost only at equal service. The Conclusion no longer cites the savings.

## Comment 5. Whether exact solution is intractable, and the solver budget

**Comment (summary).** The argument for a metaheuristic rests on exact solution being intractable, and this is not established. A limit of 180 seconds per subproblem inside an overall limit of 3,600 seconds cannot complete a 21-point grid, since 21 x 180 = 3,780. The reviewer asks for a proper budget, preferably with a commercial solver, for incumbents and bounds in place of N/A so that the comparison continues past instance 3, and points out that Section 5.2 opened by saying instances 1 to 5 were solved exactly, which contradicted Table 9.

**Response.** The reviewer is right on each point, and the re-run shows that the intractability claim was false at benchmark size. We have withdrawn it.

1. *What was wrong in the earlier version.* The two limits were inconsistent, as the reviewer computed. More seriously, the exact method had been run on instances 1 to 3 only. The entries for instances 4 and 5 ("more than 3,600 s", N/A) were not the outcome of a run. Instance 4 solves in 40 seconds. We apologise for this.
2. *New budget.* The minimum-penalty problem is solved first, then cost is minimised at 21 equally spaced bounds on Z2. Each subproblem has 300 seconds on one thread and a relative gap tolerance of 1e-4. There is no overall cutoff, so the grid always completes. Where a subproblem stops at its limit we report the incumbent and the bound.
3. *Solver.* We do not have access to Gurobi or CPLEX. The runs use HiGHS through SciPy on a laptop. The manuscript says so, and says that a commercial solver would be expected to do better. [If a licence becomes available before resubmission, panel (b) should be repeated with it.]
4. *Table 9 now continues past instance 3.* Panel (b) covers instances 4 to 8, 10, 12 and 15.

| Instance | Zones, centres | Variables | Points proved optimal | Largest gap elsewhere | Gap of min-penalty problem | Exact time (s) | Collection rule, HV ratio | Recourse in decoder, HV ratio | E-NSGA-II time (s), rule / recourse |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 12, 5 | 3,060 | 21/21 | none | optimal | 40 | 0.847 | 1.013 | 3.9 / 1.9 |
| 5 | 15, 6 | 4,590 | 21/21 | none | optimal | 495 | 0.797 | 1.010 | 4.1 / 2.3 |
| 6 | 18, 6 | 5,508 | 20/21 | one point infeasible in time | 1.39% | 705 | 0.806 | 1.012 | 4.3 / 2.5 |
| 7 | 21, 8 | 8,568 | 20/21 | one point infeasible in time | 2.93% | 888 | 0.807 | 1.010 | 4.7 / 2.9 |
| 8 | 24, 9 | 11,016 | 20/21 | one point infeasible in time | 3.19% | 892 | 0.762 | 1.009 | 5.2 / 3.4 |
| 10 | 30, 10 | 15,300 | 19/21 | 0.28% | 2.13% | 1,363 | 0.663 | 1.008 | 5.9 / 6.2 |
| 12 | 36, 12 | 22,032 | 18/21 | 0.26% | 1.47% | 2,164 | 0.557 | 1.008 | 6.5 / 6.9 |
| 15 | 100, 32 | 163,200 | 7/21 | 4.03% | incumbent, no useful bound | 6,691 | 0.284 | 1.003 | 30.0 / 8.2 |

5. *What the runs show about tractability.* Exact solution is not out of reach on this benchmark. Every grid point is proved optimal up to instance 5, and all but one to three points up to instance 12, in at most 37 minutes. The hard part is the high-service end of the front. On instance 15 the solver still returns a front with gaps of about 4 per cent at most, in a little under two hours. The manuscript no longer argues from intractability. Sections 3.3, 4, 5.2 and the Conclusion now say that the case for the evolutionary method is run time (seconds per run against 40 seconds to two hours).
6. *What the runs show about the algorithm.* The shortfall of the collection rule grows with size, from 85 per cent of the reference hypervolume on instance 4 to 56 per cent on instance 12 and 28 per cent on instance 15. With the recourse solved in the decoder the same algorithm stays at the level of the reference on every instance in 2 to 8 seconds. On instance 15 this is the one place where the evolutionary method has a clear practical advantage: 8 seconds against close to two hours for a front of comparable quality.
7. *Ratios above one.* The ratios of the recourse variant in panel (b) are slightly above one because the reference is a staircase through at most 21 incumbents, which understates the exact hypervolume by 1 to 2 per cent, and because the tightest grid point is missing on instances 6 to 12. The text states that this does not mean the algorithm improves on the exact front.
8. *The contradiction in Section 5.2.* The opening of Section 5.2 has been rewritten. It states the budget, lists the two withdrawn statements, and agrees with Table 9.

**Limits of what was done.** Instances 9, 11, 13 and 14 were not run. On instance 15 the recourse variant was run with 10 seeds, not 30, and the rule columns use the stored scalability runs. HiGHS checks its time limit between node solves, so on instance 15 single subproblems ran past 300 seconds, in one case to 1,544 seconds. The total time in the table includes this.

## Comment 8. The sorting fractions

**Comment (summary).** The sorting fractions are parameters and not decisions, and this only becomes apparent in the decoder. The reviewer asks that it be stated as a modelling assumption in Section 3.

**Response.** Done. The assumption in Section 3.1 now states that the three fractions are parameters of the model and not decisions, that they describe the condition of the returned products as revealed by inspection, and that the provider chooses how much to collect and where each stream is sent but not how the collected volume divides among the retailing, repair and recycling channels. It also states that the fractions are deterministic, the same at every collection centre and in every scenario, and sum to one.

## Comment 9. Exact solution on the smallest instances

**Comment (summary).** On instance 1 the exact method takes 0.27 seconds against 3.14 for E-NSGA-II and returns a certified front. The run-time claim holds from instance 3 onwards and should be qualified below that.

**Response.** We agree. The sentence the reviewer quotes was removed when Section 5.2 was rewritten for Comment 5, and the run-time claim that replaced it is now qualified in four places.

1. *Section 5.2.* The text gives both times for instances 1 to 3 (exact grid 0.27, 7.77 and 25.43 seconds, one run of E-NSGA-II 3.14, 3.30 and 3.67 seconds). It states that on instance 1 the exact method is more than ten times faster and certified, that on instance 2 the two are of the same order, that there is no reason to prefer the evolutionary method on these two instances, and that its advantage begins at instance 3.
2. *Section 5.2, panel (b) discussion.* The comparison of run times is now stated "from instance 4 onwards".
3. *Section 4, opening.* The run-time argument is limited to instance 3 onwards.
4. *Conclusion.* It now says that on the two smallest instances the exact method is as fast or faster and should be preferred.

## Comment 10. Reference front for IGD+ and the epsilon-indicator

**Comment (summary).** The reviewer asks how the reference front is constructed on instances 4 and above, where no exact front exists.

**Response.** A new paragraph, "Reference front", has been added to Section 5.1. The reference front of an instance is the set of non-dominated points of the pooled final fronts of the four algorithms over the 30 replications (120 fronts). The same construction is used on every instance of the comparison, including instances 1 to 3. The exact fronts are used only in Table 9. We recomputed the indicators from the stored runs with this construction for instances 1 to 12 and obtained the printed values.

The paragraph also states two consequences. IGD+ and the epsilon-indicator in Tables 11 and 13 measure the standing of each algorithm relative to the others and not its distance to the Pareto front of the model. Since all four algorithms use the collection rule, the pooled front lies inside that policy class and is itself short of the exact front by the amounts in Table 9.

[Open point before sending: the recomputation reproduces the printed values with the E-NSGA-II and SMS-EMOA columns exchanged. This must be resolved first.]

**Addition to Comment 10 (instances 13 to 15).** The "Reference front" paragraph now also covers the large instances. The pool is formed in the same way from the 120 fronts of the scalability runs, and the hypervolume reference point is 1.1 times the largest value of each objective in that pool. We recomputed the four indicators of the scalability table from the stored runs with this construction and obtained the printed values. [Same open point: the values match with the E-NSGA-II and SMS-EMOA rows exchanged.]

## The three references suggested in the second report

**Comment (summary).** The three references appended to the second referee report (Felis Catus Optimization, no-wait open-shop scheduling, resource-constrained project scheduling) are not required. They have no bearing on the problem, and the sentence introducing the first of them sits oddly in Section 4.

**Response.** All three have been removed from the text and from the bibliography.

1. *Section 2.* The sentence on problem-specific adaptation and benchmarking across algorithm families is kept, without the two citations and without the clause on new algorithms.
2. *Section 4.* The three sentences introducing the Felis Catus Optimization algorithm, the scheduling review and the project scheduling paper have been deleted. The paragraph now goes directly from the comparison of NSGA-II with its successors to where the problem-specific knowledge is invested.
3. No other reference was added in their place.
