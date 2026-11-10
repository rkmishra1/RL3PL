"""Edits for the minor comments on small-instance timing and on the reference front."""
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)


# Comment 9
a = "For instances 1--3 the 21-point grid needs 0.27, 7.77 and 25.43 seconds, with every point proved optimal."
rep(a, a + " One run of E-NSGA-II on the same instances takes 3.14, 3.30 and 3.67 seconds (Table~\\ref{tab:CPU}). "
    "On instance~1 the exact method is therefore more than ten times faster than the evolutionary method, and it "
    "returns a front with a certificate of optimality. On instance~2 the two run times are of the same order. On "
    "these two instances there is no reason to prefer the evolutionary method, and we make no such claim. Its "
    "run-time advantage begins at instance~3 and grows with size.")
rep("One run of E-NSGA-II takes between 2 and 30 seconds, against 40 seconds to almost two hours for the exact grid,",
    "From instance~4 onwards one run of E-NSGA-II takes between 2 and 30 seconds, against 40 seconds to almost two "
    "hours for the exact grid,")
rep("run time: seconds per run where the exact grid takes from under a minute to about two hours.",
    "run time: seconds per run where the exact grid takes from under a minute to about two hours. This holds from "
    "instance~3 onwards. On instances 1 and 2 the exact method is as fast as the evolutionary method or faster.")
rep("its run time of seconds and not the intractability of exact solution.",
    "its run time of seconds and not the intractability of exact solution, and on the two smallest instances the "
    "exact method is as fast or faster and should be preferred.")

# Comment 10
a = ("computed against a reference front formed by the non-dominated union of all fronts produced by all "
     "algorithms and replications on the instance")
rep(a, "computed against the reference front described in the next paragraph")
b = "front must be translated to weakly dominate the reference front."
rep(b, b + "\n\n\\paragraph{Reference front.} For the IGD+ and epsilon-indicator values in "
    "Tables~\\ref{tab:IGD} and~\\ref{tab:Eps} the reference front of an instance is built from the runs "
    "themselves. The final fronts of the four algorithms over the 30 replications, 120 fronts in all, are pooled "
    "and the non-dominated points of the pool are kept. The same construction is used on every instance, "
    "including instances 1--3, and for the large instances of Section~\\ref{scalability-analysis}. The exact "
    "fronts of Section~\\ref{validation-against-exact-solutions} are used only in Table~\\ref{tab:Exact}. Two "
    "consequences should be kept in mind when reading these tables. The reference is the best that the four "
    "algorithms found together, so IGD+ and the epsilon-indicator measure the standing of each algorithm relative "
    "to the others and not its distance to the Pareto front of the model. A value near zero means that an "
    "algorithm supplied most of the pooled front. Also, all four algorithms use the collection rule of "
    "Section~\\ref{sec:representation}, so the pooled front lies inside that policy class and is itself short of "
    "the exact front by the amounts shown in Table~\\ref{tab:Exact}.\n\n")
open(p, "w", encoding="utf-8").write(s)
print("ok")
