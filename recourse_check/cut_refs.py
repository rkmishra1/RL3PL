"""Remove the three unrelated references and add the reference-front statement for instances 13 to 15."""
import sys, re
tex, bib = sys.argv[1], sys.argv[2]
s = open(tex, encoding="utf-8").read()


def span(start, end, new):
    global s
    assert s.count(start) == 1, start
    a = s.index(start)
    b = s.index(end, a) + len(end)
    s = s[:a] + new + s[b:]


span("Since no algorithm dominates across problem classes \\citep{mirmozaffari2026minimising",
     "new algorithms continue to appear \\citep{salehi2026felis}.",
     "Since no algorithm can be expected to dominate across problem classes, problem-specific adaptation and "
     "benchmarking across algorithm families are needed.")
span(" At the same time, a stream of novel nature-inspired metaheuristics continues to appear.",
     "project scheduling\\citep{mohammadzadeh2025biobjective}.", "")
old = "Rather than adopting novelty for its own sake, we therefore invest"
assert s.count(old) == 1
s = s.replace(old, "We therefore invest")
old = " The same construction is used on every instance of the comparison, including instances 1--3."
assert s.count(old) == 1
s = s.replace(old, " The same construction is used on every instance, including instances 1--3. For the large "
              "instances 13--15 of Section~\\ref{scalability-analysis} the pool is formed in the same way from the "
              "120 fronts of the scalability runs, and the hypervolume reference point is 1.1 times the largest "
              "value of each objective in that pool.")
for k in ("mirmozaffari2026minimising", "salehi2026felis", "mohammadzadeh2025biobjective"):
    assert k not in s, k
open(tex, "w", encoding="utf-8").write(s)
b = open(bib, encoding="utf-8").read()
for k in ("mirmozaffari2026minimising", "salehi2026felis", "mohammadzadeh2025biobjective"):
    b, n = re.subn(r"@article\{" + k + r",.*?\n\}\n\n?", "", b, flags=re.S)
    assert n == 1, k
open(bib, "w", encoding="utf-8").write(b)
print("ok")
