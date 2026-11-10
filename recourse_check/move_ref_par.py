import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = s.index("\n\n\\paragraph{Reference front.}")
b = s.index(" Together the four indicators cover", a)
par = s[a:b].rstrip("\n")
par = par.replace(" The same construction is used on every instance, including instances 1--3, and for the large instances of Section~\\ref{scalability-analysis}.",
                  " The same construction is used on every instance of the comparison, including instances 1--3.")
assert "scalability-analysis" not in par
s = s[:a] + s[b:]
end = "and the Friedman test across all instances."
assert s.count(end) == 1
s = s.replace(end, end + par)
open(p, "w", encoding="utf-8").write(s)
