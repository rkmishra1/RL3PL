import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = "\\begin{tabular}{@{}llrcccrllrr@{}}"
assert s.count(a) == 1
i = s.index(a)
e = "\\end{tabular}"
j = s.index(e, i) + len(e)
s = s[:i] + "\\resizebox{\\linewidth}{!}{%\n" + s[i:j] + "}" + s[j:]
open(p, "w", encoding="utf-8").write(s)
