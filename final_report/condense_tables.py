"""Merge/trim the generated table bodies of report/tables into final_report/tables (values are copied, never retyped).

Run from the project root after `python -m headway.make_report_assets`:  python final_report/condense_tables.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "tables"
DST = ROOT / "final_report" / "tables"
DST.mkdir(parents=True, exist_ok=True)


def rows(name):
    """Rows of a generated body as lists of cells, plus whether a [0.3ex] group gap followed the row."""
    out = []
    for line in (SRC / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        gap = "[0.3ex]" in line
        cells = [c.strip() for c in re.sub(r"\\\\(\[[^\]]*\])?\s*$", "", line.rstrip()).split("&")]
        out.append((cells, gap))
    return out


def emit(name, lines):
    # all rows end in \\ except the last one: main.tex adds that terminator before \bottomrule
    (DST / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", name, len(lines), "lines")


# ---- parameters: highD and NGSIM side by side --------------------------------------------------------------
hd, ng = rows("params_hd.tex"), rows("params_ng.tex")
assert len(hd) == len(ng)
lines, dist = [], ""
for (a, gap), (b, _) in zip(hd, ng):
    assert a[1] == b[1] and a[0] == b[0]
    d = a[0] if a[0] != dist else ""
    dist = a[0] or dist
    sep = r" \\[0.3ex]" if gap else r" \\"
    lines.append(f"{d} & {a[1]} & " + " & ".join(a[2:5]) + " & " + " & ".join(b[2:5]) + sep)
lines[-1] = lines[-1].rsplit(" \\\\", 1)[0]
emit("params.tex", lines)

# ---- goodness of fit: KL, W_vec, W_time, KS D1 -------------------------------------------------------------
kl, wv, wt, ts = rows("kl.tex"), rows("wvec.tex"), rows("wtime.tex"), rows("tests.tex")
assert len(kl) == len(wv) == len(wt) == len(ts) == 7
lines = [r"\multicolumn{9}{@{}l}{\textit{highD}} \\"]
for k in range(7):
    lines.append(f"{kl[k][0][0]} & " + " & ".join(kl[k][0][1:4]) + " & " + " & ".join(wv[k][0][1:4]) + f" & {wt[k][0][1]} & {ts[k][0][1]} \\\\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{9}{@{}l}{\textit{NGSIM}} \\")
for k in range(7):
    lines.append(f"{kl[k][0][0]} & " + " & ".join(kl[k][0][4:7]) + " & " + " & ".join(wv[k][0][4:7]) + f" & {wt[k][0][3]} & {ts[k][0][5]} \\\\")
lines[-1] = lines[-1].rsplit(" \\\\", 1)[0]
emit("gof.tex", lines)

# statements about the tests that the text makes (read from the generated table, checked here)
pchi = [r[0][4] for r in ts] + [r[0][8] for r in ts]
assert sum("<\\!10^{-300}" in p for p in pchi) == 13 and any("2.5" in p and "-49" in p for p in pchi), pchi
p2_ng_burr = ts[5][0][7]
assert "6.8" in p2_ng_burr and "-4" in p2_ng_burr, p2_ng_burr

# ---- Markov validation without the rows given elsewhere ------------------------------------------------------
mv = [r for r in rows("markov_val.tex")
      if not r[0][0].startswith(("Pairs", "Chapman--Kolmogorov max error, $n=3$", "Second-order test $p$"))]
lines = [" & ".join(c) + r" \\" for c, _ in mv]
lines[-1] = lines[-1].rsplit(" \\\\", 1)[0]
emit("markov_val.tex", lines)

# ---- unchanged tables ---------------------------------------------------------------------------------------
for n in ("renewal", "markov", "diag", "autocorr", "detector_counts"):
    shutil.copy(SRC / f"{n}.tex", DST / f"{n}.tex")
shutil.copy(ROOT / "report" / "results.tex", ROOT / "final_report" / "results.tex")
print("copied renewal, markov, diag, autocorr, detector_counts, results.tex")
