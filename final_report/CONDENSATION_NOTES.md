# Condensed paper (final_report/)

`main.pdf`: 16 pages (10 pt, A4, 2.3 cm margins), compiled with `pdflatex main && bibtex main && pdflatex main && pdflatex main`
(no undefined references or citations, no overfull boxes). The original paper in `../report/` and all outputs in `../outputs/` are unchanged.

## Rebuild
```
python final_report/condense_tables.py        # merged/trimmed tables from report/tables (values copied, never retyped)
python final_report/make_figures.py --workers 6   # two-panel figures; asserts the re-created simulations equal outputs/truncated
python final_report/make_code_excerpts.py     # 4 code excerpts from the repository source
cd final_report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What was condensed or removed
* 45 -> 16 pages; abstract 268 -> 179 words; literature review rewritten to ~0.7 page with 9 references; 42 -> 13 references.
* Sections merged to the requested 8-section outline; Mathematical Background and the three models are one section; Computational Implementation is Section 5.9.
* Tables 18 -> 10: parameter tables (2) merged into one; KL, W_vec, W_time and KS D_1 merged into one; removed the terminology, verification-level, posterior-summary, detector-gap-statistics and hypothesis-test tables (the facts they held are in the text, with detector gap statistics given as numbers).
* Figures 9 -> 6 two-panel figures (fits, renewal paths, N(60) distribution, transition diagrams, transient probabilities, posteriors); removed the workflow diagram, trace plots, and detector figure. Panels were re-rendered at panel size from the same data (not shrunk), highD|NGSIM side by side. Fit figure axis cut at 15 s (stated in the caption).
* Appendix: 12 listings -> 4 short excerpts (proposed distribution, Metropolis-Hastings core, renewal simulation, Markov estimation), lines taken verbatim from the source, omissions marked "...".
* Removed: the unimplemented closed-form inverse sampler equation, textbook-level background, long per-file description of the code.

## Preserved
All three components, all key equations, every number (generated `results.tex` identical to the original; tables copied from it), the time-headway vs detector inter-arrival distinction, the R-hat 4.92 -> 1.005 account, the likelihood-truncation finding, both Wasserstein readings, and the limitations.

## Notes
* Bibliography (13 entries): Metropolis et al. 1953 is no longer cited (Hastings 1970 only); Gelman & Rubin 1992 replaced by Vehtari et al. 2021 for split R-hat.
* Leungbootnak et al. has no year/venue in the supplied manuscript (listed as n.d.).
