# Revision notes: research-paper rewrite

Deliverables in this folder: `main.tex` (source), `main.pdf` (compiled, 45 pages), `references.bib`, `figures/`, `tables/`,
`results.tex`, `code_appendix.tex` + `code/`. Everything numeric in `main.tex` comes from `results.tex` / `tables/*.tex`, which
`python -m headway.make_report_assets` writes from the CSV files in `outputs/` (not tracked by git; regenerate with the commands below).

## Rebuild

```
python run.py --dataset all --likelihood truncated --period 1 --workers 6 --out outputs/truncated   # corrected likelihood
python run.py --dataset all --likelihood paper     --period 1 --workers 6 --out outputs/paper       # published method
python -m headway.detector --highd data/highd --ngsim data/ngsim/export_ngsim_data.csv --workers 6 --out outputs/detector
python -m pytest -q tests > logs/pytest_out.txt                                                      # 52 passed
python -m headway.make_report_assets && python -m headway.make_code_appendix --out report
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
(`--workers 6`: the default of one worker per core exhausted virtual memory on the 32 GB Windows machine.)

## Numerical and mathematical corrections

| # | Issue found | Correction | Effect |
|---|---|---|---|
| 1 | Likelihood did not condition on the [0.5, 25] s window (published method; baselines penalised for mass < 0.5 s) | Re-estimated all 14 fits with the window-conditioned likelihood (already implemented as `--likelihood truncated`); kept the paper likelihood as a reproduction run | highD: shifted log-normal becomes best by KL (0.0026), proposed 2nd (0.0036); baseline KLs fall 2.4-7.4x. NGSIM unchanged (Burr best, proposed 4th). The published "proposed is best on highD" holds only under the published likelihood. |
| 2 | highD shifted exponential R-hat = 4.92 (earlier run) | Re-ran with the log-gap reparametrisation: R-hat = 1.005 (paper lik.); under the truncated likelihood its threshold is not identifiable (posterior = prior) and is reported as such | All 28 posteriors have split R-hat <= 1.006 |
| 3 | Markov results and within-trajectory autocorrelation were missing / used cross-vehicle pairs | Ran the within-run estimator on both datasets | Lag-1 autocorrelation 0.906 (highD), 0.727 (NGSIM) instead of 0.839 (old, file-order). pi vs observed fractions: TV 0.014 / 0.008, not 0.000. Chapman-Kolmogorov and first-vs-second-order results are new. |
| 4 | Renewal model equated 1 Hz time headways with detector inter-arrival times | Reconstructed detector crossing times from both trajectory sets (`headway/detector.py`, with tests) and tested the renewal model on them | highD detector gaps nearly independent (lag-1 0.031) but one-minute count dispersion is 0.96 (near Poisson) vs 0.61 from the headway-driven model; NGSIM 0.52 vs 0.46. Headway-driven rates (1,627 / 1,167 veh/h/lane) differ from detector flows (1,168 / 1,364). |
| 5 | Two Wasserstein readings conflated | Report W_vec (sorted bin probabilities, unitless, matches the published scale) and W_time (conventional, seconds) separately; they rank models differently | Stated in Sections 4.6 and 7.2.3 |
| 6 | Closed-form inverse sampler (Eq. 5.4) described as the implemented sampler | Stated as a derived identity only; the code inverts the window-conditioned CDF on a grid | Text correction |
| 7 | Markov state-probability figure labelled steps "successive vehicles", 20 steps | Steps are seconds along one trajectory; 60 steps | Figure 5 |
| 8 | Hard-coded numbers in the old `results.tex` | All numbers generated from outputs; old run-time/ESS/renewal-error macros replaced | e.g. max variance error is 7.0 % (NGSIM, t = 30 s), not 11.9 % |

## Changes to the paper
Restructured to the requested outline (Introduction ... Conclusion); expanded literature review with sub-sections; new abstract (268 words);
every table placed beside its discussion; the appendix contains only code extracted by `inspect.getsource`; Computational Implementation is
Section 6.12; author block and affiliation as specified (no e-mail addresses); two verified references added (Gelman & Rubin 1992; Treiber & Kesting 2013).

## Not resolved / limitations
* The published NGSIM sample is not reproducible; period 1 of the Data.gov export was chosen by comparing the mean headway implied by the published parameters (2.77-3.04 s for five distributions) with the two periods (3.08 s, 4.04 s). This is an inference.
* The published KS p-values (near 1) and the published W_vec for gamma on NGSIM (0.0173) are not reproduced.
* The binned multinomial likelihood treats the highly autocorrelated 1 Hz samples as independent, so credible intervals and KS p-values are too narrow.
* Detector validation uses one detector position per recording (highD) or one position and one 15-minute period (NGSIM, 6 lane-series, 2,046 gaps).
* Single seed (42); the code was not run with other seeds.
* Bibliographic details of Leungbootnak et al. are as in the supplied manuscript (no year or venue). Some older references (Greenberg 1966, Dawson & Chimini 1968, Ye & Zhang 2009, Mahmassani 2016, Wen et al. 2022) carry the details given in that manuscript's reference list and were not independently re-verified.
* `README.md` and `docs/*.md` predate this revision; the README has been updated, the `docs/` files are historical.
