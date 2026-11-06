# AI-Assisted Mathematics Practice and Subsequent Examination Performance

Analysis code and results for the paper

> **AI-Assisted Mathematics Practice and Subsequent Examination Performance: A Reproducibility and Sensitivity Analysis**
> *A reproducible secondary analysis of a classroom mathematics experiment*
> Manan Goel, October 2026. [Read the paper (PDF)](paper/AI_Assistance_and_Independent_Learning.pdf)

This is a secondary analysis of the field experiment by Bastani et al. (2025). Students at a Turkish high school
practised mathematics with no AI, a GPT-based AI assistant or a tutor with built-in learning safeguards, and then
sat an unassisted exam.
The code reproduces the original primary estimates and tests how much the conclusions change with the model
specification, single classrooms, prior attainment, session, the inference method (restricted wild cluster
bootstrap) and the equivalence margin.

The experiment and its data are by Bastani et al. This repository contains no participant data;
`download_data.py` fetches the public file from their repository.

## Repository layout

```
paper/                    the paper (PDF)
download_data.py          fetches the pinned source data and checks its SHA-256
analysis.py               primary reproduction, sample sensitivity, Holm adjustment, Figures 1-2
extended_analysis.py      specifications, classroom influence, GPA interactions, sessions,
                          wild cluster bootstrap, equivalence, Figures 3-4
verify_results.py         22 automated checks (see below)
run_all.py                runs the four Python steps above in order
analysis.R                optional R counterpart of the primary models
results/                  all outputs used in the paper (CSV, JSON, PNG)
expected_results/         reference copies of the numerical outputs used by verify_results.py
```

| Output in `results/` | Paper |
|---|---|
| `descriptive_statistics.csv`, `observed_means.png` | Table 1, Figure 1 |
| `regression_results.csv`, `adjusted_effects.png` | Tables 2-3, Figure 2 |
| `extended_specifications.csv` | Table 4 |
| `classroom_influence.csv`, `classroom_influence.png`, `wild_cluster_bootstrap.csv` | Figure 3, bootstrap inference |
| `gpa_interactions.csv`, `session_estimates.csv`, `session_effects.png` | Figure 4, prior attainment and sessions |
| `equivalence_sensitivity.csv` | Equivalence and acceptable margin |
| `sample_audit.json` | Sample construction (Appendix A) |

## Reproduce the results

You need Python 3.12 with the pinned versions in `requirements.txt`, and internet access once to download the data.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` downloads and checks the data, runs both analyses (about 20 seconds) and verifies the outputs. It
stops at the first failure. To run the steps separately, run `download_data.py`, `analysis.py`,
`extended_analysis.py` and `verify_results.py` in that order.

`verify_results.py` checks:

- the SHA-256 of the downloaded file and the analysis sample (2,848 records, 825 students, 44 classes)
- every regenerated CSV against `expected_results/` (1e-10 tolerance)
- the hand-written regressions, clustered standard errors, intervals, p values and Holm adjustment against statsmodels
- the vectorized wild cluster bootstrap against 24 separate statsmodels refits per arm
- each equivalence decision against the 90% interval rule

A rerun on another operating system or linear-algebra library can change the last digits of some estimates.
The relative differences are below 1e-12, so the regenerated files may differ from the committed ones byte for
byte while every number printed in the paper stays the same.

### Optional: R

```r
install.packages("sandwich")
```

```bash
Rscript analysis.R
```

This writes `results/r_primary_results.csv`. With R 4.6.1 it matches the Python primary estimates, standard
errors, intervals and Holm-adjusted p values to about 1e-14.

## Source data and original study

Bastani, H., Bastani, O., Sungu, A., Ge, H., Kabakcı, Ö., & Mariman, R. (2025). Generative AI without guardrails
can harm learning: Evidence from high school mathematics. *Proceedings of the National Academy of Sciences*,
122(26), e2422633122. https://doi.org/10.1073/pnas.2422633122

Data and original code: https://github.com/obastani/GenAICanHarmLearning (pinned commit
`2f63dae1a01d51453826fe07ef5cf6678e339588`, file `main_regressions/final_data.csv`). The terms of use for the
data are in that repository.

## Scope

This is a secondary analysis of existing data and has not been peer reviewed. The extended analyses are
exploratory. The checks show that the code reproduces the reported numbers; they cannot confirm the causal
assumptions or show that the findings generalize beyond this study.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite Bastani et al. (2025) for the original experiment and data.
