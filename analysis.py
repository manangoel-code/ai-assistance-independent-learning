"""Primary reproduction and sample sensitivity analyses.

Reproduces the primary regressions of Bastani et al. (2025) on the complete-case,
non-honors sample, with classroom-clustered standard errors (finite-sample
correction, t-based inference with G - 1 degrees of freedom) and Holm adjustment
across the four primary treatment-versus-control tests. Two sample sensitivity
analyses (including honors classes; excluding reported noncompliance) follow.

Outputs in results/: regression_results.csv, descriptive_statistics.csv,
sample_audit.json, adjusted_effects.png (Figure 2), observed_means.png (Figure 1).

Run download_data.py first.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DATA = ROOT / 'research_data/main_regressions/final_data.csv'

REQUIRED = ['Part2Tot', 'Part3Tot', 'GPTBase', 'GPTTutor', 'gpa_prev', 'teacher', 'Session', 'Grader', 'Year', 'Class']
FIXED_EFFECTS = ['teacher', 'Session', 'Grader', 'Year']
OUTCOMES = ['Part2Tot', 'Part3Tot']          # practice, unassisted exam
CONTRASTS = [('Base vs control', [0, 1, 0]), ('Tutor vs control', [0, 0, 1]), ('Tutor vs Base', [0, -1, 1])]
PUBLISHED_PRIMARY = [.137, .361, -.054, -.004]  # Bastani et al. (2025), three decimals

# Class-session combinations reported as noncompliant in the original study.
NONCOMPLIANT = [('11S', [2, 4]), ('10A', [2]), ('9J', [1]), ('10B', [3])]


def fit(frame, outcome):
    """OLS with classroom-clustered covariance; returns one dict per contrast."""
    d = frame.dropna(subset=REQUIRED).copy()
    X = pd.concat([pd.Series(1., index=d.index, name='Intercept'),
                   d[['GPTBase', 'GPTTutor', 'gpa_prev']],
                   pd.get_dummies(d[FIXED_EFFECTS].astype(str), drop_first=True, dtype=float)], axis=1)
    x = X.to_numpy(float)
    y = d[outcome].to_numpy(float)
    rank = np.linalg.matrix_rank(x)
    assert rank == x.shape[1]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    scores = pd.DataFrame(x * resid[:, None]).groupby(d.Class.to_numpy()).sum().to_numpy()
    g, n, k = len(scores), len(y), rank
    cov = bread @ (scores.T @ scores) @ bread * g / (g - 1) * (n - 1) / (n - k)
    result = []
    for label, c in CONTRASTS:
        v = np.zeros(k)
        v[:3] = c
        b = float(v @ beta)
        se = float(np.sqrt(v @ cov @ v))
        crit = t.ppf(.975, g - 1)
        result.append(dict(outcome=outcome, contrast=label, b=b, se=se, low=b - crit * se, high=b + crit * se,
                           p=float(2 * t.sf(abs(b / se), g - 1)), n=n, clusters=g, rank=k,
                           r2=1 - sum(resid ** 2) / sum((y - y.mean()) ** 2)))
    return result


def holm(res):
    """Holm step-down adjustment over the four primary treatment-versus-control tests."""
    res['p_holm_primary'] = np.nan
    ix = res.index[(res.model == 'Primary') & res.contrast.ne('Tutor vs Base')]
    ordered = sorted(ix, key=lambda i: res.loc[i, 'p'])
    running = 0.
    for j, i in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - j) * res.loc[i, 'p']))
        res.loc[i, 'p_holm_primary'] = running
    return ix


def plot_adjusted_effects(res):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, outcome, title in zip(axs, OUTCOMES, ['Assisted practice', 'Subsequent unassisted exam']):
        z = res[(res.model == 'Primary') & (res.outcome == outcome) & res.contrast.ne('Tutor vs Base')]
        for y, (_, r) in enumerate(z.iterrows()):
            ax.errorbar(r.b * 100, y, xerr=[[100 * (r.b - r.low)], [100 * (r.high - r.b)]], fmt='o', capsize=4,
                        color=['#ac502b', '#186e7c'][y], markersize=7)
        ax.axvline(0, color='#7d8388', lw=1, ls='--')
        ax.set_yticks([0, 1], ['Standard AI', 'Safeguarded tutor'])
        ax.set_title(title)
        ax.set_xlabel('Adjusted difference vs control (percentage points)')
        ax.set_ylim(-.6, 1.6)
        ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'adjusted_effects.png', dpi=220)
    plt.close(fig)


def plot_observed_means(desc):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = ['Control', 'Standard AI', 'Safeguarded tutor']
    keys = ['control', 'vanilla', 'augmented']
    pos = np.arange(3)
    ax.bar(pos - .18, desc.loc[keys, 'practice_mean'] * 100, .36, label='Practice', color='#186e7c')
    ax.bar(pos + .18, desc.loc[keys, 'exam_mean'] * 100, .36, label='Unassisted exam', color='#d7a46b')
    ax.set_xticks(pos, labels)
    ax.set_ylabel('Mean score (%)')
    ax.set_ylim(0, 80)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    fig.savefig(OUT / 'observed_means.png', dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    sample = raw.loc[raw.Honors.eq(0)].copy()
    complete = sample.dropna(subset=REQUIRED).copy()

    noncompliant = np.zeros(len(sample), dtype=bool)
    for cls, sessions in NONCOMPLIANT:
        noncompliant |= (sample.Class.eq(cls) & sample.Session.isin(sessions)).to_numpy()
    samples = [('Primary', complete), ('Including honors', raw),
               ('Exclude reported noncompliance', sample.loc[~noncompliant])]

    rows = []
    for name, d in samples:
        for outcome in OUTCOMES:
            for row in fit(d, outcome):
                row['model'] = name
                rows.append(row)
    res = pd.DataFrame(rows)
    ix = holm(res)
    assert np.allclose(res.loc[ix, 'b'], PUBLISHED_PRIMARY, atol=.0005)
    res.to_csv(OUT / 'regression_results.csv', index=False)

    desc = complete.groupby('Treatment arm').agg(
        observations=('Part2Tot', 'size'), students=('Student ID', 'nunique'), classes=('Class', 'nunique'),
        practice_mean=('Part2Tot', 'mean'), practice_sd=('Part2Tot', 'std'),
        exam_mean=('Part3Tot', 'mean'), exam_sd=('Part3Tot', 'std'))
    desc.to_csv(OUT / 'descriptive_statistics.csv')

    audit = dict(
        raw_rows=len(raw), raw_students=raw['Student ID'].nunique(), raw_classes=raw.Class.nunique(),
        nonhonors_rows=len(sample), nonhonors_students=sample['Student ID'].nunique(),
        nonhonors_classes=sample.Class.nunique(),
        complete_rows=len(complete), complete_students=complete['Student ID'].nunique(),
        complete_classes=complete.Class.nunique(),
        excluded_missing=len(sample) - len(complete),
        missing_by_variable=sample[REQUIRED].isna().sum().to_dict(),
        duplicates=int(raw.duplicated(['Student ID', 'Session']).sum()),
        sha256=hashlib.sha256(DATA.read_bytes()).hexdigest())
    (OUT / 'sample_audit.json').write_text(json.dumps(audit, indent=2))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    plot_adjusted_effects(res)
    plot_observed_means(desc)

    print(json.dumps(audit, indent=2))
    print(desc.to_string())
    print(res.to_string(index=False))


if __name__ == '__main__':
    main()
