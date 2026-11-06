"""Verify the regenerated results.

* checks the pinned source file and the analysis sample
* compares every regenerated CSV in results/ with expected_results/ (1e-10 tolerance)
* cross-checks the hand-written primary regressions, clustered covariance and
  Holm adjustment against statsmodels
* compares 24 vectorized bootstrap t statistics per arm with separate statsmodels fits
* checks each TOST decision against the 90 percent interval rule

Writes results/verification_results.json. Run after analysis.py and extended_analysis.py.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
EXPECTED = ROOT / 'expected_results'
SOURCE = ROOT / 'research_data/main_regressions/final_data.csv'
SOURCE_SHA256 = '430055b4633eaaf87879bb38eb4b1c9425b91e1d7f529029dc1883e33d91df0a'
REQUIRED = ['Part2Tot', 'Part3Tot', 'GPTBase', 'GPTTutor', 'gpa_prev', 'teacher', 'Session', 'Grader', 'Year', 'Class']

checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)
    print('PASS:', name)


def main():
    check('Pinned source checksum', hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256)
    raw = pd.read_csv(SOURCE)
    d = raw[raw.Honors.eq(0)].dropna(subset=REQUIRED).copy()
    check('Analysis sample is 2848 records, 825 students, 44 classes',
          (len(d), d['Student ID'].nunique(), d.Class.nunique()) == (2848, 825, 44))
    check('Student session pairs are unique', not raw.duplicated(['Student ID', 'Session']).any())
    check('Scores are in zero to one range',
          d[['Part2Tot', 'Part3Tot']].ge(0).all().all() and d[['Part2Tot', 'Part3Tot']].le(1).all().all())
    check('One arm per classroom and no overlapping treatments',
          d.groupby('Class')[['GPTBase', 'GPTTutor']].nunique().eq(1).all().all()
          and d[['GPTBase', 'GPTTutor']].sum(axis=1).le(1).all())

    for file in sorted(EXPECTED.glob('*.csv')):
        pd.testing.assert_frame_equal(pd.read_csv(file), pd.read_csv(OUT / file.name),
                                      check_exact=False, atol=1e-10, rtol=1e-10)
        check('Regenerated CSV matches supplied result: ' + file.name, True)

    # Cross-check the manual primary implementation with the library estimator.
    x = pd.concat([d[['GPTBase', 'GPTTutor', 'gpa_prev']].astype(float),
                   pd.get_dummies(d[['teacher', 'Session', 'Grader', 'Year']].astype(str), drop_first=True, dtype=float)],
                  axis=1)
    x = sm.add_constant(x)
    check('Full design has 35 columns and full rank', x.shape[1] == np.linalg.matrix_rank(x.to_numpy()) == 35)
    primary = pd.read_csv(OUT / 'regression_results.csv')
    main_rows = primary[primary.model.eq('Primary')]
    pvalues = []
    for outcome in ['Part2Tot', 'Part3Tot']:
        fit = sm.OLS(d[outcome], x).fit(cov_type='cluster', cov_kwds={'groups': d.Class}, use_t=True)
        for arm, contrast in [('GPTBase', 'Base vs control'), ('GPTTutor', 'Tutor vs control')]:
            r = main_rows[(main_rows.outcome.eq(outcome)) & (main_rows.contrast.eq(contrast))].iloc[0]
            check(outcome + ' ' + arm + ' coefficient, clustered SE, CI and p agree with statsmodels',
                  np.allclose([r.b, r.se, r.low, r.high, r.p],
                              [fit.params[arm], fit.bse[arm], *fit.conf_int().loc[arm], fit.pvalues[arm]],
                              atol=1e-10, rtol=1e-10))
            pvalues.append(fit.pvalues[arm])
    holm = multipletests(pvalues, method='holm')[1]
    check('Holm correction agrees with library implementation',
          np.allclose(holm, main_rows[main_rows.contrast.ne('Tutor vs Base')].p_holm_primary, atol=1e-12))

    # Independently refit 24 bootstrap outcomes per tested coefficient using statsmodels.
    xx = x.to_numpy()
    y = d.Part3Tot.to_numpy()
    groups = pd.factorize(d.Class)[0]
    g = groups.max() + 1
    n, k = xx.shape
    bread = np.linalg.inv(xx.T @ xx)
    factor = g / (g - 1) * (n - 1) / (n - k)
    rng = np.random.default_rng(321)
    for j in [1, 2]:
        xr = np.delete(xx, j, axis=1)
        fitted = xr @ np.linalg.lstsq(xr, y, rcond=None)[0]
        u = y - fitted
        w = rng.choice([-1., 1.], size=(g, 24))
        yb = fitted[:, None] + u[:, None] * w[groups]
        bet = bread @ xx.T @ yb
        res = yb - xx @ bet
        sc = np.zeros((g, 24))
        np.add.at(sc, groups, (xx @ bread[:, j])[:, None] * res)
        se = np.sqrt(factor * np.sum(sc ** 2, axis=0))
        fast = bet[j] / se
        slow = []
        for i in range(24):
            ref = sm.OLS(yb[:, i], xx).fit(cov_type='cluster', cov_kwds={'groups': groups}, use_t=True)
            slow.append(ref.params[j] / ref.bse[j])
        check('Vectorized bootstrap t agrees with 24 separate statsmodels fits for ' + x.columns[j],
              np.allclose(fast, slow, atol=1e-9, rtol=1e-9))

    # Check each equivalence result against the equivalent 90 percent interval rule.
    eq = pd.read_csv(OUT / 'equivalence_sensitivity.csv')
    check('TOST decisions match 90 percent interval criterion',
          all((r.p_TOST < .05) == (r.low90 > -r.margin and r.high90 < r.margin) for r in eq.itertuples()))

    report = {'status': 'passed', 'checks': checks,
              'scope': 'Python data download, main and extended analyses; independent primary and bootstrap '
                       'numerical cross-checks. Not proof of scientific validity or universal portability.'}
    (OUT / 'verification_results.json').write_text(json.dumps(report, indent=2))
    print('All', len(checks), 'checks passed.')


if __name__ == '__main__':
    main()
