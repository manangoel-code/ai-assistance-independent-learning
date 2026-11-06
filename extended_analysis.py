"""Exploratory extended analyses (paper Sections "Model specification and student
weighting" through "Equivalence and acceptable margin").

* four model specifications, including equal student weighting
* 44 leave-one-classroom-out refits per outcome
* continuous prior-GPA interactions
* session-by-session models
* restricted wild cluster bootstrap (4,999 Rademacher draws) for the exam outcome
* TOST equivalence at illustrative margins of 2, 3 and 5 percentage points

Outputs in results/: extended_specifications.csv, classroom_influence.csv,
gpa_interactions.csv, session_estimates.csv, wild_cluster_bootstrap.csv,
equivalence_sensitivity.csv, classroom_influence.png (Figure 3) and
session_effects.png (Figure 4).

Run download_data.py first.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

from analysis import DATA, OUT, REQUIRED, plt  # same sample definition and plot setup as the primary analysis

OUTCOMES = ['Part2Tot', 'Part3Tot']
ARMS = ['GPTBase', 'GPTTutor']
SPECIFICATIONS = ['Unadjusted', 'Session and GPA', 'Adjusted', 'Equal student weight']
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_DRAWS = 4999
BOOTSTRAP_BATCH = 100
EQUIVALENCE_MARGINS = [.02, .03, .05]  # set retrospectively; not established educational thresholds


def design(z, gpa_mean, mode='Adjusted'):
    """Design matrix for a specification; GPA is centred on the full-sample mean."""
    x = z[ARMS].astype(float).copy()
    if mode != 'Unadjusted':
        x['gpa_prev'] = z.gpa_prev
        cats = ['Session'] if mode == 'Session and GPA' else ['teacher', 'Session', 'Grader', 'Year']
        x = pd.concat([x, pd.get_dummies(z[cats].astype(str), drop_first=True, dtype=float)], axis=1)
    if mode == 'GPA interaction':
        c = z.gpa_prev - gpa_mean
        x['Base x GPA'] = z.GPTBase * c
        x['Tutor x GPA'] = z.GPTTutor * c
    return sm.add_constant(x).astype(float)


def fit(z, y, gpa_mean, mode='Adjusted', weights=None):
    """(Weighted) least squares with classroom-clustered covariance and t inference."""
    x = design(z, gpa_mean, mode)
    w = np.ones(len(z)) if weights is None else weights
    return sm.WLS(z[y], x, weights=w).fit(cov_type='cluster', cov_kwds={'groups': z.Class}, use_t=True)


def wild_cluster_bootstrap(d, gpa_mean, rng):
    """Restricted wild cluster bootstrap t tests under each zero treatment null (exam outcome)."""
    x = design(d, gpa_mean).to_numpy()
    y = d.Part3Tot.to_numpy()
    groups = pd.factorize(d.Class)[0]
    g = groups.max() + 1
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    factor = g / (g - 1) * (n - 1) / (n - k)
    base = np.linalg.lstsq(x, y, rcond=None)[0]
    res = y - x @ base
    summ = np.zeros((g, k))
    np.add.at(summ, groups, x * res[:, None])
    cov = bread @ summ.T @ summ @ bread * factor

    boot = []
    for j, arm in [(1, 'GPTBase'), (2, 'GPTTutor')]:
        # Impose the null by dropping the tested column.
        xr = np.delete(x, j, axis=1)
        fitted = xr @ np.linalg.lstsq(xr, y, rcond=None)[0]
        u = y - fitted
        observed = base[j] / np.sqrt(cov[j, j])
        hits = 0
        for start in range(0, BOOTSTRAP_DRAWS, BOOTSTRAP_BATCH):
            b = min(BOOTSTRAP_BATCH, BOOTSTRAP_DRAWS - start)
            w = rng.choice([-1., 1.], size=(g, b))
            ys = fitted[:, None] + u[:, None] * w[groups]
            bet = bread @ x.T @ ys
            rr = ys - x @ bet
            # Influence for the tested coefficient and cluster sums for each draw.
            influence = (x @ bread[:, j])[:, None] * rr
            sc = np.zeros((g, b))
            np.add.at(sc, groups, influence)
            se = np.sqrt(factor * np.sum(sc ** 2, axis=0))
            hits += int(np.sum(np.abs(bet[j] / se) >= abs(observed)))
        boot.append(dict(arm=arm, outcome='Part3Tot', draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                         p=(hits + 1) / (BOOTSTRAP_DRAWS + 1), observed_t=observed))
    return boot


def plot_classroom_influence(loco, exam_model):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, arm, title in zip(axs, ARMS, ['Standard AI exam estimate', 'Tutor exam estimate']):
        z = loco[(loco.outcome == 'Part3Tot') & (loco.arm == arm)]
        ax.scatter(np.arange(len(z)), z.b * 100, s=15, color='#186e7c')
        ax.axhline(exam_model.params[arm] * 100, ls='--', color='#ac502b')
        ax.axhline(0, color='gray', lw=.8)
        ax.set_title(title)
        ax.set_xlabel('Omitted classroom index')
        ax.set_ylabel('Difference versus control in points')
    fig.tight_layout()
    fig.savefig(OUT / 'classroom_influence.png', dpi=200)
    plt.close(fig)


def plot_session_effects(sessions):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title in zip(axs, OUTCOMES, ['Practice by session', 'Unassisted exam by session']):
        for arm, label, offset in [('GPTBase', 'Standard AI', -.04), ('GPTTutor', 'Tutor', .04)]:
            z = sessions[(sessions.outcome == y) & (sessions.arm == arm)]
            ax.errorbar(z.session + offset, z.b * 100, yerr=[(z.b - z.low) * 100, (z.high - z.b) * 100],
                        fmt='o-', capsize=3, label=label)
        ax.axhline(0, color='gray', lw=.8)
        ax.set_title(title)
        ax.set_xlabel('Session')
        ax.set_ylabel('Adjusted difference in points')
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'session_effects.png', dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    d = raw[raw.Honors.eq(0)].dropna(subset=REQUIRED).copy()
    gpa_mean = d.gpa_prev.mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Alternative specifications and equal student weighting.
    rows = []
    for mode in SPECIFICATIONS:
        for y in OUTCOMES:
            weights = 1 / d.groupby('Student ID')['Student ID'].transform('size') if mode == 'Equal student weight' else None
            m = fit(d, y, gpa_mean, mode, weights)
            for arm in ARMS:
                ci = m.conf_int().loc[arm]
                rows.append(dict(model=mode, outcome=y, arm=arm, b=m.params[arm], low=ci.iloc[0], high=ci.iloc[1],
                                 p=m.pvalues[arm], n=len(d)))
    pd.DataFrame(rows).to_csv(OUT / 'extended_specifications.csv', index=False)

    # Leave one classroom out.
    loco = []
    for cls in sorted(d.Class.unique()):
        for y in OUTCOMES:
            m = fit(d[d.Class.ne(cls)], y, gpa_mean)
            for arm in ARMS:
                loco.append(dict(omitted_class=cls, outcome=y, arm=arm, b=m.params[arm], p=m.pvalues[arm]))
    pd.DataFrame(loco).to_csv(OUT / 'classroom_influence.csv', index=False)

    # Prior attainment interactions.
    hetero = []
    for y in OUTCOMES:
        m = fit(d, y, gpa_mean, 'GPA interaction')
        for term in ['Base x GPA', 'Tutor x GPA']:
            ci = m.conf_int().loc[term]
            hetero.append(dict(outcome=y, term=term, b=m.params[term], low=ci.iloc[0], high=ci.iloc[1],
                               p=m.pvalues[term]))
    pd.DataFrame(hetero).to_csv(OUT / 'gpa_interactions.csv', index=False)

    # Session-by-session models.
    sessions = []
    for session, z in d.groupby('Session'):
        for y in OUTCOMES:
            m = fit(z, y, gpa_mean)
            for arm in ARMS:
                ci = m.conf_int().loc[arm]
                sessions.append(dict(session=session, outcome=y, arm=arm, b=m.params[arm], low=ci.iloc[0],
                                     high=ci.iloc[1], p=m.pvalues[arm], n=len(z)))
    pd.DataFrame(sessions).to_csv(OUT / 'session_estimates.csv', index=False)

    boot = wild_cluster_bootstrap(d, gpa_mean, rng)
    pd.DataFrame(boot).to_csv(OUT / 'wild_cluster_bootstrap.csv', index=False)

    # Two one-sided tests at illustrative equivalence margins.
    exam_model = fit(d, 'Part3Tot', gpa_mean)
    df = d.Class.nunique() - 1
    eq = []
    for arm in ARMS:
        b, se = exam_model.params[arm], exam_model.bse[arm]
        for margin in EQUIVALENCE_MARGINS:
            p = max(t.sf((b + margin) / se, df), t.cdf((b - margin) / se, df))
            eq.append(dict(arm=arm, margin=margin, p_TOST=p, low90=b - t.ppf(.95, df) * se,
                           high90=b + t.ppf(.95, df) * se))
    pd.DataFrame(eq).to_csv(OUT / 'equivalence_sensitivity.csv', index=False)

    loco = pd.DataFrame(loco)
    plot_classroom_influence(loco, exam_model)
    plot_session_effects(pd.DataFrame(sessions))

    print(pd.DataFrame(rows).to_string(index=False))
    print('LOCO', loco.groupby(['outcome', 'arm']).b.agg(['min', 'max']))
    print('BOOT', boot)
    print('GPA', hetero)
    print('EQUIV', eq)


if __name__ == '__main__':
    main()
