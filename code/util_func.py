from imports import *
from util_exgauss import *


def load_data_exp_3():

    dir_delay = '../data/delay'
    dir_immed = '../data/immed'
    dir_maip = '../data/maip'
    dir_dual = '../data/dual'

    filenames_delay = [
        os.path.join(dir_delay, f) for f in os.listdir(dir_delay)
    ]
    filenames_immed = [
        os.path.join(dir_immed, f) for f in os.listdir(dir_immed)
    ]
    filenames_maip = [os.path.join(dir_maip, f) for f in os.listdir(dir_maip)]
    filenames_dual = [os.path.join(dir_dual, f) for f in os.listdir(dir_dual)]

    col_names = ['t', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt']

    d_delay = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_delay
    ]
    d_immed = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_immed
    ]
    d_maip = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_maip
    ]

    col_names_dual = [
        't', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt', 'V8', 'V9', 'V10',
        'V11', 'V12', 'rt2'
    ]
    d_dual = [
        pd.read_csv(f, sep='\t', header=None, names=col_names_dual)
        for f in filenames_dual
    ]

    for idx, df in enumerate(d_delay):
        df['sub'] = idx + 101
    for idx, df in enumerate(d_immed):
        df['sub'] = idx + 201
    for idx, df in enumerate(d_maip):
        df['sub'] = idx + 301
    for idx, df in enumerate(d_dual):
        df['sub'] = idx + 401

    d_delay = pd.concat(d_delay, ignore_index=True)
    d_immed = pd.concat(d_immed, ignore_index=True)
    d_maip = pd.concat(d_maip, ignore_index=True)
    d_dual = pd.concat(d_dual, ignore_index=True)

    d_dual = d_dual[[
        't', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt', 'sub'
    ]]

    d_delay['cnd'] = 'Delay'
    d_immed['cnd'] = 'Long ITI'
    d_maip['cnd'] = 'Short ITI'
    d_dual['cnd'] = 'Dual'

    d = pd.concat([d_delay, d_immed, d_maip, d_dual], ignore_index=True)

    d['exp'] = 'Exp 3'

    d = d[['exp', 'sub', 'cnd', 't', 't_prob', 'rt']]

    d['experiment_duration'] = d.groupby(['cnd', 'sub'])['rt'].transform('sum')
    d['experiment_duration'] = d['experiment_duration'] * 5 / 60
    d['trials_completed'] = d.groupby(['cnd', 'sub'])['t'].transform('max')

    d['prob_num'] = d.groupby(
        ['cnd',
         'sub'])['t_prob'].transform(lambda x: (x < x.shift(1)).cumsum() + 1)
    d['t2c'] = d.groupby(['cnd', 'sub',
                          'prob_num'])['t_prob'].transform('count')
    d['nps'] = d.groupby(['cnd', 'sub'])['prob_num'].transform('max')

    return d


def load_data():

    d1 = load_data_exp_1()
    d2 = load_data_exp_2()

    d1['exp'] = 'Exp 1'
    d2['exp'] = 'Exp 2'

    d1 = d1[['exp', 'sub', 'cnd', 't', 't_prob', 'rt']]
    d2 = d2[['exp', 'sub', 'cnd', 't', 't_prob', 'rt']]

    d = pd.concat([d1, d2], ignore_index=True)

    d['experiment_duration'] = d.groupby(['cnd', 'sub'])['rt'].transform('sum')
    d['experiment_duration'] = d['experiment_duration'] * 5 / 60
    d['trials_completed'] = d.groupby(['cnd', 'sub'])['t'].transform('max')

    d['prob_num'] = d.groupby(
        ['cnd',
         'sub'])['t_prob'].transform(lambda x: (x < x.shift(1)).cumsum() + 1)
    d['t2c'] = d.groupby(['cnd', 'sub',
                          'prob_num'])['t_prob'].transform('count')
    d['nps'] = d.groupby(['cnd', 'sub'])['prob_num'].transform('max')

    return d


def load_data_exp_1():

    dir_delay = '../data/delay'
    dir_immed = '../data/immed'
    dir_maip = '../data/maip'

    filenames_delay = [
        os.path.join(dir_delay, f) for f in os.listdir(dir_delay)
    ]
    filenames_immed = [
        os.path.join(dir_immed, f) for f in os.listdir(dir_immed)
    ]
    filenames_maip = [os.path.join(dir_maip, f) for f in os.listdir(dir_maip)]

    col_names = ['t', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt']

    d_delay = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_delay
    ]
    d_immed = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_immed
    ]
    d_maip = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_maip
    ]

    for idx, df in enumerate(d_delay):
        df['sub'] = idx + 101
    for idx, df in enumerate(d_immed):
        df['sub'] = idx + 201
    for idx, df in enumerate(d_maip):
        df['sub'] = idx + 301

    d_delay = pd.concat(d_delay, ignore_index=True)
    d_immed = pd.concat(d_immed, ignore_index=True)
    d_maip = pd.concat(d_maip, ignore_index=True)

    d_delay['cnd'] = 'Delay'
    d_immed['cnd'] = 'Long ITI'
    d_maip['cnd'] = 'Short ITI'

    d = pd.concat([d_delay, d_immed, d_maip], ignore_index=True)

    return d


def load_data_exp_2():

    f_names = [
        os.path.join('../data/bin_data_v2', f)
        for f in os.listdir('../data/bin_data_v2') if f.endswith('.csv')
    ]

    d = pd.concat([pd.read_csv(f) for f in f_names])

    f_names = [
        os.path.join('../data/bin_data_v1', f)
        for f in os.listdir('../data/bin_data_v1') if f.endswith('.csv')
    ]

    d1 = pd.concat([pd.read_csv(f) for f in f_names])
    d1['sub_num'] = d1['sub_num'] + 100

    d = pd.concat([d, d1], ignore_index=True)

    d.rename(
        {
            'sub_num': 'sub',
            'condition': 'cnd',
            'num_problems_solved': 'nps',
            'current_trial_prob': 't_prob',
            'current_trial_exp': 't'
        },
        axis=1,
        inplace=True)

    d.loc[d['cnd'] == 'delay', 'cnd'] = 'Delay'
    d.loc[d['cnd'] == 'immediate', 'cnd'] = 'Long ITI'

    return d


def inspect_duration_vs_trials(d):

    # dd = d[d['nps'] < 14].copy()
    dd = d.copy()

    dd = dd.groupby(['exp', 'cnd', 'sub'])[[
        't2c', 'nps', 'trials_completed', 'experiment_duration'
    ]].mean().reset_index()

    sns.set_palette('rocket', 3)
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 4))
    sns.scatterplot(data=dd[dd['exp'] == 'Exp 1'],
                    x='trials_completed',
                    y='experiment_duration',
                    hue='cnd',
                    ax=ax[0, 0])
    sns.scatterplot(data=dd[dd['exp'] == 'Exp 2'],
                    x='trials_completed',
                    y='experiment_duration',
                    hue='cnd',
                    ax=ax[0, 1])
    plt.savefig('../figures/fig_duration_vs_trials.png')
    plt.close()

    sns.set_palette('rocket', 3)
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(10, 10))
    for i, e in enumerate(dd['exp'].unique()):
        de = dd[dd['exp'] == e]
        for j, m in enumerate(['t2c', 'nps']):
            sns.scatterplot(data=de,
                            x='trials_completed',
                            y=m,
                            hue='cnd',
                            ax=ax[i, j])
            ax[i, j].set_title(e)
    plt.savefig('../figures/fig_trials_vs_meas.png')
    plt.close()


def report_exgauss_mm(X, fig_title, fig_name):

    # ----- Bounds -----
    a, b = 12.0, 600.0  # hard support

    # ----- Stable ExGaussian log-pdf (untruncated) -----
    def exg_logpdf_stable(x, mu, sigma, tau):
        x = np.asarray(x)
        z = (mu - x) / (np.sqrt(2.0) * sigma) + (sigma / (np.sqrt(2.0) * tau))
        return (-np.log(2.0 * tau) + (sigma**2) / (2.0 * tau**2) +
                (mu - x) / tau - z**2 + np.log(erfcx(z)))

    # CDF & PPF from SciPy (stable)
    def exg_cdf(x, mu, sigma, tau):
        K = tau / sigma
        return exponnorm.cdf(x, K, loc=mu, scale=sigma)

    def exg_ppf(u, mu, sigma, tau):
        K = tau / sigma
        return exponnorm.ppf(u, K, loc=mu, scale=sigma)

    # ----- Moment start (untruncated; good enough for initials) -----
    def _exg_moment_start(x):
        x = np.asarray(x).ravel()
        m = float(np.mean(x))
        v = float(np.var(x, ddof=1))
        g1 = float(skew(x, bias=False))
        g1 = max(g1, 1e-3)
        r = (g1 / 2.0)**(2.0 / 3.0)  # r = tau^2/(sigma^2+tau^2)
        r = float(np.clip(r, 1e-6, 1 - 1e-6))
        tau0 = np.sqrt(max(r * v, 1e-12))
        sigma0 = np.sqrt(max((1 - r) * v, 1e-12))
        mu0 = m - tau0
        return np.array([mu0, np.log(sigma0), np.log(tau0)], float)

    # ==========================
    # Single truncated ExGaussian
    # ==========================
    def nll_exg_single_trunc(theta, x, a, b):
        if (x.min() < a) or (x.max() > b):
            return np.inf
        mu, ls, lt = theta
        sigma, tau = np.exp(ls), np.exp(lt)
        ll_num = np.sum(exg_logpdf_stable(x, mu, sigma, tau))
        Z = exg_cdf(b, mu, sigma, tau) - exg_cdf(a, mu, sigma, tau)
        if not np.isfinite(Z) or Z <= 0:
            return np.inf
        return -(ll_num - len(x) * np.log(Z))

    def fit_exg_single_trunc(x, a=12.0, b=600.0, n_restarts=6, rng=0):
        x = np.asarray(x).ravel()
        xm, xs = float(np.mean(x)), float(np.std(x, ddof=1) + 1e-9)
        bounds = [
            (xm - 5 * xs, xm + 5 * xs),
            (np.log(1e-3 * xs), np.log(10 * xs)),
            (np.log(1e-3 * xs), np.log(10 * xs)),
        ]
        base = _exg_moment_start(x)
        rng = np.random.default_rng(rng)
        best = None
        for _ in range(n_restarts):
            th0 = base + rng.normal(scale=[0.1 * xs, 0.2, 0.2])
            res = minimize(nll_exg_single_trunc,
                           th0,
                           args=(x, a, b),
                           method="L-BFGS-B",
                           bounds=bounds)
            if res.success and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            best = minimize(nll_exg_single_trunc,
                            base,
                            args=(x, a, b),
                            method="L-BFGS-B",
                            bounds=bounds)
        mu, ls, lt = best.x
        sigma, tau = float(np.exp(ls)), float(np.exp(lt))
        ll = -best.fun
        n, k = len(x), 3
        AIC = 2 * k - 2 * ll
        BIC = k * np.log(n) - 2 * ll
        return {
            "mu": float(mu),
            "sigma": sigma,
            "tau": tau,
            "ll": ll,
            "AIC": AIC,
            "BIC": BIC,
            "res": best
        }

    # ==========================
    # 2-component truncated ExGaussian mixture (post-mix truncation)
    # ==========================
    def nll_exg_mix2_trunc(theta, x, a, b):
        # theta = [mu1, ls1, lt1, mu2, ls2, lt2, a_logit]
        mu1, ls1, lt1, mu2, ls2, lt2, a_logit = theta
        s1, t1 = np.exp(ls1), np.exp(lt1)
        s2, t2 = np.exp(ls2), np.exp(lt2)
        w = expit(a_logit)  # weight for component 2 (comp1 weight = 1 - w)
        if (x.min() < a) or (x.max() > b):
            return np.inf
        l1 = exg_logpdf_stable(x, mu1, s1, t1) + np.log(1 - w + 1e-300)
        l2 = exg_logpdf_stable(x, mu2, s2, t2) + np.log(w + 1e-300)
        ll_num = np.sum(logsumexp(np.vstack([l1, l2]), axis=0))
        Zmix = ((1 - w) * (exg_cdf(b, mu1, s1, t1) - exg_cdf(a, mu1, s1, t1)) +
                w * (exg_cdf(b, mu2, s2, t2) - exg_cdf(a, mu2, s2, t2)))
        if not np.isfinite(Zmix) or Zmix <= 0:
            return np.inf
        return -(ll_num - len(x) * np.log(Zmix))

    def rand_starts_exg_mix2_trunc(x,
                                   a,
                                   b,
                                   n_starts=10,
                                   rng=None,
                                   bounds=None):
        rng = np.random.default_rng(None if rng is None else rng)
        xm, xs = float(np.mean(x)), float(np.std(x, ddof=1) + 1e-9)
        starts = []
        for _ in range(n_starts):
            mu1 = np.clip(xm - 0.5 * xs + rng.normal(scale=0.2 * xs),
                          bounds[0][0], bounds[0][1])
            mu2 = np.clip(xm + 0.5 * xs + rng.normal(scale=0.2 * xs),
                          bounds[3][0], bounds[3][1])
            if mu2 < mu1: mu1, mu2 = mu2, mu1
            ls1 = np.clip(
                np.log(0.5 * xs) + rng.normal(scale=0.3), bounds[1][0],
                bounds[1][1])
            ls2 = np.clip(
                np.log(0.8 * xs) + rng.normal(scale=0.3), bounds[4][0],
                bounds[4][1])
            lt1 = np.clip(
                np.log(0.5 * xs) + rng.normal(scale=0.3), bounds[2][0],
                bounds[2][1])
            lt2 = np.clip(
                np.log(0.8 * xs) + rng.normal(scale=0.3), bounds[5][0],
                bounds[5][1])
            a_logit = np.clip(
                np.log(0.3 / (1 - 0.3)) + rng.normal(scale=1.0), bounds[6][0],
                bounds[6][1])
            starts.append(
                np.array([mu1, ls1, lt1, mu2, ls2, lt2, a_logit], float))
        return starts

    def fit_exg_mix2_trunc(x, a=12.0, b=600.0, n_starts=16, rng=0):
        x = np.asarray(x).ravel()
        xm, xs = float(np.mean(x)), float(np.std(x, ddof=1) + 1e-9)
        bounds = [
            (xm - 5 * xs, xm + 5 * xs),  # mu1
            (np.log(1e-3 * xs), np.log(10 * xs)),  # ls1
            (np.log(1e-3 * xs), np.log(10 * xs)),  # lt1
            (xm - 5 * xs, xm + 5 * xs),  # mu2
            (np.log(1e-3 * xs), np.log(10 * xs)),  # ls2
            (np.log(1e-3 * xs), np.log(10 * xs)),  # lt2
            (-5.0, 5.0)  # a_logit
        ]
        best = None
        for th0 in rand_starts_exg_mix2_trunc(x,
                                              a,
                                              b,
                                              n_starts=n_starts,
                                              rng=rng,
                                              bounds=bounds):
            res = minimize(nll_exg_mix2_trunc,
                           th0,
                           args=(x, a, b),
                           method="L-BFGS-B",
                           bounds=bounds)
            if res.success and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            raise RuntimeError("Mixture optimization failed for all starts.")
        mu1, ls1, lt1, mu2, ls2, lt2, a_logit = best.x
        s1, t1 = float(np.exp(ls1)), float(np.exp(lt1))
        s2, t2 = float(np.exp(ls2)), float(np.exp(lt2))
        w = float(expit(a_logit))
        # order by mu
        if mu2 < mu1:
            mu1, mu2 = float(mu2), float(mu1)
            s1, s2 = s2, s1
            t1, t2 = t2, t1
            w = 1.0 - w
        ll = -best.fun
        n, k = len(x), 7
        AIC = 2 * k - 2 * ll
        BIC = k * np.log(n) - 2 * ll
        return {
            "mu1": float(mu1),
            "sigma1": s1,
            "tau1": t1,
            "mu2": float(mu2),
            "sigma2": s2,
            "tau2": t2,
            "w": w,
            "ll": ll,
            "AIC": AIC,
            "BIC": BIC,
            "res": best
        }

    # Responsibilities (truncation cancels in posterior weights)
    def responsibilities_exg_mix2_trunc(x, fit):
        w2 = fit["w"]
        w1 = 1 - w2
        l1 = exg_logpdf_stable(x, fit["mu1"], fit["sigma1"],
                               fit["tau1"]) + np.log(w1 + 1e-300)
        l2 = exg_logpdf_stable(x, fit["mu2"], fit["sigma2"],
                               fit["tau2"]) + np.log(w2 + 1e-300)
        den = logsumexp(np.vstack([l1, l2]), axis=0)
        r2 = np.exp(l2 - den)
        r1 = 1 - r2
        return np.vstack([r1, r2]).T

    # ==========================
    # Parametric-bootstrap LRT under truncated H0 (single ExG)
    # ==========================
    def r_exg_single_trunc(n, mu, sigma, tau, a, b, rng=None):
        rng = np.random.default_rng(None if rng is None else rng)
        Fa = exg_cdf(a, mu, sigma, tau)
        Fb = exg_cdf(b, mu, sigma, tau)
        U = rng.uniform(Fa, Fb, size=n)
        return exg_ppf(U, mu, sigma, tau)

    def lrt_parametric_bootstrap_trunc(X,
                                       fit1_single,
                                       a,
                                       b,
                                       B=500,
                                       rng=123,
                                       n_starts_mix=12):
        X = np.asarray(X).ravel()
        D_obs = 2.0 * (fit2_mix["ll"] - fit1_single["ll"])
        boot = np.empty(B)
        rng = np.random.default_rng(rng)
        for bidx in range(B):
            xb = r_exg_single_trunc(len(X),
                                    fit1_single["mu"],
                                    fit1_single["sigma"],
                                    fit1_single["tau"],
                                    a,
                                    b,
                                    rng=rng)
            fb1 = fit_exg_single_trunc(xb, a, b, n_restarts=3, rng=rng)
            try:
                fb2 = fit_exg_mix2_trunc(xb,
                                         a,
                                         b,
                                         n_starts=n_starts_mix,
                                         rng=rng)
                boot[bidx] = 2.0 * (fb2["ll"] - fb1["ll"])
            except Exception:
                boot[bidx] = 0.0
        pval = (np.sum(boot >= D_obs) + 1) / (B + 1)
        return D_obs, pval

    # ==========================
    # Fit both models on your X
    # ==========================
    X = np.asarray(X).ravel()
    if (X.min() < a) or (X.max() > b):
        raise ValueError("X contains values outside [a,b].")

    fit1_single = fit_exg_single_trunc(X, a, b, n_restarts=8, rng=1)
    fit2_mix = fit_exg_mix2_trunc(X, a, b, n_starts=24, rng=2)

    print("\n--- Truncated Single ExGaussian (MLE) ---")
    print(
        f"mu={fit1_single['mu']:.3f}, sigma={fit1_single['sigma']:.3f}, tau={fit1_single['tau']:.3f}"
    )
    print(
        f"logLik={fit1_single['ll']:.3f}, AIC={fit1_single['AIC']:.2f}, BIC={fit1_single['BIC']:.2f}"
    )

    print("\n--- Truncated 2×ExGaussian Mixture (MLE) ---")
    print(f"w2={fit2_mix['w']:.3f} (comp2), w1={1-fit2_mix['w']:.3f} (comp1)")
    print(
        f"comp1: mu={fit2_mix['mu1']:.3f}, sigma={fit2_mix['sigma1']:.3f}, tau={fit2_mix['tau1']:.3f}"
    )
    print(
        f"comp2: mu={fit2_mix['mu2']:.3f}, sigma={fit2_mix['sigma2']:.3f}, tau={fit2_mix['tau2']:.3f}"
    )
    print(
        f"logLik={fit2_mix['ll']:.3f}, AIC={fit2_mix['AIC']:.2f}, BIC={fit2_mix['BIC']:.2f}"
    )

    # NHST-style comparison
    k1, k2 = 3, 7
    delta_AIC = fit2_mix["AIC"] - fit1_single["AIC"]
    delta_BIC = fit2_mix["BIC"] - fit1_single["BIC"]

    print("\n=== NHST-style comparison (TRUNCATED) ===")
    print(f"N = {len(X)}")
    print(
        f"ΔAIC (mix - single) = {delta_AIC:.2f}  -> more negative favors mixture"
    )
    print(
        f"ΔBIC (mix - single) = {delta_BIC:.2f}  -> more negative favors mixture"
    )

    # Bootstrap LRT (H0 = truncated single ExG)
    D_obs, p_boot = lrt_parametric_bootstrap_trunc(X,
                                                   fit1_single,
                                                   a,
                                                   b,
                                                   B=500,
                                                   rng=123,
                                                   n_starts_mix=12)
    print(
        "\nLikelihood-Ratio Test (parametric bootstrap under H0=truncated single ExG):"
    )
    print(f"D = 2*(LL_mix - LL_single) = {D_obs:.2f},  p_boot ≈ {p_boot:.4f}")

    # ==========================
    # Plot densities (normalized on [a,b]) + colored rug
    # ==========================
    # Single truncated pdf
    Z1 = exg_cdf(b, fit1_single["mu"], fit1_single["sigma"], fit1_single["tau"]) - \
         exg_cdf(a, fit1_single["mu"], fit1_single["sigma"], fit1_single["tau"])

    # Mixture truncated normalizer
    Zmix = (
        (1 - fit2_mix["w"]) *
        (exg_cdf(b, fit2_mix["mu1"], fit2_mix["sigma1"], fit2_mix["tau1"]) -
         exg_cdf(a, fit2_mix["mu1"], fit2_mix["sigma1"], fit2_mix["tau1"])) +
        fit2_mix["w"] *
        (exg_cdf(b, fit2_mix["mu2"], fit2_mix["sigma2"], fit2_mix["tau2"]) -
         exg_cdf(a, fit2_mix["mu2"], fit2_mix["sigma2"], fit2_mix["tau2"])))

    x_min, x_max = float(np.min(X)), float(np.max(X))
    pad = 0.05 * (x_max - x_min + 1e-9)
    xgrid = np.linspace(max(a, x_min - pad), min(b, x_max + pad), 800)

    pdf_single_trunc = np.exp(
        exg_logpdf_stable(xgrid, fit1_single["mu"], fit1_single["sigma"],
                          fit1_single["tau"])) / Z1
    comp1 = (1 - fit2_mix["w"]) * np.exp(
        exg_logpdf_stable(xgrid, fit2_mix["mu1"], fit2_mix["sigma1"],
                          fit2_mix["tau1"])) / Zmix
    comp2 = (fit2_mix["w"]) * np.exp(
        exg_logpdf_stable(xgrid, fit2_mix["mu2"], fit2_mix["sigma2"],
                          fit2_mix["tau2"])) / Zmix
    pdf_mix_trunc = comp1 + comp2

    # Responsibilities & labels
    resp = responsibilities_exg_mix2_trunc(X, fit2_mix)
    hard = resp.argmax(axis=1)
    counts = np.bincount(hard, minlength=2)
    print(
        f"\nHard assignments (mixture, truncated): comp1={counts[0]}, comp2={counts[1]} / N={len(X)}"
    )

    colors = np.array(["tab:blue", "tab:orange"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left: Single truncated ExG
    ax = axes[0]
    ax.hist(X,
            bins=30,
            density=True,
            alpha=0.25,
            edgecolor="none",
            label="Data (density)")
    ax.plot(xgrid, pdf_single_trunc, lw=2, label="Single ExG (truncated)")
    for k in (0, 1):
        xk = X[hard == k]
        ax.scatter(xk,
                   np.full_like(xk, -0.002 - 0.001 * 1),
                   s=10,
                   alpha=0.9,
                   color=colors[0],
                   label=f"Rug (comp {k+1})")
    ax.set_title("Truncated ExGaussian (single)")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    yl = ax.get_ylim()
    ax.set_ylim(min(yl[0], -0.005), yl[1])
    ax.legend(loc="best")

    # Right: Mixture truncated ExG (components + total)
    ax = axes[1]
    ax.hist(X,
            bins=30,
            density=True,
            alpha=0.25,
            edgecolor="none",
            label="Data (density)")
    ax.plot(xgrid,
            comp1,
            ls="--",
            lw=2,
            color=colors[0],
            label="Comp 1 (weighted, trunc)")
    ax.plot(xgrid,
            comp2,
            ls="--",
            lw=2,
            color=colors[1],
            label="Comp 2 (weighted, trunc)")
    ax.plot(xgrid, pdf_mix_trunc, lw=2, label="Mixture total (trunc)")
    for k in (0, 1):
        xk = X[hard == k]
        ax.scatter(xk,
                   np.full_like(xk, -0.002 - 0.001 * 1),
                   s=10,
                   alpha=0.9,
                   color=colors[k],
                   label=f"Rug (comp {k+1})")
    ax.set_title("Truncated ExGaussian mixture (2 comps)")
    ax.set_xlabel("x")
    yl = ax.get_ylim()
    ax.set_ylim(min(yl[0], -0.005), yl[1])
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig('../figures/' + fig_name)
    plt.close()


def report_gmm(X, fig_title, fig_name):

    gmm_1 = GaussianMixture(n_components=1, random_state=1).fit(X)
    gmm_2 = GaussianMixture(n_components=2, random_state=1).fit(X)

    print("\n")
    print(f"AIC for 1 component: {gmm_1.aic(X)}\n")
    print(f"AIC for 2 components: {gmm_2.aic(X)}\n")
    print("\n")
    print(f"BIC for 1 component: {gmm_1.bic(X)}\n")
    print(f"BIC for 2 components: {gmm_2.bic(X)}\n")

    # test significance of gmm with LRT
    log_likelihood_1 = gmm_1.score(X) * len(X)
    log_likelihood_2 = gmm_2.score(X) * len(X)
    lrt_statistic = 2 * (log_likelihood_2 - log_likelihood_1)
    degrees_of_freedom = 3 * (2 - 1)
    p_value = chi2.sf(lrt_statistic, degrees_of_freedom)

    print(f"LRT statistic: {lrt_statistic}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    print(f"P-value: {p_value}")

    # visualise gmm fit
    x = np.linspace(min(X), max(X), 1000)
    logprob_1 = gmm_1.score_samples(x.reshape(-1, 1))
    logprob_2 = gmm_2.score_samples(x.reshape(-1, 1))
    pdf_1 = np.exp(logprob_1)
    pdf_2 = np.exp(logprob_2)

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fs_legend = 14
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
    ax[0, 0].hist(X, bins=50, density=True, alpha=0.6)
    ax[0, 0].plot(x, pdf_1, '-k', label='1 component')
    ax[0, 0].plot(x, pdf_2, '-r', label='2 components')
    ax[0, 0].set_title(fig_title, fontsize=fs_title)
    ax[0, 0].set_xlabel('Trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_ylabel('Proportion of participants', fontsize=fs_axis_label)
    ax[0, 0].legend(fontsize=fs_legend)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    plt.savefig('../figures/' + fig_name)
    plt.close()

    return gmm_1, gmm_2


def report_exp_1():

    print()
    print("Reporting Exp 1 data...")

    d = load_data()

    dd = d.groupby(['exp', 'cnd', 'sub'])[['t2c', 'nps']].mean().reset_index()

    dd = dd.loc[dd['exp'] == 'Exp 1'].copy()

    X = dd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))
    X = X.ravel()

    # Fit null and alternative just like GaussianMixture
    exg1 = ExGaussianMixture(n_components=1,
                             a=9,
                             b=512,
                             random_state=1,
                             n_init=8).fit(X)
    exg2 = ExGaussianMixture(n_components=2,
                             a=9,
                             b=512,
                             random_state=1,
                             n_init=16).fit(X)

    # Bootstrap LRT p-value (use a small n_boot while iterating; increase at the end)
    D_obs, p_boot, D_boot, exg2_refit = bootstrap_lrt_pvalue(
        X,
        null_model=exg1,
        alt_components=2,
        n_boot=100,
        n_init_null_boot=2,
        n_init_alt_boot=4,
        random_state=123,
        verbose=True,
    )

    print()
    print("--- ExGaussian mixture model fit ---")
    print(f"AIC(1)={exg1.aic(X):.2f}  AIC(2)={exg2.aic(X):.2f}")
    print(f"BIC(1)={exg1.bic(X):.2f}  BIC(2)={exg2.bic(X):.2f}")
    print(f"LRT D_obs = {D_obs:.2f},  p_boot ≈ {p_boot:.4f}")

    R_raw = exg2.predict_proba(X)  # shape (N, 2) in the model's native order
    order = np.argsort(
        exg2.mu_
    )  # enforce a consistent order: comp1 = lower-μ, comp2 = higher-μ
    R = R_raw[:, order]  # reorder columns to [low-μ, high-μ]

    labels = R.argmax(axis=1)  # 0 or 1 (by μ order, not internal order)
    counts = np.bincount(labels, minlength=2)
    soft_counts = R.sum(axis=0)

    dd['pred'] = labels

    fig, axes = plot_exg_compare(exg1, exg2, X, bins=30)
    plt.savefig('../figures/fig_exp_1_mm_trunc_exgauss_compare.png')
    plt.close()

    # Or just the mixture panel:
    plot_exg_mixture(exg2, X, bins=30, title="Truncated ExG mixture")
    plt.savefig('../figures/fig_exp_1_mm_trunc_exgauss.png')
    plt.close()

    # gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_1_gmm.png')
    # pred = gmm_2.predict(X)
    # dd['pred'] = pred

    ddd = dd.loc[dd['pred'] == 0].copy()

    print()
    print(
        "Sample sizes per experiment and condition --- pre outlier exclusion:")
    print(dd.groupby(['exp', 'cnd'])['sub'].nunique())

    print()
    print(
        "Sample sizes per experiment and condition --- post outlier exclusion:"
    )
    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    # print cnd means
    print(ddd.groupby(['cnd'])[['t2c', 'nps']].mean())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.375)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks())
    ax[0, 0].set_xticklabels(['Delayed FB', 'Long ITI', 'Control'],
                             fontsize=fs_axis_ticks)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].set_xticks(ax[0, 1].get_xticks())
    ax[0, 1].set_xticklabels(['Delayed FB', 'Long ITI', 'Control'],
                             fontsize=fs_axis_ticks)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_1_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print("ANOVA on trials to criterion:")
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print("Pairwise comparisons on trials to criterion:")
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print("ANOVA on number of problems solved:")
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print("Pairwise comparisons on number of problems solved:")
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])


def report_exp_2():

    print()
    print("Reporting Exp 2 data...")

    d = load_data()

    dd = d.groupby(['exp', 'cnd', 'sub'])[['t2c', 'nps']].mean().reset_index()

    dd = dd.loc[dd['exp'] == 'Exp 2'].copy()

    X = dd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))
    X = X.ravel()

    # Fit null and alternative just like GaussianMixture
    exg1 = ExGaussianMixture(n_components=1,
                             a=12,
                             b=600,
                             random_state=1,
                             n_init=8).fit(X)
    exg2 = ExGaussianMixture(n_components=2,
                             a=12,
                             b=600,
                             random_state=1,
                             n_init=16).fit(X)

    # Bootstrap LRT p-value (use a small n_boot while iterating; increase at the end)
    # D_obs, p_boot, D_boot, exg2_refit = bootstrap_lrt_pvalue(
    #     X,
    #     null_model=exg1,
    #     alt_components=2,
    #     n_boot=100,
    #     n_init_null_boot=2,
    #     n_init_alt_boot=4,
    #     random_state=123,
    #     verbose=True,
    # )

    # print()
    # print("--- ExGaussian mixture model fit ---")
    # print(f"AIC(1)={exg1.aic(X):.2f}  AIC(2)={exg2.aic(X):.2f}")
    # print(f"BIC(1)={exg1.bic(X):.2f}  BIC(2)={exg2.bic(X):.2f}")
    # print(f"LRT D_obs = {D_obs:.2f},  p_boot ≈ {p_boot:.4f}")

    R_raw = exg2.predict_proba(X)  # shape (N, 2) in the model's native order
    order = np.argsort(
        exg2.mu_
    )  # enforce a consistent order: comp1 = lower-μ, comp2 = higher-μ
    R = R_raw[:, order]  # reorder columns to [low-μ, high-μ]

    labels = R.argmax(axis=1)  # 0 or 1 (by μ order, not internal order)
    counts = np.bincount(labels, minlength=2)
    soft_counts = R.sum(axis=0)

    dd['pred'] = labels

    fig, axes = plot_exg_compare(exg1, exg2, X, bins=30)
    plt.savefig('../figures/fig_exp_2_mm_trunc_exgauss_compare.png')
    plt.close()

    # Or just the mixture panel:
    plot_exg_mixture(exg2, X, bins=30, title="Truncated ExG mixture")
    plt.savefig('../figures/fig_exp_2_mm_trunc_exgauss.png')
    plt.close()

    # gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_2_gmm.png')
    # pred = gmm_2.predict(X)
    # dd['pred'] = pred

    # Mixture model doesn't support 2 components
    ddd = dd.copy()
    # ddd = dd.loc[dd['pred'] == 0].copy()

    print()
    print(
        "Sample sizes per experiment and condition --- pre outlier exclusion:")
    print(dd.groupby(['exp', 'cnd'])['sub'].nunique())

    print()
    print(
        "Sample sizes per experiment and condition --- post outlier exclusion:"
    )
    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.25)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_xticks([0, 1])
    ax[0, 0].set_xticklabels(['Delayed FB', 'Long ITI'],
                             fontsize=fs_axis_ticks)
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_xticks([0, 1])
    ax[0, 1].set_xticklabels(['Delayed FB', 'Long ITI'],
                             fontsize=fs_axis_ticks)
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_2_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print("ANOVA on trials to criterion:")
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print("Pairwise comparisons on trials to criterion:")
    print(res[['A', 'B', 'T', 'dof', 'p-unc', 'BF10']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print("ANOVA on number of problems solved:")
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print("Pairwise comparisons on number of problems solved:")
    print(res[['A', 'B', 'T', 'dof', 'p-unc', 'BF10']])

    # ttost for delayed vs long ITI
    for bnd in [20, 30, 40]:
        res = pg.tost(x=ddd.loc[ddd['cnd'] == 'Delay', 't2c'],
                      y=ddd.loc[ddd['cnd'] == 'Long ITI', 't2c'],
                      bound=bnd,
                      paired=False,
                      correction=True)
        print(res)


def report_exp_1_vs_2(dd):

    print()
    print("Reporting Exp 1 vs 2...")

    d = load_data()

    dd = d.groupby(['exp', 'cnd', 'sub'])[['t2c', 'nps']].mean().reset_index()

    dd1 = dd.loc[dd['exp'] == 'Exp 1'].copy()
    dd2 = dd.loc[dd['exp'] == 'Exp 2'].copy()

    X = dd1['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))
    X = X.ravel()

    exg1 = ExGaussianMixture(n_components=1,
                             a=12,
                             b=600,
                             random_state=1,
                             n_init=8).fit(X)
    exg2 = ExGaussianMixture(n_components=2,
                             a=12,
                             b=600,
                             random_state=1,
                             n_init=16).fit(X)

    R_raw = exg2.predict_proba(X)
    order = np.argsort(exg2.mu_)
    R = R_raw[:, order]

    labels = R.argmax(axis=1)
    counts = np.bincount(labels, minlength=2)
    soft_counts = R.sum(axis=0)

    dd1['pred'] = labels
    dd2['pred'] = 0

    dd = pd.concat([dd1, dd2], ignore_index=True)
    dd = dd.loc[dd['pred'] == 0].copy()

    # remove cnd=='Short ITI' from dd
    dd = dd[dd['cnd'] != 'Short ITI']

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats

    # Make sure the reference levels are Exp 2 and Long ITI so the interaction coefficient = DID
    dd2 = dd.copy()
    dd2['exp'] = pd.Categorical(dd2['exp'], categories=['Exp 2', 'Exp 1'])
    dd2['cnd'] = pd.Categorical(dd2['cnd'], categories=['Long ITI', 'Delay'])

    m = smf.ols('t2c ~ C(exp)*C(cnd)',
                data=dd2).fit(cov_type='HC3')  # robust SEs
    coef_name = 'C(exp)[T.Exp 1]:C(cnd)[T.Delay]'
    did = m.params[coef_name]
    t_res = m.t_test([1.0 if p == coef_name else 0.0 for p in m.params.index])

    tval = float(np.squeeze(t_res.tvalue))  # scalar t
    df = getattr(t_res, "df_denom",
                 getattr(t_res, "df", m.df_resid))  # fallback to model df

    # One-sided test H1: DID > 0
    p_one = stats.t.sf(tval, df)  # sf = 1 - cdf

    # (optional) check via the two-sided p-value that statsmodels returns
    p_two = float(np.squeeze(t_res.pvalue))
    p_one_check = (p_two / 2.0) if tval >= 0 else (1.0 - p_two / 2.0)

    print({
        'DID_est': float(m.params[coef_name]),
        't': tval,
        'df': df,
        'p_one_sided': p_one
    })


def report_exp_3(dd):

    d = load_data_exp_3()
    dd = d3.groupby(['exp', 'cnd', 'sub'])[['t2c', 'nps']].mean().reset_index()
    print(dd.groupby(['exp', 'cnd'])['sub'].nunique())

    X = dd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_1_gmm.png')

    pred = gmm_2.predict(X)
    dd['pred'] = pred

    ddd = dd.loc[dd['pred'] == 0].copy()

    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.375)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks())
    ax[0,
       0].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                          fontsize=fs_axis_ticks)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].set_xticks(ax[0, 1].get_xticks())
    ax[0,
       1].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                          fontsize=fs_axis_ticks)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_3_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])


def report_exp_3(dd):

    ddd = dd[dd['cnd'].isin(['Delay', 'Long ITI', 'Short ITI'])].copy()
    X = ddd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_3_gmm.png')

    pred = gmm_2.predict(X)
    ddd['pred'] = pred

    ddd = ddd.loc[ddd['pred'] == 0].copy()

    # Dual-task condition
    ddd_dual = dd[dd['cnd'].isin(['Dual'])].copy()
    X = ddd_dual['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_3_gmm.png')

    pred = gmm_2.predict(X)
    ddd_dual['pred'] = pred

    ddd_dual = ddd_dual.loc[ddd_dual['pred'] == 0].copy()

    # recombine
    ddd = pd.concat([ddd, ddd_dual], ignore_index=True)

    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.375)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks())
    ax[0,
       0].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                          fontsize=fs_axis_ticks)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].set_xticks(ax[0, 1].get_xticks())
    ax[0,
       1].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                          fontsize=fs_axis_ticks)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_3_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])


def plot_cat_space():

    d = load_data_exp_1()

    d['prob_num'] = d.groupby(
        ['cnd',
         'sub'])['t_prob'].transform(lambda x: (x < x.shift(1)).cumsum() + 1)

    d['t2c'] = d.groupby(['cnd', 'sub',
                          'prob_num'])['t_prob'].transform('count')
    d['nps'] = d.groupby(['cnd', 'sub'])['prob_num'].transform('max')

    d = d[d['nps'] == 14].copy()
    d = d[d['sub'] == 103].copy()

    d = d[['prob_num', 'bnd', 'x', 'y', 'nps']].copy()


    d1 = d[d['prob_num'] <= 7].copy()
    d2 = d[d['prob_num'] > 7].copy()

    d1['prob_num'] = d1['prob_num'].astype('category')
    d2['prob_num'] = d2['prob_num'].astype('category')

    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(5, 10))
    sns.scatterplot(data=d1,
                    x='x',
                    y='y',
                    hue='prob_num',
                    ax=ax[0, 0],
                    legend=False)
    for p in d1['prob_num'].unique():
        dd = d1[d1['prob_num'] == p]
        b = dd['bnd'].unique()
        bl = b - 1
        bu = b + 1
        ax[0, 0].axvline(x=b, color='gray', linestyle='--', alpha=0.8)
        ax[0, 0].axvline(x=bl, color='gray', linestyle='-', alpha=0.8)
        ax[0, 0].axvline(x=bu, color='gray', linestyle='-', alpha=0.8)
    sns.scatterplot(data=d2,
                    x='x',
                    y='y',
                    hue='prob_num',
                    ax=ax[1, 0],
                    legend=False)
    for p in d2['prob_num'].unique():
        dd = d2[d2['prob_num'] == p]
        b = dd['bnd'].unique()
        bl = b - 1
        bu = b + 1
        ax[1, 0].axhline(y=b, color='gray', linestyle='--', alpha=0.8)
        ax[1, 0].axhline(y=bl, color='gray', linestyle='-', alpha=0.8)
        ax[1, 0].axhline(y=bu, color='gray', linestyle='-', alpha=0.8)
    [a.set_xlabel("Bar Thickness", size=14) for a in ax[:, 0]]
    [a.set_ylabel("Bar Angle", size=14) for a in ax[:, 0]]
    ax[0, 0].set_title("Example Problems 1 through 7", size=16)
    ax[1, 0].set_title("Example Problems 8 through 14", size=16)
    plt.subplots_adjust(hspace=0.3, top=0.92, bottom=0.08)
    plt.savefig('../figures/fig_design_exp_1_space.png')
    plt.close()
