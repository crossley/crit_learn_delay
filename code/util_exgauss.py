from imports import *

try:
    from sklearn.base import BaseEstimator
except Exception:

    class BaseEstimator:  # lightweight fallback
        pass


class _ExG:

    @staticmethod
    def logpdf(x, mu, sigma, tau):
        """Stable log-pdf of ExGaussian (untruncated)."""
        x = np.asarray(x)
        z = (mu - x) / (np.sqrt(2.0) * sigma) + (sigma / (np.sqrt(2.0) * tau))
        return (-np.log(2.0 * tau) + (sigma**2) / (2.0 * tau**2) +
                (mu - x) / tau - z**2 + np.log(erfcx(z)))

    @staticmethod
    def cdf(x, mu, sigma, tau):
        K = tau / sigma
        return exponnorm.cdf(x, K, loc=mu, scale=sigma)

    @staticmethod
    def ppf(u, mu, sigma, tau):
        K = tau / sigma
        return exponnorm.ppf(u, K, loc=mu, scale=sigma)


class ExGaussianMixture(BaseEstimator):
    """
    Truncated ExGaussian mixture with a scikit-learn-like API.

    Parameters
    ----------
    n_components : int
        Number of mixture components (>=1).
    a, b : float
        Truncation bounds; model is normalized on [a, b].
    n_init : int
        Number of random initializations; best (highest log-lik) is kept.
    random_state : int or np.random.Generator or None
        RNG seed or Generator.
    maxiter : int
        Max optimizer iterations per start.
    tol : float
        Optimizer tolerance.
    init_method : {'quantiles','random'}
        How to seed component means.

    Attributes (after fit)
    ----------------------
    weights_ : (K,) mixture weights
    mu_      : (K,) component Normal means
    sigma_   : (K,) component Normal std
    tau_     : (K,) component Exp means
    loglik_  : float, total log-likelihood on training X
    n_params_ : int, number of free parameters (3K + (K-1))
    converged_ : bool
    """

    def __init__(
        self,
        n_components=1,
        a=12.0,
        b=600.0,
        n_init=10,
        random_state=None,
        maxiter=1000,
        tol=1e-6,
        init_method="quantiles",
    ):
        self.n_components = int(n_components)
        self.a = float(a)
        self.b = float(b)
        self.n_init = int(n_init)
        self.random_state = random_state
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.init_method = init_method

        # set after fit
        self.weights_ = None
        self.mu_ = None
        self.sigma_ = None
        self.tau_ = None
        self.loglik_ = None
        self.n_params_ = None
        self.converged_ = False

    # ---------- public API ----------
    def fit(self, X, y=None):
        X = np.asarray(X, float).ravel()
        if X.size == 0:
            raise ValueError("X is empty.")
        if (X.min() < self.a) or (X.max() > self.b):
            raise ValueError("All X must lie within [a,b].")

        rng = _to_rng(self.random_state)
        xm, xs = float(X.mean()), float(X.std(ddof=1) + 1e-9)

        # parameter vector: [mu*K, log_sigma*K, log_tau*K, alpha*(K-1)]
        K = self.n_components
        n_params = 3 * K + (K - 1)
        self.n_params_ = n_params

        bounds = []
        # mu bounds relative to data
        mu_lo, mu_hi = xm - 5 * xs, xm + 5 * xs
        # log sigma/tau bounds relative to scale
        ls_lo, ls_hi = np.log(1e-3 * xs), np.log(10 * xs)
        lt_lo, lt_hi = np.log(1e-3 * xs), np.log(10 * xs)

        bounds += [(mu_lo, mu_hi)] * K
        bounds += [(ls_lo, ls_hi)] * K
        bounds += [(lt_lo, lt_hi)] * K
        # alpha (K-1) unbounded; we fix last logit to 0
        bounds += [(-np.inf, np.inf)] * (K - 1)

        best = None
        for _ in range(self.n_init):
            theta0 = self._random_init(X, K, xm, xs, rng)
            res = minimize(
                self._nll,
                theta0,
                args=(X, K),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": self.maxiter,
                    "ftol": self.tol
                },
            )
            if best is None or (res.success and res.fun < best.fun):
                best = res

        if best is None:
            raise RuntimeError("Optimization failed for all initializations.")

        # unpack best params
        mu, ls, lt, alpha = self._unpack(best.x, K)
        w = _softmax_with_last_zero(alpha)  # (K,)
        sigma = np.exp(ls)
        tau = np.exp(lt)

        # set attributes
        self.weights_ = w
        self.mu_ = mu
        self.sigma_ = sigma
        self.tau_ = tau
        self.loglik_ = -best.fun
        self.converged_ = bool(best.success)
        return self

    def score_samples(self, X):
        """Log-density on [a,b] for each sample (normalized). Shape (n,)."""
        X = np.asarray(X, float).ravel()
        log_num = self._log_mixture_num(X, self.weights_, self.mu_,
                                        self.sigma_, self.tau_)
        log_Zmix = self._log_Zmix(self.weights_, self.mu_, self.sigma_,
                                  self.tau_)
        return log_num - log_Zmix

    def score(self, X, y=None):
        """Mean log-likelihood per sample on [a,b]."""
        ls = self.score_samples(X)
        return float(ls.mean())

    def aic(self, X):
        """Akaike Information Criterion (like sklearn's .aic)."""
        ll = float(self.score(X) * len(np.asarray(X).ravel()))
        return 2 * self.n_params_ - 2 * ll

    def bic(self, X):
        """Bayesian Information Criterion (like sklearn's .bic)."""
        X = np.asarray(X).ravel()
        ll = float(self.score(X) * len(X))
        return self.n_params_ * np.log(len(X)) - 2 * ll

    def predict_proba(self, X):
        """Responsibilities r_{ik} (posterior component probabilities). Shape (n,K)."""
        X = np.asarray(X, float).ravel()
        K = self.n_components
        log_w = np.log(self.weights_ + 1e-300)
        logfs = np.empty((K, X.size))
        for k in range(K):
            logfs[k, :] = _ExG.logpdf(X, self.mu_[k], self.sigma_[k],
                                      self.tau_[k]) + log_w[k]
        # truncation constants cancel in responsibilities
        log_den = logsumexp(logfs, axis=0)
        r = np.exp(logfs - log_den)  # (K, n)
        return r.T  # (n, K)

    def predict(self, X):
        """Hard assignments = argmax responsibilities. Shape (n,)."""
        return self.predict_proba(X).argmax(axis=1)

    def sample(self, n_samples=1, random_state=None):
        """Draw from the truncated mixture using inverse-CDF per component."""
        rng = _to_rng(
            random_state if random_state is not None else self.random_state)
        K = self.n_components
        comp = rng.choice(K, size=n_samples, p=self.weights_)
        Xs = np.empty(n_samples)
        for i, k in enumerate(comp):
            mu, s, t = self.mu_[k], self.sigma_[k], self.tau_[k]
            Fa = _ExG.cdf(self.a, mu, s, t)
            Fb = _ExG.cdf(self.b, mu, s, t)
            u = rng.uniform(Fa, Fb)
            Xs[i] = _ExG.ppf(u, mu, s, t)
        return Xs

    # ---------- internals ----------
    def _nll(self, theta, X, K):
        """Negative log-likelihood on [a,b] for parameter vector theta."""
        mu, ls, lt, alpha = self._unpack(theta, K)
        sigma = np.exp(ls)
        tau = np.exp(lt)
        w = _softmax_with_last_zero(alpha)  # (K,)

        # numerator: sum_i log sum_k w_k f_k(x_i)
        log_num = self._log_mixture_num(X, w, mu, sigma, tau)
        ll_num = float(log_num.sum())

        # denominator: log Z_mix
        log_Zmix = self._log_Zmix(w, mu, sigma, tau)
        if not np.isfinite(log_Zmix):
            return 1e20
        nll = -(ll_num - len(X) * log_Zmix)
        return nll if np.isfinite(nll) else 1e20

    @staticmethod
    def _log_mixture_num(X, w, mu, sigma, tau):
        K = len(w)
        log_w = np.log(w + 1e-300)
        logfs = np.empty((K, X.size))
        for k in range(K):
            logfs[k, :] = _ExG.logpdf(X, mu[k], sigma[k], tau[k]) + log_w[k]
        return logsumexp(logfs, axis=0)

    def _log_Zmix(self, w, mu, sigma, tau):
        """log Z_mix = log( sum_k w_k [F_k(b)-F_k(a)] )."""
        Zk = []
        for k in range(len(w)):
            F_b = _ExG.cdf(self.b, mu[k], sigma[k], tau[k])
            F_a = _ExG.cdf(self.a, mu[k], sigma[k], tau[k])
            Zk.append(max(F_b - F_a, 0.0))
        Zmix = float(np.dot(w, Zk))
        return np.log(Zmix + 1e-300)

    @staticmethod
    def _unpack(theta, K):
        """theta -> (mu[K], log_sigma[K], log_tau[K], alpha[K-1])"""
        mu = theta[0:K]
        ls = theta[K:2 * K]
        lt = theta[2 * K:3 * K]
        alpha = theta[3 * K:]  # length K-1
        return mu, ls, lt, alpha

    def _random_init(self, X, K, xm, xs, rng):
        """Random but sensible initialization."""
        # mu: spread near quantiles or random jitter around mean
        if self.init_method == "quantiles":
            qs = np.linspace(0.15, 0.85, K)
            mu0 = np.quantile(X, qs)
            mu0 = mu0 + rng.normal(scale=0.2 * xs, size=K)
        else:
            mu0 = xm + rng.normal(scale=0.8 * xs, size=K)
        mu0 = np.clip(mu0, xm - 5 * xs, xm + 5 * xs)

        # log sigma / log tau around data scale
        ls0 = np.log(0.6 * xs + 1e-9) + rng.normal(scale=0.3, size=K)
        lt0 = np.log(0.6 * xs + 1e-9) + rng.normal(scale=0.3, size=K)

        # mixture logits (K-1); last implicit zero
        if K > 1:
            alpha0 = rng.normal(scale=0.5, size=K - 1)
        else:
            alpha0 = np.empty(0)

        return np.concatenate([mu0, ls0, lt0, alpha0], axis=0)


# ---------- helpers ----------
def _to_rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _softmax_with_last_zero(alpha):
    """Return softmax([alpha..., 0]) with last logit fixed to 0 to identify."""
    if alpha.size == 0:
        return np.array([1.0])
    logits = np.concatenate([alpha, np.array([0.0])])
    return softmax(logits)


def bootstrap_lrt_pvalue(
    X,
    null_model,  # fitted ExGaussianMixture with n_components=1
    alt_components=2,  # usually 2
    n_boot=200,  # increase at the end (e.g., 500–2000)
    n_init_null_boot=2,  # fewer starts for speed on bootstrap fits
    n_init_alt_boot=4,  # idem
    random_state=0,
    verbose=False,
):
    """
    Parametric bootstrap LRT between fitted null (1-comp) and alt (K-comp) truncated ExG mixtures.
    Returns (D_obs, p_boot, D_boot_array, alt_model_obs).

    Notes:
      - Uses null_model.sample(...) to simulate under H0 with the same [a,b].
      - Refits both models to each bootstrap sample with reduced n_init to save time.
      - If a bootstrap alt fit fails, we record D_b = 0.0 (conservative).
    """
    X = np.asarray(X, float).ravel()
    n = X.size
    rng = np.random.default_rng(random_state)

    # Fit the alternative on the real data (can be different n_init than null_model used)
    alt_model_obs = ExGaussianMixture(
        n_components=alt_components,
        a=null_model.a,
        b=null_model.b,
        n_init=max(8, n_init_alt_boot),  # use a bit more for the observed data
        random_state=rng,
        maxiter=null_model.maxiter,
        tol=null_model.tol,
        init_method="quantiles",
    ).fit(X)

    # Observed LRT statistic
    ll_null = null_model.score(X) * n
    ll_alt = alt_model_obs.score(X) * n
    D_obs = 2.0 * (ll_alt - ll_null)

    # Bootstrap distribution under H0
    D_boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        # 1) simulate from H0 (null) on [a,b]
        xb = null_model.sample(n_samples=n, random_state=rng)

        # 2) refit null (few starts for speed)
        null_b = ExGaussianMixture(
            n_components=1,
            a=null_model.a,
            b=null_model.b,
            n_init=n_init_null_boot,
            random_state=rng,
            maxiter=null_model.maxiter,
            tol=null_model.tol,
            init_method="quantiles",
        ).fit(xb)

        # 3) refit alternative (few starts for speed)
        try:
            alt_b = ExGaussianMixture(
                n_components=alt_components,
                a=null_model.a,
                b=null_model.b,
                n_init=n_init_alt_boot,
                random_state=rng,
                maxiter=null_model.maxiter,
                tol=null_model.tol,
                init_method="quantiles",
            ).fit(xb)
            ll0 = null_b.score(xb) * n
            ll1 = alt_b.score(xb) * n
            D_boot[b] = 2.0 * (ll1 - ll0)
        except Exception:
            # If the alt fit fails on a bootstrap replicate, be conservative
            D_boot[b] = 0.0

        if verbose and ((b + 1) % max(1, n_boot // 10) == 0):
            print(f" bootstrap {b+1}/{n_boot}")

    # Smoothed bootstrap p-value
    p_boot = (np.sum(D_boot >= D_obs) + 1.0) / (n_boot + 1.0)
    return D_obs, float(p_boot), D_boot, alt_model_obs


# --- Single-panel: plot mixture total + components, with rug colored by hard labels ---
def plot_exg_mixture(model, X, bins=30, title=None, ax=None):
    X = np.asarray(X).ravel()
    a, b = model.a, model.b
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    plt.subplots_adjust(left=0.15)

    # grid
    xmin, xmax = float(np.min(X)), float(np.max(X))
    pad = 0.05 * (xmax - xmin + 1e-9)
    xgrid = np.linspace(max(a, xmin - pad), min(b, xmax + pad), 800)

    # total pdf on [a,b]
    pdf_total = np.exp(model.score_samples(xgrid))

    # component pdfs (weighted, properly normalized on [a,b])
    w, mu, sigma, tau = model.weights_, model.mu_, model.sigma_, model.tau_
    from math import log
    # Z_mix = sum_k w_k [F_k(b)-F_k(a)]
    Zk = [
        _ExG.cdf(b, mu[k], sigma[k], tau[k]) -
        _ExG.cdf(a, mu[k], sigma[k], tau[k]) for k in range(len(w))
    ]
    Zmix = float(np.dot(w, Zk))
    logZmix = np.log(Zmix + 1e-300)

    comps = []
    for k in range(len(w)):
        logf = _ExG.logpdf(xgrid, mu[k], sigma[k], tau[k])
        comps.append(w[k] * np.exp(logf - logZmix))

    # histogram
    ax.hist(X,
            bins=bins,
            density=True,
            alpha=0.25,
            edgecolor="none",
            label="Data (density)")

    # lines
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for k, ck in enumerate(comps):
        ax.plot(xgrid,
                ck,
                ls="--",
                lw=2,
                color=colors[k % len(colors)],
                label=f"Component {k+1} (weighted)")
    ax.plot(xgrid, pdf_total, lw=2, label="Mixture total")

    # rug colored by hard labels (ordered by mu for consistency)
    order = np.argsort(mu)
    R = model.predict_proba(X)[:, order]
    labels = R.argmax(axis=1)
    for k in range(len(w)):
        xk = X[labels == k]
        if xk.size:
            ax.scatter(xk,
                       np.full_like(xk, -0.002 - 0.001 * k),
                       s=10,
                       alpha=0.9,
                       color=colors[k % len(colors)],
                       label=(f"Rug (comp {k+1})"
                              if k == 0 else None))  # avoid dup legend

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    yl = ax.get_ylim()
    ax.set_ylim(min(yl[0], -0.01), yl[1])
    ax.legend(loc="best")
    return ax


# --- Two-panel: left = single (n_components=1), right = mixture (n_components>=2) ---
def plot_exg_compare(exg_single, exg_mix, X, bins=30):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_exg_mixture(exg_single,
                     X,
                     bins=bins,
                     title="Single truncated ExGaussian",
                     ax=axes[0])
    plot_exg_mixture(exg_mix,
                     X,
                     bins=bins,
                     title=f"Mixture ({exg_mix.n_components} comps)",
                     ax=axes[1])
    axes[0].set_ylabel("Density")
    plt.tight_layout()
    return fig, axes
