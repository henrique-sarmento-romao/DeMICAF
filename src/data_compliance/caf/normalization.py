"""
caf_normalisation.py.

====================
Parametric normalisation of Mahalanobis Distance (MD) scores into [0, 1]
compliance scores suitable for federated deployment.

Design principles
-----------------
* Fitting is performed on a vector of *scalar MD values* from an in-distribution
  (ID) reference set — never on raw feature vectors.
* Each normaliser returns a plain dict of lightweight parameters that can be
  broadcast across federation nodes without sharing any raw scores.
* Evaluation applies a closed-form transformation using only those parameters.
 * Higher score → more compliant / in-distribution (inverted mapping: 1=in-distribution).

Available normalisers
---------------------
  "gamma"    — Gamma CDF fitted to MD² (primary recommendation).
  "lognorm"  — Log-normal CDF fitted to MD (best for heavy-tailed distributions).
  "erf"      — Gaussian-erf scaling on MD (robust fallback, two parameters).
  "logistic" — Sigmoid centred on a data-driven threshold (most intuitive).
  "gmm"      — Gaussian Mixture CDF on MD (for multimodal ID distributions).

Usage
-----
    from caf_normalisation import fit, transform, select_normaliser

    # On the reference node (fit on ID MD scores):
    params = fit(md_ref, method="gamma")          # -> shareable dict
    # Broadcast `params` to all evaluation nodes.

    # On any evaluation node:
    scores = transform(md_test, params)           # -> np.ndarray in [0, 1]

    # Automatic method selection via goodness-of-fit:
    params = fit(md_ref, method="auto")
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats
from scipy.special import erf
from scipy.stats import gamma as gamma_dist
from scipy.stats import kstest
from scipy.stats import lognorm as lognorm_dist
from sklearn.mixture import GaussianMixture

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Type alias for the shared parameter dict
# ---------------------------------------------------------------------------
Params = dict[str, Any]  # {"method": str, ...method-specific floats/lists...}

Method = Literal["gamma", "lognorm", "erf", "logistic", "gmm", "auto"]


# ===========================================================================
# Public API
# ===========================================================================


def fit(md_ref: np.ndarray, method: Method = "gamma", **kwargs: Any) -> Params:
    """Fit a normaliser to in-distribution MD scores and return its parameters.

    Parameters
    ----------
    md_ref : np.ndarray, shape (N,)
        Mahalanobis distances computed on the ID reference set.  Must contain
        only finite, non-negative values.
    method : str
        One of "gamma", "lognorm", "erf", "logistic", "gmm", or "auto".
        "auto" selects the best-fitting parametric model via KS test.
    **kwargs
        Method-specific overrides (e.g. n_components=3 for "gmm").

    Returns
    -------
    Params
        A JSON-serialisable dict containing "method" plus the fitted parameters.
        This dict is the only artefact that needs to be shared across nodes.
    """
    md_ref = _validate(md_ref)

    dispatchers: dict[str, Callable[..., Params]] = {
        "gamma": _fit_gamma,
        "lognorm": _fit_lognorm,
        "erf": _fit_erf,
        "logistic": _fit_logistic,
        "gmm": _fit_gmm,
    }

    if method == "auto":
        return _auto_select(md_ref, dispatchers, **kwargs)

    if method not in dispatchers:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(dispatchers)}")

    return dispatchers[method](md_ref, **kwargs)


def transform(md_test: np.ndarray, params: Params) -> np.ndarray:
    """Apply a fitted (or aggregated) normaliser to new MD scores.

    Accepts both single-client params (from `fit`) and aggregated params
    (from `aggregate`, which may use the mixture methods).

    Parameters
    ----------
    md_test : np.ndarray, shape (N,)
        Mahalanobis distances to be normalised.
    params : Params
        Parameter dict previously returned by `fit` or `aggregate`.

    Returns
    -------
    np.ndarray, shape (N,)
        Compliance scores in [0, 1].  Score of 1 ≈ fully in-distribution;
        score of 0 ≈ maximally out-of-distribution.
    """
    md_test = np.asarray(md_test, dtype=float).ravel()

    # Mixture transforms are defined further down the module; names resolve at call time.
    dispatchers = {
        "gamma": _transform_gamma,
        "lognorm": _transform_lognorm,
        "erf": _transform_erf,
        "logistic": _transform_logistic,
        "gmm": _transform_gmm,
        "gamma_mixture": _transform_gamma_mixture,
        "lognorm_mixture": _transform_lognorm_mixture,
    }

    method = params.get("method")
    if method not in dispatchers:
        raise ValueError(f"Unknown method in params: '{method}'")

    return dispatchers[method](md_test, params)


def select_normaliser(md_ref: np.ndarray) -> str:
    """Return the name of the best-fitting method for the given MD distribution.

    Runs KS tests for each parametric family and returns the method with the
    smallest KS statistic (highest p-value as tiebreaker).
    """
    md_ref = _validate(md_ref)
    return _best_ks(md_ref)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _validate(md: np.ndarray) -> np.ndarray:
    """Coerce input to 1-D float array and check basic validity."""
    md = np.asarray(md, dtype=float).ravel()
    if not np.all(np.isfinite(md)):
        raise ValueError("md_ref contains NaN or Inf values.")
    if np.any(md < 0):
        raise ValueError("MD values must be non-negative.")
    if len(md) < 10:
        warnings.warn("Fewer than 10 reference samples — parameter estimates may be unreliable.", stacklevel=2)
    return md


# ===========================================================================
# Per-method implementations (fit · transform · aggregate)
# ===========================================================================

# ---------------------------------------------------------------------------
# Gamma CDF  (primary recommendation)
# ---------------------------------------------------------------------------
# Theoretical basis: MD² ~ chi²(p) = Gamma(p/2, 2) when the Gaussian
# assumption holds.  Fitting a free Gamma to the *observed* MD² distribution
# relaxes the rigid chi-squared shape, accommodating model misspecification
# without saturating in the far tail.
#
# Shared parameters: alpha (shape), theta (scale) — two scalars.
#
# Aggregation (exact=False, default): moment-matched single Gamma via law of
# total variance on MD².  Returns {"method": "gamma", ...} — same format as
# a single-client fit, routed through _transform_gamma.
# Aggregation (exact=True): weighted mixture of K Gamma CDFs stored under
# method key "gamma_mixture", routed through _transform_gamma_mixture.
# ---------------------------------------------------------------------------


def _fit_gamma(md_ref: np.ndarray, **_: Any) -> Params:
    md2 = md_ref**2
    # scipy gamma.fit returns (shape, loc, scale); we fix loc=0 (MD² ≥ 0).
    alpha, loc, theta = gamma_dist.fit(md2, floc=0)
    return {"method": "gamma", "alpha": float(alpha), "theta": float(theta)}


def _transform_gamma(md_test: np.ndarray, params: Params) -> np.ndarray:
    md2 = md_test**2
    # CDF of the fitted Gamma: P(MD²_ref ≤ md²) → score in [0, 1].
    score = gamma_dist.cdf(md2, a=params["alpha"], scale=params["theta"])
    return np.clip(1.0 - score, 0.0, 1.0)


def _aggregate_gamma(client_params: list[Params], w: np.ndarray, exact: bool = False) -> Params:
    alphas = np.array([p["alpha"] for p in client_params])
    thetas = np.array([p["theta"] for p in client_params])

    if exact:
        return {
            "method": "gamma_mixture",
            "weights": w.tolist(),
            "alphas": alphas.tolist(),
            "thetas": thetas.tolist(),
        }

    # Moment-match to a single Gamma via law of total variance on MD².
    means = alphas * thetas
    mu = float(np.dot(w, means))
    var_within = float(np.dot(w, alphas * thetas**2))
    var_between = float(np.dot(w, (means - mu) ** 2))
    var = var_within + var_between
    theta_g = var / mu
    alpha_g = mu / theta_g
    return {"method": "gamma", "alpha": float(alpha_g), "theta": float(theta_g)}


def _transform_gamma_mixture(md_test: np.ndarray, params: Params) -> np.ndarray:
    """CDF of a weighted mixture of Gamma distributions, evaluated on MD²."""
    md2 = md_test**2
    score = np.zeros(len(md2))
    for wk, ak, tk in zip(params["weights"], params["alphas"], params["thetas"], strict=False):
        score += wk * gamma_dist.cdf(md2, a=ak, scale=tk)
    return np.clip(1.0 - score, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Log-normal CDF  (heavy-tailed distributions)
# ---------------------------------------------------------------------------
# MD (not MD²) is often empirically log-normally distributed in high-
# dimensional deep-feature spaces.  Fitting a normal to log(MD) gives a
# two-parameter CDF that maps smoothly onto [0, 1].
#
# Shared parameters: mu_log (mean of log MD), sigma_log (SD of log MD).
#
# Aggregation (exact=False, default): moment-matched single log-normal via
# law of total variance in log-space.  Returns {"method": "lognorm", ...},
# routed through _transform_lognorm.
# Aggregation (exact=True): weighted mixture of K log-normal CDFs stored
# under method key "lognorm_mixture", routed through _transform_lognorm_mixture.
# ---------------------------------------------------------------------------


def _fit_lognorm(md_ref: np.ndarray, **_: Any) -> Params:
    log_md = np.log(md_ref + 1e-9)  # guard against exact zeros
    mu_log = float(log_md.mean())
    sigma_log = float(log_md.std(ddof=1))
    return {"method": "lognorm", "mu_log": mu_log, "sigma_log": sigma_log}


def _transform_lognorm(md_test: np.ndarray, params: Params) -> np.ndarray:
    # lognorm.cdf(x, s=sigma, scale=exp(mu)) equals Phi((log x - mu) / sigma).
    score = lognorm_dist.cdf(
        md_test + 1e-9,
        s=params["sigma_log"],
        scale=np.exp(params["mu_log"]),
    )
    return np.clip(1.0 - score, 0.0, 1.0)


def _aggregate_lognorm(client_params: list[Params], w: np.ndarray, exact: bool = False) -> Params:
    mu_logs = np.array([p["mu_log"] for p in client_params])
    sigma_logs = np.array([p["sigma_log"] for p in client_params])

    if exact:
        return {
            "method": "lognorm_mixture",
            "weights": w.tolist(),
            "mu_logs": mu_logs.tolist(),
            "sigma_logs": sigma_logs.tolist(),
        }

    # Moment-match to a single log-normal via law of total variance in log-space.
    mu_global = float(np.dot(w, mu_logs))
    var_within = float(np.dot(w, sigma_logs**2))
    var_between = float(np.dot(w, (mu_logs - mu_global) ** 2))
    sigma_global = float(np.sqrt(var_within + var_between))
    return {"method": "lognorm", "mu_log": mu_global, "sigma_log": max(sigma_global, 1e-9)}


def _transform_lognorm_mixture(md_test: np.ndarray, params: Params) -> np.ndarray:
    """CDF of a weighted mixture of log-normal distributions."""
    score = np.zeros(len(md_test))
    for wk, mu, sigma in zip(params["weights"], params["mu_logs"], params["sigma_logs"], strict=False):
        score += wk * lognorm_dist.cdf(md_test + 1e-9, s=sigma, scale=np.exp(mu))
    return np.clip(1.0 - score, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Gaussian-erf scaling  (Kriegel et al., SDM 2011 — robust fallback)
# ---------------------------------------------------------------------------
# Does not assume a specific distributional family for MD.  Standardises the
# score using the mean and SD of the ID reference, then maps through the error
# function to [0, 1].  Values at or below the ID mean score ≈ 0; values
# progressively above approach 1 asymptotically.
#
# Shared parameters: mu_md, sigma_md — two scalars.
#
# Aggregation — law of total variance pooling:
# The pooled mean is the weighted average of client means.
# The pooled variance applies the law of total variance:
#   Var_total = E[Var_within] + Var[E_within]
#             = Σ_k w_k σ_k²  +  Σ_k w_k (μ_k − μ_global)²
# This is the only statistically correct way to merge standard deviations
# from separate groups without access to raw scores.
# ---------------------------------------------------------------------------


def _fit_erf(md_ref: np.ndarray, **_: Any) -> Params:
    # Trimmed estimators are used to reduce the influence of extreme outliers
    # in the reference set on the location/scale estimates.
    mu_md = float(stats.trim_mean(md_ref, proportiontocut=0.05))
    sigma_md = float(stats.mstats.trimmed_std(md_ref, limits=(0.05, 0.05)))
    sigma_md = max(sigma_md, 1e-9)  # prevent division by zero
    return {"method": "erf", "mu_md": mu_md, "sigma_md": sigma_md}


def _transform_erf(md_test: np.ndarray, params: Params) -> np.ndarray:
    z = (md_test - params["mu_md"]) / (params["sigma_md"] * np.sqrt(2))
    score = erf(z)
    return np.clip(1.0 - score, 0.0, 1.0)


def _aggregate_erf(client_params: list[Params], w: np.ndarray) -> Params:
    mus = np.array([p["mu_md"] for p in client_params])
    sigmas = np.array([p["sigma_md"] for p in client_params])

    mu_global = float(np.dot(w, mus))

    # Law of total variance: within-group + between-group
    var_within = np.dot(w, sigmas**2)
    var_between = np.dot(w, (mus - mu_global) ** 2)
    sigma_global = float(np.sqrt(var_within + var_between))

    return {"method": "erf", "mu_md": mu_global, "sigma_md": max(sigma_global, 1e-9)}


# ---------------------------------------------------------------------------
# Logistic / sigmoid  (expert-threshold-aware)
# ---------------------------------------------------------------------------
# Centres a logistic sigmoid on the 95th percentile of the ID MD distribution
# (tau), treating that quantile as the compliance decision boundary.  The
# steepness parameter beta is derived from the interquartile spread so that
# the transition zone adapts to the tightness of the ID distribution.
#
# Shared parameters: tau (threshold), beta (steepness).
#
# Aggregation — weighted location, harmonic steepness:
# tau is a location (threshold): weighted mean is correct.
# beta is inversely proportional to spread (beta ~ 1/IQR), so the pooled
# spread is combined via the law of total variance on (1/beta), then inverted.
# ---------------------------------------------------------------------------


def _fit_logistic(md_ref: np.ndarray, tau: float | None = None, beta: float | None = None, **_: Any) -> Params:
    if tau is None:
        # Default: treat the 95th percentile of ID as the decision boundary.
        tau = float(np.percentile(md_ref, 95))
    if beta is None:
        # Steepness derived from the IQR so it adapts to distribution spread.
        iqr = float(np.percentile(md_ref, 75) - np.percentile(md_ref, 25))
        iqr = max(iqr, 1e-9)
        beta = float(4.0 / iqr)  # transition spans roughly one IQR around tau
    return {"method": "logistic", "tau": tau, "beta": beta}


def _transform_logistic(md_test: np.ndarray, params: Params) -> np.ndarray:
    score = 1.0 / (1.0 + np.exp(-params["beta"] * (md_test - params["tau"])))
    return np.clip(1.0 - score, 0.0, 1.0)


def _aggregate_logistic(client_params: list[Params], w: np.ndarray) -> Params:
    taus = np.array([p["tau"] for p in client_params])
    betas = np.array([p["beta"] for p in client_params])

    tau_global = float(np.dot(w, taus))

    # 1/beta ~ IQR (spread); pool spreads, then invert.
    spreads = 1.0 / np.maximum(betas, 1e-9)
    mean_spread = float(np.dot(w, spreads))
    var_spread = float(np.dot(w, (spreads - mean_spread) ** 2))  # between-client
    pooled_spread = float(np.sqrt(mean_spread**2 + var_spread))  # conservative
    beta_global = float(1.0 / max(pooled_spread, 1e-9))

    return {"method": "logistic", "tau": tau_global, "beta": beta_global}


# ---------------------------------------------------------------------------
# GMM CDF  (multimodal ID distributions)
# ---------------------------------------------------------------------------
# Fits a Gaussian Mixture Model on the 1-D MD distribution.  The resulting
# CDF is a weighted sum of Gaussian CDFs, providing a smooth multi-modal
# reference.  Use when the ID set contains visually distinct sub-populations
# (e.g. two scanner protocols at one site).
#
# Shared parameters: weights, means, stds (each a list of K floats).
#
# Aggregation — component concatenation and weight renormalisation:
# Each client's GMM is already a mixture model.  The global GMM is formed by
# concatenating all components and rescaling weights to sum to 1.  No new
# fitting is required; the number of global components is K × n_components.
# ---------------------------------------------------------------------------


def _fit_gmm(md_ref: np.ndarray, n_components: int = 2, **_: Any) -> Params:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(md_ref.reshape(-1, 1))
    weights = gmm.weights_.tolist()
    means = gmm.means_.flatten().tolist()
    stds = np.sqrt(gmm.covariances_).flatten().tolist()
    return {"method": "gmm", "weights": weights, "means": means, "stds": stds}


def _transform_gmm(md_test: np.ndarray, params: Params) -> np.ndarray:
    # Weighted sum of component Gaussian CDFs.
    score = np.zeros(len(md_test))
    for w, mu, sigma in zip(params["weights"], params["means"], params["stds"], strict=False):
        score += w * stats.norm.cdf(md_test, loc=mu, scale=max(sigma, 1e-9))
    return np.clip(1.0 - score, 0.0, 1.0)


def _aggregate_gmm(client_params: list[Params], w: np.ndarray) -> Params:
    global_weights: list[float] = []
    global_means: list[float] = []
    global_stds: list[float] = []

    for wk, p in zip(w, client_params, strict=False):
        local_weights = np.array(p["weights"])
        # Rescale each client's component weights by the client's global weight
        scaled = (wk * local_weights).tolist()
        global_weights.extend(scaled)
        global_means.extend(p["means"])
        global_stds.extend(p["stds"])

    # Renormalise (should already sum to 1 up to floating-point error)
    total = sum(global_weights)
    global_weights = [ww / total for ww in global_weights]

    return {"method": "gmm", "weights": global_weights, "means": global_means, "stds": global_stds}


# ===========================================================================
# Auto-selection via KS goodness-of-fit
# ===========================================================================


def _best_ks(md_ref: np.ndarray) -> str:
    """Return the method name whose fitted CDF best matches the empirical one."""
    candidates = {}

    # Gamma (on MD²)
    try:
        p = _fit_gamma(md_ref)
        ks, _ = kstest(md_ref**2, lambda x: gamma_dist.cdf(x, a=p["alpha"], scale=p["theta"]))
        candidates["gamma"] = ks
    except Exception:  # noqa: BLE001, S110 — candidate simply drops out of the KS comparison
        pass

    # Log-normal (on MD)
    try:
        p = _fit_lognorm(md_ref)
        ks, _ = kstest(md_ref, lambda x: lognorm_dist.cdf(x, s=p["sigma_log"], scale=np.exp(p["mu_log"])))
        candidates["lognorm"] = ks
    except Exception:  # noqa: BLE001, S110 — candidate simply drops out of the KS comparison
        pass

    # erf has no distributional assumption — add as fallback with a penalty
    candidates["erf"] = 1.0  # always available, preferred only if others fail

    if not candidates:
        return "erf"

    return min(candidates, key=lambda name: candidates[name])


def _auto_select(md_ref: np.ndarray, dispatchers: dict[str, Callable[..., Params]], **kwargs: Any) -> Params:
    best_method = _best_ks(md_ref)
    return dispatchers[best_method](md_ref, **kwargs)


# ===========================================================================
# Convenience: batch fit over multiple datasets / nodes
# ===========================================================================


def fit_all(md_ref_dict: dict[str, np.ndarray], method: Method = "gamma") -> dict[str, Params]:
    """Fit a normaliser for each node in a federated registry.

    Parameters
    ----------
    md_ref_dict : dict
        Mapping of node_id -> 1-D array of reference MD scores.
    method : str
        Normalisation method (same for all nodes, or "auto").

    Returns
    -------
    dict
        Mapping of node_id -> Params dict, ready to broadcast.

    Example
    -------
    >>> registry = fit_all({"siteA": md_A, "siteB": md_B}, method="gamma")
    >>> scores_A = transform(md_test_A, registry["siteA"])
    """
    return {node_id: fit(md_ref, method=method) for node_id, md_ref in md_ref_dict.items()}


# ===========================================================================
# Federation: combine per-client parameters into a single global normaliser
# ===========================================================================
#
# Each combination function accepts a list of per-client Params dicts and an
# optional list of client weights (e.g. number of reference samples at each
# client).  If weights are omitted, uniform weighting is assumed.
#
# All methods return a single Params dict in the same format as `fit`, so the
# result can be passed directly to `transform` without any further changes.
# ===========================================================================


def aggregate(
    client_params: list[Params],
    weights: list[float] | None = None,
    exact: bool = False,
) -> Params:
    """Combine per-client normaliser parameters into a single global normaliser.

    All clients must have been fitted with the same method.

    Parameters
    ----------
    client_params : list of Params
        One parameter dict per client, as returned by `fit`.
    weights : list of float, optional
        Per-client weights, e.g. number of ID reference samples.
        Defaults to uniform weighting.
    exact : bool, default False
        For "gamma" and "lognorm" only.  When False (default), applies
        law-of-total-variance moment matching to return a single distribution
        in the same 2-parameter format as a single-client fit.  When True,
        returns the exact weighted mixture (method "gamma_mixture" /
        "lognorm_mixture"), which grows with the number of clients.
        Has no effect on "erf", "logistic", or "gmm" methods.

    Returns
    -------
    Params
        A single global parameter dict, passable directly to `transform`.

    Example
    -------
    >>> global_params = aggregate([params_A, params_B, params_C],
    ...                           weights=[5000, 3000, 8000])
    >>> scores = transform(md_test, global_params)
    """
    if not client_params:
        raise ValueError("client_params must not be empty.")

    methods = {p["method"] for p in client_params}
    if len(methods) > 1:
        raise ValueError(
            f"All clients must use the same method; got: {methods}. Re-fit with a consistent method before aggregating."
        )

    method = next(iter(methods))
    K = len(client_params)

    # Normalise weights to sum to 1
    if weights is None:
        w = np.full(K, 1.0 / K)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != K:
            raise ValueError("len(weights) must equal len(client_params).")
        w = w / w.sum()

    dispatchers: dict[str, Callable[..., Params]] = {
        "gamma": _aggregate_gamma,
        "lognorm": _aggregate_lognorm,
        "erf": _aggregate_erf,
        "logistic": _aggregate_logistic,
        "gmm": _aggregate_gmm,
    }

    if method not in dispatchers:
        raise ValueError(f"No aggregation rule for method '{method}'.")

    if method in ("gamma", "lognorm"):
        return dispatchers[method](client_params, w, exact=exact)
    return dispatchers[method](client_params, w)
