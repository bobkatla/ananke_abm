import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass

def normalize_time_minutes(x: np.ndarray, T_minutes: int) -> np.ndarray:
    return np.clip(x / float(T_minutes), 0.0, 1.0)

@dataclass
class PurposePriors:
    # Fourier coefficients for lambda_p(t): size (2*K_time_prior+1,)
    time_fourier: np.ndarray
    # Log-normal params (mu, sigma) on normalized durations (Δ in (0,1])
    dur_mu_log: float
    dur_sigma_log: float
    # Simple moments
    mu_t: float
    sigma_t: float
    mu_d: float
    sigma_d: float

def _fit_time_fourier_from_hist(t_samples: np.ndarray, K_time_prior: int, n_bins:int=240) -> Tuple[np.ndarray, float, float]:
    """
    Fit a low-frequency Fourier series to an empirical presence density on [0,1].
    Returns:
        coeffs a: shape (1 + 2*K_time_prior,), order = [1, cos1..cosK, sin1..sinK]
        mu_t, sigma_t estimated from the smoothed histogram.
    """
    # Histogram presence density
    hist, edges = np.histogram(t_samples, bins=n_bins, range=(0.0,1.0), density=True)
    centers = 0.5*(edges[:-1]+edges[1:])  # (n_bins,)

    # Build design matrix for least squares on centers
    import math
    twopi = 2.0*math.pi
    X = [np.ones_like(centers)]
    for k in range(1, K_time_prior+1):
        X.append(np.cos(twopi*k*centers))
        X.append(np.sin(twopi*k*centers))
    X = np.stack(X, axis=1)  # (n_bins, 2K+1)
    # Least squares
    a, *_ = np.linalg.lstsq(X, hist, rcond=None)
    # Moments from histogram
    mu_t = float((centers * hist).sum() / (hist.sum()+1e-12))
    var_t = float(( (centers-mu_t)**2 * hist ).sum() / (hist.sum()+1e-12))
    sigma_t = max(var_t, 1e-8) ** 0.5
    return a.astype(np.float32), mu_t, sigma_t

def _fit_duration_lognormal(d_norm: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit log-normal on normalized durations in (0,1]. Clip tiny values to avoid -inf.
    Returns:
        mu_log, sigma_log, mean_d, std_d
    """
    eps = 1e-6
    x = np.clip(d_norm, eps, 1.0)
    logx = np.log(x)
    mu = float(np.mean(logx))
    sigma = float(np.std(logx) + 1e-8)
    mean_d = float(np.mean(x))
    std_d = float(np.std(x))
    return mu, sigma, mean_d, std_d

def derive_priors_from_activities(
    activities: pd.DataFrame,
    purposes_df: pd.DataFrame,
    T_minutes: int,
    K_time_prior: int
) -> Tuple[Dict[str, PurposePriors], List[str]]:
    """
    Build priors per purpose from schedules (ignoring persons).
    activities columns: ['persid','hhid','stopno','purpose','startime','total_duration']
    purposes_df columns include 'purpose','is_primary','can_open_close_day' etc.
    """
    purposes = purposes_df["purpose"].tolist()
    priors: Dict[str, PurposePriors] = {}
    # Collect normalized presence time samples for each purpose
    for p in purposes:
        segs = activities[activities["purpose"]==p]
        if len(segs)==0:
            a = np.zeros(1+2*K_time_prior, dtype=np.float32)
            mu_log, sigma_log, mean_d, std_d = -4.0, 0.5, 0.02, 0.01
            priors[p] = PurposePriors(
                time_fourier=a, dur_mu_log=mu_log, dur_sigma_log=sigma_log,
                mu_t=0.5, sigma_t=0.25, mu_d=mean_d, sigma_d=std_d
            )
            continue
        # time samples: proportional to duration
        rows = []
        for _,r in segs.iterrows():
            t0 = r["startime"] / float(T_minutes)
            d = r["total_duration"] / float(T_minutes)
            n = max(int(10*d*T_minutes), 3)  # proportional samples; at least 3
            ts = np.linspace(t0, min(t0+d,1.0), num=n, endpoint=False)
            rows.append(ts)
        t_samples = np.concatenate(rows) if rows else np.array([0.5], dtype=float)
        a, mu_t, sigma_t = _fit_time_fourier_from_hist(t_samples, K_time_prior)
        # durations
        d_norm = segs["total_duration"].to_numpy(dtype=float)/float(T_minutes)
        mu_log, sigma_log, mean_d, std_d = _fit_duration_lognormal(d_norm)
        priors[p] = PurposePriors(
            time_fourier=a, dur_mu_log=mu_log, dur_sigma_log=sigma_log,
            mu_t=mu_t, sigma_t=sigma_t, mu_d=mean_d, sigma_d=std_d
        )
    return priors, purposes
