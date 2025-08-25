import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


def normalize_time_minutes(x: np.ndarray, T_minutes: int) -> np.ndarray:
    return np.clip(x / float(T_minutes), 0.0, 1.0)


@dataclass
class PurposePriors:
    # Fourier coefficients for λ_p(clock): shape (2*K_clock_prior+1,)
    time_fourier: np.ndarray
    # Log-normal params on normalized durations (Δ in (0,1], normalized by allocation horizon)
    dur_mu_log: float
    dur_sigma_log: float
    # Moments (clock domain for t; allocation domain for d)
    mu_t: float
    sigma_t: float
    mu_d: float
    sigma_d: float


def _fit_time_fourier_from_hist(
    t_samples01: np.ndarray,
    K_clock_prior: int,
    n_bins: int = 288,  # 5-min bins over 24h by default
) -> Tuple[np.ndarray, float, float]:
    """
    Fit a low-frequency Fourier series to an empirical presence density on [0,1] (CLOCK domain).

    Returns:
        coeffs a: shape (1 + 2*K), order = [1, cos1..cosK, sin1..sinK]
        mu_t, sigma_t: moments computed from the smoothed histogram (in [0,1])
    """
    # Histogram density on [0,1]
    hist, edges = np.histogram(t_samples01, bins=n_bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])  # (n_bins,)

    # Design matrix for LS on periodic basis
    twopi = 2.0 * math.pi
    X = [np.ones_like(centers)]
    for k in range(1, K_clock_prior + 1):
        X.append(np.cos(twopi * k * centers))
        X.append(np.sin(twopi * k * centers))
    X = np.stack(X, axis=1)  # (n_bins, 2K+1)

    # Least squares fit
    a, *_ = np.linalg.lstsq(X, hist, rcond=None)

    # Moments from histogram
    denom = hist.sum() + 1e-12
    mu_t = float((centers * hist).sum() / denom)
    var_t = float(((centers - mu_t) ** 2 * hist).sum() / denom)
    sigma_t = max(var_t, 1e-12) ** 0.5

    return a.astype(np.float32), mu_t, sigma_t


def _fit_duration_lognormal(d_norm: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit log-normal on normalized durations in (0,1] (allocation-normalized).
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
    T_alloc_minutes: int,
    K_clock_prior: int,
    T_clock_minutes: int = 1440,
) -> Tuple[Dict[str, PurposePriors], List[str]]:
    """
    Build priors per purpose from schedules (ignoring persons).

    Time handling:
      - Presence prior λ_p is fit on CLOCK time (24h), using per-segment samples that
        are mapped to clock-of-day via modulo T_clock_minutes.
      - Duration stats are fit on ALLOCATION-normalized durations (divide by T_alloc_minutes).

    Expected columns in `activities`:
      ['persid','hhid','stopno','purpose','startime','total_duration']

    `purposes_df` must include at least 'purpose'; 'can_open_close_day' is optional.
    """
    purposes = purposes_df["purpose"].tolist()
    priors: Dict[str, PurposePriors] = {}

    for p in purposes:
        segs = activities[activities["purpose"] == p]
        if len(segs) == 0:
            # Safe defaults for unseen purposes
            a = np.zeros(1 + 2 * K_clock_prior, dtype=np.float32)
            mu_log, sigma_log, mean_d, std_d = -4.0, 0.5, 0.02, 0.01
            priors[p] = PurposePriors(
                time_fourier=a,
                dur_mu_log=mu_log,
                dur_sigma_log=sigma_log,
                mu_t=0.5,
                sigma_t=0.25,
                mu_d=mean_d,
                sigma_d=std_d,
            )
            continue

        # ---- CLOCK-time presence samples (weighted by duration) ----
        rows = []
        for _, r in segs.iterrows():
            start_min = float(r["startime"])
            dur_min = float(r["total_duration"])
            # sample times inside the segment, proportional to duration (at least 3)
            n = max(int(10 * dur_min), 3)
            ts_alloc_min = np.linspace(start_min, min(start_min + dur_min, float(T_alloc_minutes)), num=n, endpoint=False)
            # map to CLOCK minutes, normalize to [0,1]
            ts_clock01 = ((ts_alloc_min % float(T_clock_minutes)) / float(T_clock_minutes)).astype(np.float32)
            rows.append(ts_clock01)

        t_samples01 = np.concatenate(rows) if rows else np.array([0.5], dtype=np.float32)
        a, mu_t, sigma_t = _fit_time_fourier_from_hist(t_samples01, K_clock_prior)

        # ---- Duration stats on allocation-normalized durations ----
        d_norm = segs["total_duration"].to_numpy(dtype=float) / float(T_alloc_minutes)
        mu_log, sigma_log, mean_d, std_d = _fit_duration_lognormal(d_norm)

        priors[p] = PurposePriors(
            time_fourier=a,
            dur_mu_log=mu_log,
            dur_sigma_log=sigma_log,
            mu_t=mu_t,
            sigma_t=sigma_t,
            mu_d=mean_d,
            sigma_d=std_d,
        )

    return priors, purposes
