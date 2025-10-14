import re
import math
from pathlib import Path
from typing import List, Dict, Tuple

import click
import numpy as np
import torch
import matplotlib.pyplot as plt

from ananke_abm.models.traj_syn.vae.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_syn.vae.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_syn.core.utils_bases import make_alloc_grid


def _decode_reg_cfg(cfg_dec: Dict) -> Tuple[float, Dict[str, float]]:
    """Support both dict and dataclass-like reg_cfg."""
    reg = cfg_dec.get("reg_cfg", {})
    if isinstance(reg, dict):
        coeff_l2_global = float(reg.get("coeff_l2_global", 0.0))
        coeff_l2_per_purpose = reg.get("coeff_l2_per_purpose", {})
    else:
        coeff_l2_global = float(getattr(reg, "coeff_l2_global", 0.0))
        coeff_l2_per_purpose = getattr(reg, "coeff_l2_per_purpose", {})
    return coeff_l2_global, coeff_l2_per_purpose


def _infer_dims_from_pds_sd(pds_sd: Dict[str, torch.Tensor], P_fallback: int) -> Tuple[int, int, int]:
    """
    Infer (feature_dim, d_p, hidden_width) from a PDS state_dict:
    - feature_dim: phi_p.shape[1]
    - d_p: out_features of the LAST mlp.*.weight
    - hidden_width: out_features of the FIRST mlp.*.weight
    """
    # feature dim from phi_p
    if "phi_p" not in pds_sd:
        raise RuntimeError("Checkpoint PDS has no 'phi_p' param; cannot infer feature dimension.")
    P, feature_dim = pds_sd["phi_p"].shape
    if P_fallback is not None and P != P_fallback:
        # Not critical, just sanity
        pass

    # collect all mlp.*.weight layers (linear layers)
    lin = []
    pat = re.compile(r"^mlp\.(\d+)\.weight$")
    for k, v in pds_sd.items():
        m = pat.match(k)
        if m and v.ndim == 2:
            idx = int(m.group(1))
            out_features, in_features = v.shape
            lin.append((idx, out_features, in_features))
    if not lin:
        # Fallback if no mlp.*.weight present (unlikely)
        d_p = feature_dim
        hidden = feature_dim
        return feature_dim, d_p, hidden

    lin_sorted = sorted(lin, key=lambda x: x[0])
    first_idx, first_out, _ = lin_sorted[0]
    last_idx, last_out, _ = lin_sorted[-1]

    hidden = first_out           # width of first hidden layer
    d_p = last_out               # embedding dimension (output of last MLP layer)

    return feature_dim, d_p, hidden


@torch.no_grad()
def rebuild_modules_from_ckpt(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    purposes: List[str] = [str(p) for p in ck["purposes"]]
    cfg_time: Dict = ck["configs"]["time"]
    cfg_basis: Dict = ck["configs"]["basis"]
    cfg_dec: Dict = ck["configs"]["dec"]
    cfg_vae: Dict = ck["configs"]["vae"]

    # -------- infer dims correctly from PDS checkpoint --------
    pds_sd = ck["model_state"]["pds"]
    feature_dim, d_p, hidden = _infer_dims_from_pds_sd(pds_sd, P_fallback=len(purposes))
    P = len(purposes)

    # -------- rebuild PDS with inferred shapes and load weights --------
    # phi placeholder must be [P, feature_dim]
    pds = PurposeDistributionSpace(torch.empty(P, feature_dim), d_p=d_p, hidden=hidden).to(device)
    pds.load_state_dict(pds_sd, strict=True)
    K_clock = int(cfg_basis["K_clock_prior"])
    pds.set_clock_prior_K(K_clock)
    pds.eval()

    # -------- rebuild decoder and load weights --------
    coeff_l2_global, coeff_l2_per_purpose = _decode_reg_cfg(cfg_dec)
    dec = TimeFieldDecoder(
        P=P,
        m_latent=int(cfg_vae["latent_dim"]),
        d_p=d_p,  # <-- must match the inferred embedding size
        K_decoder_time=int(cfg_basis["K_decoder_time"]),
        alpha_prior=float(cfg_dec["alpha_prior"]),
        time_cfg=cfg_time,
        idx2purpose=purposes,
        alpha_init_per_purpose=None,
        coeff_l2_global=coeff_l2_global,
        coeff_l2_per_purpose=coeff_l2_per_purpose,
    ).to(device)
    dec.load_state_dict(ck["model_state"]["dec"], strict=True)
    dec.eval()

    # -------- time grids & clock prior on eval grid --------
    T_alloc_minutes = int(cfg_time["ALLOCATION_HORIZON_MINS"])
    step_mins = int(cfg_time["VALID_GRID_MINS"])
    T_clock_minutes = int(cfg_time["T_clock_minutes"])

    t_alloc_minutes_eval, t_alloc01_eval = make_alloc_grid(
        T_alloc_minutes=T_alloc_minutes, step_minutes=step_mins, device=device, dtype=torch.float32
    )
    L_eval = int(t_alloc_minutes_eval.numel())

    loglam_eval = pds.lambda_log_on_alloc_grid(
        t_alloc_minutes_eval, T_clock_minutes=T_clock_minutes
    )  # [P,L]

    meta = {
        "purposes": purposes,
        "T_alloc_minutes": T_alloc_minutes,
        "step_minutes": step_mins,
        "L_eval": L_eval,
        "t_eval01": t_alloc01_eval.detach().cpu().numpy(),
    }
    return pds, dec, loglam_eval, meta


@torch.no_grad()
def sample_unaries(dec: TimeFieldDecoder,
                   pds: PurposeDistributionSpace,
                   loglam_eval: torch.Tensor,
                   latent_dim: int,
                   num_samples: int,
                   device: torch.device):
    B = num_samples
    e_p = pds()  # [P, d_p]
    s = torch.randn(B, latent_dim, device=device)
    z = s / (s.norm(dim=-1, keepdim=True) + 1e-8)

    theta = dec.utilities_on_grid(z, e_p, loglam_eval, grid_type="eval", endpoint_mask=None)  # [B,P,L]
    alpha = dec.alpha.view(1, -1, 1)            # [1,P,1]
    prior_term = alpha * loglam_eval.unsqueeze(0)  # [B,P,L]
    latent_term = theta - prior_term
    probs = torch.softmax(theta, dim=1)         # [B,P,L]
    return theta, probs, prior_term, latent_term


def plot_mean_with_bands(x01, curves, label, out_png):
    mean = np.mean(curves, axis=0)
    p10 = np.percentile(curves, 10, axis=0)
    p90 = np.percentile(curves, 90, axis=0)

    plt.figure(figsize=(10, 3))
    plt.plot(x01, mean, label=f"{label} (mean)")
    plt.fill_between(x01, p10, p90, alpha=0.25, label="10–90%")
    plt.xlabel("Allocation time (0–1)")
    plt.ylabel("Utility (θ)")
    plt.title(f"Unaries θ: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_component_pair(x01, prior_curve, latent_curve, label, out_png):
    plt.figure(figsize=(10, 3))
    plt.plot(x01, prior_curve, label="clock prior (α·log λ_p(clock))")
    plt.plot(x01, latent_curve, label="latent-driven")
    plt.xlabel("Allocation time (0–1)")
    plt.ylabel("Utility component")
    plt.title(f"θ decomposition: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_heatmap(theta_one, purposes, x01, out_png, vmax=None):
    plt.figure(figsize=(10, 2 + 0.3*len(purposes)))
    im = plt.imshow(
        theta_one, aspect='auto', interpolation='nearest',
        extent=[x01[0], x01[-1], 0, len(purposes)], vmin=None, vmax=vmax
    )
    plt.colorbar(im, fraction=0.02, pad=0.02, label="θ")
    plt.yticks(np.arange(len(purposes)) + 0.5, purposes)
    plt.xlabel("Allocation time (0–1)")
    plt.title("Unaries θ heatmap (one sampled z)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


@click.command()
@click.option("--ckpt", type=click.Path(exists=True), required=True, help="Path to vae_epXXXX.pt")
@click.option("--outdir", type=click.Path(), required=True)
@click.option("--num_samples", type=int, default=512, help="Number of z samples to average")
@click.option("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
@click.option("--heatmap_examples", type=int, default=3, help="Save this many θ heatmaps for random z")
def main(ckpt, outdir, num_samples, device, heatmap_examples):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    pds, dec, loglam_eval, meta = rebuild_modules_from_ckpt(ckpt, dev)
    purposes = meta["purposes"]
    t01 = meta["t_eval01"]
    L = int(meta["L_eval"])
    latent_dim = int(dec.m)  # saved with decoder

    theta, probs, prior_term, latent_term = sample_unaries(
        dec, pds, loglam_eval, latent_dim=latent_dim, num_samples=num_samples, device=dev
    )

    theta_np = theta.cpu().numpy()                                  # [B,P,L]
    prior_np = (prior_term.expand_as(theta)).cpu().numpy()          # [B,P,L]
    latent_np = latent_term.cpu().numpy()                           # [B,P,L]

    # Save arrays
    np.save(out / "theta_samples.npy", theta_np)
    np.save(out / "theta_prior_component.npy", prior_np)
    np.save(out / "theta_latent_component.npy", latent_np)
    np.save(out / "t_eval01.npy", t01)

    # Per-purpose mean±band
    for p_idx, p_name in enumerate(purposes):
        curves = theta_np[:, p_idx, :]  # [B,L]
        plot_mean_with_bands(t01, curves, label=p_name, out_png=out / f"unaries_{p_idx:02d}_{p_name}.png")

    # Decomposition means
    theta_mean = theta_np.mean(axis=0)
    prior_mean = prior_np.mean(axis=0)
    latent_mean = latent_np.mean(axis=0)
    for p_idx, p_name in enumerate(purposes):
        plot_component_pair(
            t01, prior_mean[p_idx], latent_mean[p_idx],
            label=p_name, out_png=out / f"unaries_decomp_{p_idx:02d}_{p_name}.png"
        )

    # A few heatmaps
    vmax = np.percentile(theta_np, 99.0)
    B = theta_np.shape[0]
    rng = np.random.default_rng(123)
    for i, b in enumerate(rng.choice(B, size=min(heatmap_examples, B), replace=False)):
        plot_heatmap(theta_np[b], purposes, t01, out_png=out / f"unaries_heatmap_{i+1}.png", vmax=vmax)

    print(f"[done] wrote unary diagnostics to: {out.resolve()}")


if __name__ == "__main__":
    main()
