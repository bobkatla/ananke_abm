from pathlib import Path
from typing import List, Dict

import click
import torch

from ananke_abm.models.traj_syn.core.data_utils.randomness import set_seed
from ananke_abm.models.traj_syn.core.data_utils.sanitize import sanitize_theta
from ananke_abm.models.traj_syn.core.utils_bases import make_alloc_grid
from ananke_abm.models.traj_syn.vae.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_syn.vae.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_syn.vae.encoder import TrajEncoderGRU
from ananke_abm.models.traj_syn.eval.traj_decode_utils import decoded_to_activities_df



@click.command()
@click.option("--ckpt", type=click.Path(exists=True), required=True)
@click.option("--out_csv", type=click.Path(), required=True)
@click.option("--num_gen", type=int, default=10000)
@click.option("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
def main(ckpt: str, out_csv: str, num_gen: int, device: str):
    set_seed(123)
    dev = torch.device(device)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)

    # --- config extraction ---
    purposes: List[str] = ck["purposes"]
    cfg_time: Dict = ck["configs"]["time"]
    cfg_basis: Dict = ck["configs"]["basis"]
    cfg_pep: Dict = ck["configs"]["pep"]
    cfg_dec: Dict = ck["configs"]["dec"]
    cfg_vae: Dict = ck["configs"]["vae"]

    time_cfg = cfg_time
    T_alloc_minutes = int(time_cfg["ALLOCATION_HORIZON_MINS"])
    step_mins = int(time_cfg["VALID_GRID_MINS"])
    T_clock_minutes = int(time_cfg["T_clock_minutes"])

    # -------------------------------
    # Build modules safely
    # -------------------------------

    # --- PurposeDistributionSpace ---
    pds_state = ck["model_state"]["pds"]
    feat_dim = pds_state["phi_p"].shape[1]  # e.g., 18
    d_p = int(cfg_pep["d_p"]) if isinstance(cfg_pep, dict) else int(getattr(cfg_pep, "d_p"))
    hidden = int(cfg_pep.get("hidden", 64)) if isinstance(cfg_pep, dict) else int(getattr(cfg_pep, "hidden", 64))

    placeholder_phi = torch.zeros(len(purposes), feat_dim)
    pds = PurposeDistributionSpace(placeholder_phi, d_p=d_p, hidden=hidden).to(dev)
    pds.load_state_dict(pds_state, strict=True)

    K_clock_prior = (cfg_basis["K_clock_prior"] 
                    if isinstance(cfg_basis, dict) 
                    else getattr(cfg_basis, "K_clock_prior", 6))
    pds.set_clock_prior_K(int(K_clock_prior))

    # --- Helper to safely access nested values whether dict or dataclass ---
    def _get(obj, key, default):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    reg_cfg = _get(cfg_dec, "reg_cfg", None)
    coeff_l2_global = _get(reg_cfg, "coeff_l2_global", 0.0)
    coeff_l2_per_purpose = _get(reg_cfg, "coeff_l2_per_purpose", {})

    # --- Decoder ---
    dec_state = ck["model_state"]["dec"].copy()
    # remove cached grid-dependent Phi_* keys
    for k in list(dec_state.keys()):
        if k.startswith("Phi_"):
            del dec_state[k]

    dec = TimeFieldDecoder(
        P=len(purposes),
        m_latent=int(_get(cfg_vae, "latent_dim", 16)),
        d_p=d_p,
        K_decoder_time=int(_get(cfg_basis, "K_decoder_time", 4)),
        alpha_prior=float(_get(cfg_dec, "alpha_prior", 1.0)),
        time_cfg=time_cfg,
        idx2purpose=purposes,
        alpha_init_per_purpose=None,
        coeff_l2_global=float(coeff_l2_global),
        coeff_l2_per_purpose=coeff_l2_per_purpose,
    ).to(dev)
    dec.load_state_dict(dec_state, strict=False)

    # --- Encoder ---
    enc_state = ck["model_state"]["enc"]
    enc = TrajEncoderGRU(
        d_p=d_p,
        K_time_token_clock=4,
        K_time_token_alloc=0,
        K_dur_token=4,
        m_latent=int(_get(cfg_vae, "latent_dim", 16)),
        gru_hidden=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ).to(dev)
    enc.load_state_dict(enc_state, strict=True)

    pds.eval()
    dec.eval()
    enc.eval()

    # -------------------------------
    # Sampling
    # -------------------------------
    t_alloc_minutes_eval, t_alloc01_eval = make_alloc_grid(
        T_alloc_minutes=T_alloc_minutes, step_minutes=step_mins, device=dev, dtype=torch.float32
    )
    with torch.no_grad():
        loglam_eval = pds.lambda_log_on_alloc_grid(t_alloc_minutes_eval, T_clock_minutes=T_clock_minutes)  # [P,L]
    L_eval = loglam_eval.shape[1]

    decoded = []
    with torch.no_grad():
        for start in range(0, num_gen, 512):
            cur = min(512, num_gen - start)
            s = torch.randn(cur, int(_get(cfg_vae, "latent_dim", 16)), device=dev)
            z = s / (s.norm(dim=-1, keepdim=True) + 1e-8)
            e_p = pds()
            theta = dec.utilities_on_grid(z, e_p, loglam_eval, grid_type="eval", endpoint_mask=None)  # [B,P,L]
            theta = sanitize_theta(theta)
            y = torch.argmax(theta, dim=1)  # [B,L]

            t = t_alloc01_eval.detach().cpu().numpy()
            for b in range(y.shape[0]):
                yb = y[b].detach().cpu().numpy()
                if L_eval == 0:
                    decoded.append([])
                    continue
                segs = []
                start_i = 0
                for i in range(1, L_eval):
                    if yb[i] != yb[i - 1]:
                        p = int(yb[i - 1])
                        t0 = float(t[start_i])
                        t1 = float(t[i])
                        segs.append((p, t0, max(t1 - t0, 0.0)))
                        start_i = i
                p = int(yb[-1])
                t0 = float(t[start_i])
                t1 = float(t[-1])
                segs.append((p, t0, max(t1 - t0, 0.0)))
                decoded.append(segs)

    gen_df = decoded_to_activities_df(decoded, purposes, T_alloc_minutes, start_persid=0, prefix="vae")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    gen_df.to_csv(out_csv, index=False)
    click.echo(f"Wrote VAE-only samples to: {out_csv}")


if __name__ == "__main__":
    main()
