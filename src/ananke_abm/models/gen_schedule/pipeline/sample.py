import click
import os
import json
import numpy as np
import torch
from ananke_abm.models.gen_schedule.utils.seed import set_seed
from ananke_abm.models.gen_schedule.models.factory import build_model


 # -------------- CSV preview reconstruction --------------
def decode_person_to_segments(seq_row, person_id_prefix, grid_minutes, inverse_purpose_map):
    """
    Convert a single generated timeline (length L of int labels)
    into segments with columns:
    persid, stopno, purpose, starttime, total_duration
    """
    out_rows = []
    current_purpose_idx = seq_row[0]
    current_start_bin = 0
    stopno = 0

    for t in range(1, len(seq_row)):
        if seq_row[t] != current_purpose_idx:
            duration_bins = t - current_start_bin
            out_rows.append({
                "persid": person_id_prefix,
                "stopno": stopno,
                "purpose": inverse_purpose_map[current_purpose_idx],
                "starttime": current_start_bin * grid_minutes,
                "total_duration": duration_bins * grid_minutes,
            })
            stopno += 1
            current_purpose_idx = seq_row[t]
            current_start_bin = t

    # flush last segment
    last_duration_bins = len(seq_row) - current_start_bin
    out_rows.append({
        "persid": person_id_prefix,
        "stopno": stopno,
        "purpose": inverse_purpose_map[current_purpose_idx],
        "starttime": current_start_bin * grid_minutes,
        "total_duration": last_duration_bins * grid_minutes,
    })
    return out_rows


def crf_decode_batch(logits_batch, crf_model, enforce_nonhome):
    """
    logits_batch: (B, T, P) torch.FloatTensor on same device as crf_model
    returns (B, T) long
    """
    assert crf_model is not None, "crf_model must be provided for CRF decoding"
    with torch.no_grad():
        return crf_model.decode(logits_batch, enforce_nonhome=enforce_nonhome)


def sample(
    ckpt_path,
    num_samples,
    outprefix,
    seed,
    csv_max_persons,
    decode_mode="argmax",
    crf_path=None,
    enforce_nonhome=False,
    reject_all_home: bool = False,
):
    """
    Generate a synthetic population from a trained checkpoint and save:
    - <prefix>.npz              : machine artifact with generated_labels and mean logits
    - <prefix>_meta.json        : metadata (purpose map, grid info, etc.)
    - <prefix>_preview.csv      : first K individuals in human-readable segment format
    """
    # Load checkpoint
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt_obj["cfg"]
    meta = ckpt_obj["meta"]

    purpose_map = meta["purpose_map"]                # dict {purpose_name: index}
    inverse_purpose_map = {v: k for k, v in purpose_map.items()}  # {index: purpose_name}
    purpose_names_ordered = [inverse_purpose_map[i] for i in range(len(inverse_purpose_map))]
    grid_min = meta["grid_min"]
    horizon_min = meta["horizon_min"]
    num_time_bins = meta["L"]
    latent_dim = cfg["model"]["z_dim"]
    P = len(purpose_map)
    home_idx = purpose_map.get("Home", None)

    # Init model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, meta).to(device)
    model.load_state_dict(ckpt_obj["model"])
    model.eval()

    # Optional CRF model (only if decode_mode == "crf")
    crf_model = None
    if decode_mode == "crf":
        if crf_path is None or crf_path == "":
            raise ValueError("decode_mode='crf' requires crf_path")
        from ananke_abm.models.gen_schedule.models.crf.model import TransitionCRF
        crf_ckpt = torch.load(crf_path, map_location="cpu")
        crf_home_idx = crf_ckpt.get("home_idx", None)
        assert crf_home_idx == home_idx, \
            f"CRF home_idx {crf_home_idx} does not match VAE home_idx {home_idx}"
        crf_model = TransitionCRF(num_purposes=P, home_idx=crf_home_idx).to(device)
        crf_model.load_state_dict(crf_ckpt["A_state_dict"])
        crf_model.eval()
        for p in crf_model.parameters():
            p.requires_grad_(False)

    # Reproducibility
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Sampling loop (batched for memory safety)
    batch_size_generate = 1024
    all_generated_batches = []

    # running stats for logits across *individuals*
    U_running_mean_flat = None          # shape (T*P,) float64
    U_running_M2_flat = None            # shape (T*P,) float64
    U_running_count = 0                 # scalar: how many individuals we've seen

    # latent stats
    latent_sum = torch.zeros(latent_dim, dtype=torch.float64)
    latent_sq_sum = torch.zeros(latent_dim, dtype=torch.float64)
    latent_total_count = 0

    model_dtype = next(model.parameters()).dtype

    def update_running_stats_batch(U_logits_batch_cpu_float64):
        """
        U_logits_batch_cpu_float64: (B, T, P) float64 on CPU
        Updates global running mean / M2 / count over individuals.
        """
        nonlocal U_running_mean_flat, U_running_M2_flat, U_running_count

        B, T, P = U_logits_batch_cpu_float64.shape
        batch_flat = U_logits_batch_cpu_float64.reshape(B, T * P)  # (B, TP)

        for b in range(B):
            x = batch_flat[b]  # (TP,)
            if U_running_count == 0:
                U_running_mean_flat = x.clone()                     # init mean
                U_running_M2_flat = torch.zeros_like(x, dtype=torch.float64)
                U_running_count = 1
            else:
                U_running_count += 1
                delta = x - U_running_mean_flat
                U_running_mean_flat = U_running_mean_flat + delta / U_running_count
                delta2 = x - U_running_mean_flat
                U_running_M2_flat = U_running_M2_flat + delta * delta2

    # --------- MAIN SAMPLING LOOP (with rejection) ---------
    with torch.no_grad():
        remaining = num_samples   ### we track how many accepted samples we still need

        while remaining > 0:
            current_bs = batch_size_generate

            # sample latent
            Z = torch.randn(current_bs, latent_dim, device=device, dtype=model_dtype)

            # decode to logits
            U_logits = model.decoder(Z)  # (B, num_time_bins, P)

            # choose decoding method
            if decode_mode == "argmax":
                y_labels = torch.argmax(U_logits, dim=-1)  # (B, num_time_bins)
            elif decode_mode == "crf":
                y_labels = crf_decode_batch(
                    U_logits, crf_model, enforce_nonhome=enforce_nonhome
                )  # (B, num_time_bins)
            else:
                raise ValueError(f"Unknown decode_mode: {decode_mode}")

            # --------- NEW: rejection of "all-Home" days ----------
            if reject_all_home and home_idx is not None:
                # keep any person for whom at least one bin != Home
                # y_labels: (B, T)
                keep_mask = (y_labels != home_idx).any(dim=1)  # (B,)
            else:
                keep_mask = torch.ones(
                    y_labels.size(0), dtype=torch.bool, device=y_labels.device
                )

            keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
            keep_count = int(keep_indices.numel())

            if keep_count == 0:
                # nothing accepted in this batch, try again
                continue

            # If we only need 'remaining' more samples, cap to that
            if keep_count > remaining:
                keep_indices = keep_indices[:remaining]
                keep_count = remaining

            # Subselect accepted individuals
            y_keep = y_labels[keep_indices]        # (keep_count, T)
            U_keep = U_logits[keep_indices]        # (keep_count, T, P)
            Z_keep = Z[keep_indices]               # (keep_count, latent_dim)

            all_generated_batches.append(
                y_keep.cpu().numpy().astype(np.int64)
            )

            # update logits running stats on accepted subset
            update_running_stats_batch(U_keep.cpu().to(torch.float64))

            # accumulate latent stats on accepted subset
            latent_sum += Z_keep.sum(dim=0).cpu().to(torch.float64)
            latent_sq_sum += (Z_keep ** 2).sum(dim=0).cpu().to(torch.float64)
            latent_total_count += keep_count

            remaining -= keep_count

        # stack all generated label sequences, should be exactly num_samples
        generated_labels = np.concatenate(all_generated_batches, axis=0)  # (>=num_samples, T)
        generated_labels = generated_labels[:num_samples]  # just in case

        # finalize logits stats
        if U_running_count > 0:
            U_mean_logits = U_running_mean_flat.reshape(num_time_bins, P)  # (T,P)
            if U_running_count > 1:
                U_var_logits = (U_running_M2_flat / (U_running_count - 1)).reshape(num_time_bins, P)
            else:
                U_var_logits = torch.zeros(num_time_bins, P, dtype=torch.float64)
            U_std_logits = torch.sqrt(torch.clamp(U_var_logits, min=0.0))
        else:
            U_mean_logits = torch.zeros(num_time_bins, P, dtype=torch.float64)
            U_std_logits = torch.zeros(num_time_bins, P, dtype=torch.float64)

        U_mean_logits = U_mean_logits.numpy().astype(np.float32)
        U_std_logits = U_std_logits.numpy().astype(np.float32)

    # summarize latent sampling stats
    latent_mean = (latent_sum / max(1, latent_total_count)).numpy()
    latent_var = (latent_sq_sum / max(1, latent_total_count)).numpy() - latent_mean ** 2
    latent_std = np.sqrt(np.maximum(latent_var, 1e-12))
    Z_stats = np.stack([latent_mean, latent_std], axis=0).astype(np.float32)  # (2, latent_dim)

    # --------- preview CSV, npz, meta ---------
    preview_rows = []
    preview_count = min(csv_max_persons, generated_labels.shape[0])
    for i in range(preview_count):
        person_id_prefix = f"gen_{i:06d}"
        person_rows = decode_person_to_segments(
            generated_labels[i],
            person_id_prefix=person_id_prefix,
            grid_minutes=grid_min,
            inverse_purpose_map=inverse_purpose_map,
        )
        preview_rows.extend(person_rows)

    # write preview CSV
    preview_csv_path = f"{outprefix}_preview.csv"
    os.makedirs(os.path.dirname(preview_csv_path), exist_ok=True)
    import csv
    with open(preview_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["persid", "stopno", "purpose", "starttime", "total_duration"]
        )
        writer.writeheader()
        for row in preview_rows:
            writer.writerow(row)

    # -------------- save npz (machine artifact) --------------
    npz_path = f"{outprefix}.npz"
    np.savez_compressed(
        npz_path,
        Y_generated=generated_labels.astype(np.int64),
        U_mean_logits=U_mean_logits.astype(np.float32),
        U_std_logits=U_std_logits.astype(np.float32),
        Z_stats=Z_stats.astype(np.float32),
    )

    # -------------- save meta json --------------
    meta_json_path = f"{outprefix}_meta.json"
    meta_out = {
        "purpose_map": purpose_map,  # {purpose_name: index}
        "purpose_names_ordered": purpose_names_ordered,  # [idx0_name, idx1_name, ...]
        "grid_min": grid_min,
        "horizon_min": horizon_min,
        "num_time_bins": num_time_bins,
        "latent_dim": latent_dim,
        "num_samples": int(num_samples),
        "seed": int(seed),
        "vae_ckpt": ckpt_path,
        "decode_mode": decode_mode,
        "crf_path": crf_path,
        "pds_method": cfg["model"].get("method", "auto_pds"),
        "reject_all_home": bool(reject_all_home),
    }
    with open(meta_json_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta_out, f_meta, indent=2)

    click.echo(f"[sample:{decode_mode}] Saved machine artifact to {npz_path}")
    click.echo(f"[sample:{decode_mode}] Saved metadata to {meta_json_path}")
    click.echo(f"[sample:{decode_mode}] Saved human-readable preview CSV to {preview_csv_path}")
