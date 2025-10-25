import click
import os
import json
import numpy as np
import torch
from ananke_abm.models.gen_schedule.utils.seed import set_seed
from ananke_abm.models.gen_schedule.models.vae import ScheduleVAE


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


def sample(ckpt_path, num_samples, outprefix, seed, csv_max_persons):
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

    # Init model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScheduleVAE(L=num_time_bins, P=P,
                        z_dim=cfg["model"]["z_dim"],
                        emb_dim=cfg["model"]["emb_dim"]).to(device)
    model.load_state_dict(ckpt_obj["model"])
    model.eval()

    # Reproducibility
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Sampling loop (batched for memory safety)
    batch_size_generate = 1024
    all_generated_batches = []
    logits_running_sum = None
    logits_batch_count = 0

    # also track latent stats
    latent_sum = torch.zeros(latent_dim, dtype=torch.float64)
    latent_sq_sum = torch.zeros(latent_dim, dtype=torch.float64)
    latent_total_count = 0

    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            current_bs = min(batch_size_generate, remaining)
            remaining -= current_bs

            # sample latent
            Z = torch.randn(current_bs, latent_dim, device=device, dtype=model_dtype)

            # forward decode (we only need decoder, but calling decoder directly is fine)
            U_logits = model.decoder(Z)  # (current_bs, num_time_bins, P)

            # argmax to get discrete schedule
            y_labels = torch.argmax(U_logits, dim=-1)  # (current_bs, num_time_bins)
            all_generated_batches.append(y_labels.cpu().numpy().astype(np.int64))

            # accumulate mean logits across individuals for sanity-plot (unaries)
            batch_mean_logits = U_logits.mean(dim=0).cpu().numpy()  # (num_time_bins, P)
            if logits_running_sum is None:
                logits_running_sum = batch_mean_logits.copy()
            else:
                logits_running_sum += batch_mean_logits
            logits_batch_count += 1

            # accumulate latent stats
            latent_sum += Z.sum(dim=0).cpu().to(torch.float64)
            latent_sq_sum += (Z**2).sum(dim=0).cpu().to(torch.float64)
            latent_total_count += current_bs

    generated_labels = np.concatenate(all_generated_batches, axis=0)  # (num_samples, num_time_bins)
    U_mean_logits = logits_running_sum / max(1, logits_batch_count)   # (num_time_bins, P)

    # summarize latent sampling stats
    latent_mean = (latent_sum / max(1, latent_total_count)).numpy()
    latent_var = (latent_sq_sum / max(1, latent_total_count)).numpy() - latent_mean**2
    latent_std = np.sqrt(np.maximum(latent_var, 1e-12))
    Z_stats = np.stack([latent_mean, latent_std], axis=0)  # (2, latent_dim)

   
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
    }
    with open(meta_json_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta_out, f_meta, indent=2)

    click.echo(f"Saved machine artifact to {npz_path}")
    click.echo(f"Saved metadata to {meta_json_path}")
    click.echo(f"Saved human-readable preview CSV to {preview_csv_path}")
