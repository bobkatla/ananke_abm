
import click
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ananke_abm.models.gen_schedule.utils.cfg import load_config, ensure_dir
from ananke_abm.models.gen_schedule.utils.seed import set_seed
from ananke_abm.models.gen_schedule.utils.ckpt import save_checkpoint
from ananke_abm.models.gen_schedule.dataio.rasterize import prepare_from_csv
from ananke_abm.models.gen_schedule.models.vae import ScheduleVAE, kl_gaussian
from ananke_abm.models.gen_schedule.losses.reg import time_total_variation
from ananke_abm.models.gen_schedule.evals.report import make_report, save_report
from ananke_abm.models.gen_schedule.evals.metrics import tod_marginals, bigram_matrix, minutes_share
from ananke_abm.models.gen_schedule.viz.plots import plot_unaries_mean, plot_minutes_share, plot_tod_marginal, plot_bigram_delta


class GridDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.Y = d["Y"].astype(np.int64)
    def __len__(self): return self.Y.shape[0]
    def __getitem__(self, i):
        y = self.Y[i]
        return torch.from_numpy(y)

@click.group()
def main():
    pass

@main.command()
@click.option("--activities", type=click.Path(exists=True), required=True)
@click.option("--grid", type=int, default=10)
@click.option("--out", type=click.Path(), required=True)
def prepare(activities, grid, out):
    prepare_from_csv(activities, out, grid_min=grid)
    click.echo(f"Prepared grid at {out}")


@main.command("fit")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), default="runs")
@click.option("--run", type=str, required=True)
@click.option("--seed", type=int, default=123)
def fit(config, output_dir, run, seed):
    class GridDataset(Dataset):
        def __init__(self, npz_path):
            d = np.load(npz_path)
            self.Y = d["Y"].astype(np.int64)
        def __len__(self):
            return self.Y.shape[0]
        def __getitem__(self, i):
            y = self.Y[i]
            return torch.from_numpy(y)

    cfg = load_config(config)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = os.path.join(output_dir, run)
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "checkpoints"))
    ensure_dir(os.path.join(outdir, "plots"))

    data_npz_path = cfg["data"]["npz"]
    meta_path = data_npz_path.replace(".npz", "_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)

    purpose_map = meta["purpose_map"]                  # {purpose_name: index}
    home_idx = purpose_map.get("Home", None)
    if home_idx is None:
        # fallback: most common first label in dataset
        tmp_ref = np.load(data_npz_path)["Y"].astype(np.int64)
        vals, counts = np.unique(tmp_ref[:, 0], return_counts=True)
        home_idx = int(vals[np.argmax(counts)])

    num_purposes = len(purpose_map)
    num_time_bins = meta["L"]

    full_dataset = GridDataset(data_npz_path)
    num_total = len(full_dataset)
    num_val = max(1, int(num_total * cfg["train"]["val_frac"]))
    num_train = num_total - num_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(cfg["train"]["batch_size"], max(1, num_train)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(cfg["train"]["batch_size"], max(1, num_val)),
        shuffle=False,
        drop_last=False,
    )

    model = ScheduleVAE(
        L=num_time_bins,
        P=num_purposes,
        z_dim=cfg["model"]["z_dim"],
        emb_dim=cfg["model"]["emb_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_val_loss = np.inf
    last_ckpt_path = os.path.join(outdir, "checkpoints", "last.pt")
    best_ckpt_path = os.path.join(outdir, "checkpoints", "best_val.pt")

    num_epochs = cfg["train"]["epochs"]
    warmup_epochs = int(max(1, num_epochs * cfg["train"]["beta_warm_frac"]))
    beta_target = cfg["train"]["beta_target"]

    lambda_tv = cfg["train"]["lambda_tv"]
    lambda_home = cfg["train"].get("lambda_home", 0.1)

    def start_end_home_loss(logits_batch, home_class_index):
        # logits_batch: (B, T, P)
        # we want high prob of home at t=0 and t=T-1
        B, T, P = logits_batch.shape
        if T < 2:
            return torch.tensor(0.0, device=logits_batch.device, dtype=logits_batch.dtype)
        logp0 = F.log_softmax(logits_batch[:, 0, :], dim=-1)      # (B,P)
        logpT = F.log_softmax(logits_batch[:, -1, :], dim=-1)     # (B,P)
        loss0 = -logp0[:, home_class_index].mean()
        lossT = -logpT[:, home_class_index].mean()
        return (loss0 + lossT) * 0.5

    for epoch in range(1, num_epochs + 1):
        model.train()
        beta = beta_target * min(1.0, epoch / max(1, warmup_epochs))

        total_train_loss = 0.0
        total_train_ce = 0.0
        total_train_kl = 0.0
        total_train_tv = 0.0
        total_train_home = 0.0
        num_train_batches = 0

        for batch_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch_labels = batch_labels.to(device)  # (B,T)
            logits_batch, mu, logvar = model(batch_labels)  # logits_batch: (B,T,P)

            ce_loss = F.cross_entropy(
                logits_batch.permute(0, 2, 1),  # (B,P,T)
                batch_labels,                   # (B,T)
                reduction="mean"
            )

            kl_loss = kl_gaussian(mu, logvar)
            tv_loss = time_total_variation(logits_batch)

            home_loss = start_end_home_loss(logits_batch, home_idx)

            loss = ce_loss + beta * kl_loss + lambda_tv * tv_loss + lambda_home * home_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            total_train_loss += float(loss.item())
            total_train_ce += float(ce_loss.item())
            total_train_kl += float(kl_loss.item())
            total_train_tv += float(tv_loss.item())
            total_train_home += float(home_loss.item())
            num_train_batches += 1

        if num_train_batches > 0:
            avg_train_loss = total_train_loss / num_train_batches
            avg_train_ce = total_train_ce / num_train_batches
            avg_train_kl = total_train_kl / num_train_batches
            avg_train_tv = total_train_tv / num_train_batches
            avg_train_home = total_train_home / num_train_batches
        else:
            avg_train_loss = 0.0
            avg_train_ce = 0.0
            avg_train_kl = 0.0
            avg_train_tv = 0.0
            avg_train_home = 0.0

        model.eval()
        total_val_loss = 0.0
        total_val_ce = 0.0
        total_val_kl = 0.0
        total_val_tv = 0.0
        total_val_home = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                logits_batch, mu, logvar = model(batch_labels)

                ce_loss = F.cross_entropy(
                    logits_batch.permute(0, 2, 1),
                    batch_labels,
                    reduction="mean"
                )
                kl_loss = kl_gaussian(mu, logvar)
                tv_loss = time_total_variation(logits_batch)
                home_loss = start_end_home_loss(logits_batch, home_idx)

                val_loss = ce_loss + beta * kl_loss + lambda_tv * tv_loss + lambda_home * home_loss

                total_val_loss += float(val_loss.item())
                total_val_ce += float(ce_loss.item())
                total_val_kl += float(kl_loss.item())
                total_val_tv += float(tv_loss.item())
                total_val_home += float(home_loss.item())
                num_val_batches += 1

        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_ce = total_val_ce / num_val_batches
            avg_val_kl = total_val_kl / num_val_batches
            avg_val_tv = total_val_tv / num_val_batches
            avg_val_home = total_val_home / num_val_batches
        else:
            avg_val_loss = 0.0
            avg_val_ce = 0.0
            avg_val_kl = 0.0
            avg_val_tv = 0.0
            avg_val_home = 0.0

        save_checkpoint(
            {"model": model.state_dict(), "meta": meta, "cfg": cfg},
            last_ckpt_path,
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                {"model": model.state_dict(), "meta": meta, "cfg": cfg},
                best_ckpt_path,
            )

        log_record = {
            "epoch": epoch,
            "beta": beta,
            "train_loss": avg_train_loss,
            "train_ce": avg_train_ce,
            "train_kl": avg_train_kl,
            "train_tv": avg_train_tv,
            "train_home": avg_train_home,
            "val_loss": avg_val_loss,
            "val_ce": avg_val_ce,
            "val_kl": avg_val_kl,
            "val_tv": avg_val_tv,
            "val_home": avg_val_home,
            "num_train_batches": num_train_batches,
            "num_val_batches": num_val_batches,
        }
        print(json.dumps(log_record))

    print("Done. Checkpoints saved:", best_ckpt_path, last_ckpt_path)


@main.command("sample-population")
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=True,
              help="Trained checkpoint to sample from.")
@click.option("--num-samples", default=10000, show_default=True,
              help="How many synthetic individuals to generate.")
@click.option("--outprefix", type=click.Path(), required=True,
              help="Prefix for output files. Will emit <prefix>.npz, <prefix>_meta.json, <prefix>_preview.csv.")
@click.option("--seed", default=123, show_default=True,
              help="Random seed for reproducibility.")
@click.option("--csv-max-persons", default=200, show_default=True,
              help="Max number of generated individuals to export into the human-readable preview CSV.")
def sample_population(ckpt_path, num_samples, outprefix, seed, csv_max_persons):
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

    # -------------- CSV preview reconstruction --------------
    def decode_person_to_segments(seq_row, person_id_prefix, grid_minutes):
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

    preview_rows = []
    preview_count = min(csv_max_persons, generated_labels.shape[0])
    for i in range(preview_count):
        person_id_prefix = f"gen_{i:06d}"
        person_rows = decode_person_to_segments(
            generated_labels[i],
            person_id_prefix=person_id_prefix,
            grid_minutes=grid_min,
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


@main.command("eval-population")
@click.option("--samples", "samples_npz_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>.npz from sample-population.")
@click.option("--samples-meta", "samples_meta_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>_meta.json from sample-population.")
@click.option("--reference", "reference_grid_path", type=click.Path(exists=True), required=True,
              help="Reference prepared grid npz (e.g. runs/data/train_10min.npz).")
@click.option("--out-json", "out_json_path", type=click.Path(), required=True,
              help="Where to write metrics report JSON.")
def eval_population(samples_npz_path, samples_meta_path, reference_grid_path, out_json_path):
    """
    Evaluate a previously sampled synthetic population against real data.
    No model or GPU required.
    """
    # load generated population
    synth_npz = np.load(samples_npz_path)
    generated_labels = synth_npz["Y_generated"].astype(np.int64)  # (N, L)

    # load metadata for purpose_map etc.
    with open(samples_meta_path, "r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)
    purpose_map = meta["purpose_map"]  # {purpose_name: index}

    # load reference (real) grid data
    ref_npz = np.load(reference_grid_path)
    reference_labels = ref_npz["Y"].astype(np.int64)
    # reference time-of-day marginals (precomputed in prepare())
    reference_tod_path = reference_grid_path.replace(".npz", "_tod.npy")
    reference_tod = np.load(reference_tod_path) if os.path.exists(reference_tod_path) else None

    # compute report
    report_dict = make_report(
        Y_synth=generated_labels,
        Y_ref=reference_labels,
        purpose_map=purpose_map,
        ref_tod=reference_tod,
    )
    save_report(report_dict, out_json_path)

    click.echo(json.dumps(report_dict, indent=2))
    click.echo(f"Saved metrics report to {out_json_path}")


@main.command("viz-population")
@click.option("--samples", "samples_npz_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>.npz from sample-population.")
@click.option("--samples-meta", "samples_meta_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>_meta.json from sample-population.")
@click.option("--outdir", "outdir_path", type=click.Path(), required=True,
              help="Directory to write plots.")
@click.option("--reference", "reference_grid_path", type=click.Path(), default="",
              help="Optional reference prepared grid npz (real data) to overlay.")
def viz_population(samples_npz_path, samples_meta_path, outdir_path, reference_grid_path):
    """
    Produce sanity plots for a sampled population:
    - Mean unaries over time (U_mean_logits)
    - Minutes share bars (synth vs ref)
    - Time-of-day marginals per purpose (synth vs ref)
    - Bigram delta heatmap
    No model or GPU required.
    """
    ensure_dir(outdir_path)
    # load generated population artifact
    synth_npz = np.load(samples_npz_path)
    generated_labels = synth_npz["Y_generated"].astype(np.int64)     # (N, L)
    U_mean_logits = synth_npz["U_mean_logits"].astype(np.float32)    # (L, P)

    with open(samples_meta_path, "r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)

    purpose_map = meta["purpose_map"]  # {purpose_name: index}
    purpose_names_ordered = meta["purpose_names_ordered"]  # [idx0_name, idx1_name, ...]
    P = len(purpose_names_ordered)

    # compute synth stats
    synth_minutes_share = minutes_share(generated_labels, P)
    synth_tod = tod_marginals(generated_labels, P)
    synth_bigram = bigram_matrix(generated_labels, P)

    # load reference stats if provided
    if reference_grid_path and os.path.exists(reference_grid_path):
        ref_npz = np.load(reference_grid_path)
        reference_labels = ref_npz["Y"].astype(np.int64)
        ref_minutes_share = minutes_share(reference_labels, P)
        ref_tod = tod_marginals(reference_labels, P)
        ref_bigram = bigram_matrix(reference_labels, P)
    else:
        # fallback: compare synth to itself just to make plots work
        ref_minutes_share = synth_minutes_share
        ref_tod = synth_tod
        ref_bigram = synth_bigram

    # 1. Mean unaries (logits over time)
    plot_unaries_mean(
        U_mean_logits,                    # (L,P)
        purpose_names_ordered,            # names aligned with P
        os.path.join(outdir_path, "unaries")
    )
    # 2. Minutes share bar chart
    plot_minutes_share(
        share_syn=synth_minutes_share,
        share_ref=ref_minutes_share,
        purposes=purpose_names_ordered,
        outpath=os.path.join(outdir_path, "minutes_share.png"),
    )
    # 3. Time-of-day marginal curves per purpose
    plot_tod_marginal(
        m_ref=ref_tod,
        m_syn=synth_tod,
        purposes=purpose_names_ordered,
        outdir=os.path.join(outdir_path, "tod"),
    )
    # 4. Bigram delta heatmap
    plot_bigram_delta(
        B_ref=ref_bigram,
        B_syn=synth_bigram,
        purposes=purpose_names_ordered,
        outdir=os.path.join(outdir_path, "bigrams"),
    )
    click.echo(f"Saved plots to {outdir_path}")

