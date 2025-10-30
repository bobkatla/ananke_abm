import os
import json
import click
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from ananke_abm.models.gen_schedule.models.factory import build_model

@click.command("prepare-crf-data")
@click.option("--vae_ckpt", type=click.Path(exists=True), required=True,
              help="Trained VAE checkpoint (.pt)")
@click.option("--split_pt", type=click.Path(exists=True), required=True,
              help="Stored train/val dataset split (.pt) with SubSet objects.")
@click.option("--outdir", type=click.Path(), required=True,
              help="Directory to save CRF-ready npz files.")
@click.option("--batch_size", type=int, default=64)
def prepare_crf_data(vae_ckpt, split_pt, outdir, batch_size):
    os.makedirs(outdir, exist_ok=True)

    # Load frozen VAE
    obj = torch.load(vae_ckpt, map_location="cpu")
    cfg = obj["cfg"]
    meta = obj["meta"]
    P, L = len(meta["purpose_map"]), meta["L"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, meta).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load split (train/val as Subset datasets)
    split_obj = torch.load(split_pt, weights_only=False)
    train_dataset = split_obj["train_dataset"]
    val_dataset   = split_obj["val_dataset"]

    def extract_logits(dataset, tag):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch_labels in tqdm(loader, desc=f"CRF prep {tag}"):
                # batch_labels: (B,T) long from your saved SubSet
                batch_labels = batch_labels.to(device)
                # forward through VAE to get logits
                logits_batch, _, _ = model(batch_labels)  # (B,T,P)
                all_logits.append(logits_batch.cpu().numpy())  # float32 later
                all_labels.append(batch_labels.cpu().numpy())  # int64 later
        U = np.concatenate(all_logits, axis=0)  # (N,T,P)
        Y = np.concatenate(all_labels, axis=0)  # (N,T)
        return U, Y

    # Build CRF train set
    U_train, Y_train = extract_logits(train_dataset, "train")
    np.savez_compressed(
        os.path.join(outdir, "crf_train.npz"),
        U=U_train.astype(np.float32),
        Y=Y_train.astype(np.int64),
    )

    # Build CRF val set
    U_val, Y_val = extract_logits(val_dataset, "val")
    np.savez_compressed(
        os.path.join(outdir, "crf_val.npz"),
        U=U_val.astype(np.float32),
        Y=Y_val.astype(np.int64),
    )

     # Resolve home_idx
    purpose_map = meta["purpose_map"]
    if "Home" in purpose_map:
        home_idx = int(purpose_map["Home"])
    else:
        raise ValueError("Purpose map does not contain 'Home' purpose.")
    
    # Meta useful for downstream sanity/debug
    meta_out = {
        "P": P,
        "L": L,
        "purpose_map": meta["purpose_map"],
        "grid_min": meta["grid_min"],
        "horizon_min": meta["horizon_min"],
        "vae_ckpt": vae_ckpt,
        "split_pt": split_pt,
        "home_idx": home_idx,
    }
    with open(os.path.join(outdir, "crf_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[prepare-crf-data] wrote crf_train.npz, crf_val.npz, crf_meta.json to {outdir}")
