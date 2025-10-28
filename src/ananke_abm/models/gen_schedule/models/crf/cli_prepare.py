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

    # Load VAE checkpoint
    obj = torch.load(vae_ckpt, map_location="cpu")
    cfg = obj["cfg"]
    meta = obj["meta"]
    P, L = len(meta["purpose_map"]), meta["L"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, meta).to(device)
    model.load_state_dict(obj["model"])
    model.eval()

    # Load dataset splits
    split_obj = torch.load(split_pt, weights_only=False)
    train_dataset = split_obj["train_dataset"]
    val_dataset = split_obj["val_dataset"]

    def extract_logits(dataset, name):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch_labels in tqdm(loader, desc=f"CRF prep {name}"):
                batch_labels = batch_labels.to(device)
                logits_batch, _, _ = model(batch_labels)  # (B,T,P)
                all_logits.append(logits_batch.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
        U = np.concatenate(all_logits, axis=0)  # (N,T,P)
        Y = np.concatenate(all_labels, axis=0)  # (N,T)
        return U, Y

    # Train set
    U_train, Y_train = extract_logits(train_dataset, "train")
    np.savez_compressed(
        os.path.join(outdir, "crf_train.npz"),
        U=U_train.astype(np.float32),
        Y=Y_train.astype(np.int64),
    )

    # Val set
    U_val, Y_val = extract_logits(val_dataset, "val")
    np.savez_compressed(
        os.path.join(outdir, "crf_val.npz"),
        U=U_val.astype(np.float32),
        Y=Y_val.astype(np.int64),
    )

    # Meta
    meta_out = {
        "P": P,
        "L": L,
        "purpose_map": meta["purpose_map"],
        "vae_ckpt": vae_ckpt,
        "split_pt": split_pt,
        "batch_size": batch_size,
    }
    with open(os.path.join(outdir, "crf_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[prepare-crf-data] Saved CRF-ready npz files to {outdir}")
